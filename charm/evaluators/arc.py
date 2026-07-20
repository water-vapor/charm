from pathlib import Path
import json
import os
from typing import Any
import numpy as np
import torch
import torch.distributed as dist

from charm.transforms import inverse_chain, grid_hash


class ARCEvaluator:
    """
    ARC Evaluator - generic, works with any model.

    Aggregates predictions across augmentations via voting.
    Supports optional Q-value weighted voting.
    Generates pass@K metrics and Kaggle-style submission.

    The evaluator uses inverse_chain to invert augmentations:
    - online_aug: transforms applied at runtime (canvas, flatten, etc.)
    - offline_aug: transforms from preprocessing (dihedral, color perm, scale)
    """

    def __init__(
        self,
        ground_truth_inputs: dict[str, list[np.ndarray]],
        ground_truth_outputs: dict[str, list] | None = None,
        pass_ks: tuple[int, ...] = (1, 2, 5, 10, 100, 1000),
        submission_k: int = 100,
        aggregated_voting: bool = True,
        recent_window_size: int | None = None,
        exp_decay: bool = False,
        decay_half_life: float = 10.0,
    ):
        """
        Args:
            ground_truth_inputs: {puzzle_id: [test_input_grid_0, ...]} for hashing
            ground_truth_outputs: {puzzle_id: [test_output_grid_0, ...]} for evaluation
                                  if None, result returns {} metrics but submission is still produced
            pass_ks: K values for pass@K metrics
            submission_k: top K predictions to include in submission
            aggregated_voting: if True, accumulate predictions across eval passes
            recent_window_size: If set, keep only the most recent N eval events when
                aggregating. None keeps full history.
            exp_decay: If True, apply exponential time-decay over aggregated events.
            decay_half_life: event-age half-life used when exp_decay=True.
        """
        self.ground_truth_inputs = ground_truth_inputs
        self.pass_ks = pass_ks
        self.submission_k = submission_k
        self.aggregated_voting = aggregated_voting
        if recent_window_size is not None and recent_window_size < 1:
            raise ValueError("recent_window_size must be >= 1 when provided")
        self.recent_window_size = recent_window_size
        self.exp_decay = exp_decay
        self.decay_half_life = decay_half_life
        if self.exp_decay and self.decay_half_life <= 0:
            raise ValueError("decay_half_life must be > 0 when exp_decay=True")

        self._gt_outputs = ground_truth_outputs

        # precompute ground truth input hashes and output hashes
        # {puzzle_id: [(input_hash, output_hash), ...]}
        # output_hash is None if no solution provided
        self._gt_hashes = {}
        for puzzle_id, inputs in ground_truth_inputs.items():
            pairs = []
            if self._gt_outputs is None:
                # inference mode: no ground truth outputs
                for inp in inputs:
                    pairs.append((grid_hash(inp), None))
            else:
                # evaluation mode: solution must have all puzzles with matching output counts
                if puzzle_id not in self._gt_outputs:
                    raise ValueError(f"solution file missing puzzle: {puzzle_id}")
                puzzle_outputs = self._gt_outputs[puzzle_id]
                if len(puzzle_outputs) != len(inputs):
                    raise ValueError(
                        f"puzzle {puzzle_id}: expected {len(inputs)} outputs, got {len(puzzle_outputs)}"
                    )
                for inp, out in zip(inputs, puzzle_outputs):
                    inp_hash = grid_hash(inp)
                    out_hash = grid_hash(np.array(out, dtype=np.uint8))
                    pairs.append((inp_hash, out_hash))
            self._gt_hashes[puzzle_id] = pairs

        # {pred_hash: pred_grid} for reconstructing submission
        self._pred_grids = {}
        # event-aware prediction storage (single source of truth)
        # {event_id: {(puzzle_id, input_hash): [(pred_hash, q_value), ...]}}
        self._events: dict[Any, dict[tuple[str, str], list[tuple[str, float]]]] = {}
        self._active_event_id: Any | None = None
        self._event_counter = 0

    def _window_event_order(self, event_order: list[Any] | None = None) -> list[Any]:
        """Return the event order filtered to recent_window_size (without mutating history)."""
        if event_order is None:
            event_order = list(self._events.keys())
        if self.recent_window_size is None:
            return list(event_order)
        return list(event_order[-self.recent_window_size:])

    def _effective_event_order(self, event_order: list[Any] | None = None) -> list[Any]:
        if event_order is None:
            event_order = list(self._events.keys())
        event_order = list(event_order)
        if not self.aggregated_voting:
            return event_order
        return self._window_event_order(event_order)

    def _event_decay_weights(self, event_order: list[Any]) -> dict[Any, float] | None:
        """
        Optional per-event exponential decay weights.

        Newest event has weight 1.0. Older events decay as:
            w(age) = 2^(-age / half_life)
        """
        if not self.aggregated_voting or not self.exp_decay:
            return None
        if not event_order:
            return {}

        n = len(event_order)
        weights: dict[Any, float] = {}
        for i, event_id in enumerate(event_order):
            age = n - 1 - i
            weights[event_id] = 2.0 ** (-age / self.decay_half_life)
        return weights

    def _rebuild_flat_predictions_from_events(
        self,
        events: dict[Any, dict[tuple[str, str], list[tuple[str, float]]]] | None = None,
        event_order: list[Any] | None = None,
        event_weights: dict[Any, float] | None = None,
    ) -> dict[tuple[str, str], list[tuple[str, float] | tuple[str, float, float]]]:
        if events is None:
            events = self._events
        if event_order is None:
            event_order = list(events.keys())

        merged: dict[tuple[str, str], list[tuple[str, float] | tuple[str, float, float]]] = {}
        for event_id in event_order:
            bucket = events.get(event_id, {})
            event_weight = 1.0 if event_weights is None else event_weights.get(event_id, 1.0)
            for key, votes in bucket.items():
                if key not in merged:
                    merged[key] = []
                if event_weights is None:
                    merged[key].extend(votes)
                    continue

                merged[key].extend((pred_hash, q_value, event_weight) for pred_hash, q_value in votes)
        return merged

    def clear_eval(self, event_id: Any | None = None):
        """Prepare buffers for a new eval event."""
        if not self.aggregated_voting:
            self._events = {}
            self._pred_grids = {}

        if event_id is None:
            event_id = f"auto_{self._event_counter}"
            self._event_counter += 1

        if event_id in self._events:
            del self._events[event_id]

        self._events[event_id] = {}
        self._active_event_id = event_id

    def _gather_predictions(self):
        """
        Gather predictions from all ranks in distributed mode.

        Returns:
            (is_main, predictions, pred_grids, events, event_order)
            - is_main: True on rank 0 (or non-distributed), False otherwise
            - predictions: merged predictions dict on main rank, empty on others
            - pred_grids: merged pred grid map on main rank, empty on others
        """
        if not dist.is_initialized():
            event_order = list(self._events.keys())
            effective_event_order = self._effective_event_order(event_order)
            event_weights = self._event_decay_weights(effective_event_order)
            preds = self._rebuild_flat_predictions_from_events(
                events=self._events,
                event_order=effective_event_order,
                event_weights=event_weights,
            )
            return True, preds, self._pred_grids, self._events, event_order

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        all_data = [None] * world_size if rank == 0 else None
        dist.gather_object(
            (self._events, self._pred_grids),
            all_data,
            dst=0
        )
        if rank != 0:
            return False, {}, {}, {}, []

        merged_events: dict[Any, dict[tuple[str, str], list[tuple[str, float]]]] = {}
        merged_grids: dict[str, np.ndarray] = {}
        for rank_events, grids in all_data:
            if isinstance(rank_events, dict):
                for event_id, bucket in rank_events.items():
                    if event_id not in merged_events:
                        merged_events[event_id] = {}
                    if not isinstance(bucket, dict):
                        continue
                    for key, votes in bucket.items():
                        if key not in merged_events[event_id]:
                            merged_events[event_id][key] = []
                        merged_events[event_id][key].extend(votes)
            merged_grids.update(grids)

        merged_event_order = list(merged_events.keys())
        effective_event_order = self._effective_event_order(merged_event_order)
        event_weights = self._event_decay_weights(effective_event_order)
        merged_preds = self._rebuild_flat_predictions_from_events(
            events=merged_events,
            event_order=effective_event_order,
            event_weights=event_weights,
        )
        return True, merged_preds, merged_grids, merged_events, merged_event_order

    @staticmethod
    def _vote_and_rank(predictions):
        """Vote on predictions and return sorted by (count, avg_q) descending."""
        vote_map = {}
        if predictions and len(predictions[0]) == 3:
            for pred_hash, q, weight in predictions:
                if pred_hash not in vote_map:
                    vote_map[pred_hash] = [0.0, 0.0]
                vote_map[pred_hash][0] += weight
                vote_map[pred_hash][1] += q * weight
        else:
            for pred_hash, q in predictions:
                if pred_hash not in vote_map:
                    vote_map[pred_hash] = [0.0, 0.0]
                vote_map[pred_hash][0] += 1.0
                vote_map[pred_hash][1] += q
        return sorted(
            vote_map.items(),
            key=lambda x: (x[1][0], x[1][1] / max(x[1][0], 1e-12)),
            reverse=True
        )

    def update(
        self,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        q_values: torch.Tensor | None = None,
    ):
        """
        Process batch predictions.

        Args:
            batch: from dataloader, must have 'puzzle_ids', 'offline_aug', 'online_aug', 'inputs'
            predictions: (B, seq_len) or (B, H, W) - raw model output
            q_values: (B,) optional confidence scores
        """
        puzzle_ids = batch["puzzle_ids"]
        offline_augs = batch["offline_aug"]
        online_augs = batch["online_aug"]
        inputs = batch["inputs"].cpu().numpy()
        preds = predictions.cpu().numpy()

        if q_values is not None:
            q_vals = torch.sigmoid(q_values).cpu().numpy()
        else:
            q_vals = np.ones(len(puzzle_ids), dtype=np.float32)

        for i, puzzle_id in enumerate(puzzle_ids):
            online_aug = online_augs[i]
            offline_aug = offline_augs[i]

            # inverse online augmentation (flatten → canvas → scale, etc.)
            inp_grid = inverse_chain(inputs[i], online_aug)
            pred_grid = inverse_chain(preds[i], online_aug)

            # inverse offline augmentation (dihedral, color perm, scale)
            inp_grid = inverse_chain(inp_grid, offline_aug)
            pred_grid = inverse_chain(pred_grid, offline_aug)

            # hash grids
            inp_hash = grid_hash(inp_grid)
            pred_hash = grid_hash(pred_grid)

            # store prediction
            key = (puzzle_id, inp_hash)
            if self._active_event_id is None:
                self.clear_eval()
            assert self._active_event_id is not None
            event_bucket = self._events[self._active_event_id]
            if key not in event_bucket:
                event_bucket[key] = []
            event_bucket[key].append((pred_hash, float(q_vals[i])))

            # store grid for submission reconstruction
            if pred_hash not in self._pred_grids:
                self._pred_grids[pred_hash] = pred_grid

    def result(
        self,
        save_path: str | Path | None = None,
        return_submission: bool = False,
    ):
        """
        Finalize evaluation: gather, compute metrics, and optionally save submission.

        Collective in distributed mode: all ranks must call.

        Args:
            save_path: If provided, saves submission.json. If the path ends with ".json",
                it is treated as a file path; otherwise as a directory.
            return_submission: If True, also return submission dict on rank 0.

        Returns:
            On rank 0 (or non-distributed):
              - metrics dict
              - (metrics, submission) if return_submission
            On other ranks:
              - {}
              - ({}, {}) if return_submission
        """
        is_main, predictions, pred_grids, _events, _event_order = self._gather_predictions()
        if not is_main:
            if return_submission:
                return ({}, {})
            return {}

        submission = self._build_submission(predictions, pred_grids)

        if save_path is not None:
            save_path = str(save_path)
            if save_path.endswith(".json"):
                out_path = save_path
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
            else:
                os.makedirs(save_path, exist_ok=True)
                out_path = os.path.join(save_path, "submission.json")
            with open(out_path, "w") as f:
                json.dump(submission, f)

        metrics = {} if self._gt_outputs is None else self._compute_pass_at_k(predictions)
        if return_submission:
            return metrics, submission
        return metrics

    def _build_submission(self, predictions, pred_grids) -> dict[str, list]:
        """Build Kaggle-style submission dict from merged predictions."""
        submission = {}
        for puzzle_id, gt_pairs in self._gt_hashes.items():
            puzzle_submission = []
            for inp_hash, _ in gt_pairs:
                key = (puzzle_id, inp_hash)
                preds = predictions.get(key, [])
                sorted_preds = self._vote_and_rank(preds)

                attempts = []
                for pred_hash, _ in sorted_preds[:self.submission_k]:
                    grid = pred_grids.get(pred_hash)
                    if grid is not None:
                        attempts.append(grid.tolist())

                while len(attempts) < self.submission_k:
                    if attempts:
                        attempts.append(attempts[0])
                    else:
                        attempts.append([[0]])

                puzzle_submission.append({
                    f"attempt_{i + 1}": grid for i, grid in enumerate(attempts)
                })

            submission[puzzle_id] = puzzle_submission
        return submission

    def _compute_pass_at_k(self, predictions) -> dict[str, float]:
        """Compute pass@K metrics from merged predictions."""
        correct = {k: 0.0 for k in self.pass_ks}
        total_puzzles = 0

        for puzzle_id, gt_pairs in self._gt_hashes.items():
            puzzle_correct = {k: 0 for k in self.pass_ks}
            num_test = len(gt_pairs)

            for inp_hash, out_hash in gt_pairs:
                if out_hash is None:
                    continue

                key = (puzzle_id, inp_hash)
                preds = predictions.get(key, [])
                if not preds:
                    continue

                sorted_preds = self._vote_and_rank(preds)
                for k in self.pass_ks:
                    for pred_hash, _ in sorted_preds[:k]:
                        if pred_hash == out_hash:
                            puzzle_correct[k] += 1
                            break

            if num_test > 0:
                for k in self.pass_ks:
                    correct[k] += puzzle_correct[k] / num_test
                total_puzzles += 1

        if total_puzzles > 0:
            for k in self.pass_ks:
                correct[k] /= total_puzzles

        return {f"pass@{k}": correct[k] for k in self.pass_ks}
