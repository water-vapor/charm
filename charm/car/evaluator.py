"""Exhaustive-D4 evaluation and seen/unseen-combination metrics for CAR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import tqdm

from charm.car.metadata import parse_car_puzzle_id
from charm.evaluators.arc import ARCEvaluator
from charm.training_utils.eval_helpers import batch_to_device


class CAREvaluator(ARCEvaluator):
    """ARC voting with CAR task groups and per-horizon metrics."""

    def __init__(
        self,
        *,
        ground_truth_inputs,
        ground_truth_outputs,
        pass_ks=(1, 2, 5, 10, 100, 1000),
        aggregated_voting: bool = True,
        recent_window_size: int | None = None,
        exp_decay: bool = False,
        decay_half_life: float = 10.0,
        puzzle_groups: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        super().__init__(
            ground_truth_inputs=ground_truth_inputs,
            ground_truth_outputs=ground_truth_outputs,
            pass_ks=pass_ks,
            submission_k=0,
            aggregated_voting=aggregated_voting,
            recent_window_size=recent_window_size,
            exp_decay=exp_decay,
            decay_half_life=decay_half_life,
        )
        self.puzzle_groups = puzzle_groups
        self._horizon_by_puzzle = {
            puzzle_id: factors.horizon
            for puzzle_id in ground_truth_inputs
            if (factors := parse_car_puzzle_id(puzzle_id)) is not None
        }

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        # CAR computes metrics only and never reconstructs ARC submissions.
        self._pred_grids.clear()

    def _compute_group(
        self,
        predictions,
        puzzle_ids,
    ) -> dict[str, float]:
        correct = {k: 0.0 for k in self.pass_ks}
        total_puzzles = 0
        for puzzle_id in puzzle_ids:
            gt_pairs = self._gt_hashes[puzzle_id]
            puzzle_correct = {k: 0 for k in self.pass_ks}
            for input_hash, output_hash in gt_pairs:
                if output_hash is None:
                    continue
                ranked = self._vote_and_rank(
                    predictions.get((puzzle_id, input_hash), [])
                )
                for k in self.pass_ks:
                    if any(
                        prediction_hash == output_hash
                        for prediction_hash, _ in ranked[:k]
                    ):
                        puzzle_correct[k] += 1
            if gt_pairs:
                for k in self.pass_ks:
                    correct[k] += puzzle_correct[k] / len(gt_pairs)
                total_puzzles += 1
        if total_puzzles:
            correct = {k: value / total_puzzles for k, value in correct.items()}
        return {f"pass@{k}": correct[k] for k in self.pass_ks}

    def result(
        self,
        save_path: str | Path | None = None,
        return_submission: bool = False,
    ):
        if save_path is not None:
            raise ValueError("CAR evaluation does not produce ARC submissions")
        is_main, predictions, _, _, _ = self._gather_predictions()
        if not is_main:
            return ({}, {}) if return_submission else {}

        metrics = {}
        groups = self.puzzle_groups or {"": tuple(self._gt_hashes)}
        for group, puzzle_ids in groups.items():
            prefix = f"{group}/" if group else ""
            metrics.update(
                {
                    f"{prefix}{name}": value
                    for name, value in self._compute_group(
                        predictions,
                        puzzle_ids,
                    ).items()
                }
            )
            horizons = sorted(
                {
                    self._horizon_by_puzzle[puzzle_id]
                    for puzzle_id in puzzle_ids
                    if puzzle_id in self._horizon_by_puzzle
                }
            )
            for horizon in horizons:
                horizon_ids = [
                    puzzle_id
                    for puzzle_id in puzzle_ids
                    if self._horizon_by_puzzle.get(puzzle_id) == horizon
                ]
                metrics.update(
                    {
                        f"{prefix}by_horizon/H={horizon}/{name}": value
                        for name, value in self._compute_group(
                            predictions,
                            horizon_ids,
                        ).items()
                    }
                )
        return (metrics, {}) if return_submission else metrics


def matching_car_evaluator(
    evaluator: CAREvaluator,
    *,
    aggregated_voting: bool,
) -> CAREvaluator:
    return CAREvaluator(
        ground_truth_inputs=evaluator.ground_truth_inputs,
        ground_truth_outputs=evaluator._gt_outputs,
        pass_ks=tuple(evaluator.pass_ks),
        aggregated_voting=aggregated_voting,
        recent_window_size=evaluator.recent_window_size,
        exp_decay=evaluator.exp_decay if aggregated_voting else False,
        decay_half_life=evaluator.decay_half_life,
        puzzle_groups=evaluator.puzzle_groups,
    )


def build_car_evaluator(config: Any, train_dataset, eval_dataset) -> CAREvaluator:
    ground_truth_inputs = eval_dataset.get_noaug_test_inputs()
    ground_truth_outputs = eval_dataset.get_noaug_test_outputs()
    train_task_ids = set(train_dataset.car_task_id_to_int)
    eval_task_ids = set(ground_truth_inputs)
    unseen_task_ids = eval_task_ids - train_task_ids
    puzzle_groups = None
    if unseen_task_ids:
        puzzle_groups = {
            "seen": tuple(sorted(eval_task_ids & train_task_ids)),
            "unseen": tuple(sorted(unseen_task_ids)),
        }
    return CAREvaluator(
        ground_truth_inputs=ground_truth_inputs,
        ground_truth_outputs=ground_truth_outputs,
        pass_ks=(1, 2, 5, 10, 100, 1000),
        recent_window_size=config.eval_recent_window_size,
        exp_decay=config.eval_exp_decay,
        decay_half_life=config.eval_decay_half_life,
        puzzle_groups=puzzle_groups,
    )


def run_car_evaluation(
    train_state: Any,
    evaluator: CAREvaluator,
    eval_loader,
    eval_sampler,
    epoch: int,
    rank: int,
    world_size: int,
    device: str,
    eval_event_id: Any | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    train_state.model.eval()
    evaluator.clear_eval(event_id=eval_event_id)
    current_evaluator = matching_car_evaluator(
        evaluator,
        aggregated_voting=False,
    )
    current_evaluator.clear_eval(event_id=eval_event_id)
    return_keys = ["preds", "q_halt_logits"]

    with torch.no_grad():
        eval_sampler.set_epoch(epoch)
        eval_iter = eval_loader
        if rank == 0:
            eval_iter = tqdm.tqdm(
                eval_loader,
                desc="Evaluating CAR",
                leave=False,
            )
        for _, batch, _ in eval_iter:
            batch_device = batch_to_device(batch, device=device)
            batch_device["labels"] = torch.full_like(
                batch_device["inputs"],
                -100,
            )
            with torch.device(device):
                carry = train_state.model.initial_carry(batch_device)
            while True:
                carry, _, _, preds, all_finish = train_state.model(
                    carry=carry,
                    batch=batch_device,
                    return_keys=return_keys,
                )
                if all_finish:
                    break
            eval_batch = {
                "puzzle_ids": batch["puzzle_ids"],
                "offline_aug": batch["offline_aug"],
                "online_aug": batch["online_aug"],
                "inputs": batch["inputs"],
            }
            predictions = preds["preds"]
            q_values = preds["q_halt_logits"]
            evaluator.update(eval_batch, predictions, q_values)
            current_evaluator.update(eval_batch, predictions, q_values)

    train_state.model.train()
    return evaluator.result(), current_evaluator.result()
