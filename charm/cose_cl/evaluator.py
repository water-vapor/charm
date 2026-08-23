"""Evaluator behavior used only by the CoSE continual-learning experiments."""

from charm.evaluators.arc import ARCEvaluator


class CoSEEvaluator(ARCEvaluator):
    """Add the canonical CoSE-CL metrics without changing CHARM's evaluator."""

    @staticmethod
    def _vote_and_rank(predictions):
        """Rank deterministically by count, rounded mean confidence, then hash."""
        vote_map = {}
        if predictions and len(predictions[0]) == 3:
            for pred_hash, q_value, weight in predictions:
                if pred_hash not in vote_map:
                    vote_map[pred_hash] = [0.0, 0.0]
                vote_map[pred_hash][0] += weight
                vote_map[pred_hash][1] += q_value * weight
        else:
            for pred_hash, q_value in predictions:
                if pred_hash not in vote_map:
                    vote_map[pred_hash] = [0.0, 0.0]
                vote_map[pred_hash][0] += 1.0
                vote_map[pred_hash][1] += q_value
        return sorted(
            vote_map.items(),
            key=lambda item: (
                item[1][0],
                round(item[1][1] / max(item[1][0], 1e-12), 6),
                item[0],
            ),
            reverse=True,
        )

    def per_puzzle_pass_at_k(
        self, ks: tuple[int, ...] = (1, 2)
    ) -> dict[str, dict[int, float]] | None:
        """Return per-puzzle pass@K; collective when distributed."""
        is_main, predictions, _, _, _ = self._gather_predictions()
        if not is_main:
            return None

        result = {}
        for puzzle_id, gt_pairs in self._gt_hashes.items():
            per_k = {k: 0 for k in ks}
            num_test = 0
            for inp_hash, out_hash in gt_pairs:
                if out_hash is None:
                    continue
                num_test += 1
                ranked = self._vote_and_rank(
                    predictions.get((puzzle_id, inp_hash), [])
                )
                for k in ks:
                    if any(pred_hash == out_hash for pred_hash, _ in ranked[:k]):
                        per_k[k] += 1
            if num_test:
                result[puzzle_id] = {k: per_k[k] / num_test for k in ks}
        return result
