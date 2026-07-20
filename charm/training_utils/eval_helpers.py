from typing import Optional, Any, Callable
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tqdm

from charm.evaluators.arc import ARCEvaluator


@dataclass
class EvalComponents:
    eval_loader: DataLoader
    eval_sampler: DistributedSampler
    eval_dataset: Any
    evaluator: ARCEvaluator


def batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Move batch tensors to device, handling nested dicts."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = {dk: dv.to(device) if isinstance(dv, torch.Tensor) else dv for dk, dv in v.items()}
        else:
            result[k] = v
    return result


def build_eval_components(
    *,
    config: Any,
    train_dataset: Any,
    rank: int,
    world_size: int,
    create_eval_dataloader_fn: Callable[..., tuple[DataLoader, DistributedSampler, Any, Any]],
    solution_output_path: Optional[str] = None,
) -> EvalComponents:
    """
    Build ARC public-evaluation dataloader and voting evaluator.
    """
    if config.task_type != "arc":
        raise ValueError("The CHARM release supports only task_type='arc'.")
    if not hasattr(train_dataset, "unique_str_to_int"):
        raise AttributeError("train_dataset must provide 'unique_str_to_int' for eval mapping")
    if not hasattr(train_dataset, "puzzle_id_to_int"):
        raise AttributeError("train_dataset must provide 'puzzle_id_to_int' for eval mapping")

    eval_loader, eval_sampler, _, eval_dataset = create_eval_dataloader_fn(
        config,
        rank=rank,
        world_size=world_size,
        solution_output_path=solution_output_path,
        unique_str_to_int=train_dataset.unique_str_to_int,
        puzzle_id_to_int=train_dataset.puzzle_id_to_int,
        track_per_aug_embeddings=getattr(train_dataset, "track_per_aug_embeddings", False),
        per_aug_param_to_int=getattr(train_dataset, "_per_aug_param_to_int", None),
    )

    gt_inputs = eval_dataset.get_noaug_test_inputs()
    gt_outputs = eval_dataset.get_noaug_test_outputs()
    evaluator = ARCEvaluator(
        ground_truth_inputs=gt_inputs,
        ground_truth_outputs=gt_outputs,
        pass_ks=(1, 2, 5, 10, 100, 1000),
        submission_k=100,
        recent_window_size=config.eval_recent_window_size,
        exp_decay=config.eval_exp_decay,
        decay_half_life=config.eval_decay_half_life,
    )
    if rank == 0:
        print(f"Eval dataset: {len(eval_loader.dataset)} test pairs, {len(gt_inputs)} puzzles")

    return EvalComponents(
        eval_loader=eval_loader,
        eval_sampler=eval_sampler,
        eval_dataset=eval_dataset,
        evaluator=evaluator,
    )


def print_eval_metrics(eval_metrics: dict[str, float], step: int, rank: int, tag: str = "eval") -> None:
    """Print evaluation metrics on rank 0 so runs are readable without W&B."""
    if rank != 0:
        return
    if not eval_metrics:
        print(f"[{tag}] step={step} no metrics returned")
        return

    preferred_keys = ("pass@1", "pass@2", "pass@5", "pass@10", "pass@100", "pass@1000")
    ordered_keys = [k for k in preferred_keys if k in eval_metrics]
    ordered_keys.extend(sorted(k for k in eval_metrics if k not in preferred_keys))
    metrics_str = " ".join(f"{k}={float(eval_metrics[k]):.6f}" for k in ordered_keys)
    print(f"[{tag}] step={step} {metrics_str}")


def print_dual_eval_metrics(
    aggregated_metrics: dict[str, float],
    current_metrics: dict[str, float],
    step: int,
    rank: int,
    tag: str = "eval",
) -> None:
    """Print both cumulative and current-only eval metrics on rank 0."""
    print_eval_metrics(aggregated_metrics, step=step, rank=rank, tag=f"{tag}-aggregated")
    print_eval_metrics(current_metrics, step=step, rank=rank, tag=f"{tag}-current")


def run_evaluation(
    train_state: Any,
    evaluator: ARCEvaluator,
    eval_loader: DataLoader,
    eval_sampler: DistributedSampler,
    epoch: int,
    rank: int,
    world_size: int,
    device: str,
    save_path: Optional[str] = None,
    num_passes: int = 1,
    eval_event_id: Optional[Any] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Run evaluation loop.

    Returns:
      - aggregated metrics (historical + current event)
      - current-only metrics (current event only)
    """
    train_state.model.eval()
    evaluator.clear_eval(event_id=eval_event_id)
    current_evaluator = ARCEvaluator(
        ground_truth_inputs=evaluator.ground_truth_inputs,
        ground_truth_outputs=getattr(evaluator, "_gt_outputs", None),
        pass_ks=tuple(evaluator.pass_ks),
        submission_k=evaluator.submission_k,
        aggregated_voting=False,
        exp_decay=False,
        decay_half_life=evaluator.decay_half_life,
    )

    # return_keys for ACTLossHead - we need preds and q_halt_logits
    return_keys = ["preds", "q_halt_logits"]

    with torch.no_grad():
        for pass_idx in range(num_passes):
            # set different epoch for each pass to get different shuffling/augmentation
            eval_sampler.set_epoch(epoch * num_passes + pass_idx)

            # progress bar for rank 0
            eval_iter = eval_loader
            if rank == 0:
                desc = f"Evaluating pass {pass_idx + 1}/{num_passes}" if num_passes > 1 else "Evaluating"
                eval_iter = tqdm.tqdm(eval_loader, desc=desc, leave=False)

            for set_name, batch, local_batch_size in eval_iter:
                batch_gpu = batch_to_device(batch, device=device)

                # add dummy labels (ACTLossHead expects labels, but we don't use them for eval)
                batch_gpu["labels"] = torch.full_like(batch_gpu["inputs"], -100)

                # initialize carry (use full model, not inner model)
                with torch.device(device):
                    carry = train_state.model.initial_carry(batch_gpu)

                # forward loop until all samples halt (matching original TRM eval logic)
                while True:
                    carry, loss, metrics, preds, all_finish = train_state.model(
                        carry=carry, batch=batch_gpu, return_keys=return_keys
                    )
                    if all_finish:
                        break

                predictions = preds["preds"]
                q_vals = preds["q_halt_logits"]

                eval_batch = {
                    "puzzle_ids": batch["puzzle_ids"],
                    "offline_aug": batch["offline_aug"],
                    "online_aug": batch["online_aug"],
                    "inputs": batch["inputs"],
                }

                evaluator.update(eval_batch, predictions, q_vals)
                current_evaluator.update(eval_batch, predictions, q_vals)

    train_state.model.train()
    # Keep aggregated submission behavior unchanged; compute current-only metrics without a second forward pass.
    aggregated_metrics = evaluator.result(save_path=save_path)
    current_metrics = current_evaluator.result(save_path=None)
    return aggregated_metrics, current_metrics
