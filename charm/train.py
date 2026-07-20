from typing import Any
import os
import math
import random
import yaml
import shutil
import copy

import numpy as np

import torch
import torch.distributed as dist
from torch import nn

import tqdm
import wandb
import hydra
from omegaconf import DictConfig
# from adam_atan2 import AdamATan2

from charm.utils.model_loading import get_model_source_path
from charm.models.ema import EMAHelper
from charm.training_utils.config_types import PretrainConfig, TrainState
from charm.training_utils.runtime import (
    DEVICE,
    BACKEND,
    MPI,
    SOCKET,
    init_distributed_environment,
    normalize_config_defaults,
    broadcast_model_state,
)
from charm.training_utils.checkpointing import (
    load_checkpoint,
)
from charm.training_utils.data_builders import (
    create_train_dataloader,
    create_eval_dataloader,
)
from charm.training_utils.model_init import init_train_state
from charm.training_utils.eval_helpers import (
    batch_to_device,
    build_eval_components,
    print_dual_eval_metrics,
    run_evaluation,
)


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def save_train_state(
    config: PretrainConfig,
    train_state: TrainState,
    ema_helper=None,
    real_epoch: int = 0,
    last_eval_step: int = 0,
):
    if config.checkpoint_path is None:
        return
    if train_state.accum_step != 0:
        print(f"Skipping checkpoint save at step {train_state.step} (accum_step={train_state.accum_step})")
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    state = {
        "step": train_state.step,
        "accum_step": train_state.accum_step,
        "real_epoch": real_epoch,
        "last_eval_step": last_eval_step,
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_states": [optim.state_dict() for optim in train_state.optimizers],
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "rng_state": torch.random.get_rng_state(),
    }
    if DEVICE == "xpu" and torch.xpu.is_available():
        state["gpu_rng_state"] = torch.xpu.get_rng_state()
    elif DEVICE == "cuda" and torch.cuda.is_available():
        state["gpu_rng_state"] = torch.cuda.get_rng_state()
    if ema_helper is not None:
        state["ema_state"] = ema_helper.state_dict()

    torch.save(state, os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt"))

def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def _get_smart_puzzle_embedding_module(model: nn.Module):
    """Best-effort unwrap for compiled/loss-head wrapped models."""
    module = model
    for _ in range(8):
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod  # type: ignore[attr-defined]
            continue
        if hasattr(module, "model"):
            module = module.model  # type: ignore[attr-defined]
            continue
        break

    puzzle_emb = getattr(module, "puzzle_emb", None)
    if puzzle_emb is None and hasattr(module, "inner"):
        puzzle_emb = getattr(module.inner, "puzzle_emb", None)  # type: ignore[attr-defined]

    if puzzle_emb is not None and hasattr(puzzle_emb, "interaction_mode"):
        return puzzle_emb
    return None


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    if train_state.step >= train_state.total_steps:
        return

    # To device (tensors to xpu, keep lists as-is, handle nested dicts)
    batch = batch_to_device(batch, device=DEVICE)

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device(DEVICE):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    # Optional gate telemetry from SmartTaskEmbedding (for interaction-mode ablations)
    gate_metrics = {}
    puzzle_emb_module = _get_smart_puzzle_embedding_module(train_state.model)
    if puzzle_emb_module is not None and getattr(puzzle_emb_module, "interaction_mode", "none") != "none":
        gate_mean = getattr(puzzle_emb_module, "last_gate_mean", None)
        gate_active = getattr(puzzle_emb_module, "last_gate_active_ratio", None)
        if isinstance(gate_mean, torch.Tensor) and isinstance(gate_active, torch.Tensor):
            mean_is_finite = bool(torch.isfinite(gate_mean).item())
            active_is_finite = bool(torch.isfinite(gate_active).item())
            if mean_is_finite and active_is_finite:
                gate_metrics = {
                    "gate_mean": gate_mean.detach(),
                    "gate_active_ratio": gate_active.detach(),
                }

    # Scale loss by effective batch size (global_batch_size * grad_accum_steps)
    effective_batch_size = global_batch_size * config.grad_accum_steps
    ((1 / effective_batch_size) * loss).backward()

    # Track accumulation
    train_state.accum_step += 1

    # Only step optimizer when accumulation is complete
    if train_state.accum_step < config.grad_accum_steps:
        return None

    # Reset accumulation counter and increment step
    train_state.accum_step = 0
    train_state.step += 1

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Apply optimizer
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step

        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        gate_metric_values = None
        gate_metric_keys = None
        if gate_metrics:
            gate_metric_keys = list(sorted(gate_metrics.keys()))
            gate_metric_values = torch.stack([gate_metrics[k] for k in gate_metric_keys]).to(metric_values.device)
            if world_size > 1:
                dist.reduce(gate_metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess (use effective_batch_size for loss normalization)
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (effective_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            if gate_metric_values is not None and gate_metric_keys is not None:
                gate_metric_values_np = gate_metric_values.cpu().numpy()
                for i, k in enumerate(gate_metric_keys):
                    reduced_metrics[f"train/{k}"] = gate_metric_values_np[i] / world_size

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore
        normalize_config_defaults(config)
        if config.eval_interval is not None and config.eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1, got {config.eval_interval}")
        has_train_steps_override = config.train_steps_override is not None
        has_eval_interval_steps_override = config.eval_interval_steps_override is not None
        if has_train_steps_override != has_eval_interval_steps_override:
            raise ValueError(
                "Step-based overrides must be both set or both unset: "
                f"train_steps_override={config.train_steps_override}, "
                f"eval_interval_steps_override={config.eval_interval_steps_override}"
            )
        if has_train_steps_override:
            assert config.train_steps_override is not None
            assert config.eval_interval_steps_override is not None
            if config.train_steps_override < 1:
                raise ValueError(
                    f"train_steps_override must be >= 1, got {config.train_steps_override}"
                )
            if config.eval_interval_steps_override < 1:
                raise ValueError(
                    "eval_interval_steps_override must be >= 1, "
                    f"got {config.eval_interval_steps_override}"
                )

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    # Initialize distributed training/runtime.
    RANK, WORLD_SIZE, _ = init_distributed_environment(
        device=DEVICE,
        backend=BACKEND,
        mpi=MPI,
        socket=SOCKET,
        log_xpu_master_addr=True,
    )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    use_step_overrides = config.train_steps_override is not None

    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)

    # Create dataloaders
    train_loader, train_sampler, train_metadata = create_train_dataloader(config, rank=RANK, world_size=WORLD_SIZE)

    if RANK == 0:
        print(f"vocab_size={train_metadata.vocab_size}, seq_len={train_metadata.seq_len}")

    # Create eval dataloader and evaluator
    # NOTE: eval dataset must use the same unique_str_to_int mapping as train dataset
    evaluator = None
    eval_loader = None
    eval_sampler = None
    should_build_eval = (config.eval_interval is not None or use_step_overrides) and bool(config.data_paths_test)
    if should_build_eval:
        solution_path = os.path.join(config.checkpoint_path, "solution.json") if config.checkpoint_path and config.task_type == "arc" else None
        train_dataset = train_loader.dataset
        eval_components = build_eval_components(
            config=config,
            train_dataset=train_dataset,
            rank=RANK,
            world_size=WORLD_SIZE,
            create_eval_dataloader_fn=create_eval_dataloader,
            solution_output_path=solution_path,
        )
        eval_loader = eval_components.eval_loader
        eval_sampler = eval_components.eval_sampler
        evaluator = eval_components.evaluator

    # Compute samples per effective epoch from dataset
    # One effective epoch = num_original_puzzles * mean_pairs_per_puzzle
    train_dataset = train_loader.dataset
    samples_per_effective_epoch = train_dataset.num_original_puzzles * train_dataset.mean_pairs_per_puzzle

    # Train state
    train_state = init_train_state(
        config,
        train_metadata,
        samples_per_effective_epoch,
        rank=RANK,
        world_size=WORLD_SIZE,
        total_steps_override=config.train_steps_override if use_step_overrides else None,
    )

    # Load checkpoint (all ranks load independently for full state restore)
    resume_info = {}
    if config.load_checkpoint:
        resume_info = load_checkpoint(train_state, config, rank=RANK, device=DEVICE)
        if RANK == 0:
            print("Resume mode: checkpoint load")

    # Sync model params across ranks to ensure identical state
    broadcast_model_state(train_state.model, world_size=WORLD_SIZE)

    if RANK == 0:
        print(f"Dataset: {len(train_dataset)} pairs, {train_dataset.num_original_puzzles} original puzzles, "
              f"{train_dataset.num_augmented_puzzles} augmented puzzles, {train_dataset.mean_pairs_per_puzzle:.2f} pairs/puzzle")
        if use_step_overrides:
            print(
                "Total steps: "
                f"{train_state.total_steps} (override; epochs-based step calculation disabled)"
            )
        else:
            print(f"Total steps: {train_state.total_steps} ({config.epochs} effective epochs)")

    # Compute steps per eval (eval_interval is in effective epochs)
    steps_per_effective_epoch = samples_per_effective_epoch / config.global_batch_size
    eval_interval_steps = None
    if use_step_overrides:
        assert config.eval_interval_steps_override is not None
        eval_interval_steps = int(config.eval_interval_steps_override)
    elif config.eval_interval is not None:
        eval_interval_steps = max(1, int(config.eval_interval * steps_per_effective_epoch))

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)
        if resume_info.get("ema_state"):
            ema_helper.load_state_dict(resume_info["ema_state"])

    # Training Loop
    last_eval_step = resume_info.get("last_eval_step", 0)
    real_epoch = resume_info.get("real_epoch", 0)
    current_epoch = real_epoch

    while train_state.step < train_state.total_steps:
        current_epoch = real_epoch
        train_sampler.set_epoch(current_epoch)
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(current_epoch)
        real_epoch += 1

        train_state.model.train()
        for set_name, batch, local_batch_size in train_loader:
            if train_state.step >= train_state.total_steps:
                break

            global_batch_size = local_batch_size * WORLD_SIZE
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)
                effective_epoch = train_state.step * config.global_batch_size / samples_per_effective_epoch
                progress_bar.set_postfix(epoch=real_epoch, effective_epoch=f"{int(effective_epoch)}")

            if config.ema:
                ema_helper.update(train_state.model)

            # Check if we should eval
            if eval_interval_steps is not None:
                should_eval = (train_state.step - last_eval_step) >= eval_interval_steps
                if should_eval:
                    last_eval_step = train_state.step

                    if RANK == 0:
                        if use_step_overrides:
                            print(f"Checkpoint at step {train_state.step}")
                        else:
                            effective_epochs_so_far = train_state.step * config.global_batch_size / samples_per_effective_epoch
                            print(f"Checkpoint at step {train_state.step} (effective epoch ~{effective_epochs_so_far:.0f})")
                        if config.checkpoint_every_eval:
                            save_train_state(
                                config,
                                train_state,
                                ema_helper,
                                current_epoch,
                                last_eval_step,
                            )

                    # Run evaluation if evaluator is configured
                    if evaluator is not None:
                        # create save path for submission.json (ARC only)
                        eval_save_path = None
                        if config.checkpoint_path is not None and evaluator is not None:
                            eval_save_path = os.path.join(
                                config.checkpoint_path,
                                f"evaluator_step_{train_state.step}",
                            )
                        # Use EMA model for evaluation when enabled
                        if config.ema and ema_helper is not None:
                            if RANK == 0:
                                print("SWITCH TO EMA")
                            train_state_eval = copy.deepcopy(train_state)
                            train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
                        else:
                            train_state_eval = train_state

                        eval_metrics_aggregated, eval_metrics_current = run_evaluation(
                            train_state_eval, evaluator, eval_loader, eval_sampler,
                            real_epoch, RANK, WORLD_SIZE, save_path=eval_save_path,
                            device=DEVICE,
                            num_passes=config.eval_passes,
                            eval_event_id=train_state.step,
                        )

                        print_dual_eval_metrics(
                            eval_metrics_aggregated,
                            eval_metrics_current,
                            step=train_state.step,
                            rank=RANK,
                            tag="eval",
                        )
                        if RANK == 0 and eval_metrics_aggregated:
                            wandb.log({f"eval/{k}": v for k, v in eval_metrics_aggregated.items()}, step=train_state.step)
                        if RANK == 0 and eval_metrics_current:
                            wandb.log({f"eval_current/{k}": v for k, v in eval_metrics_current.items()}, step=train_state.step)

    # Final checkpoint
    if RANK == 0:
        print("Training complete. Saving final checkpoint.")
        save_train_state(
            config,
            train_state,
            ema_helper,
            current_epoch,
            last_eval_step,
        )

    # Finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
