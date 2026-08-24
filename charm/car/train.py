"""Train CAR task-memory comparisons on CUDA."""

from __future__ import annotations

import copy

import hydra
from omegaconf import DictConfig
import torch
import torch.distributed as dist
import tqdm
import wandb

from charm.car.dataset import (
    create_car_eval_dataloader,
    create_car_train_dataloader,
)
from charm.car.evaluator import build_car_evaluator, run_car_evaluation
from charm.car.model_init import init_car_train_state, logical_parameter_counts
from charm.models.ema import EMAHelper
from charm.train import (
    load_synced_config,
    save_code_and_config,
    save_train_state,
    train_batch,
)
from charm.training_utils.checkpointing import load_checkpoint
from charm.training_utils.eval_helpers import print_dual_eval_metrics
from charm.training_utils.runtime import (
    DEVICE,
    broadcast_model_state,
    init_distributed_environment,
)


@hydra.main(config_path="../config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig) -> None:
    if DEVICE != "cuda":
        raise RuntimeError("The CAR code-release launcher supports CUDA only")

    rank, world_size, _ = init_distributed_environment()
    config = load_synced_config(
        hydra_config,
        rank=rank,
        world_size=world_size,
    )
    if config.eval_passes != 1:
        raise ValueError(
            "CAR evaluation already enumerates all eight D4 views; "
            "eval_passes must be 1"
        )
    use_step_overrides = config.train_steps_override is not None
    torch.random.manual_seed(config.seed + rank)

    train_loader, train_sampler, train_metadata = create_car_train_dataloader(
        config,
        rank=rank,
        world_size=world_size,
    )
    train_dataset = train_loader.dataset
    if rank == 0:
        print(
            f"vocab_size={train_metadata.vocab_size}, "
            f"seq_len={train_metadata.seq_len}, "
            f"car_tasks={train_metadata.car_num_tasks}, "
            f"car_rules={train_metadata.car_num_rules}"
        )

    evaluator = None
    eval_loader = None
    eval_sampler = None
    should_build_eval = (
        config.eval_interval is not None or use_step_overrides
    ) and bool(config.data_paths_test)
    if should_build_eval:
        eval_loader, eval_sampler, eval_dataset = create_car_eval_dataloader(
            config,
            train_dataset=train_dataset,
            rank=rank,
            world_size=world_size,
        )
        evaluator = build_car_evaluator(
            config,
            train_dataset,
            eval_dataset,
        )
        if rank == 0:
            print(
                f"Eval dataset: {len(eval_loader.dataset)} D4 views, "
                f"{len(evaluator.ground_truth_inputs)} tasks"
            )

    samples_per_effective_epoch = (
        train_dataset.num_original_puzzles * train_dataset.mean_pairs_per_puzzle
    )
    train_state = init_car_train_state(
        config,
        train_metadata,
        samples_per_effective_epoch,
        rank=rank,
        world_size=world_size,
        total_steps_override=(
            config.train_steps_override if use_step_overrides else None
        ),
    )

    resume_info = {}
    if config.load_checkpoint:
        resume_info = load_checkpoint(
            train_state,
            config,
            rank=rank,
            device=DEVICE,
        )
        if rank == 0:
            print("Resume mode: checkpoint load")
    broadcast_model_state(train_state.model, world_size=world_size)

    if rank == 0:
        print(
            f"Dataset: {len(train_dataset)} pairs, "
            f"{train_dataset.num_original_puzzles} tasks, "
            f"{train_dataset.mean_pairs_per_puzzle:.2f} pairs/task"
        )
        print(f"Total steps: {train_state.total_steps}")

    steps_per_effective_epoch = samples_per_effective_epoch / config.global_batch_size
    eval_interval_steps = None
    if use_step_overrides:
        eval_interval_steps = int(config.eval_interval_steps_override)
    elif config.eval_interval is not None:
        eval_interval_steps = max(
            1,
            int(config.eval_interval * steps_per_effective_epoch),
        )

    progress_bar = None
    ema_helper = None
    if rank == 0:
        progress_bar = tqdm.tqdm(
            total=train_state.total_steps,
            initial=train_state.step,
        )
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log(logical_parameter_counts(train_state.model), step=0)
        save_code_and_config(config)

    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)
        if resume_info.get("ema_state"):
            ema_helper.load_state_dict(resume_info["ema_state"])

    last_eval_step = resume_info.get("last_eval_step", 0)
    real_epoch = resume_info.get("real_epoch", 0)
    current_epoch = real_epoch

    while train_state.step < train_state.total_steps:
        current_epoch = real_epoch
        train_sampler.set_epoch(current_epoch)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(current_epoch)
        real_epoch += 1

        train_state.model.train()
        for _, batch, local_batch_size in train_loader:
            if train_state.step >= train_state.total_steps:
                break
            global_batch_size = local_batch_size * world_size
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=rank,
                world_size=world_size,
            )

            if rank == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)
                effective_epoch = (
                    train_state.step
                    * config.global_batch_size
                    / samples_per_effective_epoch
                )
                progress_bar.set_postfix(
                    epoch=real_epoch,
                    effective_epoch=f"{int(effective_epoch)}",
                )
            if ema_helper is not None:
                ema_helper.update(train_state.model)

            if eval_interval_steps is None:
                continue
            should_eval = (
                train_state.step - last_eval_step >= eval_interval_steps
                or train_state.step == train_state.total_steps
            )
            if not should_eval:
                continue
            last_eval_step = train_state.step

            if rank == 0:
                print(f"Checkpoint at step {train_state.step}")
                if config.checkpoint_every_eval:
                    save_train_state(
                        config,
                        train_state,
                        ema_helper,
                        current_epoch,
                        last_eval_step,
                    )

            if evaluator is None:
                continue
            if ema_helper is not None:
                if rank == 0:
                    print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            aggregated, current = run_car_evaluation(
                train_state_eval,
                evaluator,
                eval_loader,
                eval_sampler,
                real_epoch,
                rank,
                world_size,
                device=DEVICE,
                eval_event_id=train_state.step,
            )
            print_dual_eval_metrics(
                aggregated,
                current,
                step=train_state.step,
                rank=rank,
                tag="eval",
            )
            if rank == 0 and aggregated:
                wandb.log(
                    {f"eval/{key}": value for key, value in aggregated.items()},
                    step=train_state.step,
                )
            if rank == 0 and current:
                wandb.log(
                    {f"eval_current/{key}": value for key, value in current.items()},
                    step=train_state.step,
                )

    if rank == 0:
        print("Training complete. Saving final checkpoint.")
        save_train_state(
            config,
            train_state,
            ema_helper,
            current_epoch,
            last_eval_step,
        )
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
