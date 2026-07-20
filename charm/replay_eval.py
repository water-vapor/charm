"""
Replay evaluation across a checkpoint range with ARC aggregated voting.

This script re-evaluates checkpoints in step order and accumulates evaluator
events exactly like training-time ARC evaluation, but without relying on
checkpoint-adjacent evaluator sidecars.
"""

from typing import Optional, List, Tuple
import os
import re
import copy

import torch
import torch.distributed as dist
import hydra
import wandb
import tqdm
from omegaconf import DictConfig

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
    _load_checkpoint_payload,
    _resize_puzzle_embedding_if_needed,
)
from charm.training_utils.eval_helpers import (
    batch_to_device,
    build_eval_components,
    run_evaluation,
    print_dual_eval_metrics,
)
from charm.training_utils.data_builders import (
    create_train_dataloader,
    create_eval_dataloader,
)
from charm.training_utils.model_init import init_train_state
from charm.evaluators.arc import ARCEvaluator


class ReplayEvalConfig(PretrainConfig):
    # Directory or file containing checkpoints to replay.
    # If unset, falls back to load_checkpoint, then checkpoint_path.
    replay_checkpoint_source: Optional[str] = None
    # Inclusive replay bounds on checkpoint step.
    replay_start_step: Optional[int] = None
    replay_end_step: Optional[int] = None
    # Save per-checkpoint submission.json outputs.
    replay_save_submissions: bool = True
    # Optional subdir under checkpoint_path for submission outputs.
    # Example: "replay_eval" -> <checkpoint_path>/replay_eval/evaluator_step_*
    replay_submission_subdir: Optional[str] = "replay_eval"
    # Suffix appended to wandb run_name for replay clarity.
    replay_run_name_suffix: str = "_eval"
    # Optional per-checkpoint extra eval sweeps at larger recurrent step counts.
    # Example: base steps=16 and replay_halt_max_step_deltas=(0, 8)
    # evaluates at 16 and 24, then aggregates both into the same ARC event.
    replay_halt_max_step_deltas: Tuple[int, ...] = ()


def _checkpoint_step(path: str) -> int:
    name = os.path.basename(path)
    match = re.match(r"step_(\d+)\.pt$", name)
    if match is None:
        raise ValueError(f"checkpoint file does not match step_*.pt naming: '{path}'")
    return int(match.group(1))


def _collect_replay_checkpoints(
    source: str,
    start_step: Optional[int],
    end_step: Optional[int],
) -> List[Tuple[int, str]]:
    if os.path.isfile(source):
        step = _checkpoint_step(source)
        if start_step is not None and step < start_step:
            raise ValueError(f"checkpoint step {step} is below replay_start_step={start_step}")
        if end_step is not None and step > end_step:
            raise ValueError(f"checkpoint step {step} is above replay_end_step={end_step}")
        return [(step, source)]

    if not os.path.isdir(source):
        raise FileNotFoundError(f"replay checkpoint source not found: '{source}'")

    by_step: dict[int, str] = {}
    for file_name in os.listdir(source):
        if not file_name.endswith(".pt"):
            continue
        full_path = os.path.join(source, file_name)
        if not os.path.isfile(full_path):
            continue
        if not re.match(r"step_(\d+)\.pt$", file_name):
            continue
        step = _checkpoint_step(file_name)
        if step in by_step:
            raise ValueError(
                f"duplicate checkpoints for step {step}: '{by_step[step]}' and '{full_path}'"
            )
        by_step[step] = full_path

    checkpoints = sorted(by_step.items(), key=lambda x: x[0])
    if start_step is not None:
        checkpoints = [item for item in checkpoints if item[0] >= start_step]
    if end_step is not None:
        checkpoints = [item for item in checkpoints if item[0] <= end_step]

    if not checkpoints:
        raise ValueError(
            f"no checkpoints found in range "
            f"[{start_step if start_step is not None else '-inf'}, {end_step if end_step is not None else '+inf'}] "
            f"under '{source}'"
        )

    return checkpoints


def _load_checkpoint_for_replay_eval(
    train_state: TrainState,
    checkpoint_path: str,
    rank: int,
    expected_step: Optional[int] = None,
) -> dict:
    loaded = _load_checkpoint_payload(
        checkpoint_path,
        rank=rank,
        device=DEVICE,
        log_prefix="Loading checkpoint for replay-eval",
    )
    filename_step = _checkpoint_step(loaded.resolved_path)
    if expected_step is not None and filename_step != expected_step:
        raise ValueError(
            f"checkpoint step mismatch for '{loaded.resolved_path}': "
            f"expected_step={expected_step}, filename_step={filename_step}"
        )

    payload_step = loaded.step
    if payload_step is not None and payload_step != filename_step:
        raise ValueError(
            f"checkpoint step mismatch for '{loaded.resolved_path}': "
            f"filename_step={filename_step}, payload_step={payload_step}"
        )
    if payload_step is None and rank == 0:
        print(
            f"Warning: checkpoint '{loaded.resolved_path}' has no embedded step metadata; "
            f"using filename step={filename_step}."
        )

    effective_step = payload_step if payload_step is not None else filename_step

    _resize_puzzle_embedding_if_needed(train_state.model, loaded.state_dict)
    train_state.model.load_state_dict(loaded.state_dict)

    train_state.step = int(effective_step)
    train_state.accum_step = 0
    train_state.carry = None

    return {
        "step": int(effective_step),
        "real_epoch": loaded.real_epoch,
        "ema_state": loaded.ema_state,
        "resolved_checkpoint_path": loaded.resolved_path,
    }


def load_synced_replay_config(hydra_config: DictConfig, rank: int, world_size: int) -> ReplayEvalConfig:
    objects = [None]
    if rank == 0:
        config = ReplayEvalConfig(**hydra_config)  # type: ignore[arg-type]
        normalize_config_defaults(config, run_name_suffix=config.replay_run_name_suffix)

        if config.replay_checkpoint_source is None:
            if config.load_checkpoint not in (None, "latest"):
                config.replay_checkpoint_source = config.load_checkpoint
            else:
                config.replay_checkpoint_source = config.checkpoint_path

        if config.replay_checkpoint_source is None:
            raise ValueError("Unable to determine replay_checkpoint_source.")

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore[return-value]


def _submission_path(config: ReplayEvalConfig, step: int, has_evaluator: bool) -> Optional[str]:
    if not has_evaluator or not config.replay_save_submissions:
        return None
    if config.checkpoint_path is None:
        return None

    base = config.checkpoint_path
    if config.replay_submission_subdir:
        base = os.path.join(base, config.replay_submission_subdir)
    return os.path.join(base, f"evaluator_step_{step}")


def _unwrap_replay_model(model: torch.nn.Module) -> torch.nn.Module:
    """Follow known wrappers until the underlying recurrent model is reached."""
    module = model
    seen: set[int] = set()
    while True:
        module_id = id(module)
        if module_id in seen:
            raise RuntimeError("Encountered a wrapper cycle while unwrapping replay model.")
        seen.add(module_id)

        compiled_module = getattr(module, "_orig_mod", None)
        if isinstance(compiled_module, torch.nn.Module):
            module = compiled_module
            continue

        wrapped_model = getattr(module, "model", None)
        if isinstance(wrapped_model, torch.nn.Module):
            module = wrapped_model
            continue

        return module


def _get_model_eval_step_attr(model: torch.nn.Module) -> tuple[object, str]:
    model = _unwrap_replay_model(model)
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise AttributeError("Model does not expose a config object for replay eval.")

    for attr_name in ("halt_max_steps", "loops"):
        if hasattr(model_config, attr_name):
            return model_config, attr_name

    raise AttributeError(
        "Model does not expose config.halt_max_steps or config.loops for replay eval."
    )


def _get_model_eval_steps(model: torch.nn.Module) -> tuple[str, int]:
    model_config, attr_name = _get_model_eval_step_attr(model)
    return attr_name, int(getattr(model_config, attr_name))


def _set_model_eval_steps(model: torch.nn.Module, steps: int) -> None:
    model_config, attr_name = _get_model_eval_step_attr(model)
    setattr(model_config, attr_name, int(steps))


def _resolve_replay_eval_steps(
    config: ReplayEvalConfig,
    model: torch.nn.Module,
) -> List[int]:
    _, base_steps = _get_model_eval_steps(model)
    if len(config.replay_halt_max_step_deltas) == 0:
        return [base_steps]

    resolved: List[int] = []
    seen: set[int] = set()
    for delta in config.replay_halt_max_step_deltas:
        steps = base_steps + int(delta)
        if steps < 1:
            raise ValueError(
                f"Invalid replay eval steps={steps} from base_steps={base_steps} delta={delta}"
            )
        if steps in seen:
            continue
        seen.add(steps)
        resolved.append(steps)
    return resolved


def _update_arc_evaluators(
    train_state: TrainState,
    evaluators: List[ARCEvaluator],
    eval_loader,
    eval_sampler,
    epoch: int,
    rank: int,
    num_passes: int,
) -> None:
    train_state.model.eval()
    return_keys = ["preds", "q_halt_logits"]

    with torch.no_grad():
        for pass_idx in range(num_passes):
            eval_sampler.set_epoch(epoch * num_passes + pass_idx)

            eval_iter = eval_loader
            if rank == 0:
                desc = f"Evaluating pass {pass_idx + 1}/{num_passes}" if num_passes > 1 else "Evaluating"
                eval_iter = tqdm.tqdm(eval_loader, desc=desc, leave=False)

            for set_name, batch, local_batch_size in eval_iter:
                batch_gpu = batch_to_device(batch, device=DEVICE)
                batch_gpu["labels"] = torch.full_like(batch_gpu["inputs"], -100)

                with torch.device(DEVICE):
                    carry = train_state.model.initial_carry(batch_gpu)

                while True:
                    carry, loss, metrics, preds, all_finish = train_state.model(
                        carry=carry, batch=batch_gpu, return_keys=return_keys
                    )
                    if all_finish:
                        break

                eval_batch = {
                    "puzzle_ids": batch["puzzle_ids"],
                    "offline_aug": batch["offline_aug"],
                    "online_aug": batch["online_aug"],
                    "inputs": batch["inputs"],
                }

                for evaluator in evaluators:
                    evaluator.update(eval_batch, preds["preds"], preds["q_halt_logits"])

    train_state.model.train()


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    rank, world_size, _ = init_distributed_environment(
        device=DEVICE,
        backend=BACKEND,
        mpi=MPI,
        socket=SOCKET,
    )

    config = load_synced_replay_config(hydra_config, rank=rank, world_size=world_size)
    torch.random.manual_seed(config.seed + rank)

    checkpoints = _collect_replay_checkpoints(
        source=config.replay_checkpoint_source,
        start_step=config.replay_start_step,
        end_step=config.replay_end_step,
    )
    if rank == 0:
        first_step = checkpoints[0][0]
        last_step = checkpoints[-1][0]
        print(
            f"Replay checkpoints: {len(checkpoints)} "
            f"(steps {first_step}..{last_step}) from '{config.replay_checkpoint_source}'"
        )

    train_loader, _, train_metadata = create_train_dataloader(
        config, rank=rank, world_size=world_size
    )
    train_dataset = train_loader.dataset
    eval_components = build_eval_components(
        config=config,
        train_dataset=train_dataset,
        rank=rank,
        world_size=world_size,
        create_eval_dataloader_fn=create_eval_dataloader,
        solution_output_path=None,
    )
    eval_loader = eval_components.eval_loader
    eval_sampler = eval_components.eval_sampler
    evaluator = eval_components.evaluator

    samples_per_effective_epoch = (
        train_dataset.num_original_puzzles * train_dataset.mean_pairs_per_puzzle
    )
    train_state = init_train_state(
        config,
        train_metadata,
        samples_per_effective_epoch,
        rank=rank,
        world_size=world_size,
    )

    ema_helper = None
    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    if rank == 0:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)

    for expected_step, checkpoint_path in checkpoints:
        resume_info = _load_checkpoint_for_replay_eval(
            train_state,
            checkpoint_path,
            rank=rank,
            expected_step=expected_step,
        )
        broadcast_model_state(train_state.model, world_size=world_size)

        step = int(resume_info["step"])

        if config.ema and ema_helper is not None:
            ema_state = resume_info.get("ema_state")
            if ema_state is not None:
                ema_helper.load_state_dict(ema_state)
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                if rank == 0:
                    print(f"Warning: EMA enabled but checkpoint '{checkpoint_path}' has no ema_state.")
                train_state_eval = train_state
        else:
            if rank == 0 and resume_info.get("ema_state") is not None:
                print(
                    f"Warning: checkpoint '{checkpoint_path}' contains ema_state, "
                    "but config.ema=False, evaluating non-EMA weights."
                )
            train_state_eval = train_state

        eval_save_path = _submission_path(config, step=step, has_evaluator=True)
        replay_eval_steps = _resolve_replay_eval_steps(config, train_state_eval.model)
        if rank == 0 and len(replay_eval_steps) > 1:
            print(f"Checkpoint step {step}: replay ARC eval with steps={replay_eval_steps}")

        if len(replay_eval_steps) == 1:
            eval_metrics_aggregated, eval_metrics_current = run_evaluation(
                train_state_eval,
                evaluator,
                eval_loader,
                eval_sampler,
                0,  # Intentionally decoupled from training real_epoch
                rank,
                world_size,
                device=DEVICE,
                save_path=eval_save_path,
                num_passes=config.eval_passes,
                eval_event_id=step,
            )
        else:
            current_evaluator = ARCEvaluator(
                ground_truth_inputs=evaluator.ground_truth_inputs,
                ground_truth_outputs=getattr(evaluator, "_gt_outputs", None),
                pass_ks=tuple(evaluator.pass_ks),
                submission_k=evaluator.submission_k,
                aggregated_voting=False,
                exp_decay=False,
                decay_half_life=evaluator.decay_half_life,
            )
            evaluator.clear_eval(event_id=step)
            current_evaluator.clear_eval(event_id=step)

            _, original_eval_steps = _get_model_eval_steps(train_state_eval.model)
            try:
                for eval_steps in replay_eval_steps:
                    _set_model_eval_steps(train_state_eval.model, eval_steps)
                    _update_arc_evaluators(
                        train_state=train_state_eval,
                        evaluators=[evaluator, current_evaluator],
                        eval_loader=eval_loader,
                        eval_sampler=eval_sampler,
                        epoch=0,  # Intentionally decoupled from training real_epoch
                        rank=rank,
                        num_passes=config.eval_passes,
                    )
            finally:
                _set_model_eval_steps(train_state_eval.model, original_eval_steps)

            eval_metrics_aggregated = evaluator.result(save_path=eval_save_path)
            eval_metrics_current = current_evaluator.result(save_path=None)

        print_dual_eval_metrics(
            eval_metrics_aggregated,
            eval_metrics_current,
            step=step,
            rank=rank,
            tag="replay-eval",
        )
        if rank == 0 and eval_metrics_aggregated:
            wandb.log({f"eval/{k}": v for k, v in eval_metrics_aggregated.items()}, step=step)
        if rank == 0 and eval_metrics_current:
            wandb.log({f"eval_current/{k}": v for k, v in eval_metrics_current.items()}, step=step)

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
