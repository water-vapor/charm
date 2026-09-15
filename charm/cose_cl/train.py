"""CoSE-CL training loop: single GPU (python -m) or DDP (torchrun).

Sequential arms train one puzzle per stage on its augmented demo pairs, then run the
augmentation-voting eval on its held-out test pairs. Joint arms run the same number of
virtual stages with puzzle-uniform batches (one puzzle per batch, drawn uniformly), so
every puzzle's expected budget matches the sequential arms'. Full-stream re-check
sweeps every eval_every stages give the retention matrices; pass@1 and pass@2 are
reported at every measurement.

Distribution: batch_size is the global batch (each rank trains batch_size // world);
gradients of trainable parameters are summed across ranks, sparse SignSGD gathers its
row updates internally, and eval pairs are strided across ranks with the evaluator
gathering votes on rank 0. Bookkeeping, logging, and checkpoints live on rank 0.
"""

import json
import os
import time
from dataclasses import asdict
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.utils.data import DataLoader, RandomSampler

from charm.cose_cl.config import parse_args
from charm.cose_cl.data import Stream, arc1_eval_set, eval_batches
from charm.cose_cl.evaluator import CoSEEvaluator
from charm.cose_cl.model import (
    build_model, base_row_drift, base_row_snapshot, inner_module, load_expanded, setup_arm)
from charm.models.smart_task_embedding import SmartTaskEmbedding
from charm.training_utils.runtime import (
    broadcast_model_state, init_distributed_environment)
from charm.datasets.puzzle_dataloader import puzzle_collate_fn


def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def evaluate(model, ds, pair_idxs, gt_inputs, gt_outputs, puzzle_ids, cfg, rank, world):
    """Voting eval over the given pairs, strided across ranks.

    Returns (per-puzzle pass@k, mean@1, mean@2) on rank 0; (None, nan, nan) elsewhere.
    """
    evaluator = CoSEEvaluator(
        {pid: gt_inputs[pid] for pid in puzzle_ids},
        {pid: gt_outputs[pid] for pid in puzzle_ids},
        pass_ks=(1, 2), submission_k=2, aggregated_voting=False)
    evaluator.clear_eval(event_id=0)
    model.eval()
    with torch.no_grad():
        for batch, n_real in eval_batches(ds, pair_idxs[rank::world], cfg.eval_batch_size):
            gpu_batch = to_device(batch, cfg.device)
            # eval datasets carry no labels; the loss head still expects the key
            gpu_batch["labels"] = torch.full_like(gpu_batch["inputs"], -100)
            with torch.device(cfg.device):
                carry = model.initial_carry(gpu_batch)
            while True:
                carry, _, _, preds, all_finish = model(
                    carry=carry, batch=gpu_batch, return_keys=["preds", "q_halt_logits"])
                if all_finish:
                    break
            sliced = {k: batch[k][:n_real]
                      for k in ("puzzle_ids", "offline_aug", "online_aug", "inputs")}
            evaluator.update(sliced, preds["preds"][:n_real], preds["q_halt_logits"][:n_real])
    model.train()
    per_puzzle = evaluator.per_puzzle_pass_at_k((1, 2))  # collective gather
    if per_puzzle is None:
        return None, float("nan"), float("nan")
    mean1 = float(np.mean([v[1] for v in per_puzzle.values()]))
    mean2 = float(np.mean([v[2] for v in per_puzzle.values()]))
    return per_puzzle, mean1, mean2


def sweep(model, stream, order, cfg, rank, world):
    return evaluate(model, stream.eval_set, stream.eval_pair_idxs(order),
                    stream.gt_inputs, stream.gt_outputs, order, cfg, rank, world)


def arc1_probe(model, cfg, stream, rank, world):
    ds, pair_idxs, puzzle_ids = arc1_eval_set(
        cfg.data_dir, stream, cfg.arc1_eval_puzzles, cfg.arc1_eval_augs)
    return evaluate(model, ds, pair_idxs, ds.get_noaug_test_inputs(),
                    ds.get_noaug_test_outputs(), puzzle_ids, cfg, rank, world)


def joint_batches(stream, order, cfg, stage, local_batch, rank, world):
    """Puzzle-uniform batches: pick a puzzle, then batch_size of its pairs — the same
    per-puzzle expected budget and batch composition as the sequential arms. The
    schedule RNG is rank-free: every rank sees the same puzzle and the same global
    sample, sharded contiguously, so the union of rank batches is world-invariant."""
    rng = np.random.RandomState(cfg.seed * 100003 + stage)
    pools = [stream.train_pairs[stream.semantic_idx(p)] for p in order]
    batches = []
    for _ in range(cfg.stage_steps):
        pool = pools[rng.randint(len(pools))]
        sample = pool[rng.randint(len(pool), size=cfg.batch_size)]
        batches.append(sample[rank * local_batch:(rank + 1) * local_batch].tolist())
    return batches


def train_phase(model, arm, loader, cfg, gstep, world, embed_opt=None, estep=None,
                replay_iter=None, penalty=None):
    """One stage. embed_opt/estep are passed in by joint arms so semantic-row momentum
    and warmup persist across virtual stage boundaries; sequential arms use a fresh
    optimizer per stage. replay_iter and penalty support the CL baselines; replay
    and stream batches keep separate ACT carries so row-update gating by step type is
    exact (mixed carries would let pretraining samples receive updates on stream steps)."""
    model.train()
    if estep is None:  # plain-table variants have no embed_opt, so estep is the sentinel
        embed_opt = arm.fresh_embed_opt()
        estep = [0]
    trainable = [p for p in model.parameters() if p.requires_grad]
    carries = {"stream": None, "replay": None}
    exact_sum = count_sum = 0
    lm_sum = 0.0
    stream_steps = 0
    stream_iter = iter(loader)
    for step in range(1, cfg.stage_steps + 1):
        replay_step = replay_iter is not None and step % replay_iter[0] == 0
        source = "replay" if replay_step else "stream"
        _, batch, _ = next(replay_iter[1]) if replay_step else next(stream_iter)
        gpu_batch = to_device(batch, cfg.device)
        if carries[source] is None:
            with torch.device(cfg.device):
                carries[source] = model.initial_carry(gpu_batch)
        carries[source], loss, metrics, _, _ = model(
            carry=carries[source], batch=gpu_batch, return_keys=[])
        loss = loss / cfg.batch_size  # global batch; grads are summed across ranks
        if penalty is not None:
            loss = loss + penalty() / world  # penalty is per-model, grads sum over ranks
        loss.backward()

        reduced = torch.stack([metrics["exact_accuracy"], metrics["count"], metrics["lm_loss"]])
        if world > 1:
            for p in trainable:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
            dist.all_reduce(reduced)

        if not replay_step:
            estep[0] += 1
        row_scale = min(1.0, estep[0] / max(1, cfg.warmup_steps))
        if embed_opt is not None:
            if not replay_step:
                for group in embed_opt.param_groups:
                    group["lr"] = cfg.embed_lr * row_scale
                embed_opt.step()
            embed_opt.zero_grad()
        if arm.sparse_opt is not None:
            if not replay_step:
                for group in arm.sparse_opt.param_groups:
                    group["lr"] = cfg.sparse_lr * row_scale
                arm.sparse_opt.step()
            arm.sparse_opt.zero_grad()
        if arm.shared_opt is not None:
            gstep[0] += 1
            for group in arm.shared_opt.param_groups:
                group["lr"] = arm.shared_lr * min(1.0, gstep[0] / max(1, cfg.warmup_steps))
            arm.shared_opt.step()
            arm.shared_opt.zero_grad()
        model.zero_grad(set_to_none=True)

        if replay_step:
            continue
        stream_steps += 1
        exact_sum += float(reduced[0])
        count_sum += int(reduced[1])
        lm_sum += float(reduced[2]) / cfg.batch_size
    return {"exact": exact_sum / max(1, count_sum), "lm": lm_sum / max(1, stream_steps)}


def new_row_snapshot(model, stream, puzzle_ids):
    """CPU clones of the given puzzles' rows (semantic + instance span), for the
    stronger frozen-arm check: previously learned rows must stay bit-identical."""
    puzzle_emb = inner_module(model).puzzle_emb
    smart = isinstance(puzzle_emb, SmartTaskEmbedding)
    snap = {}
    for pid in puzzle_ids:
        pidx = stream.semantic_idx(pid)
        lo, hi = stream.instance_ranges[pidx]
        rows = {}
        if smart:
            rows["semantic"] = puzzle_emb.puzzle_embed.weight[pidx].detach().cpu().clone()
            if puzzle_emb.use_sparse_instance_residual:
                rows["instance"] = puzzle_emb.instance_residual_embed.weights[lo:hi].detach().cpu().clone()
        else:
            rows["instance"] = puzzle_emb.weights[lo:hi].detach().cpu().clone()
        snap[pid] = rows
    return snap


def new_row_drift(model, stream, snap):
    puzzle_emb = inner_module(model).puzzle_emb
    smart = isinstance(puzzle_emb, SmartTaskEmbedding)
    worst = 0.0
    for pid, rows in snap.items():
        pidx = stream.semantic_idx(pid)
        lo, hi = stream.instance_ranges[pidx]
        if "semantic" in rows:
            worst = max(worst, float((puzzle_emb.puzzle_embed.weight[pidx].detach().cpu()
                                      - rows["semantic"]).abs().max()))
        if "instance" in rows:
            table = puzzle_emb.instance_residual_embed.weights if smart else puzzle_emb.weights
            worst = max(worst, float((table[lo:hi].detach().cpu() - rows["instance"]).abs().max()))
    return worst


def save_state(cfg, model, arm, stage_next, gstep, current, records, order,
               R1_rows, R2_rows, event_stages, arc1_before, joint_embed_opt, joint_estep):
    torch.save(
        {"stage_next": stage_next, "gstep": gstep[0], "model": model.state_dict(),
         "shared_opt": arm.shared_opt.state_dict() if arm.shared_opt else None,
         "joint_embed_opt": joint_embed_opt.state_dict() if joint_embed_opt else None,
         "joint_estep": joint_estep,
         "current": current, "records": records, "order": order,
         "R1_rows": R1_rows, "R2_rows": R2_rows, "event_stages": event_stages,
         "arc1_before": arc1_before, "config": asdict(cfg)},
        os.path.join(cfg.out_dir, "latest.pt"))


def run(cfg, baseline=None):
    rank, world, _ = init_distributed_environment(device="cuda", backend="nccl", mpi=None)
    main = rank == 0
    assert cfg.device == "cuda", "CoSE-CL supports NVIDIA CUDA only"
    assert cfg.batch_size % world == 0, \
        "--batch_size must be divisible by the distributed world size"
    local_batch = cfg.batch_size // world
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    os.makedirs(cfg.out_dir, exist_ok=True)
    if main and not cfg.resume:
        assert not os.path.exists(os.path.join(cfg.out_dir, "log.jsonl")), \
            f"{cfg.out_dir} already holds a run; pass --resume or use a fresh out_dir"
    with open(cfg.ckpt_config) as f:
        base_cfg = yaml.safe_load(f)

    stream = Stream(base_cfg, cfg.data_dir, cfg.stream_parquet,
                    cfg.no_translation_ratio, cfg.eval_augs)
    order = stream.order(cfg.order_seed, cfg.n_puzzles)
    N = len(order)
    joint = cfg.arm in ("joint", "joint_all")
    train_shared = cfg.arm in ("naive", "reset", "joint_all") or baseline is not None
    if main:
        print(f"base vocab={stream.base_vocab} puzzles={stream.base_puzzles} "
              f"fingerprints={stream.fingerprints} | "
              f"expanded vocab={stream.vocab} puzzles={stream.num_puzzles} | "
              f"stream N={N} | world={world} local_batch={local_batch}")
        with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(asdict(cfg) | {"mapping_fingerprints": stream.fingerprints}
                           | ({"baseline": baseline.describe()} if baseline else {}), f)

    model = build_model(base_cfg["arch"], stream.vocab, stream.num_puzzles,
                        local_batch, cfg.device, halt_loss_weight=cfg.halt_loss_weight)
    info = load_expanded(model, cfg.ckpt, cfg.device)
    broadcast_model_state(model, world_size=world)
    if main:
        print(f"loaded {cfg.ckpt} step={info['step']} expanded={info['expanded']}")
    arm = setup_arm(model, cfg, train_shared,
                    comp_only=baseline.comp_only if baseline else False, world_size=world)
    if baseline:
        baseline.attach(model, arm, stream, cfg, local_batch, world)
    snap = base_row_snapshot(model, stream.base_vocab, stream.base_puzzles) if main else None
    base_state = None
    if cfg.arm == "reset":
        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    joint_embed_opt = joint_estep = None
    if joint:
        joint_embed_opt = arm.fresh_embed_opt()
        joint_estep = [0]

    current = {}
    R1_rows, R2_rows, event_stages = [], [], [-1]
    records = []
    gstep = [0]
    start_stage = 0
    arc1_before = (float("nan"), float("nan"))
    learned_snap = {}  # rows of already-learned puzzles; references are permanent
    latest = os.path.join(cfg.out_dir, "latest.pt")
    resuming = cfg.resume and os.path.exists(latest)
    if resuming:
        state = torch.load(latest, map_location=cfg.device, weights_only=False)
        saved = {k: v for k, v in state["config"].items() if k != "resume"}
        # Runs saved before this option used the fixed 0.5 coefficient.
        saved.setdefault("halt_loss_weight", 0.5)
        current_cfg = {k: v for k, v in asdict(cfg).items() if k != "resume"}
        assert saved == current_cfg and state["order"] == order, \
            "resume config/order mismatch — refusing to mix experiments"
        model.load_state_dict(state["model"])
        if arm.shared_opt and state["shared_opt"]:
            arm.shared_opt.load_state_dict(state["shared_opt"])
        if joint_embed_opt and state["joint_embed_opt"]:
            joint_embed_opt.load_state_dict(state["joint_embed_opt"])
            joint_estep = state["joint_estep"]
        records, gstep = state["records"], [state["gstep"]]
        start_stage = state["stage_next"]
        arc1_before = state["arc1_before"]
        # resume is statistical, not bitwise (online augs are unseeded): reseed
        # deterministically so the continued stream diverges from a fresh run
        torch.manual_seed(cfg.seed + rank + start_stage * 7919)
        np.random.seed((cfg.seed + rank + start_stage * 7919) % 2 ** 32)
        if main:
            current = state["current"]
            R1_rows, R2_rows = state["R1_rows"], state["R2_rows"]
            event_stages = state["event_stages"]
            learned_snap = new_row_snapshot(model, stream,
                                            order if joint else order[:start_stage])
            print(f"resumed at stage {start_stage}")

    log_f = open(os.path.join(cfg.out_dir, "log.jsonl"), "a") if main else None
    if main:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.init(project=cfg.project_name, name=cfg.run_name or os.path.basename(cfg.out_dir),
                   config=asdict(cfg) | {"trainable_params": n_trainable, "world_size": world,
                                         "mapping_fingerprints": stream.fingerprints}
                   | ({"baseline": baseline.describe()} if baseline else {}))

    def log(record):
        if log_f is None:
            return
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()
        prefix = record["type"] + ("_" + record["when"] if record["type"] == "arc1" else "")
        payload = {f"{prefix}/{k}": v for k, v in record.items()
                   if isinstance(v, (int, float)) and k != "stage"}
        # x-axis = stages completed: zero-shot/arc1_before at 0, stage i at i+1
        wandb.log(payload, step=record.get("stage", -1) + 1)

    if cfg.arc1_eval_puzzles > 0 and not resuming:
        _, p1, p2 = arc1_probe(model, cfg, stream, rank, world)
        arc1_before = (p1, p2)
        if main:
            print(f"arc1 before: pass@1={p1:.4f} pass@2={p2:.4f}")
            log({"type": "arc1", "when": "before", "stage": -1,
                 "pass1": p1, "pass2": p2})

    # zero-shot sweep = the "pure composition" baseline: blank rows, untouched base model
    # (skipped on resume: it is deterministic and already recorded as R rows[0])
    if not resuming:
        zero_pp, zero1, zero2 = sweep(model, stream, order, cfg, rank, world)
        if main:
            print(f"zero-shot (pure composition): pass@1={zero1:.4f} pass@2={zero2:.4f}")
            log({"type": "sweep", "stage": -1, "pass1": zero1, "pass2": zero2,
                 "mismatch1": 0, "mismatch2": 0})
            current = {pid: zero_pp[pid] for pid in order}
            R1_rows = [[zero_pp[p][1] for p in order]]
            R2_rows = [[zero_pp[p][2] for p in order]]
    elif main:
        zero1 = float(np.mean(R1_rows[0]))
        zero2 = float(np.mean(R2_rows[0]))

    for stage in range(start_stage, N):
        if cfg.arm == "reset":
            model.load_state_dict(base_state)
            arm = setup_arm(model, cfg, train_shared, world_size=world)
            gstep = [0]
        t0 = time.time()
        if joint:
            loader = DataLoader(
                stream.identity_view(), num_workers=2,
                batch_sampler=joint_batches(stream, order, cfg, stage, local_batch, rank, world),
                collate_fn=partial(puzzle_collate_fn, set_name="train"))
        else:
            view = stream.train_view([order[stage]])
            generator = torch.Generator()
            generator.manual_seed(cfg.seed * 100003 + stage * world + rank)
            loader = DataLoader(
                view, batch_size=local_batch, num_workers=2, drop_last=True,
                sampler=RandomSampler(view, replacement=True,
                                      num_samples=local_batch * cfg.stage_steps,
                                      generator=generator),
                collate_fn=partial(puzzle_collate_fn, set_name="train"))
        stats = train_phase(model, arm, loader, cfg, gstep, world,
                            embed_opt=joint_embed_opt, estep=joint_estep,
                            **(baseline.phase_kwargs() if baseline else {}))
        t_train = time.time() - t0

        record = {"type": "stage", "stage": stage, **stats, "t_train": t_train}
        if not joint:
            pid = order[stage]
            pp, _, _ = evaluate(model, stream.eval_set,
                                stream.eval_pairs[stream.semantic_idx(pid)],
                                stream.gt_inputs, stream.gt_outputs, [pid], cfg, rank, world)
            if main:
                current[pid] = pp[pid]
                record |= {"puzzle_id": pid, "pass1": pp[pid][1], "pass2": pp[pid][2]}
                if cfg.arm != "reset":
                    # permanent post-learning reference: the drift check must cover a
                    # puzzle from the moment its rows are written
                    learned_snap |= new_row_snapshot(model, stream, [pid])
        if main:
            cum1 = float(np.mean([current[p][1] for p in order]))
            cum2 = float(np.mean([current[p][2] for p in order]))
            record |= {"cum1": cum1, "cum2": cum2}
            records.append(record)
            log(record)
            label = "joint" if joint else record["puzzle_id"]
            print(f"stage {stage + 1:3d}/{N} {label} "
                  f"exact={stats['exact']:.2f} lm={stats['lm']:.3f} "
                  f"pass@1/2={record.get('pass1', float('nan')):.2f}/"
                  f"{record.get('pass2', float('nan')):.2f} cum@1/2={cum1:.3f}/{cum2:.3f} "
                  f"{t_train:.0f}s")

        checkpoint = (stage + 1) % cfg.eval_every == 0 or stage == N - 1
        # No sweeps for reset: each stage's model is independent, so re-checking all
        # puzzles with the last model would clobber the oracle's per-puzzle tally.
        if checkpoint and cfg.arm != "reset":
            sweep_pp, s1, s2 = sweep(model, stream, order, cfg, rank, world)
            if main:
                seen = set(order) if joint else set(order[: stage + 1])
                mismatch1 = sum(1 for p in order if p in seen and sweep_pp[p][1] != current[p][1])
                mismatch2 = sum(1 for p in order if p in seen and sweep_pp[p][2] != current[p][2])
                current = {p: sweep_pp[p] for p in order}
                R1_rows.append([current[p][1] for p in order])
                R2_rows.append([current[p][2] for p in order])
                event_stages.append(stage)
                drift = base_row_drift(model, snap)
                learned_drift = new_row_drift(model, stream, learned_snap)
                if joint:  # joint rows train throughout; sequential references are permanent
                    learned_snap = new_row_snapshot(model, stream, sorted(seen))
                log({"type": "sweep", "stage": stage, "pass1": s1, "pass2": s2,
                     "mismatch1": mismatch1, "mismatch2": mismatch2,
                     "learned_row_drift": learned_drift,
                     **{f"drift_{k}": v for k, v in drift.items()}})
                print(f"  sweep@{stage}: pass@1={s1:.4f} pass@2={s2:.4f} "
                      f"mismatch@1/2={mismatch1}/{mismatch2} base_row_drift={drift} "
                      f"learned_row_drift={learned_drift}")
        if checkpoint:
            if main:
                save_state(cfg, model, arm, stage + 1, gstep, current, records, order,
                           R1_rows, R2_rows, event_stages, arc1_before,
                           joint_embed_opt, joint_estep)
            if dist.is_initialized():
                dist.barrier()

    arc1_after = (float("nan"), float("nan"))
    if cfg.arc1_eval_puzzles > 0:
        _, p1, p2 = arc1_probe(model, cfg, stream, rank, world)
        arc1_after = (p1, p2)
        if main:
            print(f"arc1 after: pass@1={p1:.4f} pass@2={p2:.4f}")
            log({"type": "arc1", "when": "after", "stage": N - 1,
                 "pass1": p1, "pass2": p2})

    if main:
        stage_recs = [r for r in records if "puzzle_id" in r]
        np.savez(
            os.path.join(cfg.out_dir, "results.npz"),
            order=np.array(order),
            zero1=np.array(R1_rows[0]),
            zero2=np.array(R2_rows[0]),
            final1=np.array([current[p][1] for p in order]),
            final2=np.array([current[p][2] for p in order]),
            R1=np.array(R1_rows), R2=np.array(R2_rows), event_stages=np.array(event_stages),
            stage_pass1=np.array([r["pass1"] for r in stage_recs]),
            stage_pass2=np.array([r["pass2"] for r in stage_recs]),
            arc1_before=np.array(arc1_before), arc1_after=np.array(arc1_after),
        )
        log_f.close()
        final1 = float(np.mean([current[p][1] for p in order]))
        final2 = float(np.mean([current[p][2] for p in order]))
        wandb.summary.update({
            "final/pass1": final1, "final/pass2": final2,
            "zero/pass1": zero1, "zero/pass2": zero2,
            "arc1/before_pass2": arc1_before[1], "arc1/after_pass2": arc1_after[1],
            "arc1/retention": arc1_after[1] / arc1_before[1] if arc1_before[1] else float("nan")})
        wandb.finish()
        print(f"done: final all-{N} pass@1={final1:.4f} pass@2={final2:.4f}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
