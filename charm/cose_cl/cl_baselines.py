"""Classical CL-algorithm baselines on the same harness, for the comparison table.

One representative per family, each at two scopes (all shared weights, or only the
CoSE composition params), with rows always training as in the core arms:
  ewc        Elastic Weight Consolidation: quadratic penalty weighted by a diagonal
             Fisher estimated on ARC1 pretraining data
  l2sp       L2-SP: plain quadratic anchor to the base model weights (EWC with identity Fisher)
  rehearsal  experience replay: every N-th step trains on a pretraining batch.
             Stream puzzles are never revisited.
             Replay replaces stream steps, keeping the total number of updates fixed

Kept separate from train.py so the core method carries no baseline machinery.
Function-space methods (LwF) are omitted: replay of real source data dominates
distillation when that data is available, as it is here.
"""

import argparse
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

from charm.cose_cl.config import Config, add_shared_args
from charm.cose_cl.data import ARCPerPairDataset, train_transforms
from charm.cose_cl.train import run, to_device
from charm.datasets.puzzle_dataloader import puzzle_collate_fn


class Baseline:
    def __init__(self, algo: str, scope: str, reg_lambda: float, rehearsal_every: int,
                 fisher_batches: int):
        self.algo = algo
        self.scope = scope
        self.comp_only = scope == "comp"
        self.reg_lambda = reg_lambda
        self.rehearsal_every = rehearsal_every
        self.fisher_batches = fisher_batches
        self.anchors = []
        self.replay_iter = None

    def describe(self) -> dict:
        return {"algo": self.algo, "scope": self.scope, "reg_lambda": self.reg_lambda,
                "rehearsal_every": self.rehearsal_every, "fisher_batches": self.fisher_batches}

    def attach(self, model, arm, stream, cfg: Config, local_batch: int, world: int):
        if self.algo == "rehearsal":
            pretrain_set = ARCPerPairDataset(
                stream.pretrain_paths, pair_types=stream.pretrain_pair_types,
                path_multiplicities=stream.pretrain_multiplicities, eval_mode=False,
                online_transforms=train_transforms(cfg.no_translation_ratio))
            loader = DataLoader(
                pretrain_set, batch_size=local_batch, num_workers=2, drop_last=True,
                sampler=RandomSampler(pretrain_set, replacement=True, num_samples=10 ** 9),
                collate_fn=partial(puzzle_collate_fn, set_name="replay"))
            self.replay_iter = (self.rehearsal_every, iter(loader))
            return

        fisher = [torch.ones_like(p) for p in arm.shared_params]
        if self.algo == "ewc":
            fisher = self._estimate_fisher(model, arm, stream, cfg, local_batch, world)
        self.anchors = [(p, p.detach().clone(), f)
                        for p, f in zip(arm.shared_params, fisher)]

    def _estimate_fisher(self, model, arm, stream, cfg: Config, local_batch: int, world: int):
        """Diagonal Fisher of the task loss on pretraining data, normalized to mean 1."""
        pretrain_set = ARCPerPairDataset(
            stream.pretrain_paths, pair_types=stream.pretrain_pair_types,
            path_multiplicities=stream.pretrain_multiplicities, eval_mode=False,
            online_transforms=train_transforms(cfg.no_translation_ratio))
        loader = DataLoader(
            pretrain_set, batch_size=local_batch, num_workers=2, drop_last=True,
            sampler=RandomSampler(pretrain_set, replacement=True,
                                  num_samples=local_batch * self.fisher_batches),
            collate_fn=partial(puzzle_collate_fn, set_name="fisher"))
        fisher = [torch.zeros_like(p) for p in arm.shared_params]
        model.train()
        carry = None  # persistent, so Fisher sees the multi-step ACT state distribution
        for _, batch, _ in loader:
            gpu_batch = to_device(batch, cfg.device)
            if carry is None:
                with torch.device(cfg.device):
                    carry = model.initial_carry(gpu_batch)
            carry, loss, _, _, _ = model(carry=carry, batch=gpu_batch, return_keys=[])
            (loss / cfg.batch_size).backward()
            for f, p in zip(fisher, arm.shared_params):
                f += p.grad.detach() ** 2
            model.zero_grad(set_to_none=True)
        if arm.sparse_opt is not None:
            # sparse local_weights are grad-bearing buffers, not parameters: clear them
            # or the Fisher pass's pretraining gradients pollute the first stream update
            arm.sparse_opt.zero_grad()
        if world > 1:
            for f in fisher:  # identical penalties on every rank
                dist.all_reduce(f)
        total = sum(f.sum() for f in fisher)
        numel = sum(f.numel() for f in fisher)
        for f in fisher:
            f *= numel / total  # mean 1, so reg_lambda is comparable to l2sp
        print(f"fisher estimated on {self.fisher_batches} pretraining batches")
        return fisher

    def penalty(self):
        return 0.5 * self.reg_lambda * sum(
            (f * (p - ref).pow(2)).sum() for p, ref, f in self.anchors)

    def phase_kwargs(self) -> dict:
        if self.algo == "rehearsal":
            return {"replay_iter": self.replay_iter}
        return {"penalty": self.penalty}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", required=True, choices=["ewc", "l2sp", "rehearsal"])
    p.add_argument("--scope", default="all", choices=["all", "comp"])
    p.add_argument("--reg_lambda", type=float, default=1e-2)
    p.add_argument("--rehearsal_every", type=int, default=4)
    p.add_argument("--fisher_batches", type=int, default=64)
    add_shared_args(p)
    args = vars(p.parse_args())
    baseline = Baseline(args.pop("algo"), args.pop("scope"), args.pop("reg_lambda"),
                        args.pop("rehearsal_every"), args.pop("fisher_batches"))
    cfg = Config(arm=f"{baseline.algo}_{baseline.scope}", **args)
    run(cfg, baseline=baseline)


if __name__ == "__main__":
    main()
