import math
from typing import Any, Iterable

import torch
from torch.optim import AdamW


def _materialize_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return list(parameters)


def _split_muon_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon_parameters = []
    aux_adam_parameters = []
    for parameter in parameters:
        if parameter.ndim == 2:
            muon_parameters.append(parameter)
        else:
            aux_adam_parameters.append(parameter)
    return muon_parameters, aux_adam_parameters


def _muon_newton_schulz_step(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    gram = x @ x.mT
    poly = torch.addmm(gram, gram, gram, alpha=c, beta=b)
    return torch.addmm(x, poly, x, alpha=1.0, beta=a)


def _muon_orthogonalize(gradient: torch.Tensor, steps: int) -> torch.Tensor:
    if gradient.ndim != 2:
        raise ValueError(f"Muon expects 2D tensors, got shape {tuple(gradient.shape)}")

    x = gradient.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT

    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    coeffs = [
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ]
    for step in range(steps):
        a, b, c = coeffs[min(step, len(coeffs) - 1)]
        x = _muon_newton_schulz_step(x, a, b, c)

    return x.mT if transposed else x


def _adjust_muon_lr(lr: float, matched_adamw_rms: float, shape: torch.Size) -> float:
    rows, cols = shape[:2]
    return lr * math.sqrt(max(rows, cols)) * matched_adamw_rms


class _AuxAdamMixin:
    def _step_aux_adam(self) -> None:
        for group in self.param_groups:
            if group.get("use_muon", False):
                continue

            group["step"] = group.get("step", 0) + 1
            step = group["step"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue

                state = self.state[parameter]
                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(gradient)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(gradient)

                exp_avg = state["adamw_exp_avg"]
                exp_avg_sq = state["adamw_exp_avg_sq"]
                exp_avg.lerp_(gradient, 1 - beta1)
                exp_avg_sq.lerp_(gradient.square(), 1 - beta2)

                update = exp_avg / (exp_avg_sq.sqrt() + eps)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                parameter.mul_(1 - lr * weight_decay)
                parameter.add_(update, alpha=-(lr * math.sqrt(bias_correction2) / bias_correction1))


class MuonWithAuxAdam(_AuxAdamMixin, torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict[str, Any]]):
        for group in param_groups:
            if group.get("use_muon", False):
                group.setdefault("lr", 0.0)
                group.setdefault("weight_decay", 0.1)
                group.setdefault("matched_adamw_rms", 0.2)
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
            else:
                group.setdefault("lr", 0.0)
                group.setdefault("weight_decay", 0.1)
                group.setdefault("adamw_betas", (0.9, 0.95))
                group.setdefault("adamw_eps", 1e-8)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            matched_adamw_rms = group["matched_adamw_rms"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue

                state = self.state[parameter]
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(gradient)

                muon_buffer = state["muon_buffer"]
                muon_buffer.mul_(momentum).add_(gradient)
                update = gradient.add(muon_buffer, alpha=momentum) if nesterov else muon_buffer
                update = _muon_orthogonalize(update, steps=ns_steps)

                parameter.mul_(1 - lr * weight_decay)
                parameter.add_(
                    update,
                    alpha=-_adjust_muon_lr(lr, matched_adamw_rms, update.shape),
                )

        self._step_aux_adam()
        return loss


def build_dense_optimizer(parameters: Iterable[torch.nn.Parameter], config: Any) -> torch.optim.Optimizer:
    parameters = _materialize_parameters(parameters)
    optimizer_name = getattr(config, "optimizer", "adamw")

    if optimizer_name == "adamw":
        return AdamW(
            parameters,
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

    if optimizer_name != "muon":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")

    muon_parameters, aux_adam_parameters = _split_muon_parameters(parameters)

    if not muon_parameters:
        return AdamW(
            parameters,
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

    optimizer_groups = [
        {
            "params": muon_parameters,
            "use_muon": True,
            "lr": 0.0,
            "weight_decay": config.weight_decay,
        }
    ]
    if aux_adam_parameters:
        optimizer_groups.append(
            {
                "params": aux_adam_parameters,
                "use_muon": False,
                "lr": 0.0,
                "weight_decay": config.weight_decay,
                "adamw_betas": (config.beta1, config.beta2),
                "adamw_eps": 1e-8,
            }
        )

    return MuonWithAuxAdam(optimizer_groups)
