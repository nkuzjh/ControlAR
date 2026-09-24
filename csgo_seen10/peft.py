"""Explicit LoRA and parameter policy for the independent Seen-10 PEFT run."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 64, dropout: float = 0.05):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires positive rank/alpha and dropout in [0, 1)")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight.add_((self.lora_B @ self.lora_A).to(self.base.weight.dtype), alpha=self.scaling)
        return self.base


class QKVLoRALinear(nn.Module):
    """One frozen fused base, with separate rank-r Q, K and V updates."""

    def __init__(self, base: nn.Linear, q_size: int, kv_size: int, rank: int = 32,
                 alpha: float = 64, dropout: float = 0.05):
        super().__init__()
        if base.out_features != q_size + 2 * kv_size:
            raise ValueError("Fused QKV dimensions disagree with attention layout")
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires positive rank/alpha and dropout in [0, 1)")
        self.base = base
        self.base.requires_grad_(False)
        self.q_size, self.kv_size = q_size, kv_size
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.dropout = nn.Dropout(dropout)
        for label, out_size in (("q", q_size), ("k", kv_size), ("v", kv_size)):
            a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
            b = nn.Parameter(torch.zeros(out_size, rank, device=base.weight.device, dtype=base.weight.dtype))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            setattr(self, f"{label}_A", a)
            setattr(self, f"{label}_B", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(x)
        delta = torch.cat(tuple(
            F.linear(F.linear(dropped, getattr(self, f"{label}_A")), getattr(self, f"{label}_B"))
            for label in ("q", "k", "v")
        ), dim=-1)
        return self.base(x) + delta * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        delta = torch.cat(tuple(
            getattr(self, f"{label}_B") @ getattr(self, f"{label}_A")
            for label in ("q", "k", "v")
        ), dim=0)
        self.base.weight.add_(delta.to(self.base.weight.dtype), alpha=self.scaling)
        return self.base


def inject_lora(gpt: nn.Module, rank: int = 32, alpha: float = 64,
                dropout: float = 0.05) -> nn.Module:
    layers = gpt.layers
    if len(layers) != 36:
        raise ValueError(f"PEFT GPT-XL requires 36 blocks, got {len(layers)}")
    for block in layers:
        attention = block.attention
        if isinstance(attention.wqkv, (LoRALinear, QKVLoRALinear)):
            raise ValueError("LoRA is already injected")
        kv_size = attention.n_kv_head * attention.head_dim
        attention.wqkv = QKVLoRALinear(attention.wqkv, attention.dim, kv_size, rank, alpha, dropout)
        attention.wo = LoRALinear(attention.wo, rank, alpha, dropout)
        for name in ("w1", "w3", "w2"):
            feed_forward = block.feed_forward
            setattr(feed_forward, name, LoRALinear(getattr(feed_forward, name), rank, alpha, dropout))
    return gpt


@torch.no_grad()
def merge_lora_(gpt: nn.Module) -> nn.Module:
    for block in gpt.layers:
        attention = block.attention
        for name in ("wqkv", "wo"):
            module = getattr(attention, name)
            if not isinstance(module, (LoRALinear, QKVLoRALinear)):
                raise ValueError(f"Cannot merge {name}: LoRA is absent")
            setattr(attention, name, module.merge())
        for name in ("w1", "w3", "w2"):
            module = getattr(block.feed_forward, name)
            if not isinstance(module, LoRALinear):
                raise ValueError(f"Cannot merge {name}: LoRA is absent")
            setattr(block.feed_forward, name, module.merge())
    return gpt


def parameter_role(name: str) -> str:
    if name.startswith("pose_map_embedder."):
        return "pose"
    if name.startswith("gpt.layers."):
        if any(name.endswith(suffix) for suffix in (
            ".lora_A", ".lora_B", ".q_A", ".q_B", ".k_A", ".k_B", ".v_A", ".v_B"
        )):
            return "lora"
        return "frozen_base"
    if name.startswith(("gpt.cls_embedding.cap_proj.", "gpt.condition_mlp.cap_proj.",
                        "gpt.condition_layers.", "gpt.output.")):
        return "pretrained_full"
    return "frozen_other"


def configure_trainable(model: nn.Module) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        role = parameter_role(name)
        parameter.requires_grad_(role in ("pose", "lora", "pretrained_full"))
        totals[role] += parameter.numel()
    if totals["lora"] != 28_606_464 or totals["pose"] != 2_371_584 or totals["pretrained_full"] != 38_338_560:
        raise AssertionError(f"Unexpected PEFT parameter roles: {dict(totals)}")
    return dict(totals)


def build_optimizer(model: nn.Module, *, lora_lr: float = 1e-4, pose_lr: float = 1e-4,
                    pretrained_lr: float = 5e-5, weight_decay: float = 0.05,
                    betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                    fused: bool | None = None) -> torch.optim.AdamW:
    grouped: dict[str, list[nn.Parameter]] = {
        "lora": [], "pose": [], "pretrained_decay": [], "pretrained_no_decay": []
    }
    for name, parameter in model.named_parameters():
        role = parameter_role(name)
        if not parameter.requires_grad:
            continue
        if role == "pretrained_full":
            key = "pretrained_decay" if parameter.ndim >= 2 else "pretrained_no_decay"
        elif role in ("lora", "pose"):
            key = role
        else:
            raise AssertionError(f"Unexpected trainable parameter: {name}")
        grouped[key].append(parameter)
    spec = (("lora", lora_lr, 0.0), ("pose", pose_lr, 0.0),
            ("pretrained_decay", pretrained_lr, weight_decay),
            ("pretrained_no_decay", pretrained_lr, 0.0))
    groups = [{"params": grouped[name], "lr": lr, "weight_decay": wd, "role": name,
               "peak_lr": lr} for name, lr, wd in spec if grouped[name]]
    use_fused = torch.cuda.is_available() if fused is None else fused
    return torch.optim.AdamW(groups, betas=betas, eps=eps, fused=use_fused)


def audit_parameters(model: nn.Module, optimizer: torch.optim.Optimizer,
                     *, vq_model: nn.Module | None = None) -> dict[str, Any]:
    blocks = model.gpt.layers
    if len(blocks) != 36:
        raise AssertionError(f"Expected 36 GPT LoRA blocks, got {len(blocks)}")
    lora_coverage = []
    for index, block in enumerate(blocks):
        expected = (("attention.wqkv", block.attention.wqkv, QKVLoRALinear),
                    ("attention.wo", block.attention.wo, LoRALinear),
                    ("feed_forward.w1", block.feed_forward.w1, LoRALinear),
                    ("feed_forward.w3", block.feed_forward.w3, LoRALinear),
                    ("feed_forward.w2", block.feed_forward.w2, LoRALinear))
        for suffix, module, expected_type in expected:
            if not isinstance(module, expected_type) or module.rank != 32 or module.alpha != 64 or module.dropout.p != 0.05:
                raise AssertionError(f"Missing or noncanonical LoRA at gpt.layers.{index}.{suffix}")
            if module.scaling != 2 or module.base.bias is not None:
                raise AssertionError(f"LoRA scaling/bias changed at gpt.layers.{index}.{suffix}")
            lora_coverage.append(f"gpt.layers.{index}.{suffix}")
    indexed: dict[int, tuple[int, dict[str, Any]]] = {}
    for index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            if id(parameter) in indexed:
                raise AssertionError("Parameter appears in multiple optimizer groups")
            indexed[id(parameter)] = (index, group)
    totals: Counter[str] = Counter()
    rows = []
    for name, parameter in model.named_parameters():
        role = parameter_role(name)
        expected_trainable = role in ("lora", "pose", "pretrained_full")
        if parameter.requires_grad != expected_trainable or (id(parameter) in indexed) != expected_trainable:
            raise AssertionError(f"PEFT trainability/optimizer mismatch: {name}")
        totals[role] += parameter.numel()
        index, group = indexed.get(id(parameter), (None, None))
        expected_group = ("lora" if role == "lora" else "pose" if role == "pose" else
                          "pretrained_decay" if parameter.ndim >= 2 else "pretrained_no_decay")
        if group is not None and group["role"] != expected_group:
            raise AssertionError(f"Wrong optimizer group for {name}")
        rows.append({"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
                     "requires_grad": parameter.requires_grad, "role": role, "optimizer_group": index,
                     "peak_lr": None if group is None else group["peak_lr"],
                     "weight_decay": None if group is None else group["weight_decay"]})
    if totals["lora"] != 28_606_464 or sum(totals[role] for role in ("lora", "pose", "pretrained_full")) != 69_316_608:
        raise AssertionError(f"PEFT parameter count changed: {dict(totals)}")
    vq_rows = []
    if vq_model is not None:
        for name, parameter in vq_model.named_parameters():
            if parameter.requires_grad or id(parameter) in indexed:
                raise AssertionError(f"VQ parameter must be frozen and excluded from optimizer: {name}")
            vq_rows.append({"name": name, "shape": list(parameter.shape),
                            "numel": parameter.numel(), "requires_grad": False,
                            "optimizer_group": None})
    return {"total_numel": sum(totals.values()), "trainable_numel": 69_316_608,
            "optimizer_numel": sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
            "roles": dict(totals), "vq_frozen": True if vq_model is not None else None,
            "vq_parameters": vq_rows,
            "lora_coverage": lora_coverage,
            "optimizer_defaults": {"betas": list(optimizer.defaults["betas"]), "eps": optimizer.defaults["eps"]},
            "parameters": rows}
