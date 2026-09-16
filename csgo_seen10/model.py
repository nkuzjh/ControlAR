from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from autoregressive.models.gpt_t2i import GPT_models


class NumericPoseMapEmbedder(nn.Module):
    """Project normalized 5DoF and fixed-map identity into native caption tokens."""

    def __init__(self, *, caption_dim: int = 2048, token_count: int = 120, map_count: int = 10):
        super().__init__()
        self.caption_dim = int(caption_dim)
        self.token_count = int(token_count)
        self.pose_mlp = nn.Sequential(
            nn.Linear(5, 1024),
            nn.SiLU(),
            nn.Linear(1024, self.caption_dim),
        )
        self.map_embedding = nn.Embedding(map_count, self.caption_dim)
        self.token_embedding = nn.Parameter(torch.empty(1, self.token_count, self.caption_dim))
        nn.init.normal_(self.map_embedding.weight, std=0.02)
        nn.init.normal_(self.token_embedding, std=0.02)

    def forward(self, pose: torch.Tensor, map_id: torch.Tensor) -> torch.Tensor:
        pose = pose.to(dtype=self.pose_mlp[0].weight.dtype)
        map_id = map_id.to(dtype=torch.long)
        feature = self.pose_mlp(pose) + self.map_embedding(map_id)
        return feature[:, None, :] + self.token_embedding


class Seen10GenerationModel(nn.Module):
    def __init__(self, gpt: nn.Module, *, caption_dim: int = 2048, token_count: int = 120):
        super().__init__()
        self.gpt = gpt
        self.pose_map_embedder = NumericPoseMapEmbedder(caption_dim=caption_dim, token_count=token_count)

    def forward(
        self,
        *,
        pose: torch.Tensor,
        map_id: torch.Tensor,
        idx: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        caption = self.pose_map_embedder(pose, map_id)
        return self.gpt(
            cond_idx=caption,
            idx=idx,
            targets=targets,
            mask=mask,
            condition=condition,
        )


def build_gpt(
    *,
    model_name: str = "GPT-XL",
    image_size: int = 448,
    downsample_size: int = 16,
    token_count: int = 120,
    adapter_size: str = "small",
    condition_type: str = "radar",
    dropout: float = 0.1,
    token_dropout: float = 0.1,
) -> nn.Module:
    latent_size = image_size // downsample_size
    if latent_size * downsample_size != image_size:
        raise ValueError(f"image_size={image_size} must divide evenly by stride {downsample_size}")
    return GPT_models[model_name](
        vocab_size=16384,
        block_size=latent_size**2,
        num_classes=1000,
        cls_token_num=token_count,
        model_type="t2i",
        resid_dropout_p=dropout,
        ffn_dropout_p=dropout,
        token_dropout_p=token_dropout,
        adapter_size=adapter_size,
        condition_type=condition_type,
    )


def _checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "module", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    raise ValueError(f"Could not find a model state_dict in checkpoint {path}")


def load_official_gpt_weights(gpt: nn.Module, checkpoint_path: str | Path) -> None:
    """Load official ControlAR weights strictly, resizing only a zero-valued buffer."""
    source = _checkpoint_state(checkpoint_path)
    target = gpt.state_dict()

    # Accommodate common wrapper prefixes without weakening key validation.
    normalized: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in normalized:
            raise ValueError(f"Duplicate checkpoint key after prefix normalization: {key}")
        normalized[key] = value

    missing = sorted(set(target) - set(normalized))
    unexpected = sorted(set(normalized) - set(target))
    if missing or unexpected:
        raise RuntimeError(
            "Official GPT checkpoint keys do not match the requested GPT-XL model; "
            f"missing={missing[:12]} ({len(missing)} total), "
            f"unexpected={unexpected[:12]} ({len(unexpected)} total)"
        )

    loadable: dict[str, torch.Tensor] = {}
    resized_buffer = False
    for key, expected in target.items():
        value = normalized[key]
        if value.shape != expected.shape:
            if key == "condition_mlp.uncond_embedding" and value.ndim == 2 and expected.ndim == 2:
                if value.shape[1] != expected.shape[1]:
                    raise RuntimeError(
                        f"Cannot resize {key}: feature dimensions differ {tuple(value.shape)} vs {tuple(expected.shape)}"
                    )
                # This registered unconditional control buffer is all zeros in
                # the native model. Its first dimension follows image block_size
                # (1024 for 512px, 784 for 448px), so keep the target-size zeros.
                if torch.count_nonzero(value).item() != 0:
                    raise RuntimeError(f"Expected {key} to be an all-zero native buffer before resizing")
                loadable[key] = expected
                resized_buffer = True
                continue
            raise RuntimeError(
                f"Official GPT tensor {key} has shape {tuple(value.shape)}, "
                f"expected {tuple(expected.shape)}"
            )
        loadable[key] = value

    gpt.load_state_dict(loadable, strict=True)
    if resized_buffer:
        print("Loaded official GPT weights strictly; kept the 448px-sized zero control buffer")
    else:
        print("Loaded official GPT weights strictly")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
