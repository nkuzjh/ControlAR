"""PEFT-only compiled decode boundary for native CUDA inverse-CDF sampling."""

from __future__ import annotations

from typing import Any

import torch

from csgo_seen10.compiled_inference import COMPILE_MODE, _decode_step_logits


def prepare_peft_compiled_logits_step() -> Any:
    """Compile Transformer, CFG and top-k without including the CDF scan."""

    return torch.compile(_decode_step_logits, mode=COMPILE_MODE, fullgraph=True)
