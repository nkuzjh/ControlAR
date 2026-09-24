#!/usr/bin/env python3
"""Test-only deterministic backend for a small exact-resume acceptance run."""
from __future__ import annotations
import os
import runpy
import sys
from pathlib import Path

if '--smoke' not in sys.argv:
    raise SystemExit('This acceptance wrapper requires --smoke; formal training is not supported')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
# Math SDPA avoids nondeterministic fast attention backward in this test only.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
print('Acceptance-only: deterministic_algorithms=True, SDPA=math, CUBLAS_WORKSPACE_CONFIG=:4096:8',flush=True)
runpy.run_path(str(ROOT/'train_seen10_peft.py'),run_name='__main__')
