#!/usr/bin/env python3
"""Small CPU-only correctness checks for PEFT math, batch budgets and target isolation."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from csgo_seen10.peft import LoRALinear, QKVLoRALinear
from csgo_seen10.data import Seen10GenerationDataset, read_benchmark_rows
from scripts.validate_csgo_seen10_peft import check_batch, ContractError
from train_seen10_peft import lr_factor, validate_batch


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(73)
    errors = {}
    for label, wrapper in (
        ("linear", LoRALinear(nn.Linear(7, 9, bias=False).double(), rank=3, alpha=6, dropout=0)),
        # Intentionally use unequal Q/K/V widths to exercise the fused split.
        ("qkv", QKVLoRALinear(nn.Linear(7, 10, bias=False).double(), q_size=6, kv_size=2,
                              rank=3, alpha=6, dropout=0)),
    ):
        x = torch.randn(4, 5, 7, dtype=torch.float64)
        wrapper.eval()
        assert torch.equal(wrapper(x), wrapper.base(x)), label
        with torch.no_grad():
            if label == "linear":
                wrapper.lora_B.normal_(std=0.1)
                delta = wrapper.lora_B @ wrapper.lora_A
            else:
                for part in ("q", "k", "v"):
                    getattr(wrapper, part + "_B").normal_(std=0.1)
                delta = torch.cat([getattr(wrapper, p + "_B") @ getattr(wrapper, p + "_A")
                                   for p in ("q", "k", "v")], dim=0)
        actual = wrapper(x)
        manual = F.linear(x, wrapper.base.weight + 2 * delta)
        torch.testing.assert_close(actual, manual, rtol=1e-12, atol=1e-12)
        actual.square().mean().backward()
        assert wrapper.base.weight.grad is None
        for name, p in wrapper.named_parameters():
            if not name.startswith("base."):
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        before = wrapper(x).detach()
        merged = wrapper.merge()
        after = merged(x).detach()
        errors[label] = (before-after).abs().max().item()
        torch.testing.assert_close(before, after, rtol=1e-12, atol=1e-12)
    # Accumulation must average losses, not add an extra factor of micro batch.
    base = LoRALinear(nn.Linear(7, 5, bias=False).double(), rank=3, alpha=6, dropout=0)
    whole, accumulated = copy.deepcopy(base), copy.deepcopy(base)
    x, target = torch.randn(8, 7, dtype=torch.float64), torch.randn(8, 5, dtype=torch.float64)
    F.mse_loss(whole(x), target).backward()
    for start in (0, 4):
        (F.mse_loss(accumulated(x[start:start+4]), target[start:start+4]) / 2).backward()
    for (name, a), (_, b) in zip(whole.named_parameters(), accumulated.named_parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    combinations = [check_batch(*values) for values in ((1,1,128),(1,4,32),(1,8,16),(1,16,8),(2,8,8))]
    for row in combinations:
        actual = validate_batch(row['world_size'], row['micro_batch'], row['accumulation'], formal=True)
        assert actual == (128, 390, row['epoch_microsteps'])
    lr_points = {step: lr_factor(step) for step in (1, 195, 196, 19500)}
    assert lr_points[1] == 1 / 195 and lr_points[195] == 1
    assert 0.1 < lr_points[196] < 1 and lr_points[19500] == 0.1
    for values in ((1,8,8),(1,0,128),(1,-8,-16),(1,8,16.0)):
        try:
            check_batch(*values)
        except ContractError:
            pass
        else:
            raise AssertionError(f"Accepted invalid batch combination {values}")
    isolation = []
    for split in ("seen_discrete_test", "seen_continuous"):
        row = dict(read_benchmark_rows('/home/jiahao/task/UniLIP/data/csgo_benchmark_v2', split,
                                       max_samples=1, require_images=False)[0])
        row['image_path'] = '/nonexistent/peft_target_fpv_must_never_be_opened.jpg'
        item = Seen10GenerationDataset([row], include_target=False)[0]
        assert 'target' not in item and item['sample_id'] == row['sample_id']
        assert list(item['radar'].shape) == [3,448,448] and torch.isfinite(item['pose']).all()
        if split == 'seen_continuous':
            assert item['clip_id'] == row['clip_id'] and item['frame_index'] == row['frame_index']
        isolation.append({'split':split,'sample_id':row['sample_id'],'missing_target_read':False})
    print(json.dumps({'passed':True,'device':'cpu','nonzero_merge_max_abs_error':errors,
                      'accumulation_gradient_equivalence':True,'batch_combinations':combinations,
                      'lr_factor_by_optimizer_step':lr_points,
                      'target_isolation':isolation},indent=2))


if __name__ == '__main__':
    main()
