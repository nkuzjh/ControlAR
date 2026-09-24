#!/usr/bin/env python3
"""Compare unmerged/merged trained PEFT logits on one condition (CUDA, no training)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from csgo_seen10.model import Seen10GenerationModel, build_gpt, load_checkpoint
from csgo_seen10.peft import inject_lora, merge_lora_
from csgo_seen10.data import Seen10GenerationDataset, read_benchmark_rows
from train_seen10 import latent_mask


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint=load_checkpoint(args.checkpoint)
    if checkpoint.get('format')!='csgo_seen10_exp32gen_aligned_peft_v1':
        raise ValueError('Expected PEFT checkpoint')
    model_config=checkpoint['model_config']
    gpt=build_gpt(dropout=0,token_dropout=0).cuda()
    inject_lora(gpt,**{k:model_config['peft'][k] for k in ('rank','alpha','dropout')})
    model=Seen10GenerationModel(gpt).cuda()
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    model.eval()
    nonzero_b=sum(int(torch.count_nonzero(p)) for n,p in model.named_parameters()
                  if n.endswith(('.lora_B','.q_B','.k_B','.v_B')))
    assert nonzero_b>0,'Merge acceptance must use trained, nonzero LoRA increments'
    rows=read_benchmark_rows('/home/jiahao/task/UniLIP/data/csgo_benchmark_v2',
                             'seen_validation',max_samples=1,require_images=False)
    item=Seen10GenerationDataset(rows,include_target=False)[0]
    torch.manual_seed(42)
    values=dict(pose=item['pose'][None].cuda(),map_id=item['map_id'][None].cuda(),
                condition=item['radar'][None].cuda(),idx=torch.randint(0,16384,(1,783),device='cuda'),
                targets=None,mask=latent_mask(1,120,784,torch.device('cuda')))
    with torch.inference_mode():
        before=model(**values)[0]
        merge_lora_(model.gpt)
        after=model(**values)[0]
    error=(before-after).abs()
    # Parallel low-rank and merged GEMMs differ in floating point reduction order.
    torch.testing.assert_close(before,after,rtol=1e-3,atol=5e-4)
    result={'passed':True,'checkpoint':str(args.checkpoint.resolve()),'dtype':'float32',
            'tf32':False,'sample_id':item['sample_id'],'nonzero_lora_B_elements':nonzero_b,
            'logits_max_abs_error':error.max().item(),'logits_mean_abs_error':error.mean().item(),
            'argmax_agreement':(before.argmax(-1)==after.argmax(-1)).float().mean().item(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
