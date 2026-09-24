#!/usr/bin/env python3
"""CPU comparison of uninterrupted and restarted PEFT training checkpoint states."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uninterrupted',type=Path)
    parser.add_argument('resumed',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    left=torch.load(args.uninterrupted,map_location='cpu',weights_only=False)
    right=torch.load(args.resumed,map_location='cpu',weights_only=False)
    counts={'tensors':0,'tensor_elements':0,'arrays':0,'scalars':0}
    differences=[]
    def compare(a,b,path):
        if isinstance(a,torch.Tensor):
            counts['tensors']+=1;counts['tensor_elements']+=a.numel()
            if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or not torch.equal(a,b):
                differences.append(path)
        elif isinstance(a,np.ndarray):
            counts['arrays']+=1
            if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b):differences.append(path)
        elif isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys():differences.append(path+'.keys');return
            for key in a:compare(a[key],b[key],f'{path}.{key}')
        elif isinstance(a,(list,tuple)):
            if type(a)!=type(b) or len(a)!=len(b):differences.append(path+'.length');return
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
        else:
            counts['scalars']+=1
            if a!=b:differences.append(path)
    keys=['model','optimizer','scheduler','scaler','steps','global_optimizer_step','consumed_samples',
          'epoch','batch_in_epoch','micro_step_in_accum','best_val_loss','best_step','val_loss',
          'world_size','rng_states','dataloader_generator_state','sampler_state','training_config','model_config']
    # Include additional generator/checkpoint state introduced by the trainer.
    keys += [k for k in left if 'generator' in k and k not in keys]
    for key in keys:compare(left[key],right[key],key)
    result={'passed':not differences,'comparison':'bitwise tensor and exact non-tensor state',
            'compared_keys':keys,'counts':counts,'differences':differences,
            'global_optimizer_step':left['global_optimizer_step'],'consumed_samples':left['consumed_samples'],
            'uninterrupted':str(args.uninterrupted.resolve()),'resumed':str(args.resumed.resolve())}
    print(json.dumps(result,indent=2))
    if differences:raise SystemExit(1)


if __name__=='__main__':main()
