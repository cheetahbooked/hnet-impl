import os
from contextlib import nullcontext

import torch
from torch import nested, Tensor as TT
from torch.distributed import device_mesh as tdm, fsdp

from hnet_impl import HNetLM, HNetConfig, ByteTokenizer, completion_sync

## dist init
r,ws = int(os.environ['WORLD_SIZE']), int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
mesh = tdm.init_device_mesh('cuda', (ws,), mesh_dim_names=('dp',))

## create model
t = ByteTokenizer()
c = HNetConfig.create_reasonable_config(D=[512,1024], arch=['m4','T9'])
with torch.device('cuda'): m = HNetLM(c)

## fsdp/compile
m.backbone.block_compile(ac=False)
m.apply_fsdp( # default: BF16, ZeRO2, 1D mesh
    m,
    mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    reshard_after_forward=False,
    mesh=mesh['dp'],
) if ws > 1 else m

## optim / lr sched
base_lr,max_steps = 3e-4,1000
opt = torch.optim.AdamW([
    dict(params=ls, lr=base_lr * lr_mod)
    for ls, lr_mod in zip(m.split_params_by_hierachy(), c.lambda_s())
], betas=(0.9,0.95), weight_decay=0.01)
lrs = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (pct:=step/max_steps) and (
    pct*10 if pct < .1 else (1 if pct < .9 else (1-pct)*10)
))

## example dumb task: random number of repeating letters
def generate_random_letters():
    from string import ascii_lowercase
    from random import randint
    return ''.join(randint(0,10)*c for c in ascii_lowercase)
def NJT(ls: list[TT]): return nested.nested_tensor(ls, layout=torch.jagged)
def random_batches():
    while True:
        tokens = t.encode([generate_random_letters() for _ in range(32)])
        iids = [s[:-1] for s in tokens]
        lbls = [s[1: ] for s in tokens]
        yield NJT(iids), NJT(lbls).long()

with m.sampling_mode():
    print(completion_sync('',t,m,max_new=99))

## training loop
zero = torch.tensor(0.,device='cuda')
for step,(iids,lbls) in zip(range(max_steps),random_batches()):
    with torch.autocast('cuda',torch.bfloat16) if ws > 1 else nullcontext():
        (l_avg,l_sum),extra = m(iids.cuda(),lbls.cuda())
        l_ratio = sum([e.loss_ratio for e in extra], zero)
        loss = l_avg + l_ratio
    loss.backward()

    opt.step()
    opt.zero_grad()
    lrs.step()

    if step % 10 == 0 == r: print(f'{step=}: {l_avg.item()=:.3f} {l_ratio.item()=:.3f}')

with m.sampling_mode():
    print(completion_sync('',t,m,max_new=99))
