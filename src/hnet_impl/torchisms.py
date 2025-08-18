# Common torch aliases.

from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
from typing import Callable
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
from torch import nn, Tensor as TT, distributed as dist, multiprocessing as mp, nested

from torch._dynamo import OptimizedModule
from torch.distributed import device_mesh as tdm, fsdp, checkpoint as dcp
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.checkpoint import state_dict as dcps
from torch.distributed.tensor import DTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.profiler import schedule, profile, ProfilerActivity, record_function
from torch.nested._internal.ops import (
    register_jagged_func,
    normalize_function,
    raggedness_matches,
    _wrap_jagged_dim,
    extract_kwargs,
    NestedTensor,
)


def NJT(ls: list[TT]):
    return nested.nested_tensor(ls, layout=torch.jagged)


def dupe_fn(fn: Callable, salt: int):
    co_new = fn.__code__.replace(co_consts=fn.__code__.co_consts + (salt,))
    return FunctionType(
        co_new, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )


@contextmanager
def ensure_no_cuda_sync():
    torch.cuda.set_sync_debug_mode("error")
    yield
    torch.cuda.set_sync_debug_mode("default")


@contextmanager
def unsafe_reduce_optimizedmodule_overhead():
    # extremely unsafe strategy to reduce overhead of executing torch.compiled blocks
    ocs = dynamo.utils.call_size
    dynamo.utils.call_size = dynamo.decorators.skip(lambda x, i: x.shape[i])
    with dynamo.set_stance(skip_guard_eval_unsafe=True):
        yield
    dynamo.utils.call_size = ocs


@contextmanager
def summon_full_params(model: FSDPModule):
    handles = [
        m.unshard(async_op=True)
        for m in reversed(list(model.modules()))
        if isinstance(m, FSDPModule)
    ]
    for h in handles:
        h.wait() if h is not None else 0

    yield

    for m in reversed(list(model.modules())):
        if isinstance(m, FSDPModule):
            m.reshard()


def rand_njt_iids(docs: int, slen: range, v: int = 256):
    # generate random iids of fixed batch size
    return NJT(
        [
            torch.randint(v, (seqlen,))
            for seqlen in torch.randint(slen.start, slen.stop, (docs,))
        ]
    )


def random_iids(
    msl: int,
    *,
    vsize: int = 256,
    s_min: int = 256,
    s_max: int = 1024,
    rng: torch.Generator = torch.default_generator,
):
    """Generate a varlen batch of input IDs, with a total length of up to `msl`.

    Arguments:
        msl: maximum (total) sequence length of batch
        vsize: vocab size of iids
        s_min: minimum sqeuence length of individual document
        s_max: maximum sqeuence length of individual document
    """
    assert s_min <= s_max <= msl
    samples, total = [], 0
    while True:
        # generate random length && iids on CPU.
        i = torch.randint(s_min, s_max + 1, (1,), generator=rng).item()
        t = torch.randint(vsize, (i,), dtype=torch.int, generator=rng).to(
            "cuda", non_blocking=True
        )
        # if the current random document would cause total batch to exceed msl, yield
        if i + total > msl:
            yield samples
            samples, total = [], 0
        samples.append(t)
        total += i


def random_x(d: int, msl: int, *, device="cuda", dtype=torch.bfloat16, **k):
    for samples in random_iids(msl, **k):
        yield NJT(
            [torch.randn(t.numel(), d, device=device, dtype=dtype) for t in samples]
        )


### profiling tools
def tqdm_with_step(prof: profile, iters: int, **k):
    for i in tqdm(range(iters)):
        yield i
        prof.step()


@contextmanager
def profiler_setup(path_ct: Path, iters: int, skip_first=10):
    path_ct.mkdir(parents=True, exist_ok=True)
    sched = schedule(skip_first=skip_first, wait=5, warmup=1, active=3, repeat=3)
    activ = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    def handler(p: profile):
        p.export_chrome_trace(str(path_ct / f"step-{p.step_num}.json"))

    with profile(
        activities=activ, schedule=sched, on_trace_ready=handler, with_stack=True
    ) as prof:
        yield tqdm_with_step(prof, iters)


def make_chrometrace(
    name: str, dl: iter, m: nn.Module, fwd_to_loss: callable, *, iters: int = 30
):
    dataset = [next(dl) for _ in range(iters)]  # prefetch data
    with profiler_setup(Path(f"./chrometrace-{name}"), 30) as g:
        for i, inputs in zip(g, dataset):
            torch.cuda.synchronize()
            with record_function("fwd"):
                loss = fwd_to_loss(inputs)
            with record_function("bwd"):
                loss.backward()
            for p in m.parameters():
                p.grad = None
