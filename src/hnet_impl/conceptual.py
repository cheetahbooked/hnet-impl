from collections import deque

from .torchisms import torch, TT, nn, OptimizedModule, fsdp


def get_seq_idx(cu_seqlens: TT, flatlen: int) -> TT:
    # convert cu_seqlens -> seq_idx
    seq_idx = torch.zeros(flatlen, dtype=torch.int, device=cu_seqlens.device)
    seq_idx[cu_seqlens[1:-1]] = torch.ones_like(
        cu_seqlens[1:-1], dtype=torch.int
    )  # avoid cpu sync
    return seq_idx.cumsum_(0)[None].int()  # most downstream kernels want int, not long


class BlockBoundaryMixin:
    @classmethod
    def is_block(cls, m: nn.Module):
        # determine if a module is a Block, while counting torch wrapper modules
        if isinstance(m, OptimizedModule):
            m = m._orig_mod
        if hasattr(m, "_checkpoint_wrapped_module"):
            m = m._checkpoint_wrapped_module
        return isinstance(m, BlockBoundaryMixin)

    def child_blocks(self):
        # obtain all child modules that are also BlockBoundary, without recursing into blocks
        blocks = []
        q = deque(self.children())
        while q:
            if BlockBoundaryMixin.is_block(m := q.popleft()):
                blocks.append(m)
            else:
                q += m.children()
        return blocks

    @staticmethod
    def apply_fsdp(self, **kwargs):
        for c in self.child_blocks():
            getattr(c, "apply_fsdp")(c, **kwargs)
        fsdp.fully_shard(self, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        assert issubclass(cls, nn.Module)
