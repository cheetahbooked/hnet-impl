import triton
import triton.language as tl

from .torchisms import torch, nn, TT, fsdp, F
from .conceptual import BlockBoundaryMixin


# Default: unbiased linear
def Lin(*a, bias=False, **k):
    return nn.Linear(*a, bias=bias, **k)


# Routing module: tf32 + bias
class HighPrecLinear(BlockBoundaryMixin, nn.Linear):
    def __init__(self, in_features, out_features, device=None):
        super().__init__(in_features, out_features, True, device, torch.float32)
        # NOTE: authors do uniform random bias init, despite 0 weight init.
        nn.init.zeros_(self.bias)
        # I think that is bad, and zero-init bias instead.
        nn.init.zeros_(self.weight)
        # NOTE: Be wary of your global settings elsewhere.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def forward(self, x: TT):
        # NOTE: amp.autocast will still impl this as bf16. This will be fp32 iff fsdp mixed prec is used.
        return super().forward(x.to(self.weight.dtype))

    @staticmethod
    def apply_fsdp(self, **kw):
        fsdp.fully_shard(
            self,
            **kw | {"mp_policy": fsdp.MixedPrecisionPolicy(param_dtype=torch.float32)},
        )


### LMHead: Fused Linear + Cross Entropy ###
@triton.heuristics({"HAS_SCALE": lambda args: args["scale"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def logsumexp_fwd_kernel(
    x, z, scale, D: tl.constexpr, B: tl.constexpr, HAS_SCALE: tl.constexpr
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float("inf"))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)


def logsumexp_fwd(x, scale: float | None = None, dtype: torch.dtype | None = None):
    r"""Compute the logsumexp of the input tensor over the last dimension."""
    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[(N, ND)](x=x, z=z, scale=scale, D=D, B=B)
    z = z.logsumexp(-1).view(*shape[:-1])
    return z.to(dtype) if dtype is not None and dtype != torch.float else z


MAX_FUSED_SIZE, STATIC_WARPS = 65536 // 2, 32


@triton.jit
def cross_entropy_kernel(
    logits,
    lse,
    target,
    loss,
    ignore_index,
    label_smoothing: tl.constexpr,
    logit_scale: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    b_y = tl.load(target + i_n)
    logits += i_n * V
    if b_y == ignore_index:
        for i in range(0, V, BV):
            o_v = i + tl.arange(0, BV)
            tl.store(logits + o_v, 0.0, mask=o_v < V)
        return
    b_l = tl.load(logits + b_y) * logit_scale
    b_lse = tl.load(lse + i_n)
    b_loss = b_lse - b_l
    b_z = 0.0
    eps = label_smoothing / V
    tl.debug_barrier()
    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        b_logits = (
            tl.load(logits + o_v, mask=o_v < V, other=float("-inf")) * logit_scale
        )
        if label_smoothing > 0:
            b_z += tl.sum(tl.where(o_v < V, -eps * b_logits, 0.0))
        b_p = (tl.exp(b_logits - b_lse) - eps) * logit_scale
        tl.store(logits + o_v, b_p, mask=o_v < V)
        tl.debug_barrier()
    if label_smoothing > 0:
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing * b_lse)
    b_l = tl.load(logits + b_y)
    b_l += (label_smoothing - 1) * logit_scale
    tl.store(loss + i_n, b_loss)
    tl.store(logits + b_y, b_l)


def fused_linear_cross_entropy_forward(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
):
    device = x.device
    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    dx = torch.zeros_like(x, device=device)
    dw = (
        torch.zeros_like(weight, device=device, dtype=torch.float)
        if weight is not None
        else None
    )
    db = (
        torch.zeros_like(bias, device=device, dtype=torch.float)
        if bias is not None
        else None
    )
    loss = torch.zeros(N, device=device, dtype=torch.float)

    # NOTE: if you do label masking, you need to d2h (target!=ignore_index).sum() and use 1/that as scale
    torch._assert_async((target != ignore_index).all())
    numel_scale = 1.0 / N
    scale = (
        logit_scale * numel_scale
    )  # instead of passing logit_scale to kernel, we scale in mm for numerical stability

    w = weight.to(x.dtype)  # support AMP
    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_x = x[start:end]
        c_loss = loss[start:end]
        c_target = target[start:end]

        # instead of using kernel, we apply logit scale at fwd mm to support uP numerical stability
        assert bias is None
        c_logits = torch.empty(
            c_x.shape[0], weight.shape[0], dtype=c_x.dtype, device=device
        )
        c_logits.addmm_(c_x, w.mT, beta=0, alpha=logit_scale)

        # keep lse in fp32 to maintain precision
        c_lse = logsumexp_fwd(c_logits, dtype=torch.float)

        # Here we calculate the gradient of c_logits in place so we can save memory.
        cross_entropy_kernel[(c_logits.shape[0],)](
            logits=c_logits,
            lse=c_lse,
            target=c_target,
            loss=c_loss,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            logit_scale=1.0,
            V=V,
            BV=BV,
            num_warps=STATIC_WARPS,
        )

        # gradient of logits is computed in-place by the above triton kernel and is of shape: C x V
        dx[start:end].addmm_(c_logits, w, beta=1.0, alpha=scale)  # (C,V) @ (V,H)
        # keep dw in fp32 to maintain precision
        # if weight is not None: dw.addmm_(c_logits.mT, c_x, out_dtype=torch.float, alpha=scale) # TODO: pytorch 2.8
        if weight is not None:
            dw += torch.empty_like(dw, dtype=c_x.dtype).addmm_(
                c_logits.mT, c_x, beta=0, alpha=scale
            )
        if bias is not None:
            db.add_(c_logits.sum(0), alpha=scale)

    if dw is not None:
        dw = dw.to(weight)
    if db is not None:
        db = db.to(bias)
    loss_sum = loss.sum()
    return loss_sum * numel_scale, loss_sum, dx, dw, db


## Unlike FLA's, I strip the following:
# - no support for L2 warp / penalty
# - celoss is always **not** reduced; furthermore I return mean (differentiable) & sum (non-differentiable) losses
# - celoss must always be unscaled when provided to autograd backward. a grad not-equal to 1.0 will cause a device-side assertion.
class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        logit_scale: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        num_chunks: int = 8,
    ):
        assert all(t.is_contiguous() for t in [x, target, weight])
        loss_mean, loss_sum, dx, dw, db = fused_linear_cross_entropy_forward(
            x,
            target,
            weight,
            bias,
            ignore_index,
            label_smoothing,
            logit_scale,
            num_chunks,
        )
        # create expected loss grads in a non-blocking manner
        dloss_expected = torch.cat(
            [
                torch.ones(
                    1, device=loss_mean.device, dtype=loss_mean.dtype
                ),  # celoss mean should get grad 1.
                torch.zeros(
                    1, device=loss_mean.device, dtype=loss_mean.dtype
                ),  # celoss sum should get grad 0.0
            ]
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            dloss_expected,
            dx.detach(),
            dw.detach() if weight is not None else None,
            db.detach() if bias is not None else None,
        )
        return (
            loss_mean,
            loss_sum,
        )  # <-- sum should only be used for metric calculation (bpb)

    @staticmethod
    def backward(ctx, do_mean, do_sum):
        dloss_expected, dx, dw, db = ctx.saved_tensors
        torch._assert_async((dloss_expected == torch.stack([do_mean, do_sum])).all())
        return dx, None, dw, db, None, None, None, None, None


class LMHead(nn.Linear):
    logit_scale = 1.0  # NOTE: modify this externally if you use muP

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: TT, labels: TT | None = None) -> TT:
        if labels is None:
            return F.linear(x, self.weight) * self.logit_scale
        l_mean, l_sum = FusedLinearCrossEntropyFunction.apply(
            x, labels, self.weight, self.bias, self.logit_scale
        )
        return l_mean, l_sum.detach()


__all__ = ["LMHead", "HighPrecLinear", "Lin"]


def test_lmhead(d=256, V=32000):
    with torch.random.fork_rng(["cuda"]), torch.device("cuda"):
        # init
        torch.manual_seed(0)
        m0 = nn.Linear(d, V, bias=False)
        torch.manual_seed(0)
        m1 = LMHead(d, V)

        # samples
        x = torch.randn(77, d)
        t = torch.randint(0, V, size=(77,))

        # vanilla torch
        x0 = x.clone().requires_grad_()
        y0 = m0(x0)
        l_avg_0 = F.cross_entropy(y0, t)
        l_sum_0 = F.cross_entropy(y0, t, reduction="sum").detach()
        l_avg_0.backward()

        # fused kernel
        y1 = m1(x)
        x1 = x.clone().requires_grad_()
        l_avg_1, l_sum_1 = m1(x1, t)
        l_avg_1.backward()

        # check
        assert torch.allclose(y0, y1)
        assert torch.allclose(l_avg_0, l_avg_1)
        assert torch.allclose(l_sum_0, l_sum_1)
        assert torch.allclose(m0.weight.grad, m1.weight.grad)
        assert torch.allclose(x0.grad, x1.grad)


if __name__ == "__main__":
    test_lmhead()
