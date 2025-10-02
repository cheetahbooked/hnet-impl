from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from .torchisms import torch, TT, nn, F, nested, NJT, summon_full_params
from .conceptual import BlockBoundaryMixin, get_seq_idx
from .config_hnet import HNetConfig
from .xf import Isotropic
from .lin import Lin, HighPrecLinear, LMHead

# ACT imports
from hrm.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

### ################
### H-Net submodules
### ################


@dataclass(frozen=True)
class HNetExtra:
    b: TT  # (B,j1) boolean label for whether byte was selected
    loss_ratio: TT  # scalar tensor -- routing loss for this block
    compress_ratio: float  # scalar float -- compression ratio


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


class QProjPadded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat: TT, w: TT, k_flat: TT, cu: TT):
        slen = x_flat.shape[0]
        # compute x@w.T, but padded left by 1seqlen
        q_padded = torch.empty(
            slen + 1, *x_flat.shape[1:], dtype=x_flat.dtype, device=x_flat.device
        )
        torch.mm(x_flat, w.T.type_as(x_flat), out=q_padded[1:])
        ctx.save_for_backward(x_flat, w, cu)
        return q_padded.index_copy_(0, cu[:-1], -k_flat[cu[:-1]])[:slen]

    @staticmethod
    def backward(ctx, dq_flat: TT):
        x_flat, w, cu = ctx.saved_tensors
        zero_grad = torch.zeros(
            cu.shape[0] - 1,
            dq_flat.shape[-1],
            device=dq_flat.device,
            dtype=dq_flat.dtype,
        )
        dq_flat = dq_flat.index_copy(0, cu[:-1], zero_grad)

        dx_flat = torch.zeros_like(x_flat)
        torch.mm(dq_flat[1:], w.type_as(dq_flat), out=dx_flat[:-1])
        dw = dq_flat[1:].mT @ x_flat[:-1]

        return dx_flat, dw, None, None


# NOTE: it's possible to fuse q/k/res proj into a single gemm kernel, but only iff they are of equal precision.
class RoutingModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q_proj_layer = Lin(d, d)
        self.k_proj_layer = Lin(d, d)
        # https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py#L49
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d))
            self.k_proj_layer.weight.copy_(torch.eye(d))

    def forward(self, r_flat: TT, r_cu: TT):
        k_flat = self.k_proj_layer(r_flat)
        q_flat = QProjPadded.apply(r_flat, self.q_proj_layer.weight, k_flat, r_cu)
        cos_sim = F.cosine_similarity(q_flat, k_flat, dim=-1)
        p_flat = (0.5 - cos_sim / 2).clamp(0.0, 1.0)
        b_flat = p_flat >= 0.5
        p_select_cu = F.pad(b_flat.cumsum(0), (1, 0))[r_cu]
        return p_flat, b_flat, p_select_cu


class DeChunkLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # for EMA scan kernel.
        self.block_size = 256
        self.headdim = 32
        self.nheads, _r = divmod(d, self.headdim)
        assert _r == 0
        A = -torch.ones(self.nheads, device="cuda", dtype=torch.float32)
        self.register_buffer("A", A, persistent=False)

    @staticmethod
    def forward_flat(
        h_flat: TT,
        b_flat: TT,
        p_selected_flat: TT,
        h_seq_idx: TT,
        *,
        eps=1e-4,
        nheads: int,
        headdim: int,
        block_size: int,
        A: TT,
    ):
        p = p_selected_flat.float().clamp(eps, 1 - eps)

        dt = -torch.log1p(-p.float())[..., None]
        h = (h_flat.float() / dt).type_as(h_flat)
        c = torch.ones_like(p := p.type_as(h)[None, :, None, None])

        z_bar_flat = mamba_chunk_scan_combined(
            h.view(1, -1, nheads, headdim),
            dt.expand(-1, nheads).to(h.dtype)[None],
            A,
            p,
            c,
            chunk_size=block_size,
            seq_idx=h_seq_idx,
        )[0].view(-1, h.shape[-1])

        inner2outer_idx = b_flat.cumsum(0) - 1
        return z_bar_flat.index_select(0, inner2outer_idx)

    def forward(
        self, h_flat: TT, b_flat: TT, p_selected_flat: TT, h_seq_idx: TT, *, eps=1e-4
    ):
        return self.forward_flat(
            h_flat,
            b_flat,
            p_selected_flat,
            h_seq_idx,
            eps=eps,
            nheads=self.nheads,
            headdim=self.headdim,
            block_size=self.block_size,
            A=self.A,
        )


### #################
### Final HNet Module
### #################


class HNet(nn.Module):
    def __init__(self, c: HNetConfig, stage_idx: int):
        super().__init__()
        self.stage_idx = stage_idx
        self.d = c.d_model[stage_idx]
        try:
            self.n = c.N_compress[stage_idx + 1] / c.N_compress[stage_idx]
        except IndexError:
            self.n = None

        arch_layout = c.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert len(arch_layout) in [3, 1]
        self.is_innermost = len(arch_layout) == 1

        if self.is_innermost:
            if "ACT" in arch_layout[0]:
                # Construct config dict for HierarchicalReasoningModel_ACTV1
                act_config_for_hrm = {
                    "H_cycles": c.act_cfg.H_cycles,
                    "L_cycles": c.act_cfg.L_cycles,
                    "H_layers": c.act_cfg.H_layers,
                    "L_layers": c.act_cfg.L_layers,
                    "halt_max_steps": c.act_cfg.halt_max_steps,
                    "halt_exploration_prob": c.act_cfg.halt_exploration_prob,
                    "expansion": c.act_cfg.expansion,
                    
                    "hidden_size": self.d,
                    "rms_norm_eps": 1e-5, # HRM default based on code inspection
                    "forward_dtype": "bfloat16", # Consistent with HNet usage
                    "batch_size": 1, # Dummy batch size for initialization
                }
                self.main_network = HierarchicalReasoningModel_ACTV1(act_config_for_hrm)
            else:
                self.main_network = Isotropic(
                    c, arch_layout[0], stage_idx=stage_idx
                )  # <-- don't increment
        else:
            self.encoder = Isotropic(c, arch_layout[0], stage_idx=stage_idx)
            self.main_network = HNet(c, stage_idx + 1)
            self.decoder = Isotropic(c, arch_layout[2], stage_idx=stage_idx)

            self.routing_module = RoutingModule(self.d)
            self.dechunk_layer = DeChunkLayer(self.d)
            self.residual_proj = HighPrecLinear(self.d, self.d)

        d_gain = self.d - c.d_model[stage_idx - 1] if stage_idx else None
        self.pad_dimension = (
            nn.Parameter(torch.zeros(d_gain, device="cuda")) if d_gain else None
        )

    # only compile blocks within a hnet, not the hnet itself
    def block_compile(self, ac: bool):
        self.main_network.block_compile(ac)
        if self.is_innermost:
            return
        self.encoder.block_compile(ac)
        self.decoder.block_compile(ac)
        self.register_module(
            "routing_module",
            torch.compile(self.routing_module, backend="inductor", fullgraph=True),
        )
        self.register_module(
            "residual_proj",
            torch.compile(self.residual_proj, backend="inductor", fullgraph=True),
        )
        self.ratio_loss = torch.compile(
            self.ratio_loss, backend="inductor", fullgraph=True, dynamic=True
        )

    def ratio_loss(self, b_flat: TT, p_flat: TT):
        assert self.n, "HNetConfig did not receive valid N_compress; please edit it"
        l = b_flat.numel()
        f = b_flat.sum().float() / l
        g = p_flat.float().sum() / l
        drop_experts = self.n * (1 - f) * (1 - g) / (self.n - 1)
        keep_expert = self.n * f * g
        return keep_expert + drop_experts

    @contextmanager
    def least_blocking_masked_select(
        self, *outer_flat_tensors: list[TT], mask_flat: TT, cu_seqlens: TT
    ):
        # WARNING: do not try to compile this. inductor will just wipe all pin memory & copy & Event & etc.
        inner_stats_cuda = torch.stack([cu_seqlens.diff().max(), cu_seqlens[-1]])
        inner_stats_cpu = torch.empty_like(
            inner_stats_cuda, device="cpu", pin_memory=True
        )
        inner_stats_cpu.copy_(inner_stats_cuda, non_blocking=True)
        d2h_event = torch.cuda.Event()
        d2h_event.record()

        # in the yield region, the end-user is expected to enqueue as much GPU work as possible, to make the CPU sync cheap.
        yield (mutable_res := []), inner_stats_cpu

        d2h_event.synchronize()
        inner_flatlen = inner_stats_cpu[1].item()
        idx_flat = mask_flat.nonzero_static(size=inner_flatlen).squeeze(-1)
        for outer in outer_flat_tensors:
            mutable_res.append(outer.index_select(0, idx_flat))

    def forward(self, x_flat: TT, flat_cu: TT, msl: int):
        d_orig = x_flat.shape[-1]
        x_flat = (
            x_flat
            if self.pad_dimension is None
            else torch.cat(
                [x_flat, self.pad_dimension.expand(x_flat.shape[0], -1)], dim=-1
            )
        )
        x_flat = x_flat.bfloat16()

        if self.is_innermost:
            if isinstance(self.main_network, HierarchicalReasoningModel_ACTV1):
                flat_len = x_flat.shape[0]
                batch_hrm = {"vector_input": x_flat}
                
                # Initialize carry state
                carry = self.main_network.initial_carry(batch_hrm)
                
                final_x_flat = torch.zeros_like(x_flat)
                halted_mask = torch.zeros((flat_len,), dtype=torch.bool, device=x_flat.device)
                
                for _t in range(self.main_network.config.halt_max_steps):
                    # Perform one step of ACT
                    carry, outputs = self.main_network(carry, batch_hrm)
                    
                    # Identify sequences newly halted in this step
                    newly_halted = carry.halted & ~halted_mask
                    
                    # Store the final vector for newly halted sequences
                    if newly_halted.any():
                        final_x_flat[newly_halted] = outputs["final_vector"][newly_halted]
                        
                    halted_mask |= carry.halted
                    
                    if halted_mask.all():
                        break
                        
                final_x_flat = final_x_flat[..., :d_orig]
                return final_x_flat, []

            else: # Isotropic fallback
                return self.main_network(x_flat, flat_cu, msl)[..., :d_orig], []

        r_flat = self.encoder(x_flat, flat_cu, msl)
        p_flat, b_flat, select_cu = self.routing_module(r_flat, flat_cu)

        # obtaining r_select/p_select would require a cpu-sync'ing .masked_select in normal circumstances.
        # To avoid this, we initiate a D2H of the inner H-Net's seqlen ASAP, and enqueue work to let the GPU race ahead.
        # Note that, if you are **already CPU bound** prior to this (e.g. in really small runs), this code is detrimental.
        with self.least_blocking_masked_select(
            p_flat, r_flat, mask_flat=b_flat, cu_seqlens=select_cu
        ) as (pending_selected_tensors, pending_cpu_stats):
            ratio_loss = (
                self.ratio_loss(b_flat, p_flat) if torch.is_grad_enabled() else 0
            )
            c_flat = torch.where(b_flat, p_flat, 1 - p_flat)[..., None]
            residual = self.residual_proj(r_flat)
        p_select, r_select = pending_selected_tensors

        h_select, extras = self.main_network(
            r_select, select_cu, pending_cpu_stats[0].item()
        )

        x_flat = self.dechunk_layer(
            h_select, b_flat, p_select, get_seq_idx(select_cu, p_select.shape[0])
        )
        x_flat = (residual + x_flat.float() * ste_func(c_flat)).type_as(x_flat)
        x_flat = self.decoder(x_flat, flat_cu, msl)[..., :d_orig]

        extra = HNetExtra(
            nested.nested_tensor_from_jagged(b_flat, flat_cu, max_seqlen=msl),
            ratio_loss,
            p_select.numel() / p_flat.numel(),
        )

        return x_flat, [extra] + extras


class HNetLM(BlockBoundaryMixin, nn.Module):
    def __init__(self, c: HNetConfig):
        super().__init__()
        self.c, v, d = c, c.vocab_size, c.d_model[0]
        self.embeddings = nn.Embedding(v, d)
        self.backbone = HNet(c, stage_idx=0)
        self.lm_head = LMHead(d, v)

    # Top-level contract:
    # 1. if lbls is provided, return (loss_mean,loss_sum),extras[]
    #    use loss_mean for autograd, and loss_sum for bpb calc.
    #    use extras[] to grab ratio loss && log compression ratio
    # 2. if lbls is None,     return logits,extras[]
    #    use logits for autoregressive sampling.
    #    use extras[] to grab selected token IDs (b) for sampling pretty-printing.
    def forward(
        self, iids: TT, lbls: TT | None = None
    ) -> tuple[TT | tuple[TT, TT], list]:
        assert iids.is_nested and iids.ndim == 2
        cu_s, msl = iids.offsets(), iids._max_seqlen
        x_flat = self.embeddings(iids.values())
        x_flat, extra = self.backbone(x_flat, cu_s, msl)
        res = self.lm_head(x_flat, lbls if lbls is None else lbls.values())
        if lbls is None:
            res = nested.nested_tensor_from_jagged(res, cu_s, max_seqlen=msl)
        return res, extra

    def split_params_by_hierachy(self) -> list[list[nn.Parameter]]:
        # for each param, count the number of times ".main_network" appears in it.
        d = defaultdict(list)
        for n, p in self.named_parameters():
            d[n.count("main_network")].append(p)
        # special-case innermost hnet which has redundant .main_network
        max_depth = max(d.keys())
        assert 1 == len(d[max_depth - 1]), (
            f"expected single .pad_dimension at {max_depth - 1}"
        )
        d[max_depth - 1] += d.pop(max_depth)

        return [d[k] for k in range(len(d))]

    def load_goomba_ckpt(self, path: str | None):
        from omegaconf import ListConfig

        if path is None:
            return
        with torch.serialization.safe_globals([ListConfig]):
            d = torch.load(path, mmap=True, weights_only=False)
        self.load_state_dict(d)

    @contextmanager
    def sampling_mode(self):
        with (
            summon_full_params(self),
            torch.compiler.set_stance("force_eager"),
            torch.autocast("cuda", torch.bfloat16, cache_enabled=False),
        ):
            yield


def test_fwd_correctness():
    import re
    from .sampling import ByteTokenizer, completion_sync

    ## load hardcoded model
    c = HNetConfig.load_config("hnet_2stage_XL.json")
    t = ByteTokenizer()
    with torch.device("cuda"):
        m = HNetLM(c).bfloat16()
    m.load_goomba_ckpt("hnet_2stage_XL.pt")

    ## check randint fwd logits
    torch.manual_seed(0)
    iids = torch.randint(0, 256, (77,), dtype=torch.long, device="cuda")
    with torch.no_grad():
        pfill = m(NJT([iids]))[0].values()
    # original: "tensor[77, 256] bf16 n=19712 (38Kb) x∈[-12.562, 15.188] μ=-2.047 σ=3.875 cuda:0"
    assert -12.75 < pfill.min().item() < -12.25 and 15 < pfill.max().item() < 15.5, (
        pfill
    )
    assert -2.1 < pfill.mean().item() < -2.0 and 3.87 < pfill.std().item() < 3.88, pfill
    print(f"{pfill=}")

    ## check greedy sampling result
    comp = completion_sync("Hello world!", t, m, max_new=200, temp=0.0001, min_p=0.0001)
    comp = re.sub(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]", "", comp)  # )
    assert (
        comp
        == " I hope you are doing well. In this article, we will discuss the basics of the Python programming language. We will start with the basics of Python and then move on to more advanced topics. So, let"
    ), comp


__all__ = ["HNetLM"]
if __name__ == "__main__":
    test_fwd_correctness()
