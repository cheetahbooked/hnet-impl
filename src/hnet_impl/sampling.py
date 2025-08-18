from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from termcolor import colored
from tqdm import tqdm

from .torchisms import torch, nn, TT, NJT, F


### Tokenizer ###
@contextmanager
def allow_immutable_frombuffer():
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable.*")
        yield


class Tokenizer(ABC):
    vsize_padded: int
    vocab_size: int
    bos_idx: int

    @classmethod
    @abstractmethod
    def construct(cls, *a, **k): ...
    @abstractmethod
    def encode(self, seqs: list[str]) -> list[TT]: ...
    @abstractmethod
    def decode(self, iids: list[TT]) -> list[str]: ...


@dataclass(frozen=True)
class ByteTokenizer(Tokenizer):
    vsize_padded: int = 256
    vocab_size: int = 256
    bos_idx: int = 254

    @classmethod
    def construct(cls):
        return cls()

    def _encode_one(self, s: str) -> TT:
        return (
            F.pad(
                torch.frombuffer(s.encode(), dtype=torch.uint8),
                (1, 0),
                value=self.bos_idx,
            ).int()
            if s
            else torch.tensor([self.bos_idx], dtype=torch.int)
        )

    def encode(self, seqs: list[str]) -> list[TT]:
        with allow_immutable_frombuffer():
            return [self._encode_one(s) for s in seqs]

    def decode(self, iids: list[TT]) -> list[str]:
        return [bytearray(t[1:].tolist()).decode() for t in iids]


### Sampling ###
@torch.inference_mode
def sample_nocache(
    m: nn.Module,
    iids: TT,
    bos: int,
    *,
    max_new: int = 512,
    temp: float = 1.0,
    min_p: float = 0.0001,
):
    # contract: m must accept iids (ndim 1/2) directly, and produce logits as its only output.
    assert iids.device.type == "cuda" and not iids.is_floating_point() and iids.ndim < 3
    assert min_p > 0 and 0 < temp <= 1.0 and max_new > 1
    assert iids.ndim == 1 or iids.size(0) == 1, "bsz1 only"

    def obtain_probs(l: TT):
        # start with min-p
        p = l.softmax(-1)
        top_p, _ = p.max(-1, keepdim=True)
        l = torch.where(p < top_p * min_p, -float("inf"), l)
        # apply temp
        return (l / temp).softmax(-1)

    for _ in range(max_new):
        logits = m(iids)
        probs = obtain_probs(logits[..., -1, :])
        next_token_t = torch.multinomial(probs, 1)  # [1] or [b 1]
        yield (next_token := next_token_t.view(-1).tolist())
        if bos == next_token[0]:
            return
        iids = torch.cat([iids, next_token_t], dim=-1)


# The chunk boundaries of a H-Net are defined by its encoder, so
# to obtain the chunk location of a given *output* token at $x_t$,
# we must pass $x_t$ to the model, i.e. execute one future forward pass.
def prefetched(g):  # prefetch=1 generator that omits the last output
    prev = next(g)
    for x in g:
        yield prev
        prev = x


"""HNet coloring rules
To visualize HNet outputs, we want two things to be easily visible:
a. the byte at which a given chunk *starts*, as that is what the main net sees.
b. the bytes occupied by a given chunk.

To solve (a), I adopt the following scheme:
0. The display of a codepoint is always primarily determined by its most "powerful" depth.
   If a 4-byte codepoint has bytes reaching [s=3, s=2, s=1, s=0], we print it as an s=3 codepoint.
1. For the exceptional case of a codepoint which strides *any chunk boundary*, we set it to blink.
   Let's say a 4-byte codepoint hit depths [0, 0->1->2, 0->1, 0]. That means,
   - it is part of *two* unique s=2 chunks, *three* unique s=1 chunks, and ofc 4 bytes 
   so that codepoint would be printed as blinking, and with the formatting of an s=2 codepoint.
2. To avoid having to consider different terminal colorschemes, I only use attribute mappings:
   s=0: 'dark'
   s=1: normal
   s=2: 'underline'

There are some downsides to the above scheme:
 - Although (b) is technically implied by (a), my brain is quite bad at eyeballing (b) from just (a).
 - 'dark' and normal text are indistinguishable for whitespace, which is often the starting point of a chunk :/
 - Underscores are necessarily mangled by 'underline'.

So I adopt an additional per-chunk coloring scheme, where:
1. groups of s=1 chunks round-robin through various text colors
2. groups of s=2 chunks round-robin through various background colors
"""


def get_termcolor_from_boundaries(bounds: list[TT], cid: int, hid: int) -> list[str]:
    if len(bounds) > 2:
        raise RuntimeError("this scheme only handles up to stage2")
    elem_per_depth = [b.sum() for b in bounds]
    should_blink = any(n > 1 for n in elem_per_depth)
    max_depth = sum(n != 0 for n in elem_per_depth)

    attrs = ["blink"] if should_blink else []
    attrs += [["dark"], [], ["underline"]][max_depth]
    cid += elem_per_depth[0] if len(bounds) > 0 else 0
    hid += elem_per_depth[1] if len(bounds) > 1 else 0
    return attrs, cid, hid


COLOR_CYCLE = ["light_yellow", "light_green"]
HIGHLIGHT_CYCLE = ["on_black", "on_dark_grey"]


def sample_hnet(
    m: "HNetLM", iids: TT, bos: int, tok2str_stream: Callable[[TT], TT], **sample_kwargs
):
    # yields [newchar, termcolor attributes for this char]
    assert iids.ndim == 1

    # model hook -- remaps TT<->NJT, caches bpred.b for chunk labelling
    b_prev, b_next = None, None

    def hook(iids: TT):
        nonlocal b_prev, b_next
        l, extras = m(NJT([iids]))
        b_prev, b_next = b_next, [e.b.values() for e in extras]
        return l.values()

    hook(iids)  # grab chunk labels

    cid, hid = 0, 0
    tok_stream = prefetched(sample_nocache(hook, iids, bos, **sample_kwargs))
    for s in tok2str_stream(tok_stream):
        # code below is just to trace chunk boundries / termcolor labels
        boundaries = [nxt[prv.shape[-1] :] for prv, nxt in zip(b_prev, b_next)]
        attrs, cid, hid = get_termcolor_from_boundaries(boundaries, cid, hid)
        yield (
            s,
            dict(
                color=COLOR_CYCLE[cid % 2],
                on_color=HIGHLIGHT_CYCLE[hid % 2],
                attrs=attrs,
            ),
        )


def colorize_prefill(m: "HNetLM", iids: TT, tok2str_stream: callable):
    assert iids.ndim == 1
    l, extras = m(NJT([iids]))
    ls_b = [e.b.values() for e in extras]
    b_prev = [torch.zeros(0) for t in ls_b]
    b_next = []

    def tid_tracker():
        for i, t in enumerate(iids.tolist()):
            b_next[:] = [ls_b[0][: i + 1]]
            for b in ls_b[1:]:
                b_next.append(b[: b_next[-1].sum()])
            yield [t]

    cid, hid = 0, 0
    for s in tok2str_stream(tid_tracker()):
        boundaries = [nxt[prv.shape[-1] :] for prv, nxt in zip(b_prev, b_next)]
        b_prev = b_next[:]
        attrs, cid, hid = get_termcolor_from_boundaries(boundaries, cid, hid)
        yield (
            s,
            dict(
                color=COLOR_CYCLE[cid % 2],
                on_color=HIGHLIGHT_CYCLE[hid % 2],
                attrs=attrs,
            ),
        )


### Aggregators (of tokens/bytes -> string streams)
def aggregate_bytes_to_utf8(g):
    # assuming g is a generator that yields a length 1 list[uint8],
    # this generator is a many-to-one stream of bytes to utf8
    bytelist = []
    for nt in g:
        bytelist += nt
        try:
            yield bytearray(bytelist).decode()
            bytelist = []
        except UnicodeDecodeError:  # only let UDC raise up iff exceeds max byte length
            if len(bytelist) > 4:
                raise


def completion_iter(p: str, t: Tokenizer, m: "HNetLM", **k):
    iids = t.encode([p])[0].cuda()
    AGGREGATORS = {ByteTokenizer: aggregate_bytes_to_utf8}
    aggregator = AGGREGATORS[t.__class__]

    for c, color_kwargs in sample_hnet(m, iids, t.bos_idx, aggregator, **k):
        yield colored(c, **color_kwargs)


def completion_sync(p: str, t: Tokenizer, m: "HNetLM", **k) -> str:
    try:
        return "".join(
            tqdm(completion_iter(p, t, m, **k), total=k.get("max_new", None))
        )
    except UnicodeDecodeError:
        return colored("[failed to decode UTF-8]", "red")
    except KeyError:
        return colored("[failed to decode BPE]", "red")


def colorize_byte_prefill(p: str, t: ByteTokenizer, m: "HNetLM") -> str:
    iids = t.encode([p])[0][1:].cuda()
    return "".join(
        colored(c, **k) for c, k in colorize_prefill(m, iids, aggregate_bytes_to_utf8)
    )


__all__ = [
    "ByteTokenizer",
    "completion_iter",
    "completion_sync",
    "colorize_byte_prefill",
]
