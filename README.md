# H-Net Implementation

Efficient trainable H-Net implementation.

## Install
In an external project:
```bash
src=git+https://github.com/main-horse/hnet-impl
uv add "hnet-impl @ $src"
uv add "hnet-impl[build] @ $src" --no-cache
```
For local development:
```bash
git clone https://github.com/main-horse/hnet-impl && cd hnet-impl
uv sync && uv sync --extra build
```

## Usage
```python
import torch
from hnet_impl import HNetLM, HNetConfig, completion_sync
from hnet_impl.torchisms import rand_njt_iids

c = HNetConfig.create_reasonable_config(D=[512,1024], arch=['m4','T9'])
with torch.device('cuda'): m = HNetLM(c).bfloat16()

# inference
iids = rand_njt_iids(docs=16, slen=range(128,1024)).cuda()
logits,_ = m(iids)

# training
lbls = iids.long() # i.e. torch.randint_like(iids)
(celoss,_),extra = m(iids,lbls)
```

## Dummy Training Example
To train a simple H-net for dynamically encoding letter repetition:

```python
$ uv run torchrun --nproc-per-node 2 --standalone example.py
  0%|                                                                                                                    | 0/99 [00:32<?, ?it/s]
[failed to decode UTF-8]
step=0: l_avg.item()=5.698 l_ratio.item()=1.064
...
step=990: l_avg.item()=0.462 l_ratio.item()=1.016
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▉ | 98/99 [00:02<00:00, 38.86it/s]
accddddddeeeeffggggggggghhhhhhhhhhiiiiijkkkkkkkkkklllllllllmmmmmmmmmmnnnoooooopppppqqrrrrrrrssssss
```
Where the resultant text should be rendered like:

${{\small{\color{Goldenrod}\texttt{a}\color{lightgreen}\texttt{c}\color{ForestGreen}\texttt{c}\color{Goldenrod}\texttt{d}\color{tan}\texttt{ddddd}\color{lightgreen}\texttt{e}\color{ForestGreen}\texttt{eee}\color{Goldenrod}\texttt{f}\color{tan}\texttt{f}\color{lightgreen}\texttt{g}\color{ForestGreen}\texttt{gggggggg}\color{Goldenrod}\texttt{h}\color{tan}\texttt{hhhhhhhhh}\color{lightgreen}\texttt{i}\color{ForestGreen}\texttt{iiii}\color{Goldenrod}\texttt{j}\color{lightgreen}\texttt{k}\color{ForestGreen}\texttt{kkkkkkkkk}\color{Goldenrod}\texttt{l}\color{tan}\texttt{llllllll}\color{lightgreen}\texttt{m}\color{ForestGreen}\texttt{mmmmmmmmm}\color{Goldenrod}\texttt{n}\color{tan}\texttt{nn}\color{lightgreen}\texttt{o}\color{ForestGreen}\texttt{ooooo}\color{Goldenrod}\texttt{p}\color{tan}\texttt{pppp}\color{lightgreen}\texttt{q}\color{ForestGreen}\texttt{q}\color{Goldenrod}\texttt{r}\color{tan}\texttt{rrrrrr}\color{lightgreen}\texttt{s}\color{ForestGreen}\texttt{sssss}}}}$

## Testing
kernels:
```python
uv run -m hnet_impl.norm
uv run -m hnet_impl.lin
```
$S=1$ block profiling:
```python
TORCH_LOGS=recompiles uv run -m hnet_impl.xf --s0=9289 --s1=2048 --d0=512 --d1=768 --lm=4 --lt=10
```
`2stage_XL` fwd equiv check:
```python
uv run -m hnet_impl.modeling_hnet
```
