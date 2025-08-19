To obtain scaling laws for any architecture, we must implement hardware-efficient training.

## Desiderata
For small scale research runs, the following criteria are ideal:
1. Compute blocks are torch.compile'able, with `fullgraph=True`.
2. Minimal CPU overhead / CUDA syncronization each train step.

It is possible to optimize small scale runs further than this. For example, you could compile an entire model's train step, or you could capture a train step as a CUDAGraph.

But doing so for H-Nets of arbitrary height is quite unrealistic, due to the inherent shape dynamism, so I avoid doing so.

## Block Compilation
In my [original code from July](https://github.com/main-horse/hnet/commit/8bf821ef6bb64a3e415b63fbe9cf1495f5add012), I described my code as "block-compilable".

While technically true (it did not produce errors), it did not produce anything close to optimal performance, as it involved many graph breaks. The published MFU graphs were also simply wrong due to drastic overestimation of the inner hierarchy's sequence lengths.

In reality, it takes significant effort to make an individual H-Net block compilable, especially for arbitrary hierarchies && sequence lengths.
### Mamba2
The first major roadblock against efficient H-Net training is the **Mamba2 layer**, and specifically its gigantic `mamba_split_conv1d_scan_combined` method.

![[Pasted image 20250726200755.png]]

Many past users of Mamba2 have complained about the [impossibility of compiling](https://github.com/state-spaces/mamba/issues/740) && the extreme slowness of it at small scale, to which the official reply is to "[use a large model](https://github.com/state-spaces/mamba/issues/355#issuecomment-2147597457)".

This seemed quite unreasonable to me, so I attempted to compile the model anyway. As it turns out, this is quite hard.

#### causal-conv1d
The first issue that presents itself is the incompatibility of torch.compile with `custom_op`s that have non-contiguous inputs.

I patch this by [subsuming the expected transpose into the `custom_op` wrapper](https://github.com/main-horse/mamba/commit/3c4fe71357bdfcf6773a51f70b97d6f2e2f0affb).

#### Autotuning dependency
The next issue is a recurring problem throughout mamba's triton kernels.

In many case, mamba's kernels have the following programming pattern:
1. create output tensor `y` of size `cdiv(seqlen,min(BLOCK_SIZES))`
2. run `kernel[...](y)`, with autotuning across `BLOCK_SIZES`
3. *extract* `bs = kernel.best_config.kwargs[BLOCK_SIZE]` from the best autotuned kernel, and use it to minimize future work on output `y`, e.g. `y[:cdiv(seqlen,bs)].sum()`

This is a disaster for torch.compile, as initial dynamo tracing **does not execute any kernels**, which means `kernel.best_config` no longer exists at compile time, causing code execution to error and explode...

Quick solution: bite the slight compute overhead, and [sum over all blocks](https://github.com/main-horse/mamba/commit/e6b84e4aec3e12d981e833c337af03da763696df) regardless of block size.

Also, many kernels use `pre_hook` to zero inputs during auto-tuning, but `pre_hook` is [not supported by torch.compile](https://github.com/pytorch/pytorch/issues/139059). Luckily, `reset_to_zero` is, so switching to it is sufficient to permit compilation for all mamba kernels.

#### Additional misc problems
[This commit](https://github.com/main-horse/mamba/commit/659d55367a9464f72cba9ef5c6270869831304d8) addresses even more problems:
* Somehow, inductor inlines user-defined kernels in a way that forgets to import `math` if it is used. Manual implementations of math methods resolves this.
* Inductor is unhappy if `reset_to_zero` tensors are only sometimes `None`. Use redundant tensors to avoid this.
* There is some internal torch confusion about [the ordering of named & parametric arguments to triton kernels](https://github.com/main-horse/mamba/commit/15e913426c3bbfaf0ed5c47f4c62b98b9e67f0a5), which sorting solves.



#### Redundant clones
After all of that bullshit, the `mamba_split_conv1d_scan_combined` is fullgraph compilable.

![[Pasted image 20250726200831.png]]

However, the naive compiled result is quite inefficient from a memory bandwidth perspective:

![[Pasted image 20250814183548.png]]
Each of those magenta `triton_poi*` blocks are wholly unnecessary copies from one memory address to another.

For example, the highlighted `triton_poi_fused_2` above implemented a redundant copy of one `empty` tensor to another (i.e. copying **uninitialized data**):
![[Pasted image 20250814183538.png]]

The origins of this stem from another programming pattern in mamba's source code, where
1. a large fused output tensor (e.g. `dzxbcdt`) is created
2. that tensor is `.split(...,dim=-1)` into subtensors, which is "free" in eager as it merely creates views with different strides
3. those split tensors are modified in-place, under the assumption that modifying them will also propagate writes to the root fused tensor `dzxbcdt`.

Unfortunately, real world kernels do not perform well if you have gaping holes between every row. Therefore, inductor tries to manually copy memory between fused<->chunk tensors.

Ultimately, this is highly redundant, and can be [solved with some reasonable redefinition](https://github.com/main-horse/mamba/commit/c1ccd18c3088ffb572811895e61935a5e3ee3b68). This improves the execution time of small-scale mamba layers by roughly 20%, as they are primarily memory bandwidth bound.


### Transformer
Since the rest of the isotropic block is defined the same way as a standard llama2 transformer, there are zero issues in fullgraphing each block.

One minor nit is that I patch/reimplement certain tridao kernels (rope, rmsnorm) to be more torch-compile friendly, as the existing public approach to supporting compile is "[not wrapping it right](https://github.com/Dao-AILab/flash-attention/issues/1680#issuecomment-2931241853)".

### Block generalization
If you thought fixing all of that was sufficient to make block compilation work correctly, think again.

$S=0$ nets work correctly. And -- in the pure "mamba $s=0$, transformer $s=1$" case -- $S=1$ nets as well, by the undefined behavior of pytorch 2.7.

$S=2$ nets are broken. I don't have a saved image, but it will produce some gobbledygook about hitting recompile limits on dynamic shapes.

The underlying issue is the original code's treatment of [all Isotropic layers as implementations of the same `Block`](https://github.com/goombalab/hnet/blob/main/hnet/modules/block.py#L108). Because,
- torch.compile treats all functions with the same code object (hash) as identical
- All `Block.forward` methods are the same object, and hence the same "function"

Dynamo's interpretation of compiled H-Net blocks is thus equivalent to a ridiculously dynamic function which:
* has varying sequence length (ok)
- can either implement a mamba2 *or* transformer forward (ok...?)
- has parameters with dynamically varying hidden dim (very bad)

But I only need the first item on that list to be true. So, to sidestep this behavior, I transform the Block into a metaclass:
![[Pasted image 20250814202650.png]]
Whenever a new _kind_ of Block is required, it copies the implementation of `.forward` into a new `FunctionType` with a different hash.

That ensures dynamo correctly specializes the compiled code, depending on the behavior of each block, and helps to fix $S>1$ nets by proxy.

### Block overhead
[In torchtitan](https://github.com/pytorch/torchtitan/blob/7354848dfb6dd2d67727a4702130f75c5985ed94/torchtitan/experiments/llama4/infra/parallelize.py#L449), block compilation works quite well, as a motivated strategy to:
- obtain reasonable performance
- reduce compile time
- keep compilation composable with parallelism APIs

All those points are still true for H-Nets of sufficient size.

When you have insufficient size, this occurs:
![[Pasted image 20250814184724.png]]
A naive approach to block compilation leads to only ~50% of execution time occurring inside the actual inductor code produced for the block, and the rest of it spent on safety barriers like shape size counting and dynamo guards.

Of course, those barriers are important, as they are responsible for triggering recompilation in the event of varied sequence lengths, and/or accidental external modifications to the global torch environment. So it would be bad to remove them completely.

On the other hand, executing them *every* block (which torch must do for generalized correctness) is a waste, if we know with certainty that a block is repeated. Therefore, we ~~use the 2.8 `nested_compile_region` feature~~ manually implement a bespoke strategy to evade guards in the event of a repeated layer variant:
![[Pasted image 20250814185517.png]]

After doing so, the compile guards only exist for the first layer an Isotropic:
![[Pasted image 20250814185751.png]]

Obviously, this is highly unsafe, and should only be done with the confident backing of someone who has worked on the torch.compile ecosystem extensively.

---

So, after we deal with all of those horrors, the performance of Isotropics are quite well-optimized.

Yet, if we look at the graph on a reasonably small $S=1$ model:
![[Pasted image 20250814203212.png]]
There is still obscene overhead from the external modules -- close to 40% of execution time, which creates obvious compute bubbles at the top.
## CPU Overhead
Essentially I do four things:
0. train with 1gpu (so no FSDP2 overhead)
1. nuke NJT
2. compile other modules
3. overlap masked_select with compute

This makes the forward pass very ugly. I go from this (beautiful, clean code):
![[Pasted image 20250814203658.png]]

To this garbage:
![[Pasted image 20250814203729.png]]

This is not fun at all, so I don't really want to elaborate on it.

## Conclusion
After that work, even my `S2-small.yaml` config is incredibly CUDA bound:
![[Pasted image 20250814204035.png]]

