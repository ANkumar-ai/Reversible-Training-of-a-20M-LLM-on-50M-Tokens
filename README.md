# ERA V5 · Session 13 — Reversible Training of a 20M LLM on 50M Tokens

Notebook: [`session13_reversible_20M.ipynb`](session13_reversible_20M.ipynb). Open in Colab, pick a GPU runtime, then Run all. It needs only `torch`, `matplotlib` and `datasets` (all Colab defaults). It uses bf16 on A100/L4 and fp16 with GradScaler on T4, with no code change.

## Setup
1. Model: GPT, 21.5M params. 12 layers, width 384, 6 heads, context 512, tied embeddings.
2. Data: TinyStories, streamed. 50M bytes for training, 1M held out for validation.
3. Tokenizer: raw bytes, vocab 256. I chose this so the logits (256 × tokens) stay tiny and don't hide the activation memory that reversibility removes. A 50K BPE vocab would make the logits the biggest tensor.
4. Training: AdamW (β 0.9/0.95), learning rate 1e-3 with 200 warmup steps then cosine decay to 1e-4, gradient clipping at 1.0, init std 0.02.
5. Weight decay is 0 in every run. Reversibility needs it.
6. There is no dropout, because the backward pass must recompute each block exactly as the forward pass ran it.
7. Hardware: Colab T4, 15 GB, fp16.

## Runs
1. **Standard**: a normal residual stream `p = p + f(p)`, batch 32.
2. **Midpoint**: `p[l+1] = p[l-1] + 2h·f(p[l])`, with h = 0.25, batch 32.
3. **Symplectic Euler**: two streams, `v += h·f(x); x += h·v`, with h = 0.5, batch 32.
4. **Midpoint with fp32 blocks**: same as run 2 but the blocks run in fp32, to check whether rounding in the rebuilt activations costs loss.
5. **Midpoint at max batch**: the largest power-of-two batch that fits, same 50M tokens.

Here `f(p)` is the block's update only (attention plus MLP), without the residual add.

## Results

| run | batch | steps | train loss | val loss | tok/s | peak GiB | rebuild error |
|---|---|---|---|---|---|---|---|
| standard | 32 | 3051 | 0.601 | **0.631** | 61,156 | 3.04 | — |
| midpoint | 32 | 3051 | 0.615 | 0.643 | 46,424 | **0.88** | 9.3e-2 |
| symplectic Euler | 32 | 3051 | 0.623 | 0.651 | 46,177 | 0.88 | 2.7e-1 |
| midpoint, fp32 blocks | 32 | 3051 | 0.616 | 0.644 | 18,093 | 1.11 | 1.3e-4 |
| midpoint, max batch | 512 | 190 | 2.249 | 2.224 | 47,899 | 7.57 | 2.5e-1 |

- Losses are in nats per byte.
- Rebuild error = run the stack forward, invert it back to the input, then take the max difference relative to the input's size.
- Max batch: standard 128, midpoint 512. That is 4x.

## Findings
1. **Midpoint worked better than Euler.** Val loss was 0.643 vs 0.651, and rebuild error was 3x smaller. Midpoint is the variant I'd use.
2. **Memory is 3.5x lower at the same batch** (3.04 → 0.88 GiB). Only the final two residual states are stored, plus one layer's working memory during backward.
3. **Speed cost is 24%** (61K → 46K tok/s). Each block runs its forward pass a second time during backward. That fits the paper's 30–50% estimate.
4. **The loss cost is small:** 0.643 vs 0.631, about 2%.
5. **Rebuild error does not explain the loss gap.** fp32 blocks cut rebuild error about 700x (9.3e-2 → 1.3e-4) but val loss stayed the same (0.644 vs 0.643). The fp16 rounding makes gradients noisier without hurting training. The 2% gap comes from the reversible update rule itself, not from precision. fp32 was also 2.6x slower on the T4, so it isn't worth it.
6. **Max batch is 4x larger** (128 → 512, peak 7.6 GiB). On a T4, throughput rose only 3%, because the GPU was already fully busy computing at batch 32. An earlier A100 run showed +20%, because the A100 still had unused compute at batch 32.
7. **The max-batch loss (2.224) is not comparable to the others.** The same 50M tokens at batch 512 is only 190 optimizer steps instead of 3051. A bigger batch buys throughput, not learning per token, unless the learning rate and token budget are rescaled.

## What I understood

**Why activations are the problem.** Autograd keeps every layer's input for the backward pass. That memory grows with batch × sequence × depth. ZeRO shards weights, gradients and optimizer state, but never activations.

**Why a normal residual block can't be reversed.** Recovering `p[l]` from `p[l+1] = p[l] + f(p[l])` needs `f(p[l])`. That in turn needs `p[l]`, the very thing we're trying to recover.

**How midpoint gets around it.** It adds the update to the state two layers back and evaluates `f` at the state in between: `p[l+1] = p[l-1] + 2h·f(p[l])`. In backward we already hold `p[l]` and `p[l+1]`, so `p[l-1] = p[l+1] − 2h·f(p[l])` can be computed exactly.

**How symplectic Euler works.** It keeps a second "velocity" stream `v`. Each step can be undone in reverse order: `x -= h·v`, then `v -= h·f(x)`.

**What the backward pass does, per layer, from top to bottom:**
1. Rebuild the layer's input from the two states I'm holding.
2. Rerun that one block with autograd on.
3. Backprop through it.
4. Throw it away.

So memory no longer depends on depth. That is where the 3.5x saving comes from, and it grows with more layers.

**The costs:**
1. About 25% more compute, from one extra forward pass per layer.
2. No dropout and no other randomness in the forward pass.
3. Weight decay is set to 0.
4. The step size `h` needs tuning, because the scheme is only marginally stable.

**What it changes for parallelism.** Once activations are small, the memory limit becomes the training state (weights, gradients, optimizer). That brings ZeRO back as the first tool to use, and pipeline and context parallelism are needed later than before.

## Limits
1. It's a single-GPU experiment. It doesn't combine reversibility with ZeRO, tensor, pipeline or context parallelism.
2. The step sizes `h` are starting values, not tuned.
3. The "Euler" variant is my two-stream symplectic form. It may differ from the paper's exact formulation.
4. The max-batch run keeps the learning rate and token budget fixed, so its loss only shows what a bigger batch does with the same tokens. It isn't a tuned large-batch result.
