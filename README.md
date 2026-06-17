# mtlattn

Fused **flash-attention (forward + backward)** for Apple Silicon — a Metal
compute kernel for PyTorch MPS tensors, with online softmax, fp32 accumulation,
**no padding and no materialized `[L, L]` score matrix**.

Variable-length (`cu_seqlens`) attention is the core; it also does **causal
masking, GQA/MQA, sliding-window, and arbitrary additive attention bias**
(prefix-LM / ALiBi / custom masks), **auto-accelerates decode** (splitKV /
FlashDecoding — 8–20× on few-query/long-KV generation), and ships a
`scaled_dot_product_attention` drop-in so existing models use it unchanged.

- **Two runtime paths, selected automatically**: the Metal 4 `matmul2d`
  accelerator path (any GPU on macOS 26.2+ — M5's Neural Accelerator where
  present, regular matrix units on M3/M4) for head_dim 64/128, and a portable
  `simdgroup_matrix` kernel everywhere else. One wheel covers both.
- **Forward + backward** — `varlen_attention` is differentiable, so it trains
  (the one thing the MFA-based MPS kernels don't do for ragged sequences). The
  backward uses a simdgroup-per-row kernel (~3.5× the forward, near the FLOPs
  ideal). head_dim 64/96/128/256 on the fast path, any ≤128 on the fallback;
  fp16 / bf16 / fp32.

Built for [pixal3d-mac](https://github.com/lastowl/pixal3d-mac) (image-to-3D on
Mac), but standalone: a drop-in for the `flash_attn` varlen API and for
`F.scaled_dot_product_attention` on any MPS workload — sparse / 3D transformers,
and LLM inference (Llama / Mistral / Qwen-class: causal + GQA + sliding-window).

## Why this exists

On Apple Silicon there is no `flash_attn`. The usual fallback is to pad
ragged sequences into a dense `[B, H, Lmax, D]` batch and call
`scaled_dot_product_attention` with a mask. That has two problems this
kernel fixes:

1. **Wasted compute + memory** on padding, and an `O(B·H·Lmax²)` score
   tensor that blows up unified memory (a real 49K-token workload needed a
   54 GiB allocation).
2. **A silent-correctness bug in PyTorch's MPS SDPA.** When the score
   matrix `B·H·Nq·Nkv` exceeds ~2³² elements, MPS SDPA returns physically
   impossible values with *no error* (the corruption hits later query rows
   first, so naive spot-checks of the first rows miss it). This is a known
   upstream bug — reported as
   [pytorch/pytorch#179352](https://github.com/pytorch/pytorch/issues/179352)
   and fixed by the in-progress PR
   [#179592](https://github.com/pytorch/pytorch/pull/179592); it reproduces
   on torch ≤ 2.12 until that lands. The root cause is a 32-bit index inside
   Apple's MPSGraph (reachable via `sdpa_general_mps`). This kernel streams
   in constant memory and matches a CPU fp32 reference at those sizes, so
   it's correct today regardless. (Repro:
   [`tests/test_mps_sdpa_bug.py`](tests/test_mps_sdpa_bug.py).)

## Install

```bash
pip install mtlattn
```

Requires macOS on Apple Silicon and PyTorch with MPS. The published wheels are
built against **torch 2.12** for **Python 3.11–3.13** (a torch C++ extension is
tied to the torch version it was built against). One arm64 wheel covers every
Apple Silicon Mac — the Metal 4 `matmul2d` accelerator path runs on any GPU with
macOS 26.2+ (M3/M4/M5; confirmed on M4), the portable simdgroup path covers M1/M2
and pre-26.2; selected at runtime. Both metallibs are bundled and the MPP
framework is weak-linked, so it loads on macOS 13+.

On other torch versions, build from source:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  pip install --no-build-isolation .
```

Building from source needs Xcode with the Metal Toolchain (`xcodebuild
-downloadComponent MetalToolchain`); the metal4.0 MPP (M5) path additionally
needs Xcode 26 / macOS 26.2, else the build falls back to the portable
simdgroup kernel.

## Use

```python
import mtlattn, torch

# q, k, v: [total_tokens, num_heads, head_dim] bf16/fp16/fp32 on MPS.
# cu_seqlens_*: int32 [num_seqs + 1], cumulative sequence lengths.
# GQA/MQA: k/v may have fewer heads than q (q heads must be a multiple of
# kv heads); each query head reads kv head (q_head // (H_q / H_kv)).
out = mtlattn.varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                               max_seqlen_q, scale=None, causal=False)

# causal=True: query i attends key j iff j <= i + (kv_len - q_len), the
# flash_attn end-aligned convention (self-attention and cached-decode both
# work). Fully-masked KV tiles are skipped, so causal self-attention is ~2x
# faster than full.

# window=W: sliding-window / local attention — query i attends only its last
# W keys (relative to the causal diagonal). Mistral-style SWA is causal=True
# with window=W. The kernel jumps straight to each block's window band, so
# cost is O(W) not O(seqlen): ~7-14x faster than full at long sequences.

# attn_bias=B: arbitrary ADDITIVE mask, [total_q, H or 1, max_kv] fp32, added to
# the logits before softmax (logit = scale*(q·k) + bias). dim1==1 broadcasts
# across heads. Use it for prefix-LM, ALiBi, custom/soft patterns — anything not
# expressible as causal/window. Indexed by global query row and seq-local key;
# composes with causal/window/GQA. Applied in forward AND backward (the bias is
# treated as constant — a grad-requiring bias raises). MPP-only (macOS 26.2+,
# fp16/bf16, head_dim 64/96/128/256); a bool mask becomes 0 / -inf additive.
out = mtlattn.varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                               max_seqlen_q, attn_bias=B)

# flash_attn-compatible wrappers (differentiable — forward + backward):
out = mtlattn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen)
out = mtlattn.flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, max_q, max_k)
```

### Drop-in for `scaled_dot_product_attention`

```python
import mtlattn

mtlattn.replace_sdpa()   # patches F.scaled_dot_product_attention globally
# ... run any PyTorch/HF model on MPS; large forward calls now use mtlattn ...
mtlattn.restore_sdpa()   # undo
```

`replace_sdpa()` routes dense `[B, H, N, D]` attention to mtlattn only where it
wins — long/ragged sequences, and the cases native MPS SDPA pads, OOMs, or hits
the `>2^32` MPSGraph bug on — and falls back to native SDPA otherwise (small
shapes, autograd/training, unsupported dtype/head_dim). A self-attention
**key-padding `attn_mask`** is converted to varlen (the valid tokens are packed
and the padding is skipped, not just masked). Any **other `attn_mask`** (a
general bool or additive-float mask, `[Nq,Nkv]` or broadcastable `[B,H,Nq,Nkv]`)
is applied as an **additive bias** on the kernel's MPP path — prefix-LM, ALiBi,
custom/soft patterns all work — and falls back to native SDPA only if MPP is
unavailable or the mask shape can't be mapped. The crossover length is
`replace_sdpa(min_seqlen=...)`. `mtlattn.sdpa(...)` is the same adapter callable
directly.

**head_dim**: 64/80/88/96/128/160/256 run on the accelerator (MPP) path
(`matmul2d` is dimension-general, so the non-32-multiple dims used by some image
models — SD1.5 / Hunyuan DiT 80/88, SD1.5 160 — get the fast path too, ~7× the
simdgroup kernel); any other `head_dim ≤ 128` runs on the portable simdgroup
kernel; `head_dim` 160/256 need the MPP path (macOS 26.2+). **Differentiable** —
if `q`/`k`/`v` require grad, `varlen_attention` routes through the backward
kernel (training), composing with causal / GQA / sliding-window / additive mask;
otherwise it uses the fast inference path. On the MPP path the backward also runs
on `matmul2d` (two flash-attn-2 kernels: a per-Q-block dQ and a per-KV-block
dK/dV); it's the slower half (~5.8 vs ~9.5 TF on M5) but still ~11× the
simdgroup-per-row fallback used on M1/M2.

Runnable tour of all of the above: [`examples/quickstart.py`](examples/quickstart.py).

## Using it in your project

**Existing PyTorch / Hugging Face model — no code changes.** Route attention
through mtlattn for the large forward passes it wins on; everything else falls
back to native SDPA:

```python
import mtlattn
mtlattn.replace_sdpa()        # patch F.scaled_dot_product_attention (inference)
# ... load and run your model on device="mps" as usual ...
```

**Already using `flash_attn`.** The varlen entry points are signature-compatible
and differentiable (forward + backward), so it's an import swap:

```python
# from flash_attn import flash_attn_varlen_qkvpacked_func
from mtlattn import flash_attn_varlen_qkvpacked_func
```

**Custom transformer / new code.** Call the kernel directly with the flags you
need — ragged batches, causal, GQA, sliding window all compose:

```python
out = mtlattn.varlen_attention(q, k, v, cu_q, cu_kv, max_seqlen_q,
                               causal=True, window=4096)
```

**Sparse / 3D transformers (TRELLIS-family).** The ragged `cu_seqlens` path is
the original use case — packed variable-length sequences with no padding and no
materialized score matrix (e.g. [pixal3d-mac](https://github.com/lastowl/pixal3d-mac)).

Inference only: training/autograd, sub-threshold shapes, and unsupported
dtype/head_dim fall back to native SDPA rather than erroring.

## Performance

Two kernel paths, selected automatically at runtime:
- **MPP path** (default on **any GPU with macOS 26.2+**, head_dim 64/96/128/256,
  fp16/bf16): fused varlen attention through Metal 4 Metal Performance Primitives
  `matmul2d`. The path is OS-gated, not GPU-family-gated — on M5 `matmul2d`
  targets the per-core **Neural Accelerator** (~9 TFLOPS); on M3/M4 it runs on the
  regular GPU matrix units (**confirmed on an M4: ~1.9 TFLOPS, ~3–4× the
  simdgroup kernel and ~2–3× native SDPA**). The size-adaptive query tile (TM
  16↔32) is gated to Apple10+ GPUs — M3/M4 always use TM=16, which is fastest
  there at every length.
- **simdgroup path** (portable, M1+): the fallback used on older GPUs, older
  macOS, head dims other than 64/128, or when `MTLATTN_NO_MPP=1`. For head_dim
  128 it's a register-resident kernel — scores/probs/output live in
  simdgroup-matrix registers (in-register softmax + online rescale, no
  threadgroup score buffers), mixed-precision matmul (half/bf16 operands, fp32
  accumulators).

M5 Pro, fp16, 12 heads, head_dim 128, through the API:

| Path | TFLOPS | notes |
|---|---|---|
| simdgroup (register-resident) | ~1.2 | portable M1+; ~0.4× native MPS SDPA on *dense* shapes |
| **MPP (M5 accelerator)** | **~10** | **~8× the simdgroup path; ~3.4× native SDPA (~2.9)** |

The MPP path streams K/V in TM=16 query tiles with the online-softmax output
accumulated in threadgroup memory via `matmul2d` multiply-accumulate, a
2-threads-per-row parallel softmax between the matmuls, and a TN=48 key tile.
~33% of the M5 Neural Accelerator's ~30-TFLOPS fp16 matmul peak, ~77% of the
practical flash-attention ceiling. The backward runs on the same `matmul2d` path
(~12 TFLOPS, faster than the forward — it's more matmul-dense).

The simdgroup path is a portable fallback, not a dense-SDPA competitor: on equal-
length dense attention native MPS SDPA is faster (~2.9 TFLOPS). mtlattn wins
where SDPA can't go — ragged/windowed/varlen shapes (no padding, no `[L,L]`
matrix), the M5 accelerator, training (backward), and the >2³² correctness bug.

vs padded SDPA (the usual MPS fallback): mtlattn runs windowed/ragged
attention ~20× faster, handles 49K-token sequences in constant memory where
SDPA needs 54 GiB, and is correct where SDPA silently corrupts (see below).

Reproduce on your machine with the benchmark CLI:

```bash
python -m mtlattn.bench --paths both --vs-sdpa --causal   # MPP, simdgroup, native SDPA
python -m mtlattn.bench --sizes 8192 --causal --window 256
```

> **Benchmarking caveat:** Apple-Silicon GPU clocks are load-state-dependent —
> a cold call after idle can read ~¼ of the warmed number, and short bursts
> understate sustained large-N throughput. Warm up and take a median. Full
> measured ceilings, what bounds each kernel, the tuning constants, and the
> known dead-ends (levers that are API-blocked or lose to occupancy) are in
> [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Correctness

`python tests/test_correctness.py` — 34 cases vs a per-sequence fp32
reference across fp16/bf16/fp32, ragged self/cross attention, packed forms,
odd head dims, thousands of tiny windows, causal / GQA-MQA / sliding-window
(and their combinations), the `sdpa()` / key-padding adapters, and an
outlier-channel overflow regression
(transformer activations spike to ~10²–10³; fp32 fragment accumulation is
required — half fragments overflow to NaN).

## Notes

- Fragments accumulate in fp32 for all input dtypes. The softmax scale is
  folded into Q at staging so scores live near ±1 (better fragment
  precision than at raw logit magnitude).
- The kernel encodes into PyTorch's `MPSStream`, so it sequences correctly
  with surrounding torch ops without a per-call CPU sync.

## Credits

- The torch↔Metal buffer bridge pattern follows
  [Pedro Naugusto's mtlgemm](https://github.com/pedronaugusto/mtlgemm).
- The MPP path targets the M5 Neural Accelerator via Metal 4 `matmul2d`;
  the simdgroup fallback draws on the tiling approach of
  [Philip Turner's metal-flash-attention](https://github.com/philipturner/metal-flash-attention).
- The `replace_sdpa()` drop-in and the causal / GQA / sliding-window feature
  set were inspired by
  [mpsops/mps-flash-attention](https://github.com/mpsops/mps-flash-attention),
  a related flash-attention-for-PyTorch-MPS project; mtlattn's own focus is
  variable-length (`cu_seqlens`) attention and the M5 accelerator path.

## License

MIT.
