# mtlattn

Fused **flash-attention (forward)** for Apple Silicon — a Metal compute kernel
for PyTorch MPS tensors, with online softmax, fp32 accumulation, **no padding
and no materialized `[L, L]` score matrix**.

Variable-length (`cu_seqlens`) attention is the core; it also does **causal
masking, GQA/MQA, and sliding-window** attention, and ships a
`scaled_dot_product_attention` drop-in so existing models use it unchanged.

- **Two runtime paths, selected automatically**: the M5 per-core Neural
  Accelerator (Metal 4 `matmul2d`) where available, a portable
  `simdgroup_matrix` kernel on M1–M4. One wheel covers both.
- **Forward only** (inference); `head_dim ≤ 128`; fp16 / bf16 / fp32.

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
Apple Silicon Mac — M1–M4 use the simdgroup path, M5 the accelerator path,
selected at runtime; both metallibs are bundled and the MPP framework is
weak-linked, so it loads on macOS 13+.

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

# flash_attn-compatible wrappers (forward only):
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
and the padding is skipped, not just masked); any other `attn_mask` falls back.
The crossover length is `replace_sdpa(min_seqlen=...)`. `mtlattn.sdpa(...)` is
the same adapter callable directly.

(Arbitrary per-position `attn_mask` support in the kernel itself — prefix-LM,
custom patterns — is on the roadmap; today only padding/causal/window are
accelerated.)

`head_dim <= 128`. Forward only (inference); no backward pass.

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
(forward only), so it's an import swap:

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
- **MPP path** (default on M5 + macOS 26.2+): fused varlen attention through
  Metal 4 Metal Performance Primitives `matmul2d`, targeting the M5 per-core
  **Neural Accelerator** (fp16/bf16 operands, fp32 accumulate).
- **simdgroup path** (portable, M1+): the fallback used on older GPUs, older
  macOS, non-128 head dims, or when `MTLATTN_NO_MPP=1`. Tiled at 4 simdgroups
  / BK=8 so 4 resident simdgroups hide device-load latency (~1.5× a naive
  2-simdgroup tiling).

M5 Pro, bf16, 12 heads, head_dim 128, through the API:

| Path | TFLOPS | vs naive simdgroup |
|---|---|---|
| simdgroup (tiled) | ~0.8 | 1.5× |
| **MPP (M5 accelerator)** | **~5.0** | **~10×** |

vs padded SDPA (the usual MPS fallback): mtlattn runs windowed/ragged
attention ~20× faster, handles 49K-token sequences in constant memory where
SDPA needs 54 GiB, and is correct where SDPA silently corrupts (see below).

Reproduce on your machine with the benchmark CLI:

```bash
python -m mtlattn.bench --paths both --vs-sdpa --causal   # MPP, simdgroup, native SDPA
python -m mtlattn.bench --sizes 8192 --causal --window 256
```

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
