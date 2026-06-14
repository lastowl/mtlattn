# mtlattn

Fused **variable-length attention** (forward) for Apple Silicon, as a Metal
compute kernel for PyTorch MPS tensors. A flash-attention-style kernel:
online softmax, `simdgroup_matrix` tiling, fp32 accumulation, **no padding
and no materialized `[L, L]` score matrix**.

Built for [pixal3d-mac](https://github.com/lastowl/pixal3d-mac) (image-to-3D
on Mac), but standalone — it's a drop-in for the `flash_attn` varlen API on
any MPS workload with ragged sequences (sparse transformers, packed batches,
windowed attention).

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

Requires macOS on Apple Silicon, Xcode with the Metal Toolchain
(`xcodebuild -downloadComponent MetalToolchain`), and PyTorch with MPS.

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  pip install --no-build-isolation .
```

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

# flash_attn-compatible wrappers (forward only):
out = mtlattn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen)
out = mtlattn.flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, max_q, max_k)
```

`head_dim <= 128`. Forward only (inference); no backward pass.

## Performance

Two kernel paths, selected automatically at runtime:
- **MPP path** (default on M5 + macOS 26.2+): fused varlen attention through
  Metal 4 Metal Performance Primitives `matmul2d`, targeting the M5 per-core
  **Neural Accelerator** (fp16/bf16 operands, fp32 accumulate).
- **simdgroup path** (portable, M1+): the fallback used on older GPUs, older
  macOS, non-128 head dims, or when `MTLATTN_NO_MPP=1`.

M5 Pro, bf16, 12 heads, head_dim 128, through the API:

| Path | TFLOPS | vs simdgroup |
|---|---|---|
| simdgroup | ~0.5 | 1× |
| **MPP (M5 accelerator)** | **~5.0** | **~10×** |

vs padded SDPA (the usual MPS fallback): mtlattn runs windowed/ragged
attention ~20× faster, handles 49K-token sequences in constant memory where
SDPA needs 54 GiB, and is correct where SDPA silently corrupts (see below).

## Correctness

`python tests/test_correctness.py` — 14 cases vs a per-sequence fp32
reference across fp16/bf16/fp32, ragged self/cross attention, packed forms,
odd head dims, thousands of tiny windows, and an outlier-channel overflow
regression (transformer activations spike to ~10²–10³; fp32 fragment
accumulation is required — half fragments overflow to NaN).

## Notes

- Fragments accumulate in fp32 for all input dtypes. The softmax scale is
  folded into Q at staging so scores live near ±1 (better fragment
  precision than at raw logit magnitude).
- The kernel encodes into PyTorch's `MPSStream`, so it sequences correctly
  with surrounding torch ops without a per-call CPU sync.

## License

MIT. The torch↔Metal buffer bridge pattern follows
[Pedro Naugusto's mtlgemm](https://github.com/pedronaugusto/mtlgemm).
