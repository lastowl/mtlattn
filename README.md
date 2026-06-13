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
2. **A silent-correctness bug in PyTorch's MPS SDPA.** For a large score
   matrix (empirically `heads·L·L` ≳ 7e9 elements — e.g. ≥ ~24K tokens at
   12 heads), MPS SDPA returns physically impossible values with *no error*
   — both masked and unmasked. Smaller is correct; even larger OOMs. Any
   large attention on MPS is exposed. This kernel streams in constant
   memory and matches a CPU fp32 reference at those sizes. (Minimal repro
   with the measured threshold sweep in
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
out = mtlattn.varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                               max_seqlen_q, scale=None)

# flash_attn-compatible wrappers (forward only):
out = mtlattn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen)
out = mtlattn.flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, max_q, max_k)
```

`head_dim <= 128`. Forward only (inference); no backward pass.

## Performance

M5 Pro, bf16, 12 heads, head_dim 128:

| Workload | padded SDPA | mtlattn | speedup |
|---|---|---|---|
| windowed (2000 windows ≤512 tok) | 50.5 s | 2.5 s | **20×** |
| full attention, 49K tokens | OOM (54 GiB) | constant memory | runs at all |
| full attention, < ~19K tokens | faster | slower (~2.5×) | use SDPA below threshold |

So the intended use is **hybrid**: SDPA for small sequences where it's both
fast and correct, mtlattn for large/ragged ones where SDPA is slow, OOMs, or
silently corrupts.

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
