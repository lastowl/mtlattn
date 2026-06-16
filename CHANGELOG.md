# Changelog

## Unreleased

- **Backward pass** — `varlen_attention` is now differentiable (a
  `torch.autograd.Function` routes through new backward kernels when q/k/v
  require grad), so it trains. Composes with causal / GQA/MQA / sliding-window
  and varlen — variable-length training the MFA-based MPS kernels don't offer.
  The forward emits log-sum-exp on demand for the backward to recompute P from.
  The backward kernel is correct (validated vs autograd across the feature
  matrix) but currently naive (one thread per output row) — a tiled version is
  future work. Inference is unaffected (still the MPP path on M5).

## 0.1.0

First release. Fused forward flash-attention for PyTorch/MPS on Apple Silicon.

- **Variable-length** (`cu_seqlens`) attention — ragged/packed sequences, no
  padding, no materialized score matrix; `flash_attn`-compatible wrappers.
- **Causal** masking (flash end-aligned convention; KV-tile skipping ~2×).
- **GQA / MQA** — fewer KV heads than query heads, inferred from shapes.
- **Sliding-window / local** attention (`window=W`); loop jumps to the band so
  cost is O(window), ~7–14× faster than full at long sequences.
- **`scaled_dot_product_attention` drop-in** (`replace_sdpa()` / `mtlattn.sdpa`)
  with size-based routing and key-padding-mask → varlen conversion.
- **Two runtime paths**: Metal 4 `matmul2d` on the M5 Neural Accelerator
  (~10× the simdgroup path), portable `simdgroup_matrix` on M1–M4 (tuned to
  4 simdgroups / BK=8, ~1.5× a naive tiling). Selected automatically.
- fp16 / bf16 / fp32, `head_dim ≤ 128`, fp32 fragment accumulation.
- Benchmark CLI (`python -m mtlattn.bench`), 34-case correctness suite, and a
  GitHub Actions wheel-build workflow.

Forward only (no backward pass).

### Known limitations / roadmap

- No backward pass (inference only).
- Arbitrary per-position `attn_mask` in the kernel (prefix-LM, custom patterns)
  — today only padding / causal / window are accelerated; others fall back.
- `head_dim > 128`.
