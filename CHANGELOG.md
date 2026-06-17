# Changelog

## Unreleased

- **splitKV / FlashDecoding — decode is now 8–20× faster.** At decode (few
  queries against a long KV cache) the normal forward gives one threadgroup per
  (seq, head), each streaming the whole cache serially — ~1000× below prefill
  efficiency (~7 GFLOP/s). splitKV adds a KV-split grid dimension: each
  threadgroup attends only its KV chunk and writes a normalized partial output +
  LSE, then a tiny combine kernel merges the splits via the online-softmax
  identity. `varlen_attention` **auto-detects** decode shapes (small
  `max_seqlen_q`, long KV, D 64/128, half/bf16, no window/bias) and routes them
  through it; prefill/training are unchanged. Measured on M5: B=1/KV=8192
  7.06→0.47 ms (15×), B=1/KV=16384 13.9→0.69 ms (20×), B=8/KV=8192 16.9→1.87 ms
  (9×). Exact vs the reference (relerr ~3e-5). Override with `MTLATTN_NO_SPLITKV`.
- **head_dim 80, 88, 160 on the accelerator (MPP) path** — `matmul2d` is
  dimension-general, so these non-32-multiple dims (SD1.5 / Hunyuan DiT use
  80/88; SD1.5 uses 160) no longer fall to the ~8×-slower simdgroup path. 80/88
  go from ~1.1 TF → ~8 TF (≈7×); 160 previously **errored entirely** and now runs
  at ~9 TF. Forward + backward, fp16/bf16, validated against the fp32 reference.
  head_dim coverage on MPP is now 64/80/88/96/128/160/256.
- Verified the matmul2d **backward is already at its tile optimum** — smaller
  tiles (dq BQ 32→16, dkv BK 16→8) were *slower*, matching the docs' finding that
  bigger tiles also lose. The ~5.8 TF is structural (2.5× the forward's FLOPs +
  accumulator occupancy), not a missed optimization.

## 0.3.0 (2026-06-17)

Headline: **arbitrary additive attention masks** (prefix-LM / ALiBi / custom
patterns) now run on the accelerator path, forward and backward, with zero
throughput cost when unused.

- **Arbitrary additive attention bias (`attn_bias`)** — a per-(query, key)
  additive mask, `[total_q, H or 1, max_kv]` fp32, added to the logits before
  softmax (`logit = scale*(q·k) + bias`); `dim1==1` broadcasts across heads.
  Covers prefix-LM, ALiBi, and arbitrary custom/soft masks — anything not
  expressible as causal/window. Applied in **both forward and backward**
  (dQ/dK/dV); the bias is treated as constant (a grad-requiring bias raises —
  no `dbias` yet). The `sdpa()` / `replace_sdpa()` adapter now routes any
  general bool or additive-float `attn_mask` (`[Nq,Nkv]` or broadcastable
  `[B,H,Nq,Nkv]`) through this path, falling back to native SDPA only if MPP is
  unavailable. **MPP-only** (macOS 26.2+, fp16/bf16, head_dim 64/96/128/256);
  the simdgroup fallback refuses a bias rather than silently dropping it.
  - Implemented with a `HAS_BIAS` Metal **function constant**, so the no-bias
    pipelines are dead-branch-eliminated — measured **zero** throughput impact
    on the (common) no-bias forward (9.66 vs ~9.5 TF). The bias path abandons
    the raw-max softmax shortcut (the per-element bias breaks `max(S·scl) =
    scl·max(S)`) and reduces the max over the full `S·scl + bias·log2(e)` logit;
    the resulting LSE includes the bias, so the backward just adds `+bias` in
    its P recompute. Prior art cross-checked against MLX's SDPA (same base-2
    `×log2(e)` bias fold) — and mtlattn's fused **masked backward** is ahead of
    MLX, which falls back to unfused autograd for training.

## 0.2.0 (2026-06-16)

Headline: both the forward **and** the backward now run on the Metal 4
accelerator (`matmul2d`), the fast path is confirmed on M3/M4 (not just M5), and
head_dim 64/96/128/256 are supported. Forward ~10 TFLOPS / backward ~12.5 TFLOPS
on M5; a training step is ~18× faster than 0.1.x. All hardware-validated on M5
and M4.

- **head_dim 256 backward** — the matmul2d backward now covers 256 as well
  (BK=8 / BQ=16 so the `[·,256]` dK/dV/dQ accumulators fit threadgroup memory),
  matching the forward's head dims. Validated end-to-end incl GQA and short
  sequences. Also: head_dim 256 forward now runs at any sequence length (the
  MPP `min_seq` gate is bypassed for 256, which has no simdgroup fallback).

- **Forward key tile TN 32 → 48 (~+13–33% on M5)** — the optimal MPP key tile
  shifted once the softmax got cheaper (exp2 / raw-max scan) and the PV matmul
  started accumulating in place, so the old TN=32 (tuned before those) was
  leaving throughput on the table. Bigger key tiles mean fewer, larger QK/PV
  matmul2d calls and fewer softmax passes — the same "bigger tiles when the
  GPU is busy" lesson the backward surfaced. The bigger tile also moved the
  TM=16→32 crossover up (~10K → ~14K), so the 12–18K-token (3D-sparse) regime
  now stays on the faster TM=16. Forward now peaks at ~10 TFLOPS (N=2048), +32%
  at N=8192, **+29% at N=12288** (6.4 → 8.3), +12% at N=16384. Neutral on M4
  (±3%). Correctness unchanged.

- **Backward pass on `matmul2d` (~12× the simdgroup backward; training step ~18×)**
  — the varlen backward now runs on the Metal 4 accelerator, like the forward.
  Two flash-attn-2-style kernels: a per-Q-block dQ kernel (loops KV tiles,
  recomputing S=Q·Kᵀ, dP=dO·Vᵀ, dS=P∘(dP−delta), accumulating dQ+=scale·dS·K) and
  a per-KV-block dK/dV kernel (loops the GQA query heads and all Q tiles,
  accumulating dV+=Pᵀ·dO and dK+=scale·dSᵀ·Q). `delta=Σ_d dO·O` is a torch
  reduction (ordered on the MPS stream). Handles causal / GQA-MQA /
  sliding-window / varlen; validated vs fp32 autograd end-to-end. ~1.0 → ~11.9
  TFLOPS at head_dim 128; a full training step (fwd+bwd, N=2048) drops ~144 ms →
  ~8 ms. head_dim 64/96/128 half/bf16 where MPP is available (BQ=32/BK=16 tiles,
  measured optimum); fp32 / other head dims / no-MPP keep the simdgroup-per-row
  backward. Validated end-to-end including packed ragged (multi-sequence varlen)
  on M5 and M4 — the unique *varlen* backward (training on packed ragged
  sequences on Mac) is now fast, not just correct.
- **LSE on the MPP forward (training forward ~8× faster)** — the Metal 4 kernel
  now emits log-sum-exp on demand, so requesting it (the forward half of a
  training step) no longer forces the slow simdgroup path. It already tracks the
  running max/sum per row; `lse = m·ln(2) + log(l)` converts the base-2 max (from
  the exp2 softmax) back to natural units. Training forward 22 ms → 2.7 ms at
  N=2048 (8.3×); MPP LSE matches the reference exactly and backward gradients are
  unchanged (validated to fp16 tolerance at large N). Backward itself is still
  the simdgroup-per-row kernel — next.

- **MPP softmax micro-opts** — folded `log2(e)` into the scale and switched the
  online softmax to `exp2` (Apple GPUs compute `exp` as `exp2(x·log2e)`, so this
  drops a multiply per element), and moved the `*scale` out of the row-max scan
  (`max(S·scl) == scl·max(S)`, so scan raw and scale the reduced max once). ~1–4%
  at long sequences. Profiling (neutered-softmax floor) showed the softmax costs
  ~16% of large-N throughput and the matmul/bandwidth floor is ~89% of the AI=32
  memory ceiling; the bulk of the softmax cost is the per-element exp loop + the
  threadgroup round-trip between the two `matmul2d` calls, which would need a
  register-resident cooperative-tensor rewrite (`reduce_rows`) to remove —
  deferred (cooperative-tensor O previously collapsed occupancy ~10×).

- **head_dim 64 / 96 / 128 / 256 on the MPP fast path** — `matmul2d` is
  dimension-general, so the Metal 4 accelerator kernel now serves head_dim 64,
  96, 128 (both TM tiles) and 256 (TM=16; closes the head_dim>128 gap) in
  half/bf16 with all features (causal / GQA / sliding-window / varlen). head_dim
  64 jumps from the simdgroup fallback (~1.3 TFLOPS) to ~8.2 (~6.3×); 96 ~8.6;
  256 ~6.8 (M5). Covers essentially every transformer head dim. Other dims ≤128
  (80, 112, …) keep working on the general simdgroup kernel; >128 other than 256
  errors clearly. Validated on M5 and M4.
- **MPP path runs on M3/M4, not just M5 (confirmed on hardware)** — `matmul2d` is
  gated on macOS 26.2, not GPU family, so the fast path loads on any Metal 4 GPU.
  Verified on an **M4 (Mac16,10, macOS 26.5.1)**: `mpp_available()` true, full
  40-case suite passes, and it's **~3–4× the hand-written simdgroup kernel**
  (head_dim 128: ~1.9 vs 0.58 TFLOPS; head_dim 64: ~1.7 vs 0.45) and ~2–3× native
  MPS SDPA. (M4 has no Neural Accelerator, so ~1.9 TFLOPS vs M5's ~9, but still
  well ahead.) The simdgroup path is now only the fallback for M1/M2 and
  pre-26.2 systems.
- **Chip-aware TM tile** — the TM=16→32 size-adaptive switch is M5-specific: on M5
  the NA-fast matmul goes bandwidth-bound at long sequences (TM=32 wins ≥~10K),
  but on M3/M4 the regular matrix units stay compute-bound so TM=16 wins at every
  size (measured through 20K). TM=32 is now gated on GPU family Apple10+ (M5 and
  newer); M3/M4 always use TM=16. Detected once at init via `supportsFamily`.

- **Split-D simdgroup forward (v4, opt-in `MTLATTN_SPLITD`)** — head_dim 128,
  half/bf16. Splits the output D dimension across the 4 simdgroups (each owns a
  32-col slice) so the fp32 `Ofrag` accumulator shrinks from 16 to 12 fragments
  (RB=3/BQ=24), raising occupancy. ~+8% over v3 on M5 (1.18→1.27 TFLOPS). Left as
  opt-in rather than default: the gain is modest and hardware-dependent (the
  simdgroup path is `simdgroup_load`-throughput bound at ~1.2–1.3 TFLOPS on M5 —
  ~6% of the register-resident matmul peak — regardless of tiling; M1–M4 may
  differ). Correctness validated across the full suite.

- **MPP path (M5 Neural Accelerator) ~1.7–1.9× faster** — ~4.9 → ~9.2 TFLOPS at
  head_dim 128, mid-range sequences (~3.2× native MPS SDPA, up from ~1.7×). Three
  changes, all flowing from the kernel being occupancy-bound (not
  bandwidth-bound):
  - PV matmul now uses `matmul2d` `multiply_accumulate` straight into the
    threadgroup O accumulator, eliminating the separate `PVb` buffer (~8 KB of
    threadgroup memory freed → higher occupancy). +34% alone.
  - Smaller key tile (TN 64 → 32): less `Sb`/`Pb` footprint, more resident
    threadgroups. Measured optimum TM=16/TN=32/SG=4.
  - Parallel online softmax: the per-tile softmax was one-thread-per-row (16 of
    128 threads active between two fast matmuls); now 2 threads cooperate per row
    with `simd_shuffle_xor` reductions. +15–17% at N≤4096 (where the GPU is
    under-saturated); ~neutral at very long single sequences.
  - Size-adaptive query tile: TM=16 mid-range (occupancy), TM=32 at ≥~10K tokens
    (override `MTLATTN_TM32_MIN`) — there the GPU is already saturated, so halving
    the K/V re-reads wins. +5–11% at 12–18K tokens (the 3D-sparse-transformer
    regime), recovering the large-N taper. Net at 16K tokens: ~6.5 TFLOPS,
    ~1.3× the pre-change MPP path.
  Correctness unchanged (full suite passes).

- **Register-resident forward (v3, head_dim=128 half/bf16)** — the score/prob/
  output pipeline now lives entirely in simdgroup-matrix registers via
  `thread_elements()` and the (measured) Apple 8×8 fragment layout: softmax
  max/exp/sum and the online rescale run per-lane with `simd_shuffle_xor`
  reductions, P is written straight into a fragment, and O is written straight to
  device — so the threadgroup score buffers (`Ss`/`Ps`), the diagonal-matrix
  rescale, and ~3 of the 4 barriers per KV tile are gone (only the shared K/V
  load barrier remains). ~8% over v2; default for its shapes (opt out with
  `MTLATTN_NO_REG`), v2 stays for fp32 / other head dims. Measurement showed the
  simdgroup path is register-occupancy/load-throughput bound near ~1.2 TFLOPS
  (~5.5% of the 21.5-TFLOPS fp16 simdgroup_matrix peak), not barrier- or
  bandwidth-bound; closing the rest of the gap to MFA needs split-D or
  async-copy rewrites (tracked).

- **Mixed-precision simdgroup forward (M1–M4 path)** — half/bf16 inputs now run
  the simdgroup matmul with operands in the input dtype (the fast matrix path)
  and fp32 accumulators (scores, output, online rescale), instead of all-fp32
  fragments. The fp32 accumulators keep partial sums from overflowing (the
  outlier-activation NaN the all-fp32 kernel guarded against), while halving the
  operand footprint frees threadgroup memory for a 4× larger KV tile (BK 8→32).
  Net ~1.4× on the portable path (~0.78→~1.1 TFLOPS at head_dim 128). This is a
  fallback path: native MPS SDPA is still faster on dense equal-length shapes
  (~2.9 TFLOPS); mtlattn's edge remains ragged/windowed/varlen, the M5
  accelerator, and backward. Unchanged numerically for fp32 input and the M5 MPP
  path.

- **Backward pass** — `varlen_attention` is now differentiable (a
  `torch.autograd.Function` routes through new backward kernels when q/k/v
  require grad), so it trains. Composes with causal / GQA/MQA / sliding-window
  and varlen — variable-length training the MFA-based MPS kernels don't offer.
  The forward emits log-sum-exp on demand for the backward to recompute P from.
  The backward kernels are correct (validated vs autograd across the feature
  matrix) and use a simdgroup-per-row design — 32 lanes cooperate on the
  head_dim via `simd_sum`, ~4.6× faster than the initial thread-per-row version
  (dQ/dK/dV now ~3.5× the forward at N=2048, near the ~2.5× FLOPs ideal). A
  fully simdgroup-matrix-tiled backward is possible future work. Inference is
  unaffected (still the MPP path on M5).

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
