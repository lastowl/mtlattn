# Experiments (built, measured, shelved)

Substantial explorations that were fully implemented and benchmarked, found **not**
to beat the tuned production kernels on current hardware, and parked on
`experiments/*` branches. They're cataloged here so the *conclusions* (and the
hard-won low-level findings) aren't lost — each is recoverable with
`git checkout experiments/<name>`.

**The recurring lesson:** on the M5, the real bottleneck for these kernels is
**occupancy** (register/threadgroup pressure) and **matmul tile size**, *not* the
things the theory or other libraries optimize for. The well-tuned threadgroup
`matmul2d` kernels (forward ~9.5 TF, backward ~5.8 TF) are genuinely hard to beat.
See `PERFORMANCE.md` for the production kernels' ceilings and the inline dead-ends
(fp8, backward tile sweeps).

---

## int8-quantized attention — `experiments/int8-attention`

**Goal:** exploit the M5 Neural Accelerator's ~2× int8 `matmul2d` throughput
(measured 61 TOPS vs 32 TFLOPS fp16) for attention.

**Result: shelved — ~1.3× ceiling at best, naive build 6–12× *slower*.**
- Accuracy is fine (~0.6% on outlier-channel inputs) **but requires group=32
  quantization** for Q/K — a single per-row scale gives 13% error because
  attention activations have outlier channels. group=32 fragments the QK matmul
  into 4 tiny K=32 calls.
- After optimization (pre-quantize K, fp16 PV, parallel softmax, device-direct
  reads) the best variant reached **~1.23× slower** than fp16 — int8 only helps
  the QK matmul (≈ half the FLOPs); PV stays fp16; the fast per-row path also
  needs Hadamard rotation for accuracy. Best *possible* is ~1.3× faster, not worth
  the machinery vs a 9.5 TF baseline.
- fp8 isn't a native `matmul2d` operand until macOS 27; int4 has no native operand.

**Keeper findings:** int8 = 2× fp16 at the matmul level on the M5 NA; group=32 is
mandatory for attention; prior art (SageAttention, Draw Things) realizes only
~1.2–1.4× *with* smoothing/rotation machinery. Full notes in the `int8-attention`
memory.

---

## Register-resident forward (MLX NAX-style) — `experiments/register-resident`

**Goal:** close the forward's ~24% gap to the practical ceiling by removing the
threadgroup-softmax round-trip — keep the QK scores and the O accumulator in
**register cooperative tensors**, with the softmax row reduction done as a manual
`simd_shuffle_xor` butterfly (sidestepping `matmul2d`'s single-simdgroup
`reduce_rows`), following MLX's `nax.h`.

**Result: shelved — correct but 3× *slower* (2.9–3.1 TF vs 9.2).** The "round-trip
gap" was a mirage:
- **Occupancy collapse.** The register cooperative-tensor O accumulator
  (`[16,128]` fp32 ≈ 64 elems/thread) tanks occupancy, so memory latency can't be
  hidden. The production forward keeps O in *threadgroup* memory **precisely
  because** low registers → high occupancy → faster. The round-trip we removed is
  what *enables* the speed; the bottleneck is occupancy, not the round-trip.
- **Small matmuls** — 16×16 QK (vs the production 16×48) is less efficient on the
  NA and means ~3× more calls.
- MLX publishes **no** perf numbers for its NAX path, so there was never evidence
  it beats threadgroup either.

**Keeper findings (non-obvious, verified by a lane-layout value-probe):** for a
single-simdgroup `matmul2d` cooperative tensor, the **query lives on `idx1`, not
`idx0`** — for *both* the `transpose_b` QK output (idx0=key) and the PV output
(idx0=d). The softmax butterfly masks are **1 and 8** (a query is spread across 4
lanes). Full notes in the `register-resident` memory.
