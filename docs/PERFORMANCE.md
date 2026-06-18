# Performance notes

Engineering reference for mtlattn's kernels: measured hardware ceilings, where the
kernels stand, what bounds them, the tuning constants and why, and — most
importantly — the **dead-ends** (so nobody re-tries a lever that's already known
to lose or to be API-blocked).

All numbers measured on an **M5 Pro** (Apple10 GPU family, has the per-core Neural
Accelerator) and an **M4 Mac mini** (Mac16,10, Apple9, no NA), fp16, H=12,
head_dim=128, macOS 26.x, torch 2.12, unless noted. **Read the measurement
caveats at the bottom before trusting any single number** — M-series GPU clocks
are load-state-dependent and burst benchmarks lie.

## Two kernel paths

| Path | When | head_dim | Where |
|---|---|---|---|
| **MPP** — Metal 4 `matmul2d` | macOS 26.2+, fp16/bf16 | 64/96/128 (256 fwd; 256 bwd) | `src/attn_mpp.metal` |
| **simdgroup** — `simdgroup_matrix` | M1/M2 or pre-26.2, or other head dims | any ≤128 | `src/attention.metal` |

The MPP path is **OS-gated (macOS 26.2), not GPU-family-gated** — confirmed
running on M4, not just M5. On M5 `matmul2d` targets the Neural Accelerator; on
M3/M4 it runs on the regular GPU matrix units (still Apple-tuned, ~3–4× the
hand-written simdgroup kernel).

## Measured hardware ceilings (M5 Pro)

| Ceiling | Value | How measured |
|---|---|---|
| `matmul2d` / NA fp16 GEMM peak | **~30 TFLOPS** | MPS large square GEMM |
| `simdgroup_matrix` fp16 peak | ~21.5 TFLOPS | register-resident MAC microbench |
| `simdgroup_matrix` fp32 peak | ~32 TFLOPS | (fp32 is *faster* than fp16 here!) |
| Memory bandwidth | ~269 GB/s | large copy |
| Practical flash-attention ceiling | ~13 TFLOPS | ~40–60% of GEMM peak (flash-attn literature) |

Roofline: attention at head_dim=128 has arithmetic intensity ≈ 2000 FLOP/byte
**globally**, so the operation is compute-bound — but a *kernel's* effective AI is
≈ its tile reuse (≈ TM/BQ), which can drop below the ridge (~80 FLOP/byte) at
small tiles, making it bandwidth- or occupancy-bound in practice.

## Where the kernels stand (sustained, warmed, median)

| | M5 fwd | M5 bwd | M4 fwd | M4 bwd |
|---|---|---|---|---|
| mid-range (N≤8K) | ~9.5 TF | ~5.8 TF | ~1.9 TF | ~2.8 TF |
| large-N (12–18K) | ~7.5–9.5* | ~5.6 TF | ~1.9 TF | — |

\* large-N forward is clock-ramp-sensitive; see caveats. Reference: native MPS
SDPA ≈ 2.9 TF (M5). So MPP forward ≈ **3× SDPA**, backward ≈ **11× the
simdgroup-per-row backward** (apples-to-apples, same session).

- **Forward** ≈ 9.5 TF = ~73% of the practical ceiling, ~32% of the NA fp16 peak.
  Near its structural limit; the residual ~16% is softmax (see below).
- **Backward** ≈ 5.8 TF — the **slower half**, occupancy-bound. It recomputes
  S=Q·Kᵀ and re-reads Q/dO per KV-block, but bigger tiles (to cut re-reads)
  *lose* to occupancy, so it's tuned to BK=16/BQ=32.

## Tuning constants (and why)

Forward (`attn_vl`, MPP):
- **TM=16, TN=48, SG=4.** TN was 32 until the softmax got cheaper (exp2 / raw-max
  scan + in-place PV accumulate); the optimum then shifted to **TN=48** (+13–33%).
- **Size-adaptive TM**: TM=16 below ~14K tokens, TM=32 at/above — **only on
  Apple10+ (M5+)**, gated via `supportsFamily`. M3/M4 always TM=16 (TM=32 never
  wins there). Override: `MTLATTN_TM32_MIN`.
- **LPR=2** (softmax threads/row) — LPR≥4 helps N=2048 slightly but regresses
  ~20% at N=8192 (threadgroup contention).
- **exp2** with `log2(e)` folded into the scale; **raw max-scan** (`max(S·scl) =
  scl·max(S)`, scale once not per key).
- PV uses `matmul2d` `multiply_accumulate` straight into the threadgroup O
  accumulator — no separate PV buffer (that 8 KB freed is what enables the tile).
- **Additive bias (`attn_bias`) is a `HAS_BIAS` function constant**, not a runtime
  flag — the no-bias pipeline is dead-branch-eliminated, so it keeps its exact
  register footprint (measured: **no** throughput change, 9.66 vs ~9.5 TF). A
  runtime branch was rejected: this kernel is occupancy-bound, so the bias
  branch's extra registers would pressure occupancy even when bias is off. The
  bias path can't use the raw-max shortcut (per-element bias breaks `max(S·scl) =
  scl·max(S)`), so it scans the full `S·scl + bias·log2(e)` logit and reads the
  bias twice (max scan + exp) — modestly slower, only when a bias is supplied.

Backward (`bwd_dq_mpp` / `bwd_dkv_mpp`, MPP):
- **dQ: BQ=32 grid / BK=16 inner. dK/dV: BK=16 grid / BQ=32 inner.** head_dim 256
  drops to BK=8 / BQ=16 (the `[·,256]` accumulators are threadgroup-tight).
- `delta = Σ_d dO·O` is a torch reduction (ordered on the MPS stream).

Decode (splitKV / FlashDecoding, auto-dispatched when `max_seqlen_q ≤ 16` and
`avg_kv ≥ 2048`):
- **`num_splits` is core-count-aware** (`Context::gpu_cores`, read once from the
  IORegistry `AGXAccelerator` `gpu-core-count`). It's the max of a *chunk-driven*
  term (`avg_kv/512`, the serial-depth driver) and a *fill-driven* term
  (`cores·6 / (num_seqs·H)`, enough KV-groups to cover this GPU), clamped to a min
  chunk (~256 keys) and a total-groups budget (`cores·50`). Falls back to 20 cores
  if the count is unavailable.
- **Why core-aware:** the old fixed `avg_kv/512` (capped `1024/num_seqs`) under-split
  short-KV decode on every chip. Measured `B1/Lkv=2048`: M5 Pro (20 cores)
  **0.121 → 0.051 ms** picking 8 splits not 4 (2.4×); M4 (10 cores)
  **0.206 → 0.157 ms** picking 5 (1.3×). The per-chip optimum *differs* (Pro wants
  8, M4 wants 5) — a fixed constant can't be right for both, and a bigger chip
  (M5 Max) wants still more. Long-KV points are chunk-limited and unchanged.
- The split count is verified per-chip: read the actual `gpu_core_count()` (Pro=20,
  M4=10) rather than inferring from the family.

Simdgroup fallback (`attn_mpp.metal` → no; `attention.metal`):
- v3 register-resident kernel for head_dim 128 (`thread_elements()` + the measured
  8×8 fragment layout: lane holds 2 elements of one row;
  `row=((l&16)>>2)|((l>>1)&3)`, `col0=((l&8)>>1)|((l&1)<<1)`; the 4 lanes of a row
  reduce with `simd_shuffle_xor(·,1)` and `(·,8)`).

## Dead-ends — measured, do NOT re-try

- **`matmul2d` `reduce_rows` for a register-resident softmax**: **API-blocked** —
  it `static_assert`s `execution_simdgroups<1>`. Our matmuls use SG=4 for
  throughput; SG=1 would cripple them. The clean register-resident softmax is
  therefore unavailable.
- **Cooperative-tensor O accumulator** (`attn_vl_coop`, kept as a reference):
  numerically correct but **~10× slower** — two register-resident `[64,128]` fp32
  tensors cost ~128 regs/thread and collapse occupancy. (Scores `[TM,TN]` are
  small enough to be register-resident; the *output* accumulator is not.)
- **TM=64 register-resident output**: same occupancy collapse.
- **Bigger backward grid tiles** (BK=24, BQ=48) to cut Q/dO re-reads: *slower*
  (4.0 vs 5.8 TF) — occupancy loss beats bandwidth saving. The backward is
  occupancy-bound.
- **Q hoisted into registers** in the forward (v3): slightly *slower* —
  `Ofrag[16]` already pressures occupancy; adding registers hurts.
- **More simdgroups (SG>4)** on either path: neutral-to-worse.
- **Mixed-precision fp16 *operands* are NOT faster than fp32** on the M5 simdgroup
  units (microbench: fp16 21.5 vs fp32 32 TFLOPS). The mixed-precision win in the
  fallback came purely from **halving operand *memory*** (enabling a 4× larger KV
  tile), not faster MACs.
- **Simdgroup-local barriers** (downgrading `threadgroup_barrier` to
  `simdgroup_barrier` in the v3 softmax): broke causal correctness — keep full
  barriers around the cross-lane `c_run` exchange.
- **head_dim>128 on the simdgroup path**: kernels are sized for ≤128; 256 is
  MPP-only.

## Measurement caveats (learned the hard way)

- **GPU clock is load-state-dependent.** Idle → low clock; sustained load →
  boosted. A *cold* call (after idle) can read ~¼ of the warm number. **Always
  warm up (sustained load) + take a median.**
- **Burst benchmarks understate sustained large-N.** Short bursts at N≥12K don't
  fully boost the clock; a 40s sustained run reached ~9.5 TF where median-of-6
  bursts showed ~7.5. pixal3d-style minute-long generation runs see the higher,
  sustained number.
- **No thermal throttling observed** — the GPU *boosts* under load, it doesn't
  throttle down over tens of seconds. Mid-range forward is rock-steady (±1% over
  60s).
- **NEVER issue an unsynced GPU loop** (`while ...: kernel()` with no periodic
  `torch.mps.synchronize()`). It floods the Metal command queue; killing it
  leaves orphaned command buffers that **wedge the device to ~¼ throughput** and
  require a **reboot/logout** to clear. Sync every ~10–20 iterations.

## Low-precision matmul2d — fp8 vs int8 (measured)

We're at ~33% of the NA fp16 peak, so a lower-precision operand path is the one
*large* lever left. Investigated which precision actually buys throughput on the
M5 NA. **Microbench: M=N=K=64 `matmul2d multiply_accumulate`, 40000 iters, 2048
threadgroups, SG=4, M5 Pro:**

| operand | accumulator | throughput | vs fp16 |
|---|---|---|---|
| `half` | float | **32.3 TFLOPS** | 1.0× |
| `bfloat` | float | 32.2 TFLOPS | 1.0× |
| `int8_t` | int | **61.3 TOPS** | **~2.0×** |

So **int8 is a real, native 2× on the M5 NA** — and `int4b_format`/`uint4b_format`
are also `matmul2d` operand types (potential ~4×, not yet benched; packed-format
handling is fiddly).

**fp8 is NOT a lever right now:**
- **Not in the `matmul2d` operand set** on this toolchain (`-std=metal4.0`). The
  accepted element types are `uint8_t / int8_t / uint4b_format / int4b_format /
  float / half / bfloat` — a `static_assert` rejects anything else. No fp8.
- **Native fp8/fp4 arrive in Metal 4.1 / macOS 27.0**, not 26.x: headers expose
  `Float8E5M2`, `Float8E4M3`, `Float4E2M1`, `UInt2`, `Int2` (OCP mxfp8/mxfp4 with
  an E8M0 per-32-block scale), all `API_AVAILABLE(macos(27.0))`. **Whether the M5
  NA hardware-accelerates them in `matmul2d` is undocumented and untested** — could
  be HW or could be emulation. Revisit when on macOS 27.
- **Software-emulated fp8 is *slower* than fp16.** `tashiscool/fp8-mps-metal`
  stores e4m3 in `uint8_t` and decodes in-register (needs PyTorch 2.10+
  `compile_shader`, monkey-patches `torch._scaled_mm`); its fused kernel runs
  **~4–26× slower than fp16**. Useful for memory/compat, never for speed.

**Takeaway:** int8 is a real 2× *at the matmul level*, but on *attention* it caps
at a **~1.3× ceiling** and isn't worth the machinery right now — built, measured,
and **parked** (see below).

## Parked: int8-quantized attention (built, benchmarked, ~1.3× ceiling)

Implemented and benchmarked an int8 forward on `matmul2d` (code on branch
`experiments/int8-attention`). It is **correct** (0.7% error, group=32 even
handles ×300 outlier channels) and — *after optimization* — **~1.2–1.3× slower**
than fp16, with a best-possible ceiling of only ~1.3× *faster*. The journey is the
lesson:

| stage | g32 (robust acc) | g128 (per-row, 1 matmul) |
|---|---|---|
| naive prototype | 0.15× (6× slower) | 0.17× |
| − per-q-tile K re-staging (device-direct read) | 0.43× | 0.57× |
| + parallel (LPR) softmax | 0.55× | **0.81× (1.23× slower)** |

What this established (and corrected):
- **The first "6× slower → it's the 4 matmuls" reading was WRONG.** A diagnostic
  (group=128 = a *single* K=128 matmul) was still 5× slower, so the matmul *count*
  was not the wall. The real cost was **occupancy collapse** (the int8 kernel's
  threadgroup footprint — staged K + int32 scratch + fp32 O — is ~2× the fp16
  kernel's, and this family is occupancy-bound) plus **serial overhead** (one
  thread/row softmax, a per-tile K-staging copy). Removing the staging (read the
  pre-quantized K device-direct) and parallelizing the softmax closed most of the
  gap. *Lesson: never conclude perf from an unoptimized kernel.*
- The **ceiling is ~1.3× faster** regardless: int8 only accelerates the QK matmul
  (≈ half the FLOPs); PV stays fp16 (SageAttention-v1 — quantizing P/V was a net
  loss). Folding the int32→fp32 dequant into the softmax *regressed* (it moves the
  device K-scale read into the hot loop) — the separate pass to threadgroup is
  better.
- Accuracy needs **group=32** for robustness — a single per-row scale is 13% off on
  ×300 outlier channels (group=32: 0.0001). The faster **per-row (g128)** path
  needs **Hadamard rotation** (QuaRot/FA3) for accuracy on real multiplicative
  outliers (smooth-K handles only mean-offset ones). So the fast path has *two*
  unfinished problems (close the last ~1.3× of speed AND add Hadamard).

Net: a real but modest ~1.3× best case, needing more occupancy work + Hadamard, on
an fp16 path already at ~73% of its ceiling. Parked on the branch; revisit if fp16
is exhausted or hardware changes. fp8/int4 don't rescue it: fp8 isn't a native
operand until macOS 27, int4 has no native `matmul2d` operand (unpack-to-int8 costs
ALU). Prior art agrees on the magnitude — SageAttention/Draw Things realize only
~1.2–1.4×, *with* smoothing/rotation machinery.

## Remaining headroom (all large/uncertain)

1. **One-pass fused backward** — major rewrite, uncertain payoff.
2. **int8 (Hadamard-rotated per-row QK + more occupancy work)** — ~1.3× ceiling,
   parked on `experiments/int8-attention`; revisit only if fp16 is exhausted.

**Register-resident softmax: tried and SHELVED (measured 3× slower).** Building the
MLX NAX-style version (scores + O in register cooperative tensors, manual
`simd_shuffle_xor` butterfly instead of `reduce_rows`) was correct but **3× slower**
— the register O accumulator collapses occupancy. This proved the forward's
threadgroup "round-trip" is *not* the bottleneck; **occupancy is**, which is exactly
why the threadgroup design wins. Parked on `experiments/register-resident`. See
`EXPERIMENTS.md` for the full catalog of shelved explorations.

Otherwise: the kernels are near their practical limits for the
`matmul2d`-with-threadgroup structure on current hardware. The easy wins are gone.
