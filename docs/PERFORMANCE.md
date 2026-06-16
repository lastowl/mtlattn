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

Backward (`bwd_dq_mpp` / `bwd_dkv_mpp`, MPP):
- **dQ: BQ=32 grid / BK=16 inner. dK/dV: BK=16 grid / BQ=32 inner.** head_dim 256
  drops to BK=8 / BQ=16 (the `[·,256]` accumulators are threadgroup-tight).
- `delta = Σ_d dO·O` is a torch reduction (ordered on the MPS stream).

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

## Remaining headroom (all large/uncertain)

1. **fp8** — if the M5 NA does fp8 `matmul2d`, fp8 operands could ~2× throughput
   (we're at ~33% of the NA fp16 peak). Precision risk; unproven here. The one
   potentially large lever, and it would help both forward and backward.
2. **Manual register-resident softmax** (cooperative-tensor scores + hand-rolled
   cross-simdgroup reduction, since `reduce_rows` is out) — could shave the
   forward's threadgroup round-trip, but high occupancy/correctness risk.
3. **One-pass fused backward** — major rewrite, uncertain payoff.

Otherwise: the kernels are near their practical limits for the
`matmul2d`-with-threadgroup structure on current hardware. The easy wins are gone.
