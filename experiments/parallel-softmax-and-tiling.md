# Shelved kernel experiments

Correct, measured, but **not** in the production kernels — kept here to re-apply
when the kernel is tuned for compute-bound configs (older Apple GPUs, fewer
heads). On M5 Pro they are null because the kernels are **memory-bandwidth
bound** on K/V re-reads (~310 GB/s, the DRAM ceiling), so compute-side wins are
invisible. All variants below pass `tests/test_correctness.py` (14 cases).

Re-validate with the correctness suite and a per-config benchmark before
shipping any of these — the right tile/parallelism depends on the target GPU.

---

## 1. Parallel online softmax — MPP kernel (`attn_mpp.metal`, `attn_vl`)

Replaces the one-row-per-thread scan (only `TM` of `NT` threads busy) with
`LPR` lanes cooperating per row, reduced via `simd_shuffle_xor`. Generic over
`TM`/`SG` as long as `TM % SG == 0` and `(TM/SG)` divides 32.

Measured M5 Pro, bf16, 12 heads: 5.05 → 5.06 TFLOPS (null — bandwidth bound).

Replace the serial block:

```cpp
        for (uint r = tid; r < TM; r += NT) {
            float m_old = mb[r], tmax = m_old;
            for (int c = 0; c < tk; ++c) tmax = max(tmax, Sb[r*TN+c]*scale);
            float corr = exp(m_old - tmax), tsum = 0.0f;
            for (uint c = 0; c < TN; ++c) { float w=(int(c)<tk)?exp(Sb[r*TN+c]*scale-tmax):0.0f; tsum+=w; Pb[r*TN+c]=T(w); }
            mb[r] = tmax; lb[r] = lb[r]*corr + tsum; cb[r] = corr;
        }
```

with:

```cpp
        // All lanes active: LPR lanes cooperate per row, reduce via shuffles.
        {
            constexpr uint RPS = TM / SG;        // rows per simdgroup
            constexpr uint LPR = 32u / RPS;      // lanes cooperating per row
            const uint sgid = tid >> 5, lane = tid & 31u;
            const uint r = sgid * RPS + (lane / LPR);
            const uint sub = lane % LPR;
            const float m_old = mb[r];
            float tmax = m_old;
            for (uint c = sub; c < uint(tk); c += LPR) tmax = max(tmax, Sb[r*TN+c]*scale);
            for (uint s = 1u; s < LPR; s <<= 1) tmax = max(tmax, simd_shuffle_xor(tmax, s));
            const float corr = exp(m_old - tmax);
            float tsum = 0.0f;
            for (uint c = sub; c < TN; c += LPR) { float w=(c<uint(tk))?exp(Sb[r*TN+c]*scale-tmax):0.0f; tsum+=w; Pb[r*TN+c]=T(w); }
            for (uint s = 1u; s < LPR; s <<= 1) tsum += simd_shuffle_xor(tsum, s);
            if (sub == 0u) { mb[r] = tmax; lb[r] = lb[r]*corr + tsum; cb[r] = corr; }
        }
```

---

## 2. Parallel online softmax — simdgroup kernel (`attention.metal`, `varlen_attn_impl`)

Same idea for the portable path. With `BQ = 8*SGS` (8 rows/simdgroup), 32 lanes
give 4 lanes/row, so the reduction is a fixed `xor 1, xor 2`.

Measured M5 Pro, bf16, 12 heads (MPP forced off): 0.51 → 0.53 TFLOPS (~4%).
Expected to matter more on GPUs where the matmul is relatively faster.

Replace the serial block:

```cpp
        // Online softmax, one thread per row.
        if (tid < BQ) {
            const uint r = tid;
            const float m_old = m_run[r];
            float row_max = m_old;
            for (int kk = 0; kk < tk; ++kk) {
                row_max = max(row_max, Ss[r * BK + uint(kk)]);  // scale folded into Q
            }
            const float c = exp(m_old - row_max);
            float row_sum = 0.0f;
            for (uint kk = 0; kk < BK; ++kk) {
                float w = 0.0f;
                if (int(kk) < tk) {
                    w = exp(Ss[r * BK + kk] - row_max);
                    row_sum += w;
                }
                Ps[r * BK + kk] = T_f(w);
            }
            m_run[r] = row_max;
            l_run[r] = l_run[r] * c + row_sum;
            c_run[r] = c;
        }
```

with:

```cpp
        // All lanes active: 4 lanes cooperate per row (8 rows/simdgroup),
        // each scans a strided quarter of the keys, reduced via simd shuffles.
        {
            const uint row_in_sg = lane >> 2;        // 0..7 within this simdgroup
            const uint sub = lane & 3u;              // 0..3: which key-quarter
            const uint r = sg_row0 + row_in_sg;      // sgid*8 + row
            const float m_old = m_run[r];
            float lmax = m_old;                       // scale already folded into Q
            for (uint kk = sub; kk < uint(tk); kk += 4) lmax = max(lmax, Ss[r * BK + kk]);
            lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
            lmax = max(lmax, simd_shuffle_xor(lmax, 2u));
            const float row_max = lmax;
            const float c = exp(m_old - row_max);
            float lsum = 0.0f;
            for (uint kk = sub; kk < BK; kk += 4) {
                float w = (kk < uint(tk)) ? exp(Ss[r * BK + kk] - row_max) : 0.0f;
                lsum += w;
                Ps[r * BK + kk] = T_f(w);
            }
            lsum += simd_shuffle_xor(lsum, 1u);
            lsum += simd_shuffle_xor(lsum, 2u);
            if (sub == 0u) {
                m_run[r] = row_max;
                l_run[r] = l_run[r] * c + lsum;
                c_run[r] = c;
            }
        }
```

---

## 3. Larger KV tile — MPP kernel (`TN = 64 -> 128`)

In `attn_mpp_varlen_half` and `attn_mpp_varlen_bfloat`, set
`constexpr int TM=16,TN=128,D=128,SG=4;`. Halves the KV-tile count (fewer
barriers, wider matmul N); threadgroup use rises to ~28 KB (fits 32 KB).

Measured M5 Pro, bf16, 12 heads: 5.06 → 5.06 TFLOPS (null — bandwidth bound).
May help where per-tile/barrier overhead is the bottleneck rather than DRAM.

---

## Why these are null on M5 Pro

At `TM=16` each query block re-reads all of K/V from device, so K/V traffic is
`(M/TM)·L·D·2·2bytes·H`. Divided by measured time that is ~290-316 GB/s at every
size from M=1024 to 16384 — the M5 Pro DRAM bandwidth ceiling. The accelerator
waits on memory, so softmax/tiling/barrier wins don't show. The only lever is
**fewer K/V re-reads = bigger `TM`**, which needs register-resident O without the
occupancy hit (the hard flash-attention-style rewrite; `cooperative_tensor` was
tried and is 10× slower — see `src/attn_mpp.metal`).
