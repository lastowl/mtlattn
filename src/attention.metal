// Fused variable-length attention forward (flash-attention style), v2:
// simdgroup_matrix tiling.
//
// Geometry: one threadgroup per (q-block, sequence, head); SGS simdgroups of
// 32 lanes, each owning 8 query rows of the BQ-row block (BQ = 8 * SGS).
// K/V tiles of BK keys are staged in threadgroup memory (K transposed for
// direct fragment loads). Scores are computed as 8x8 simdgroup matmul
// fragments, softmax runs in fp32 scalar code with running (m, l) per row,
// and the online rescale of the output accumulator fragments uses
// multiplication by a diagonal fragment. Output = diag(1/l) * O, stored
// directly via simdgroup_store when the whole block is in-bounds, else
// through a threadgroup bounce buffer with scalar masking.
//
// Precision: half/bf16 inputs use mixed-precision simdgroup matmul — operands
// in the input dtype (the fast matrix path), accumulators (scores, output,
// rescale) in fp32 so partial sums never overflow. fp32 input stays all-fp32.
// Halving the operand footprint vs all-fp32 fragments frees threadgroup memory
// for a 4x larger KV tile (BK=8 -> 32): ~1.4x the all-fp32 path on M1-M4.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint num_heads;
    uint head_dim;
    float scale;
    uint q_row_stride;  // elements between rows (head stride == head_dim)
    uint k_row_stride;
    uint v_row_stride;
    uint o_row_stride;
    uint causal;        // 0 = full attention; 1 = causal mask
    uint gqa_group;     // query heads per kv head (1 = standard MHA)
    uint window;        // 0 = unlimited; >0 = attend last `window` keys (SWA)
    uint return_lse;    // 1 = also write per-query log-sum-exp (for backward)
    uint num_seqs;      // batch size (for the backward kernels' sequence lookup)
};

constant constexpr uint DMAX = 128;
constant constexpr uint DBLK_MAX = DMAX / 8;

// T_in: device storage type; T_f: fragment type; SGS: simdgroups per
// threadgroup (threadgroup size = 32*SGS, BQ = 8*SGS); BK: staged tile keys;
// DIRECT: fragment stores may write device memory directly (T_in == T_f).
template <typename T_in, typename T_f, uint SGS, uint BK, bool DIRECT>
void varlen_attn_impl(
    device const T_in* Q,
    device const T_in* K,
    device const T_in* V,
    device T_in* O,
    device const int* cu_q,
    device const int* cu_kv,
    constant Params& p,
    device float* lse,          // [total_q, H] log-sum-exp, written if return_lse
    threadgroup T_f* Qs,        // [BQ * DMAX]; reused as O bounce
    threadgroup T_f* KsT,       // [DMAX * BK]  K transposed [D][BK]
    threadgroup T_f* Vs,        // [BK * DMAX]  V [BK][D]
    threadgroup float* Ss,      // [BQ * BK]    scores fp32
    threadgroup T_f* Ps,        // [BQ * BK]    probabilities / score staging
    threadgroup float* Diag,    // [SGS * 64]   per-sg 8x8 diagonal (fp32 accum)
    threadgroup float* m_run,   // [BQ]
    threadgroup float* l_run,   // [BQ]
    threadgroup float* c_run,   // [BQ]
    uint3 tgid,
    uint tid,
    uint sgid,
    uint lane
) {
    constexpr uint BQ = 8 * SGS;
    constexpr uint TGS = 32 * SGS;

    const uint seq = tgid.y;
    const uint head = tgid.z;                       // query head
    const uint kv_head = head / p.gqa_group;        // GQA: query head -> kv head
    const uint D = p.head_dim;
    const uint dblks = (D + 7) / 8;

    const int q_start = cu_q[seq];
    const int q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq];
    const int kv_len = cu_kv[seq + 1] - kv_start;

    const int q0 = q_start + int(tgid.x * BQ);
    if (q0 >= q_end) return;
    const int q_rows = min(int(BQ), q_end - q0);
    // Causal (flash_attn convention): query i attends key j iff j <= i + coff.
    const int coff = kv_len - (q_end - q_start);
    const int q_hi = (q0 - q_start) + (q_rows - 1) + coff;  // furthest key this block sees

    // Fold the softmax scale into Q at staging time: scores then live near
    // +-1, where half fragments have far better absolute precision than at
    // the raw logit magnitude (~sqrt(D)).
    for (uint idx = tid; idx < BQ * D; idx += TGS) {
        const uint r = idx / D;
        const uint d = idx % D;
        Qs[r * D + d] = (int(r) < q_rows)
            ? T_f(float(Q[ulong(q0 + int(r)) * p.q_row_stride + head * D + d]) * p.scale)
            : T_f(0.0f);
    }
    if (tid < BQ) {
        m_run[tid] = -INFINITY;
        l_run[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulators are fp32 regardless of operand precision: half/bf16 operands
    // ride the fast simdgroup-matrix path, but partial sums (QK dot products,
    // and the unnormalised P.V running output, which can reach ~1e5 on outlier
    // activations) stay in fp32 — so no overflow/NaN, unlike a half accumulator.
    simdgroup_matrix<float, 8, 8> Ofrag[DBLK_MAX];
    for (uint i = 0; i < dblks; ++i) {
        Ofrag[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const uint sg_row0 = sgid * 8;          // block-local first row of this sg

    // Sliding window: jump the loop to row 0's window bottom so cost is
    // O(window), not O(seqlen).
    int t0_start = 0;
    if (p.window > 0) {
        int s = (q0 - q_start) + coff - int(p.window) + 1;   // lowest key any row needs
        if (s > 0) t0_start = (s / int(BK)) * int(BK);
    }
    for (int t0 = t0_start; t0 < kv_len; t0 += int(BK)) {
        if (p.causal && t0 > q_hi) break;   // tile fully beyond causal horizon
        const int tk = min(int(BK), kv_len - t0);

        for (uint idx = tid; idx < BK * D; idx += TGS) {
            const uint kk = idx / D;
            const uint d = idx % D;
            if (int(kk) < tk) {
                const ulong krow = ulong(kv_start + t0 + int(kk));
                KsT[d * BK + kk] = T_f(K[krow * p.k_row_stride + kv_head * D + d]);
                Vs[kk * D + d] = T_f(V[krow * p.v_row_stride + kv_head * D + d]);
            } else {
                KsT[d * BK + kk] = T_f(0.0f);
                Vs[kk * D + d] = T_f(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q . K^T for this simdgroup's 8 rows, staged through Ps.
        {
            simdgroup_matrix<float, 8, 8> Sfrag[BK / 8];
            for (uint kb = 0; kb < BK / 8; ++kb) {
                Sfrag[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            for (uint db = 0; db < dblks; ++db) {
                simdgroup_matrix<T_f, 8, 8> A;
                simdgroup_load(A, &Qs[sg_row0 * D + db * 8], D);
                for (uint kb = 0; kb < BK / 8; ++kb) {
                    simdgroup_matrix<T_f, 8, 8> B;
                    simdgroup_load(B, &KsT[db * 8 * BK + kb * 8], BK);
                    // mixed-precision MAC: T_f operands, fp32 accumulator
                    simdgroup_multiply_accumulate(Sfrag[kb], A, B, Sfrag[kb]);
                }
            }
            for (uint kb = 0; kb < BK / 8; ++kb) {
                // fp32 accumulator -> fp32 Ss buffer (no Ps->Ss copy / barrier).
                simdgroup_store(Sfrag[kb], &Ss[sg_row0 * BK + kb * 8], BK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax — all lanes active: 4 lanes cooperate per row (8 rows
        // per simdgroup), each scanning a strided quarter of the keys, reduced
        // with simd shuffles. (scale was folded into Q at staging.)
        {
            const uint row_in_sg = lane >> 2;        // 0..7 within this simdgroup
            const uint sub = lane & 3u;              // 0..3: which key-quarter
            const uint r = sg_row0 + row_in_sg;
            int lim = tk;
            if (p.causal) { int h = (q0 - q_start) + int(r) + coff - t0 + 1; lim = clamp(h, 0, tk); }
            int lo = 0;
            if (p.window > 0) { int dl = (q0 - q_start) + int(r) + coff - int(p.window) + 1 - t0; lo = clamp(dl, 0, lim); }
            const float m_old = m_run[r];
            float lmax = m_old;
            for (int kk = int(sub); kk < lim; kk += 4) if (kk >= lo) lmax = max(lmax, Ss[r * BK + uint(kk)]);
            lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
            lmax = max(lmax, simd_shuffle_xor(lmax, 2u));
            const float row_max = lmax;
            // -inf when the windowed band skips this tile for the row: no-op (1).
            const float c = (row_max == -INFINITY) ? 1.0f : exp(m_old - row_max);
            float lsum = 0.0f;
            for (uint kk = sub; kk < BK; kk += 4) {
                float w = (int(kk) >= lo && int(kk) < lim) ? exp(Ss[r * BK + kk] - row_max) : 0.0f;
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
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(c) * O
        for (uint i = lane; i < 64; i += 32) {
            const uint rr = i / 8;
            const uint cc = i % 8;
            Diag[sgid * 64 + i] = (rr == cc) ? c_run[sg_row0 + rr] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_matrix<float, 8, 8> Dg;
            simdgroup_load(Dg, &Diag[sgid * 64], 8);
            for (uint db = 0; db < dblks; ++db) {
                simdgroup_multiply(Ofrag[db], Dg, Ofrag[db]);
            }
        }

        // O += P . V
        for (uint kb = 0; kb < BK / 8; ++kb) {
            simdgroup_matrix<T_f, 8, 8> Pf;
            simdgroup_load(Pf, &Ps[sg_row0 * BK + kb * 8], BK);
            for (uint db = 0; db < dblks; ++db) {
                simdgroup_matrix<T_f, 8, 8> B;
                simdgroup_load(B, &Vs[kb * 8 * D + db * 8], D);
                simdgroup_multiply_accumulate(Ofrag[db], Pf, B, Ofrag[db]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Finalize: O = diag(1/l) * O
    for (uint i = lane; i < 64; i += 32) {
        const uint rr = i / 8;
        const uint cc = i % 8;
        const float l = l_run[sg_row0 + rr];
        Diag[sgid * 64 + i] = (rr == cc) ? (l > 0.0f ? 1.0f / l : 0.0f) : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    {
        simdgroup_matrix<float, 8, 8> Dg;
        simdgroup_load(Dg, &Diag[sgid * 64], 8);
        for (uint db = 0; db < dblks; ++db) {
            simdgroup_multiply(Ofrag[db], Dg, Ofrag[db]);
        }
    }

    // Log-sum-exp per query (over scaled scores; the softmax scale is folded
    // into Q) — the backward pass recomputes P = exp(S - lse) from this.
    if (p.return_lse) {
        for (uint r = tid; r < BQ; r += TGS) {
            if (int(r) < q_rows) {
                const float l = l_run[r];
                lse[ulong(q0 + int(r)) * p.num_heads + head] =
                    (l > 0.0f) ? (m_run[r] + log(l)) : -INFINITY;
            }
        }
    }

    // Store. The path choice must be threadgroup-uniform (barriers inside),
    // so it depends only on block-level facts.
    const bool direct_ok = DIRECT && (q_rows == int(BQ)) && (D % 8 == 0);
    if (direct_ok) {
        // T_in == fp32 here (DIRECT), so the fp32 Ofrag stores straight to device.
        device float* o_base = reinterpret_cast<device float*>(O)
            + ulong(q0 + int(sg_row0)) * p.o_row_stride + head * D;
        for (uint db = 0; db < dblks; ++db) {
            simdgroup_store(Ofrag[db], o_base + db * 8, p.o_row_stride);
        }
    } else {
        // Half/bf16 output: the fp32 Ofrag can't store into the T_f buffers, so
        // bounce through the (now-free) fp32 Ss scratch one column-band at a
        // time (BK/8 db-blocks fit in Ss = [BQ][BK]) and convert on write-out.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nb_per = BK / 8;
        for (uint db0 = 0; db0 < dblks; db0 += nb_per) {
            const uint nb = min(nb_per, dblks - db0);
            for (uint j = 0; j < nb; ++j) {
                simdgroup_store(Ofrag[db0 + j], &Ss[sg_row0 * BK + j * 8], BK);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < BQ * nb * 8; idx += TGS) {
                const uint r = idx / (nb * 8);
                const uint cc = idx % (nb * 8);
                const uint d = db0 * 8 + cc;
                if (int(r) < q_rows && d < D) {
                    O[ulong(q0 + int(r)) * p.o_row_stride + head * D + d] =
                        T_in(Ss[r * BK + cc]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define INSTANTIATE(NAME, T_IN, T_F, SGS, BK, DIRECT)                         \
    kernel void NAME(                                                         \
        device const T_IN* Q [[buffer(0)]],                                   \
        device const T_IN* K [[buffer(1)]],                                   \
        device const T_IN* V [[buffer(2)]],                                   \
        device T_IN* O [[buffer(3)]],                                         \
        device const int* cu_q [[buffer(4)]],                                 \
        device const int* cu_kv [[buffer(5)]],                                \
        constant Params& p [[buffer(6)]],                                     \
        device float* lse [[buffer(7)]],                                      \
        uint3 tgid [[threadgroup_position_in_grid]],                          \
        uint tid [[thread_index_in_threadgroup]],                             \
        uint sgid [[simdgroup_index_in_threadgroup]],                         \
        uint lane [[thread_index_in_simdgroup]])                              \
    {                                                                         \
        threadgroup T_F Qs[8 * SGS * DMAX];                                   \
        threadgroup T_F KsT[DMAX * BK];                                       \
        threadgroup T_F Vs[BK * DMAX];                                        \
        threadgroup float Ss[8 * SGS * BK];                                   \
        threadgroup T_F Ps[8 * SGS * BK];                                     \
        threadgroup float Diag[SGS * 64];                                     \
        threadgroup float m_run[8 * SGS];                                     \
        threadgroup float l_run[8 * SGS];                                     \
        threadgroup float c_run[8 * SGS];                                     \
        varlen_attn_impl<T_IN, T_F, SGS, BK, DIRECT>(                         \
            Q, K, V, O, cu_q, cu_kv, p, lse, Qs, KsT, Vs, Ss, Ps, Diag,       \
            m_run, l_run, c_run, tgid, tid, sgid, lane);                      \
    }

// Mixed precision: half/bf16 OPERANDS on the fast simdgroup-matrix path, but
// fp32 ACCUMULATORS (scores, output, rescale) so partial sums never overflow —
// the earlier all-fp32-fragment kernel was correct but fell off the fast matrix
// path and, at 4x the operand footprint, was capped at BK=8. Halving the operand
// width frees threadgroup memory for BK=32 (4x the KV tile, far more matmul per
// device load): ~1.4x the all-fp32 path on M1-M4 (~0.78 -> ~1.1 TFLOPS at
// head_dim=128). fp32 input keeps fp32 fragments (BK=8) and stores its output
// directly (DIRECT). NB: SGS=4 (4 simdgroups) is the measured sweet spot here —
// more simdgroups don't help (the kernel is tile-reuse-bound, not occupancy-
// bound) and host tg_threads in ext.mm MUST equal 32*SGS or upper rows go
// uncomputed.
//   half/bf16: SGS=4 (BQ=32, 128 threads), BK=32 -> ~31 KB threadgroup memory.
INSTANTIATE(varlen_attn_half, half, half, 4, 32, false)
INSTANTIATE(varlen_attn_bfloat, bfloat, bfloat, 4, 32, false)
INSTANTIATE(varlen_attn_float, float, float, 4, 8, true)


// ============================================================================
// Register-resident forward (v3) — head_dim==128, half/bf16.
//
// The v2 kernel above stages scores through threadgroup memory (Sfrag->Ss,
// softmax, ->Ps) and rescales the output accumulator with a diagonal-matrix
// matmul: ~3 barriers and two threadgroup round-trips per KV tile, plus 16
// rescale matmuls/tile. v3 keeps the whole score/prob/output pipeline in
// registers using simdgroup_matrix::thread_elements() and the (measured) Apple
// 8x8 fragment layout — each lane owns 2 elements of one row:
//   row  = ((lane & 16) >> 2) | ((lane >> 1) & 3)     // 0..7 query row
//   col0 = ((lane & 8)  >> 1) | ((lane & 1)  << 1)    // 0,2,4,6  (e0=col0, e1=col0+1)
// The 4 lanes sharing a row reduce with simd_shuffle_xor(.,1) and (.,8). Softmax
// max/exp/sum and the online rescale run on thread_elements() with no barrier;
// P is written straight into a fragment for P.V; O is written straight to device
// (the lane knows its row+cols), so Ss/Ps/Diag and ~3 barriers/tile are gone.
// Only the shared K/V tile keeps a threadgroup staging buffer + one barrier.
template <typename T_in, uint SGS, uint BK>
void varlen_attn_reg_impl(
    device const T_in* Q,
    device const T_in* K,
    device const T_in* V,
    device T_in* O,
    device const int* cu_q,
    device const int* cu_kv,
    constant Params& p,
    device float* lse,
    threadgroup T_in* Qs,        // [BQ * D]   scaled Q staging
    threadgroup T_in* KsT,       // [D * BK]   K transposed [D][BK]
    threadgroup T_in* Vs,        // [BK * D]   V [BK][D]
    uint3 tgid, uint tid, uint sgid, uint lane
) {
    constexpr uint BQ = 8 * SGS;
    constexpr uint TGS = 32 * SGS;
    constexpr uint NKB = BK / 8;
    const uint D = p.head_dim;
    const uint dblks = D / 8;

    const uint head = tgid.z;
    const uint kv_head = head / p.gqa_group;
    const uint seq = tgid.y;
    const int q_start = cu_q[seq], q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq], kv_len = cu_kv[seq + 1] - kv_start;
    const int q0 = q_start + int(tgid.x * BQ);
    if (q0 >= q_end) return;
    const int q_rows = min(int(BQ), q_end - q0);
    const int coff = kv_len - (q_end - q_start);

    // This lane's (row, col0) within an 8x8 fragment, and its query row/index.
    const uint frow = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);
    const uint c0   = ((lane & 8u)  >> 1) | ((lane & 1u) << 1);
    const uint qrow = sgid * 8 + frow;                 // block-local query row
    const int  qg   = q0 + int(qrow);                  // global query index
    const int  il   = (qg - q_start) + coff;           // causal ref: keys <= il
    const bool valid = int(qrow) < q_rows;

    // Stage scaled Q for this block (scale folded so scores live near +-1).
    for (uint idx = tid; idx < BQ * D; idx += TGS) {
        const uint r = idx / D, d = idx % D;
        Qs[r * D + d] = (int(r) < q_rows)
            ? T_in(float(Q[ulong(q0 + int(r)) * p.q_row_stride + head * D + d]) * p.scale)
            : T_in(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> Ofrag[16];
    for (uint db = 0; db < dblks; ++db) Ofrag[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m_run = -INFINITY, l_run = 0.0f;   // per-lane (== per-row) running stats

    const uint sg_row0 = sgid * 8;
    int t0_start = 0;
    if (p.window > 0) {
        int s = (q0 - q_start) + coff - int(p.window) + 1;
        if (s > 0) t0_start = (s / int(BK)) * int(BK);
    }
    for (int t0 = t0_start; t0 < kv_len; t0 += int(BK)) {
        const int q_hi = (q0 - q_start) + (q_rows - 1) + coff;
        if (p.causal && t0 > q_hi) break;
        const int tk = min(int(BK), kv_len - t0);

        for (uint idx = tid; idx < BK * D; idx += TGS) {
            const uint kk = idx / D, d = idx % D;
            if (int(kk) < tk) {
                const ulong kr = ulong(kv_start + t0 + int(kk));
                KsT[d * BK + kk] = K[kr * p.k_row_stride + kv_head * D + d];
                Vs[kk * D + d]   = V[kr * p.v_row_stride + kv_head * D + d];
            } else { KsT[d * BK + kk] = T_in(0.0f); Vs[kk * D + d] = T_in(0.0f); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q.K^T (fp32 accumulators), kept in registers.
        simdgroup_matrix<float, 8, 8> Sfrag[NKB];
        for (uint kb = 0; kb < NKB; ++kb) Sfrag[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint db = 0; db < dblks; ++db) {
            simdgroup_matrix<T_in, 8, 8> A;
            simdgroup_load(A, &Qs[sg_row0 * D + db * 8], D);
            for (uint kb = 0; kb < NKB; ++kb) {
                simdgroup_matrix<T_in, 8, 8> B;
                simdgroup_load(B, &KsT[db * 8 * BK + kb * 8], BK);
                simdgroup_multiply_accumulate(Sfrag[kb], A, B, Sfrag[kb]);
            }
        }

        // In-register softmax for this lane's row over the tile's keys, masked.
        float sv[NKB][2];
        float tmax = -INFINITY;
        for (uint kb = 0; kb < NKB; ++kb) {
            const auto e = Sfrag[kb].thread_elements();
            for (uint h = 0; h < 2; ++h) {
                const int jl = t0 + int(kb * 8 + c0 + h);     // seq-local key index
                bool keep = jl < kv_len;
                if (p.causal) keep = keep && (jl <= il);
                if (p.window > 0) keep = keep && (jl > il - int(p.window));
                const float s = keep ? float(e[h]) : -INFINITY;
                sv[kb][h] = s;
                tmax = max(tmax, s);
            }
        }
        tmax = max(tmax, simd_shuffle_xor(tmax, 1u));
        tmax = max(tmax, simd_shuffle_xor(tmax, 8u));   // row max across the 4 lanes

        const float m_new = max(m_run, tmax);
        const float corr  = (m_new == -INFINITY) ? 1.0f : exp(m_run - m_new);
        simdgroup_matrix<T_in, 8, 8> Pfrag[NKB];
        float lsum = 0.0f;
        for (uint kb = 0; kb < NKB; ++kb) {
            auto pe = Pfrag[kb].thread_elements();
            for (uint h = 0; h < 2; ++h) {
                const float pw = (sv[kb][h] == -INFINITY) ? 0.0f : exp(sv[kb][h] - m_new);
                pe[h] = T_in(pw);
                lsum += pw;
            }
            Pfrag[kb].thread_elements() = pe;
        }
        lsum += simd_shuffle_xor(lsum, 1u);
        lsum += simd_shuffle_xor(lsum, 8u);             // row sum across the 4 lanes
        l_run = l_run * corr + lsum;

        // Online rescale O by corr (per row == per lane), then O += P.V.
        for (uint db = 0; db < dblks; ++db) {
            auto oe = Ofrag[db].thread_elements();
            oe[0] *= corr; oe[1] *= corr;
            Ofrag[db].thread_elements() = oe;
        }
        for (uint kb = 0; kb < NKB; ++kb) {
            for (uint db = 0; db < dblks; ++db) {
                simdgroup_matrix<T_in, 8, 8> Bv;
                simdgroup_load(Bv, &Vs[kb * 8 * D + db * 8], D);
                simdgroup_multiply_accumulate(Ofrag[db], Pfrag[kb], Bv, Ofrag[db]);
            }
        }
        m_run = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // before next K/V load
    }

    // Finalize O = O / l, write straight to device (lane owns its row + cols).
    const float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    if (valid) {
        for (uint db = 0; db < dblks; ++db) {
            auto oe = Ofrag[db].thread_elements();
            const uint d0 = db * 8 + c0;
            O[ulong(qg) * p.o_row_stride + head * D + d0]     = T_in(oe[0] * inv);
            O[ulong(qg) * p.o_row_stride + head * D + d0 + 1] = T_in(oe[1] * inv);
        }
        if (p.return_lse && c0 == 0) {     // one lane per row writes LSE
            lse[ulong(qg) * p.num_heads + head] = (l_run > 0.0f) ? (m_run + log(l_run)) : -INFINITY;
        }
    }
}

#define INSTANTIATE_REG(NAME, T_IN, SGS, BK)                                  \
    kernel void NAME(                                                         \
        device const T_IN* Q [[buffer(0)]], device const T_IN* K [[buffer(1)]], \
        device const T_IN* V [[buffer(2)]], device T_IN* O [[buffer(3)]],     \
        device const int* cu_q [[buffer(4)]], device const int* cu_kv [[buffer(5)]], \
        constant Params& p [[buffer(6)]], device float* lse [[buffer(7)]],    \
        uint3 tgid [[threadgroup_position_in_grid]],                          \
        uint tid [[thread_index_in_threadgroup]],                            \
        uint sgid [[simdgroup_index_in_threadgroup]],                        \
        uint lane [[thread_index_in_simdgroup]])                             \
    {                                                                         \
        threadgroup T_IN Qs[8 * SGS * 128];                                   \
        threadgroup T_IN KsT[128 * BK];                                       \
        threadgroup T_IN Vs[BK * 128];                                        \
        varlen_attn_reg_impl<T_IN, SGS, BK>(                                  \
            Q, K, V, O, cu_q, cu_kv, p, lse, Qs, KsT, Vs, tgid, tid, sgid, lane); \
    }

// head_dim==128 only (dblks fixed by D at runtime, buffers sized for DMAX=128).
INSTANTIATE_REG(varlen_attn_reg_half, half, 4, 32)
INSTANTIATE_REG(varlen_attn_reg_bfloat, bfloat, 4, 32)


// ============================================================================
// Split-D forward (v4) — head_dim==128, half/bf16.
//
// v3 is occupancy-bound: each simdgroup holds the full D=128 output, so
// Ofrag[16] (32 fp32 regs/lane) caps resident threadgroups. v4 splits the OUTPUT
// D across the SG=4 simdgroups — each owns a D/4=32-col slice — so Ofrag shrinks
// to (BQ/8)*(32/8). At BQ=16 that's 8 fragments (half of v3), roughly doubling
// occupancy. The cost: scores/probs must be shared across simdgroups (each needs
// the full key row for its D-slice), so S/P go back through threadgroup memory
// (Ss/Ps) as in v2 — but the register relief outweighs it. QK is split by KEYS
// (simdgroup s computes its BK/4-key slice); softmax is cooperative (LPR threads
// per row, simd_shuffle_xor reductions); PV is split by D. Lane->(row,col) uses
// the same measured 8x8 fragment layout as v3.
template <typename T_in, uint RB, uint BK>
void varlen_attn_splitd_impl(
    device const T_in* Q, device const T_in* K, device const T_in* V, device T_in* O,
    device const int* cu_q, device const int* cu_kv, constant Params& p, device float* lse,
    threadgroup T_in* Qs,        // [BQ * 128]
    threadgroup T_in* KsT,       // [128 * BK]
    threadgroup T_in* Vs,        // [BK * 128]
    threadgroup float* Ss,       // [BQ * BK]  scores (fp32)
    threadgroup T_in* Ps,        // [BQ * BK]  probs
    threadgroup float* m_run, threadgroup float* l_run, threadgroup float* c_run,  // [BQ]
    uint3 tgid, uint tid, uint sgid, uint lane)
{
    constexpr uint SG = 4, D = 128, BQ = RB * 8, TGS = 32 * SG, NKB = BK / 8;
    constexpr uint DSLICE = D / SG, DCB = DSLICE / 8;     // 32, 4
    constexpr uint BKSL = BK / SG, KCB = BKSL / 8;        // keys/sg in QK, blocks
    constexpr uint LPR = 4;                                // softmax threads per row
    const uint dblks = D / 8;

    const uint head = tgid.z, kv_head = head / p.gqa_group, seq = tgid.y;
    const int q_start = cu_q[seq], q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq], kv_len = cu_kv[seq + 1] - kv_start;
    const int q0 = q_start + int(tgid.x * BQ);
    if (q0 >= q_end) return;
    const int q_rows = min(int(BQ), q_end - q0);
    const int coff = kv_len - (q_end - q_start);

    const uint frow = ((lane & 16u) >> 2) | ((lane >> 1) & 3u);   // 0..7 row in fragment
    const uint c0   = ((lane & 8u)  >> 1) | ((lane & 1u) << 1);   // 0,2,4,6 col in fragment

    // Stage scaled Q (scale folded so scores live near +-1; Ss then needs no scale).
    for (uint idx = tid; idx < BQ * D; idx += TGS) {
        const uint r = idx / D, d = idx % D;
        Qs[r * D + d] = (int(r) < q_rows)
            ? T_in(float(Q[ulong(q0 + int(r)) * p.q_row_stride + head * D + d]) * p.scale)
            : T_in(0.0f);
    }
    for (uint r = tid; r < BQ; r += TGS) { m_run[r] = -INFINITY; l_run[r] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> Ofrag[RB * DCB];
    for (uint i = 0; i < RB * DCB; ++i) Ofrag[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    int t0_start = 0;
    if (p.window > 0) { int s = (q0 - q_start) + coff - int(p.window) + 1; if (s > 0) t0_start = (s / int(BK)) * int(BK); }
    const int q_hi = (q0 - q_start) + (q_rows - 1) + coff;
    for (int t0 = t0_start; t0 < kv_len; t0 += int(BK)) {
        if (p.causal && t0 > q_hi) break;
        const int tk = min(int(BK), kv_len - t0);
        for (uint idx = tid; idx < BK * D; idx += TGS) {
            const uint kk = idx / D, d = idx % D;
            if (int(kk) < tk) {
                const ulong kr = ulong(kv_start + t0 + int(kk));
                KsT[d * BK + kk] = K[kr * p.k_row_stride + kv_head * D + d];
                Vs[kk * D + d]   = V[kr * p.v_row_stride + kv_head * D + d];
            } else { KsT[d * BK + kk] = T_in(0.0f); Vs[kk * D + d] = T_in(0.0f); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK: simdgroup `sgid` computes scores for its key-slice [sgid*BKSL, +BKSL).
        for (uint rb = 0; rb < RB; ++rb) {
            for (uint kc = 0; kc < KCB; ++kc) {
                simdgroup_matrix<float, 8, 8> Sf = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                for (uint db = 0; db < dblks; ++db) {
                    simdgroup_matrix<T_in, 8, 8> A, B;
                    simdgroup_load(A, &Qs[(rb * 8) * D + db * 8], D);
                    simdgroup_load(B, &KsT[db * 8 * BK + sgid * BKSL + kc * 8], BK);
                    simdgroup_multiply_accumulate(Sf, A, B, Sf);
                }
                simdgroup_store(Sf, &Ss[(rb * 8) * BK + sgid * BKSL + kc * 8], BK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative online softmax: LPR threads per row, reduced with shuffles.
        {
            const uint row = tid / LPR, sub = tid % LPR;
            if (row < BQ) {
                int lim = tk;
                if (p.causal) { int h = (q0 - q_start) + int(row) + coff - t0 + 1; lim = clamp(h, 0, tk); }
                int lo = 0;
                if (p.window > 0) { int dl = (q0 - q_start) + int(row) + coff - int(p.window) + 1 - t0; lo = clamp(dl, 0, lim); }
                const float m_old = m_run[row];
                float tmax = m_old;
                for (uint c = sub; c < uint(BK); c += LPR)
                    if (int(c) >= lo && int(c) < lim) tmax = max(tmax, Ss[row*BK + c]);
                for (uint o = 1; o < LPR; o <<= 1) tmax = max(tmax, simd_shuffle_xor(tmax, o));
                const float corr = (tmax == -INFINITY) ? 1.0f : exp(m_old - tmax);
                float tsum = 0.0f;
                for (uint c = sub; c < uint(BK); c += LPR) {
                    float w = (int(c) >= lo && int(c) < lim) ? exp(Ss[row*BK + c] - tmax) : 0.0f;
                    tsum += w; Ps[row*BK + c] = T_in(w);
                }
                for (uint o = 1; o < LPR; o <<= 1) tsum += simd_shuffle_xor(tsum, o);
                if (sub == 0) { m_run[row] = tmax; l_run[row] = l_run[row]*corr + tsum; c_run[row] = corr; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online rescale this simdgroup's O slice by corr (per row), then O += P@V.
        for (uint rb = 0; rb < RB; ++rb)
            for (uint dc = 0; dc < DCB; ++dc) {
                const float corr = c_run[rb*8 + frow];
                auto oe = Ofrag[rb*DCB + dc].thread_elements();
                oe[0] *= corr; oe[1] *= corr;
                Ofrag[rb*DCB + dc].thread_elements() = oe;
            }
        for (uint kb = 0; kb < NKB; ++kb)
            for (uint rb = 0; rb < RB; ++rb) {
                simdgroup_matrix<T_in, 8, 8> Pf;
                simdgroup_load(Pf, &Ps[(rb*8) * BK + kb*8], BK);
                for (uint dc = 0; dc < DCB; ++dc) {
                    simdgroup_matrix<T_in, 8, 8> Vf;
                    simdgroup_load(Vf, &Vs[kb*8 * D + sgid*DSLICE + dc*8], D);
                    simdgroup_multiply_accumulate(Ofrag[rb*DCB + dc], Pf, Vf, Ofrag[rb*DCB + dc]);
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Finalize: O = O/l, written straight to device (lane owns its row + cols).
    for (uint rb = 0; rb < RB; ++rb) {
        const uint row = rb*8 + frow;
        const bool valid = int(row) < q_rows;
        const float l = l_run[row];
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (uint dc = 0; dc < DCB; ++dc) {
            auto oe = Ofrag[rb*DCB + dc].thread_elements();
            const uint col = sgid*DSLICE + dc*8 + c0;
            if (valid) {
                O[ulong(q0 + int(row)) * p.o_row_stride + head * D + col]     = T_in(oe[0] * inv);
                O[ulong(q0 + int(row)) * p.o_row_stride + head * D + col + 1] = T_in(oe[1] * inv);
            }
        }
        if (p.return_lse && sgid == 0 && c0 == 0 && valid)
            lse[ulong(q0 + int(row)) * p.num_heads + head] = (l > 0.0f) ? (m_run[row] + log(l)) : -INFINITY;
    }
}

#define INSTANTIATE_SPLITD(NAME, T_IN, RB, BK)                                \
    kernel void NAME(                                                         \
        device const T_IN* Q [[buffer(0)]], device const T_IN* K [[buffer(1)]], \
        device const T_IN* V [[buffer(2)]], device T_IN* O [[buffer(3)]],     \
        device const int* cu_q [[buffer(4)]], device const int* cu_kv [[buffer(5)]], \
        constant Params& p [[buffer(6)]], device float* lse [[buffer(7)]],    \
        uint3 tgid [[threadgroup_position_in_grid]],                          \
        uint tid [[thread_index_in_threadgroup]],                            \
        uint sgid [[simdgroup_index_in_threadgroup]],                        \
        uint lane [[thread_index_in_simdgroup]])                             \
    {                                                                         \
        threadgroup T_IN Qs[RB * 8 * 128];                                    \
        threadgroup T_IN KsT[128 * BK];                                       \
        threadgroup T_IN Vs[BK * 128];                                        \
        threadgroup float Ss[RB * 8 * BK];                                    \
        threadgroup T_IN Ps[RB * 8 * BK];                                     \
        threadgroup float m_run[RB * 8], l_run[RB * 8], c_run[RB * 8];        \
        varlen_attn_splitd_impl<T_IN, RB, BK>(                                \
            Q, K, V, O, cu_q, cu_kv, p, lse, Qs, KsT, Vs, Ss, Ps,            \
            m_run, l_run, c_run, tgid, tid, sgid, lane);                      \
    }

// head_dim==128 only. RB=3 (BQ=24), BK=32, SG=4 (128 threads): measured best.
INSTANTIATE_SPLITD(varlen_attn_splitd_half, half, 3, 32)
INSTANTIATE_SPLITD(varlen_attn_splitd_bfloat, bfloat, 3, 32)


// ============================================================================
// Backward pass (v0: correct-first, naive — one thread per output row).
//
// Recomputes S = scale*Q.K^T and P = exp(S - lse) from the stored LSE, so no
// score matrix is materialised. Slower than a tiled kernel (no simdgroup_matrix
// reuse), but straightforward to validate; a tiled version can replace it later.
//   delta[i] = sum_d dO[i,d] * O[i,d]
//   dV_j     = sum_i P_ij dO_i ;  dP_ij = dO_i.V_j ;  dS_ij = P_ij(dP_ij - delta_i)
//   dQ_i     = scale sum_j dS_ij K_j ;  dK_j = scale sum_i dS_ij Q_i
// Outputs dQ/dK/dV are contiguous: dQ row = H*D, dK/dV row = (H/gqa_group)*D.
// ============================================================================

static inline int bwd_find_seq(device const int* cu, uint nseq, int gidx) {
    for (uint s = 0; s < nseq; ++s) if (gidx < cu[s + 1]) return int(s);
    return int(nseq) - 1;
}

// delta[gq, h] = sum_d dO * O   (grid: total_q * H threads)
template <typename T>
inline void bwd_delta_impl(device const T* O, device const T* dO, device float* delta,
                           constant Params& p, uint gq, uint h) {
    const uint D = p.head_dim, H = p.num_heads;
    const ulong base = ulong(gq) * p.o_row_stride + h * D;
    float s = 0.0f;
    for (uint d = 0; d < D; ++d) s += float(dO[base + d]) * float(O[base + d]);
    delta[gq * H + h] = s;
}

// dQ[gq, h] = scale * sum_j dS_ij K_j
// One simdgroup (32 lanes) per (gq, h): lanes split the D dimension, the two
// dot products reduce with simd_sum, and each lane keeps only its slice of the
// dQ accumulator (~4 floats) — far better occupancy than one thread per row.
constant constexpr uint NLMAX = (DMAX + 31) / 32;   // D-elements per lane
template <typename T>
inline void bwd_dq_impl(device const T* Q, device const T* K, device const T* V,
                        device const T* dO, device const float* lse, device const float* delta,
                        device T* dQ, device const int* cu_q, device const int* cu_kv,
                        constant Params& p, uint gq, uint h, uint lane) {
    const uint D = p.head_dim, H = p.num_heads;
    const int s = bwd_find_seq(cu_q, p.num_seqs, int(gq));
    const int q_start = cu_q[s], q_end = cu_q[s + 1];
    const int kv_start = cu_kv[s], kv_len = cu_kv[s + 1] - kv_start;
    const int qi = int(gq) - q_start;
    const int coff = kv_len - (q_end - q_start);
    const uint hk = h / p.gqa_group;
    const float Li = lse[gq * H + h], di = delta[gq * H + h], scale = p.scale;
    const ulong qbase = ulong(gq) * p.q_row_stride + h * D;
    const ulong obase = ulong(gq) * p.o_row_stride + h * D;

    float dq[NLMAX], Qc[NLMAX], dOc[NLMAX];
    for (uint t = 0; t < NLMAX; ++t) {
        const uint d = lane + t * 32;
        dq[t] = 0.0f;
        Qc[t]  = (d < D) ? float(Q[qbase + d])  : 0.0f;
        dOc[t] = (d < D) ? float(dO[obase + d]) : 0.0f;
    }
    const int hi = p.causal ? (qi + coff) : (kv_len - 1);
    const int lo = (p.window > 0) ? max(0, qi + coff - int(p.window) + 1) : 0;
    for (int j = lo; j <= hi && j < kv_len; ++j) {
        const ulong kbase = ulong(kv_start + j) * p.k_row_stride + hk * D;
        const ulong vbase = ulong(kv_start + j) * p.v_row_stride + hk * D;
        float Kj[NLMAX], sp = 0.0f, dpp = 0.0f;
        for (uint t = 0; t < NLMAX; ++t) {
            const uint d = lane + t * 32;
            Kj[t] = (d < D) ? float(K[kbase + d]) : 0.0f;
            if (d < D) { sp += Qc[t] * Kj[t]; dpp += dOc[t] * float(V[vbase + d]); }
        }
        const float Pij = exp(simd_sum(sp) * scale - Li);
        const float dS = Pij * (simd_sum(dpp) - di);
        for (uint t = 0; t < NLMAX; ++t)
            if (lane + t * 32 < D) dq[t] += scale * dS * Kj[t];
    }
    const ulong dqbase = ulong(gq) * (H * D) + h * D;   // dQ contiguous
    for (uint t = 0; t < NLMAX; ++t) {
        const uint d = lane + t * 32;
        if (d < D) dQ[dqbase + d] = T(dq[t]);
    }
}

// dK[gk, hk], dV[gk, hk] — one simdgroup (32 lanes) per (gk, hk), lanes split D.
template <typename T>
inline void bwd_dkv_impl(device const T* Q, device const T* K, device const T* V,
                         device const T* dO, device const float* lse, device const float* delta,
                         device T* dK, device T* dV, device const int* cu_q, device const int* cu_kv,
                         constant Params& p, uint gk, uint hk, uint lane) {
    const uint D = p.head_dim, H = p.num_heads, g = p.gqa_group, Hkv = H / g;
    const int s = bwd_find_seq(cu_kv, p.num_seqs, int(gk));
    const int q_start = cu_q[s], q_len = cu_q[s + 1] - q_start;
    const int kv_start = cu_kv[s], kv_len = cu_kv[s + 1] - kv_start;
    const int kj = int(gk) - kv_start;
    const int coff = kv_len - q_len;
    const float scale = p.scale;
    const ulong kbase = ulong(gk) * p.k_row_stride + hk * D;
    const ulong vbase = ulong(gk) * p.v_row_stride + hk * D;

    float dk[NLMAX], dv[NLMAX], Kc[NLMAX], Vc[NLMAX];
    for (uint t = 0; t < NLMAX; ++t) {
        const uint d = lane + t * 32;
        dk[t] = 0.0f; dv[t] = 0.0f;
        Kc[t] = (d < D) ? float(K[kbase + d]) : 0.0f;
        Vc[t] = (d < D) ? float(V[vbase + d]) : 0.0f;
    }
    for (uint hh = hk * g; hh < (hk + 1) * g; ++hh) {       // query heads sharing this kv head
        for (int i = 0; i < q_len; ++i) {
            if (p.causal && kj > i + coff) continue;
            if (p.window > 0 && kj <= i + coff - int(p.window)) continue;
            const uint gq = uint(q_start + i);
            const ulong qbase = ulong(gq) * p.q_row_stride + hh * D;
            const ulong obase = ulong(gq) * p.o_row_stride + hh * D;
            const float Li = lse[gq * H + hh], di = delta[gq * H + hh];
            float sp = 0.0f, dpp = 0.0f;
            for (uint t = 0; t < NLMAX; ++t) {
                const uint d = lane + t * 32;
                if (d < D) { sp += float(Q[qbase + d]) * Kc[t]; dpp += float(dO[obase + d]) * Vc[t]; }
            }
            const float Pij = exp(simd_sum(sp) * scale - Li);
            const float dS = Pij * (simd_sum(dpp) - di);
            for (uint t = 0; t < NLMAX; ++t) {
                const uint d = lane + t * 32;
                if (d < D) { dv[t] += Pij * float(dO[obase + d]); dk[t] += scale * dS * float(Q[qbase + d]); }
            }
        }
    }
    const ulong dkbase = ulong(gk) * (Hkv * D) + hk * D;    // dK/dV contiguous
    for (uint t = 0; t < NLMAX; ++t) {
        const uint d = lane + t * 32;
        if (d < D) { dK[dkbase + d] = T(dk[t]); dV[dkbase + d] = T(dv[t]); }
    }
}

#define INSTANTIATE_BWD(SUF, T)                                                          \
    kernel void bwd_delta_##SUF(                                                         \
        device const T* O [[buffer(0)]], device const T* dO [[buffer(1)]],              \
        device float* delta [[buffer(2)]], constant Params& p [[buffer(3)]],            \
        uint2 gid [[thread_position_in_grid]])                                          \
    { bwd_delta_impl<T>(O, dO, delta, p, gid.x, gid.y); }                                \
    kernel void bwd_dq_##SUF(                                                            \
        device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]],              \
        device const T* V [[buffer(2)]], device const T* dO [[buffer(3)]],             \
        device const float* lse [[buffer(4)]], device const float* delta [[buffer(5)]],\
        device T* dQ [[buffer(6)]], device const int* cu_q [[buffer(7)]],              \
        device const int* cu_kv [[buffer(8)]], constant Params& p [[buffer(9)]],       \
        uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) \
    { bwd_dq_impl<T>(Q, K, V, dO, lse, delta, dQ, cu_q, cu_kv, p, tgid.x, tgid.y, lane); } \
    kernel void bwd_dkv_##SUF(                                                           \
        device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]],              \
        device const T* V [[buffer(2)]], device const T* dO [[buffer(3)]],             \
        device const float* lse [[buffer(4)]], device const float* delta [[buffer(5)]],\
        device T* dK [[buffer(6)]], device T* dV [[buffer(7)]],                         \
        device const int* cu_q [[buffer(8)]], device const int* cu_kv [[buffer(9)]],    \
        constant Params& p [[buffer(10)]],                                             \
        uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) \
    { bwd_dkv_impl<T>(Q, K, V, dO, lse, delta, dK, dV, cu_q, cu_kv, p, tgid.x, tgid.y, lane); }

INSTANTIATE_BWD(half, half)
INSTANTIATE_BWD(bfloat, bfloat)
INSTANTIATE_BWD(float, float)
