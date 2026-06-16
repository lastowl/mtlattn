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
// Precision: bfloat inputs use half fragments (11-bit mantissa — above
// bf16's 8-bit noise floor); softmax statistics are fp32. half and float
// inputs use float fragments at a smaller threadgroup size (memory budget)
// and are fp32-exact.

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
    threadgroup T_f* Diag,      // [SGS * 64]   per-sg 8x8 diagonal
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

    simdgroup_matrix<T_f, 8, 8> Ofrag[DBLK_MAX];
    for (uint i = 0; i < dblks; ++i) {
        Ofrag[i] = make_filled_simdgroup_matrix<T_f, 8, 8>(T_f(0.0f));
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
            simdgroup_matrix<T_f, 8, 8> Sfrag[BK / 8];
            for (uint kb = 0; kb < BK / 8; ++kb) {
                Sfrag[kb] = make_filled_simdgroup_matrix<T_f, 8, 8>(T_f(0.0f));
            }
            for (uint db = 0; db < dblks; ++db) {
                simdgroup_matrix<T_f, 8, 8> A;
                simdgroup_load(A, &Qs[sg_row0 * D + db * 8], D);
                for (uint kb = 0; kb < BK / 8; ++kb) {
                    simdgroup_matrix<T_f, 8, 8> B;
                    simdgroup_load(B, &KsT[db * 8 * BK + kb * 8], BK);
                    simdgroup_multiply_accumulate(Sfrag[kb], A, B, Sfrag[kb]);
                }
            }
            for (uint kb = 0; kb < BK / 8; ++kb) {
                // T_f == float for every instantiation, so store scores straight
                // to the fp32 Ss buffer (saves a Ps->Ss copy and a barrier).
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
            Diag[sgid * 64 + i] = (rr == cc) ? T_f(c_run[sg_row0 + rr]) : T_f(0.0f);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_matrix<T_f, 8, 8> Dg;
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
        Diag[sgid * 64 + i] = (rr == cc) ? T_f(l > 0.0f ? 1.0f / l : 0.0f) : T_f(0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    {
        simdgroup_matrix<T_f, 8, 8> Dg;
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
        device T_f* o_base = reinterpret_cast<device T_f*>(O)
            + ulong(q0 + int(sg_row0)) * p.o_row_stride + head * D;
        for (uint db = 0; db < dblks; ++db) {
            simdgroup_store(Ofrag[db], o_base + db * 8, p.o_row_stride);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // Qs reuse as bounce
        for (uint db = 0; db < dblks; ++db) {
            simdgroup_store(Ofrag[db], &Qs[sg_row0 * DMAX + db * 8], DMAX);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < BQ * D; idx += TGS) {
            const uint r = idx / D;
            const uint d = idx % D;
            if (int(r) < q_rows) {
                O[ulong(q0 + int(r)) * p.o_row_stride + head * D + d] =
                    T_in(Qs[r * DMAX + d]);
            }
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
        threadgroup T_F Diag[SGS * 64];                                       \
        threadgroup float m_run[8 * SGS];                                     \
        threadgroup float l_run[8 * SGS];                                     \
        threadgroup float c_run[8 * SGS];                                     \
        varlen_attn_impl<T_IN, T_F, SGS, BK, DIRECT>(                         \
            Q, K, V, O, cu_q, cu_kv, p, lse, Qs, KsT, Vs, Ss, Ps, Diag,       \
            m_run, l_run, c_run, tgid, tid, sgid, lane);                      \
    }

// All dtypes use float fragments (2 simdgroups, BQ=16, BK=16, ~27 KB
// threadgroup memory): fp32-exact accumulation. Half fragments were tried
// for bfloat (faster) but real transformer activations have outlier
// channels — QK partial sums transiently overflow half's 65504 ceiling,
// poisoning the softmax with inf-inf = NaN. bf16's 8-bit exponent makes its
// inputs immune to overflow, but any half-precision accumulation of them
// is not. Speed lives in tiling, not fragment width.
// 4 simdgroups (BQ=32, 128 threads) with BK=8 keys/tile: 4 resident
// simdgroups hide device-load latency far better than 2, and the small BK
// keeps threadgroup memory (~27 KB) under the 32 KB cap that affords them.
// ~1.5x the old 2-simdgroup/BK=16 tiling on the portable path.
INSTANTIATE(varlen_attn_half, half, float, 4, 8, false)
INSTANTIATE(varlen_attn_bfloat, bfloat, float, 4, 8, false)
INSTANTIATE(varlen_attn_float, float, float, 4, 8, true)


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

// dQ[gq, h] = scale * sum_j dS_ij K_j     (grid: total_q * H threads)
template <typename T>
inline void bwd_dq_impl(device const T* Q, device const T* K, device const T* V,
                        device const T* dO, device const float* lse, device const float* delta,
                        device T* dQ, device const int* cu_q, device const int* cu_kv,
                        constant Params& p, uint gq, uint h) {
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

    float dq[DMAX];
    for (uint d = 0; d < D; ++d) dq[d] = 0.0f;
    const int hi = p.causal ? (qi + coff) : (kv_len - 1);
    const int lo = (p.window > 0) ? max(0, qi + coff - int(p.window) + 1) : 0;
    for (int j = lo; j <= hi && j < kv_len; ++j) {
        const ulong kbase = ulong(kv_start + j) * p.k_row_stride + hk * D;
        const ulong vbase = ulong(kv_start + j) * p.v_row_stride + hk * D;
        float S = 0.0f, dP = 0.0f;
        for (uint d = 0; d < D; ++d) {
            S  += float(Q[qbase + d])  * float(K[kbase + d]);
            dP += float(dO[obase + d]) * float(V[vbase + d]);
        }
        const float Pij = exp(S * scale - Li);
        const float dS = Pij * (dP - di);
        for (uint d = 0; d < D; ++d) dq[d] += scale * dS * float(K[kbase + d]);
    }
    const ulong dqbase = ulong(gq) * (H * D) + h * D;   // dQ contiguous
    for (uint d = 0; d < D; ++d) dQ[dqbase + d] = T(dq[d]);
}

// dK[gk, hk], dV[gk, hk]     (grid: total_kv * H_kv threads)
template <typename T>
inline void bwd_dkv_impl(device const T* Q, device const T* K, device const T* V,
                         device const T* dO, device const float* lse, device const float* delta,
                         device T* dK, device T* dV, device const int* cu_q, device const int* cu_kv,
                         constant Params& p, uint gk, uint hk) {
    const uint D = p.head_dim, H = p.num_heads, g = p.gqa_group, Hkv = H / g;
    const int s = bwd_find_seq(cu_kv, p.num_seqs, int(gk));
    const int q_start = cu_q[s], q_len = cu_q[s + 1] - q_start;
    const int kv_start = cu_kv[s], kv_len = cu_kv[s + 1] - kv_start;
    const int kj = int(gk) - kv_start;
    const int coff = kv_len - q_len;
    const float scale = p.scale;
    const ulong kbase = ulong(gk) * p.k_row_stride + hk * D;
    const ulong vbase = ulong(gk) * p.v_row_stride + hk * D;

    float dk[DMAX], dv[DMAX];
    for (uint d = 0; d < D; ++d) { dk[d] = 0.0f; dv[d] = 0.0f; }
    for (uint hh = hk * g; hh < (hk + 1) * g; ++hh) {       // query heads sharing this kv head
        for (int i = 0; i < q_len; ++i) {
            // mask: key kj seen by query i?  causal: kj <= i+coff ; window: kj > i+coff-window
            if (p.causal && kj > i + coff) continue;
            if (p.window > 0 && kj <= i + coff - int(p.window)) continue;
            const uint gq = uint(q_start + i);
            const ulong qbase = ulong(gq) * p.q_row_stride + hh * D;
            const ulong obase = ulong(gq) * p.o_row_stride + hh * D;
            const float Li = lse[gq * H + hh], di = delta[gq * H + hh];
            float S = 0.0f, dP = 0.0f;
            for (uint d = 0; d < D; ++d) {
                S  += float(Q[qbase + d])  * float(K[kbase + d]);
                dP += float(dO[obase + d]) * float(V[vbase + d]);
            }
            const float Pij = exp(S * scale - Li);
            const float dS = Pij * (dP - di);
            for (uint d = 0; d < D; ++d) {
                dv[d] += Pij * float(dO[obase + d]);
                dk[d] += scale * dS * float(Q[qbase + d]);
            }
        }
    }
    const ulong dkbase = ulong(gk) * (Hkv * D) + hk * D;    // dK/dV contiguous
    for (uint d = 0; d < D; ++d) { dK[dkbase + d] = T(dk[d]); dV[dkbase + d] = T(dv[d]); }
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
        uint2 gid [[thread_position_in_grid]])                                          \
    { bwd_dq_impl<T>(Q, K, V, dO, lse, delta, dQ, cu_q, cu_kv, p, gid.x, gid.y); }       \
    kernel void bwd_dkv_##SUF(                                                           \
        device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]],              \
        device const T* V [[buffer(2)]], device const T* dO [[buffer(3)]],             \
        device const float* lse [[buffer(4)]], device const float* delta [[buffer(5)]],\
        device T* dK [[buffer(6)]], device T* dV [[buffer(7)]],                         \
        device const int* cu_q [[buffer(8)]], device const int* cu_kv [[buffer(9)]],    \
        constant Params& p [[buffer(10)]], uint2 gid [[thread_position_in_grid]])       \
    { bwd_dkv_impl<T>(Q, K, V, dO, lse, delta, dK, dV, cu_q, cu_kv, p, gid.x, gid.y); }

INSTANTIATE_BWD(half, half)
INSTANTIATE_BWD(bfloat, bfloat)
INSTANTIATE_BWD(float, float)
