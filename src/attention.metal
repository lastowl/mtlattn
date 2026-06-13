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
    const uint head = tgid.z;
    const uint D = p.head_dim;
    const uint dblks = (D + 7) / 8;

    const int q_start = cu_q[seq];
    const int q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq];
    const int kv_len = cu_kv[seq + 1] - kv_start;

    const int q0 = q_start + int(tgid.x * BQ);
    if (q0 >= q_end) return;
    const int q_rows = min(int(BQ), q_end - q0);

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

    for (int t0 = 0; t0 < kv_len; t0 += int(BK)) {
        const int tk = min(int(BK), kv_len - t0);

        for (uint idx = tid; idx < BK * D; idx += TGS) {
            const uint kk = idx / D;
            const uint d = idx % D;
            if (int(kk) < tk) {
                const ulong krow = ulong(kv_start + t0 + int(kk));
                KsT[d * BK + kk] = T_f(K[krow * p.k_row_stride + head * D + d]);
                Vs[kk * D + d] = T_f(V[krow * p.v_row_stride + head * D + d]);
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
                simdgroup_store(Sfrag[kb], &Ps[sg_row0 * BK + kb * 8], BK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < BQ * BK; idx += TGS) {
            Ss[idx] = float(Ps[idx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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
            Q, K, V, O, cu_q, cu_kv, p, Qs, KsT, Vs, Ss, Ps, Diag,            \
            m_run, l_run, c_run, tgid, tid, sgid, lane);                      \
    }

// All dtypes use float fragments (2 simdgroups, BQ=16, BK=16, ~27 KB
// threadgroup memory): fp32-exact accumulation. Half fragments were tried
// for bfloat (faster) but real transformer activations have outlier
// channels — QK partial sums transiently overflow half's 65504 ceiling,
// poisoning the softmax with inf-inf = NaN. bf16's 8-bit exponent makes its
// inputs immune to overflow, but any half-precision accumulation of them
// is not. Speed lives in tiling, not fragment width.
INSTANTIATE(varlen_attn_half, half, float, 2, 16, false)
INSTANTIATE(varlen_attn_bfloat, bfloat, float, 2, 16, false)
INSTANTIATE(varlen_attn_float, float, float, 2, 16, true)
