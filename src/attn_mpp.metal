#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Single-tile fused attention (one threadgroup, L<=TN keys, one sequence):
//   S = (Q @ K^T) * scale ; P = softmax(S) ; O = P @ V
// QK via matmul2d transpose_right; softmax in threadgroup; PV via matmul2d.
// Proves the fused chain runs on the M5 accelerator. TM queries x D head dim.
template <int TM, int TN, int D, int SG>
inline void attn1(device const half* Q, device const half* K, device const half* V,
                  device half* O, uint M, uint L, float scale,
                  threadgroup float* S,   // [TM*TN]
                  threadgroup half*  P,   // [TM*TN]
                  uint tid, uint tgid_m)
{
    using HT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using TGS = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
    using TGH = tensor<threadgroup half,  dextents<int32_t, 2>, tensor_inline>;

    // Device views: Q[M,D] -> extents (D, M); K[L,D]; V[L,D]; O[M,D].
    HT tQ((device half*)Q, dextents<int32_t, 2>(D, int32_t(M)));
    HT tK((device half*)K, dextents<int32_t, 2>(D, int32_t(L)));
    HT tV((device half*)V, dextents<int32_t, 2>(D, int32_t(L)));
    HT tO(O, dextents<int32_t, 2>(D, int32_t(M)));
    TGS tS(S, dextents<int32_t, 2>(TN, TM));
    TGH tP(P, dextents<int32_t, 2>(TN, TM));

    // S[TM,TN] = Q[TM,D] @ K[TN,D]^T  (transpose_right=true)
    {
        constexpr auto d = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
        matmul2d<d, execution_simdgroups<SG>> op;
        auto mQ = tQ.slice(0, tgid_m * TM);
        auto mK = tK.slice(0, 0);
        auto mS = tS.slice(0, 0);
        op.run(mQ, mK, mS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // softmax over the first L columns of each row; write probs to P (half).
    for (uint r = tid; r < TM; r += (SG * 32)) {
        float m = -INFINITY;
        for (uint c = 0; c < L; ++c) m = max(m, S[r * TN + c] * scale);
        float s = 0.0f;
        for (uint c = 0; c < TN; ++c) {
            float w = (c < L) ? exp(S[r * TN + c] * scale - m) : 0.0f;
            s += w; P[r * TN + c] = half(w);
        }
        float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
        for (uint c = 0; c < TN; ++c) P[r * TN + c] = half(float(P[r * TN + c]) * inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O[TM,D] = P[TM,TN] @ V[TN,D]
    {
        constexpr auto d = matmul2d_descriptor(TM, D, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<SG>> op;
        auto mP = tP.slice(0, 0);
        auto mV = tV.slice(0, 0);
        auto mO = tO.slice(0, tgid_m * TM);
        op.run(mP, mV, mO);
    }
}

kernel void attn_mpp_1tile(
    device const half* Q [[buffer(0)]], device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]], device half* O [[buffer(3)]],
    constant uint& M [[buffer(4)]], constant uint& L [[buffer(5)]], constant float& scale [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM = 64, TN = 64, D = 128, SG = 4;
    threadgroup float S[TM * TN];
    threadgroup half  P[TM * TN];
    attn1<TM, TN, D, SG>(Q, K, V, O, M, L, scale, S, P, tid, tgid.x);
}

// Online (flash) softmax over KV tiles. One threadgroup -> TM queries of one
// sequence, streaming all L keys in TN-key chunks. O accumulator in threadgroup.
// matmul2d OVERWRITES C (doesn't accumulate across separate run() calls), so
// PV goes to a threadgroup temp and O is accumulated MANUALLY: O = O*corr + PV.
template <int TM, int TN, int D, int SG>
inline void attn_flash(device const half* Q, device const half* K, device const half* V,
                       device half* O, device float* Oacc, uint M, uint L, float scale,
                       threadgroup float* Sb, threadgroup half* Pb, threadgroup float* PVb,
                       threadgroup float* mb, threadgroup float* lb,
                       threadgroup float* cb, uint tid, uint q0)
{
    using HT  = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using TGSf = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
    using TGSh = tensor<threadgroup half,  dextents<int32_t, 2>, tensor_inline>;
    constexpr uint NT = SG * 32;

    HT tQ((device half*)Q, dextents<int32_t, 2>(D, int32_t(M)));
    TGSf tS(Sb, dextents<int32_t, 2>(TN, TM));
    TGSh tP(Pb, dextents<int32_t, 2>(TN, TM));
    TGSf tPV(PVb, dextents<int32_t, 2>(D, TM));

    for (uint r = tid; r < TM; r += NT) {
        mb[r] = -INFINITY; lb[r] = 0.0f;
        uint gr = q0*TM + r; if (gr < M) for (uint dd = 0; dd < D; ++dd) Oacc[ulong(gr)*D+dd] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kv = 0; kv < L; kv += TN) {
        uint tk = min(uint(TN), L - kv);
        HT tK((device half*)K + ulong(kv) * D, dextents<int32_t, 2>(D, int32_t(tk)));
        HT tV((device half*)V + ulong(kv) * D, dextents<int32_t, 2>(D, int32_t(tk)));
        { constexpr auto d = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
          matmul2d<d, execution_simdgroups<SG>> op;
          auto a = tQ.slice(0, q0 * TM); auto b = tK.slice(0, 0); auto c = tS.slice(0, 0);
          op.run(a, b, c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = tid; r < TM; r += NT) {
            float m_old = mb[r], tmax = m_old;
            for (uint c = 0; c < tk; ++c) tmax = max(tmax, Sb[r*TN+c]*scale);
            float corr = exp(m_old - tmax), tsum = 0.0f;
            for (uint c = 0; c < TN; ++c) {
                float w = (c < tk) ? exp(Sb[r*TN+c]*scale - tmax) : 0.0f;
                tsum += w; Pb[r*TN+c] = half(w);
            }
            mb[r] = tmax; lb[r] = lb[r]*corr + tsum; cb[r] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        { constexpr auto d = matmul2d_descriptor(TM, D, static_cast<int>(dynamic_extent), false, false, false);
          matmul2d<d, execution_simdgroups<SG>> op;
          auto a = tP.slice(0, 0); auto b = tV.slice(0, 0); auto c = tPV.slice(0, 0);
          op.run(a, b, c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = tid; r < TM; r += NT) {
            uint gr = q0*TM + r; if (gr >= M) continue;
            for (uint dd = 0; dd < D; ++dd)
                Oacc[ulong(gr)*D+dd] = Oacc[ulong(gr)*D+dd]*cb[r] + PVb[r*D+dd];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = tid; r < TM; r += NT) {
        float inv = (lb[r] > 0.0f) ? 1.0f/lb[r] : 0.0f;
        uint gr = q0 * TM + r;
        if (gr < M) for (uint dd = 0; dd < D; ++dd) O[ulong(gr)*D+dd] = half(Oacc[ulong(gr)*D+dd]*inv);
    }
}

kernel void attn_mpp_flash(
    device const half* Q [[buffer(0)]], device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]], device half* O [[buffer(3)]], device float* Oacc [[buffer(7)]],
    constant uint& M [[buffer(4)]], constant uint& L [[buffer(5)]], constant float& scale [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM = 32, TN = 64, D = 128, SG = 4;
    threadgroup float Sb[TM * TN];
    threadgroup half  Pb[TM * TN];
    threadgroup float PVb[TM * D];
    threadgroup float mb[TM], lb[TM], cb[TM];
    attn_flash<TM, TN, D, SG>(Q, K, V, O, Oacc, M, L, scale, Sb, Pb, PVb, mb, lb, cb, tid, tgid.x);
}

// ---- varlen + multi-head: Q,K,V,O as [total_tokens, H, D] ----
// grid (q_tiles, num_seqs, H). Head-strided tensor_inline (strides {1, H*D}).
// O accumulated in THREADGROUP (TM=16) via manual rescale+add (no device scratch).
template <typename T, int TM, int TN, int D, int SG>
inline void attn_vl(device const T* Q, device const T* K, device const T* V, device T* O,
                    device const int* cu_q, device const int* cu_kv, uint H, float scale,
                    uint q_rs, uint k_rs, uint v_rs,
                    threadgroup float* Sb, threadgroup T* Pb, threadgroup float* PVb,
                    threadgroup float* Ob, threadgroup float* mb, threadgroup float* lb, threadgroup float* cb,
                    uint tid, uint qtile, uint seq, uint head)
{
    using HT  = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
    using TGSf = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
    using TGSh = tensor<threadgroup T,  dextents<int32_t, 2>, tensor_inline>;
    constexpr uint NT = SG * 32;
    const int qrs = int(q_rs), krs = int(k_rs), vrs = int(v_rs);  // input row strides
    const int ho = int(head * D);                    // head offset (stride(1)==D guaranteed)
    const int o_rs = int(H * D);                     // output is contiguous [M,H,D]
    const int q_start = cu_q[seq],  q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq], kv_end = cu_kv[seq + 1];
    const int q0 = q_start + int(qtile) * TM;
    if (q0 >= q_end) return;
    const int q_cnt = min(TM, q_end - q0);
    thread array<int32_t, 2> stq = {1, qrs}, stk = {1, krs}, stv = {1, vrs};

    HT tQ((device T*)Q + ulong(q0) * qrs + ho, dextents<int32_t, 2>(D, q_cnt), stq);
    TGSf tS(Sb, dextents<int32_t, 2>(TN, TM));
    TGSh tP(Pb, dextents<int32_t, 2>(TN, TM));
    TGSf tPV(PVb, dextents<int32_t, 2>(D, TM));

    for (uint i = tid; i < TM * D; i += NT) Ob[i] = 0.0f;
    for (uint r = tid; r < TM; r += NT) { mb[r] = -INFINITY; lb[r] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kv = kv_start; kv < kv_end; kv += TN) {
        int tk = min(TN, kv_end - kv);
        HT tK((device T*)K + ulong(kv) * krs + ho, dextents<int32_t, 2>(D, tk), stk);
        HT tV((device T*)V + ulong(kv) * vrs + ho, dextents<int32_t, 2>(D, tk), stv);
        { constexpr auto d = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
          matmul2d<d, execution_simdgroups<SG>> op;
          auto a = tQ.slice(0, 0); auto b = tK.slice(0, 0); auto c = tS.slice(0, 0); op.run(a, b, c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = tid; r < TM; r += NT) {
            float m_old = mb[r], tmax = m_old;
            for (int c = 0; c < tk; ++c) tmax = max(tmax, Sb[r*TN+c]*scale);
            float corr = exp(m_old - tmax), tsum = 0.0f;
            for (uint c = 0; c < TN; ++c) { float w=(int(c)<tk)?exp(Sb[r*TN+c]*scale-tmax):0.0f; tsum+=w; Pb[r*TN+c]=T(w); }
            mb[r] = tmax; lb[r] = lb[r]*corr + tsum; cb[r] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        { constexpr auto d = matmul2d_descriptor(TM, D, static_cast<int>(dynamic_extent), false, false, false);
          matmul2d<d, execution_simdgroups<SG>> op;
          auto a = tP.slice(0, 0); auto b = tV.slice(0, 0); auto c = tPV.slice(0, 0); op.run(a, b, c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < TM * D; i += NT) { uint r = i / D; Ob[i] = Ob[i]*cb[r] + PVb[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = tid; r < TM; r += NT) {
        if (int(r) >= q_cnt) continue;
        float inv = (lb[r] > 0.0f) ? 1.0f/lb[r] : 0.0f;
        ulong base = ulong(q0 + int(r)) * o_rs + head * D;
        for (uint dd = 0; dd < D; ++dd) O[base + dd] = T(Ob[r*D+dd]*inv);
    }
}

kernel void attn_mpp_varlen_half(
    device const half* Q [[buffer(0)]], device const half* K [[buffer(1)]], device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]], device const int* cu_q [[buffer(4)]], device const int* cu_kv [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant float& scale [[buffer(7)]],
    constant uint& q_rs [[buffer(8)]], constant uint& k_rs [[buffer(9)]], constant uint& v_rs [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM=16,TN=64,D=128,SG=4;
    threadgroup float Sb[TM*TN]; threadgroup half Pb[TM*TN];
    threadgroup float PVb[TM*D]; threadgroup float Ob[TM*D];
    threadgroup float mb[TM], lb[TM], cb[TM];
    attn_vl<half,TM,TN,D,SG>(Q,K,V,O,cu_q,cu_kv,H,scale,q_rs,k_rs,v_rs,Sb,Pb,PVb,Ob,mb,lb,cb,tid,tgid.x,tgid.y,tgid.z);
}

kernel void attn_mpp_varlen_bfloat(
    device const bfloat* Q [[buffer(0)]], device const bfloat* K [[buffer(1)]], device const bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]], device const int* cu_q [[buffer(4)]], device const int* cu_kv [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant float& scale [[buffer(7)]],
    constant uint& q_rs [[buffer(8)]], constant uint& k_rs [[buffer(9)]], constant uint& v_rs [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM=16,TN=64,D=128,SG=4;
    threadgroup float Sb[TM*TN]; threadgroup bfloat Pb[TM*TN];
    threadgroup float PVb[TM*D]; threadgroup float Ob[TM*D];
    threadgroup float mb[TM], lb[TM], cb[TM];
    attn_vl<bfloat,TM,TN,D,SG>(Q,K,V,O,cu_q,cu_kv,H,scale,q_rs,k_rs,v_rs,Sb,Pb,PVb,Ob,mb,lb,cb,tid,tgid.x,tgid.y,tgid.z);
}
