#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Optional arbitrary additive attention bias. Specialized per-pipeline at build
// time so the (overwhelmingly common) no-bias kernels are dead-branch-eliminated
// — they keep their original register footprint, which matters because the
// forward is occupancy-bound and register-sensitive (a runtime flag would
// pressure occupancy even when bias is off). The bias is added to the logit
// before softmax: logit = scale*(Q·K) + bias[q, head, key]. See the host
// (try_mpp_varlen) for the buffer layout and stride convention.
constant bool HAS_BIAS [[function_constant(0)]];
constexpr constant float MPP_LOG2E = 1.4426950408889634f;  // log2(e)

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
// O lives in THREADGROUP memory; the online rescale is a manual O*=corr, then
// O += P@V via matmul2d multiply_accumulate straight into it (no PV scratch
// buffer). TM=16/TN=32/SG=4 is the measured occupancy optimum on M5 — this path
// is occupancy-bound, so the small threadgroup footprint matters more than tile
// size (TM=32 and TM=64-register both regress; see attn_vl_coop note below).
template <typename T, int TM, int TN, int D, int SG>
inline void attn_vl(device const T* Q, device const T* K, device const T* V, device T* O,
                    device const int* cu_q, device const int* cu_kv, uint H, float scale,
                    uint q_rs, uint k_rs, uint v_rs, uint causal, uint g, uint window,
                    device float* lse, uint return_lse,
                    device const float* bias, uint bias_qs, uint bias_hs,
                    threadgroup float* Sb, threadgroup T* Pb,
                    threadgroup float* Ob, threadgroup float* mb, threadgroup float* lb, threadgroup float* cb,
                    uint tid, uint qtile, uint seq, uint head)
{
    using HT  = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
    using TGSf = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
    using TGSh = tensor<threadgroup T,  dextents<int32_t, 2>, tensor_inline>;
    constexpr uint NT = SG * 32;
    const int qrs = int(q_rs), krs = int(k_rs), vrs = int(v_rs);  // input row strides
    const int ho_q  = int(head * D);                 // Q/O head offset (H_q layout)
    const int ho_kv = int((head / g) * D);           // K/V head offset (GQA: maps to kv head)
    const int o_rs = int(H * D);                     // output is contiguous [M,H_q,D]
    const int q_start = cu_q[seq],  q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq], kv_end = cu_kv[seq + 1];
    const int q0 = q_start + int(qtile) * TM;
    if (q0 >= q_end) return;
    const int q_cnt = min(TM, q_end - q0);
    // Causal (flash_attn convention): query i attends key j iff j <= i + coff,
    // where coff = kv_len - q_len aligns the two sequences at the end.
    const int coff = (kv_end - kv_start) - (q_end - q_start);
    const int q_hi = (q0 - q_start) + (q_cnt - 1) + coff;  // furthest key this block sees
    thread array<int32_t, 2> stq = {1, qrs}, stk = {1, krs}, stv = {1, vrs};

    HT tQ((device T*)Q + ulong(q0) * qrs + ho_q, dextents<int32_t, 2>(D, q_cnt), stq);
    TGSf tS(Sb, dextents<int32_t, 2>(TN, TM));
    TGSh tP(Pb, dextents<int32_t, 2>(TN, TM));
    TGSf tOb(Ob, dextents<int32_t, 2>(D, TM));

    for (uint i = tid; i < TM * D; i += NT) Ob[i] = 0.0f;
    for (uint r = tid; r < TM; r += NT) { mb[r] = -INFINITY; lb[r] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sliding window: query i attends keys (diag_i - window, diag_i]. The whole
    // block's band starts at row 0's window bottom, so jump the loop there
    // (windowed attention costs O(window), not O(seqlen)).
    int kv0 = kv_start;
    if (window > 0) {
        int s = (q0 - q_start) + coff - int(window) + 1;     // lowest key any row needs
        if (s > 0) kv0 = kv_start + (s / int(TN)) * int(TN);
    }
    for (int kv = kv0; kv < kv_end; kv += TN) {
        if (causal && (kv - kv_start) > q_hi) break;   // tile fully beyond horizon
        int tk = min(TN, kv_end - kv);
        const int kvbase = kv - kv_start;              // seq-pos of first key in tile
        HT tK((device T*)K + ulong(kv) * krs + ho_kv, dextents<int32_t, 2>(D, tk), stk);
        HT tV((device T*)V + ulong(kv) * vrs + ho_kv, dextents<int32_t, 2>(D, tk), stv);
        { constexpr auto d = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
          matmul2d<d, execution_simdgroups<SG>> op;
          auto a = tQ.slice(0, 0); auto b = tK.slice(0, 0); auto c = tS.slice(0, 0); op.run(a, b, c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Parallel softmax: LPR = NT/TM threads cooperate per row (vs the old
        // one-thread-per-row, which left NT-TM threads idle between two fast
        // matmuls). The LPR threads of a row are consecutive lanes in one
        // simdgroup, so the row max/sum reduce with simd_shuffle_xor.
        constexpr uint LPR = 2u;   // threads cooperating per row (measured optimum:
                                   // >2 causes threadgroup-memory contention here)
        const uint row = tid / LPR, sub = tid % LPR;
        // Apple GPUs compute exp() as exp2(x*log2e), so fold log2(e) into the
        // scale and call exp2 directly — saves a multiply per element on the
        // exp-heavy softmax. m/scores are then in base-2 units internally (fine;
        // this path emits no LSE, and the weights/sums are mathematically equal).
        const float scl = scale * 1.4426950408889634f;
        if (row < uint(TM)) {
            int lim = tk;
            if (causal) { int h = (q0 - q_start) + int(row) + coff - kvbase + 1; lim = clamp(h, 0, tk); }
            int lo = 0;
            if (window > 0) { int dl = (q0 - q_start) + int(row) + coff - int(window) + 1 - kvbase; lo = clamp(dl, 0, lim); }
            const float m_old = mb[row];
            float tmax, corr, tsum = 0.0f;
            if (!HAS_BIAS) {
                // max(S*scl) == scl*max(S) for scl>0, so scan raw scores and scale
                // the reduced max once (saves a multiply per key in the hot scan).
                float rawmax = -INFINITY;
                for (uint c = sub; c < uint(TN); c += LPR)
                    if (int(c) >= lo && int(c) < lim) rawmax = max(rawmax, Sb[row*TN+c]);
                for (uint o = 1; o < LPR; o <<= 1) rawmax = max(rawmax, simd_shuffle_xor(rawmax, o));
                tmax = (rawmax == -INFINITY) ? m_old : max(m_old, rawmax * scl);
                // tmax stays -inf when this tile has no keys for the row (windowed
                // band skips it); rescale is then a no-op (1), not exp2(-inf+inf)=NaN.
                corr = (tmax == -INFINITY) ? 1.0f : exp2(m_old - tmax);
                for (uint c = sub; c < uint(TN); c += LPR) {
                    float w = ((int(c) >= lo) && (int(c) < lim)) ? exp2(Sb[row*TN+c]*scl - tmax) : 0.0f;
                    tsum += w; Pb[row*TN+c] = T(w);
                }
            } else {
                // Additive bias breaks the raw-max trick (the per-element logit is
                // S*scl + bias*log2e, not a uniform scaling of S), so reduce the
                // max over the full logit. bias is indexed by global query row,
                // head, and seq-local key (kvbase + c). mb/lse stay in base-2 units
                // and the resulting LSE correctly includes the bias, so the backward
                // just adds +bias in its P recompute.
                const ulong bb = ulong(q0 + int(row)) * bias_qs + ulong(head) * bias_hs + ulong(kvbase);
                float l2max = -INFINITY;
                for (uint c = sub; c < uint(TN); c += LPR)
                    if (int(c) >= lo && int(c) < lim)
                        l2max = max(l2max, Sb[row*TN+c]*scl + bias[bb + c]*MPP_LOG2E);
                for (uint o = 1; o < LPR; o <<= 1) l2max = max(l2max, simd_shuffle_xor(l2max, o));
                tmax = (l2max == -INFINITY) ? m_old : max(m_old, l2max);
                corr = (tmax == -INFINITY) ? 1.0f : exp2(m_old - tmax);
                for (uint c = sub; c < uint(TN); c += LPR) {
                    float w = ((int(c) >= lo) && (int(c) < lim))
                                ? exp2(Sb[row*TN+c]*scl + bias[bb + c]*MPP_LOG2E - tmax) : 0.0f;
                    tsum += w; Pb[row*TN+c] = T(w);
                }
            }
            for (uint o = 1; o < LPR; o <<= 1) tsum += simd_shuffle_xor(tsum, o);
            if (sub == 0) { mb[row] = tmax; lb[row] = lb[row]*corr + tsum; cb[row] = corr; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Rescale the O accumulator by corr, then O += P@V via matmul2d
        // multiply_accumulate straight into Ob — no separate PV buffer, so TM can
        // double (16->32) within the 32 KB budget, halving the K/V re-reads that
        // bound this kernel.
        for (uint i = tid; i < TM * D; i += NT) Ob[i] *= cb[i / D];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        { constexpr auto d = matmul2d_descriptor(TM, D, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
          matmul2d<d, execution_simdgroups<SG>> op;
          auto a = tP.slice(0, 0); auto b = tV.slice(0, 0); auto c = tOb.slice(0, 0); op.run(a, b, c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = tid; r < TM; r += NT) {
        if (int(r) >= q_cnt) continue;
        const float l = lb[r];
        float inv = (l > 0.0f) ? 1.0f/l : 0.0f;
        ulong base = ulong(q0 + int(r)) * o_rs + head * D;
        for (uint dd = 0; dd < D; ++dd) O[base + dd] = T(Ob[r*D+dd]*inv);
        // mb is the running max in base-2 units (exp2 softmax), so convert back:
        // lse = m_natural + log(l) = mb*ln(2) + log(l). Backward recomputes
        // P = exp(scale*S - lse) from this.
        if (return_lse)
            lse[ulong(q0 + int(r)) * H + head] =
                (l > 0.0f) ? (mb[r] * 0.69314718055994531f + log(l)) : -INFINITY;
    }
}

// TM is dispatched by sequence length: TM=16 maximizes occupancy at mid-range
// (the win below ~10K tokens), TM=32 halves the K/V re-reads and wins at long
// sequences where the GPU is already saturated (>=~10K, e.g. 3D-sparse
// transformers). The host picks the kernel by max_seqlen_q.
#define INSTANTIATE_MPP_VL(NAME, T, TM_, D_)                                    \
kernel void NAME(                                                               \
    device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]], device const T* V [[buffer(2)]], \
    device T* O [[buffer(3)]], device const int* cu_q [[buffer(4)]], device const int* cu_kv [[buffer(5)]], \
    constant uint& H [[buffer(6)]], constant float& scale [[buffer(7)]],        \
    constant uint& q_rs [[buffer(8)]], constant uint& k_rs [[buffer(9)]], constant uint& v_rs [[buffer(10)]], \
    constant uint& causal [[buffer(11)]], constant uint& g [[buffer(12)]],      \
    constant uint& window [[buffer(13)]],                                       \
    device float* lse [[buffer(14)]], constant uint& return_lse [[buffer(15)]], \
    device const float* bias [[buffer(16)]],                                    \
    constant uint& bias_qs [[buffer(17)]], constant uint& bias_hs [[buffer(18)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]]) \
{                                                                               \
    constexpr int TM=TM_,TN=48,D=D_,SG=4;                                       \
    threadgroup float Sb[TM*TN]; threadgroup T Pb[TM*TN];                       \
    threadgroup float Ob[TM*D];                                                 \
    threadgroup float mb[TM], lb[TM], cb[TM];                                   \
    attn_vl<T,TM,TN,D,SG>(Q,K,V,O,cu_q,cu_kv,H,scale,q_rs,k_rs,v_rs,causal,g,window,lse,return_lse,bias,bias_qs,bias_hs,Sb,Pb,Ob,mb,lb,cb,tid,tgid.x,tgid.y,tgid.z); \
}
// matmul2d is dimension-general, so the same kernel serves any head_dim. We
// instantiate the common dims: 64/96/128 (cover ~all transformers) get both TM
// tiles; 256 (the realistic >128 case) gets TM=16 only — at TM=32 its [32,256]
// fp32 O accumulator alone is 32 KB, the whole threadgroup budget.
INSTANTIATE_MPP_VL(attn_mpp_varlen_half,            half,   16, 128)
INSTANTIATE_MPP_VL(attn_mpp_varlen_half_tm32,       half,   32, 128)
INSTANTIATE_MPP_VL(attn_mpp_varlen_bfloat,          bfloat, 16, 128)
INSTANTIATE_MPP_VL(attn_mpp_varlen_bfloat_tm32,     bfloat, 32, 128)
INSTANTIATE_MPP_VL(attn_mpp_varlen_half_d64,        half,   16, 64)
INSTANTIATE_MPP_VL(attn_mpp_varlen_half_d64_tm32,   half,   32, 64)
INSTANTIATE_MPP_VL(attn_mpp_varlen_bfloat_d64,      bfloat, 16, 64)
INSTANTIATE_MPP_VL(attn_mpp_varlen_bfloat_d64_tm32, bfloat, 32, 64)
INSTANTIATE_MPP_VL(attn_mpp_varlen_half_d96,        half,   16, 96)
INSTANTIATE_MPP_VL(attn_mpp_varlen_half_d96_tm32,   half,   32, 96)
INSTANTIATE_MPP_VL(attn_mpp_varlen_bfloat_d96,      bfloat, 16, 96)
INSTANTIATE_MPP_VL(attn_mpp_varlen_bfloat_d96_tm32, bfloat, 32, 96)
INSTANTIATE_MPP_VL(attn_mpp_varlen_half_d256,       half,   16, 256)
INSTANTIATE_MPP_VL(attn_mpp_varlen_bfloat_d256,     bfloat, 16, 256)


// ============================================================================
// Backward dK/dV on matmul2d (flash-attn-2 structure). One threadgroup owns a
// BK-key block of one (seq, kv-head); it loops the g query heads that map to
// this kv-head and all BQ-query tiles, recomputing S=Q@K^T and P=exp(scale*S-lse)
// and accumulating dV += P^T@dO and dK += scale*dS^T@Q, with dS=P∘(dP-delta),
// dP=dO@V^T. delta[i]=Σ_d dO[i,d]·O[i,d] is precomputed (bwd_delta). Masks
// (causal/window) are applied per-element when forming P. dK/dV accumulate in
// threadgroup fp32; matmul operands are the input dtype (NA fast path).
template <typename T, int BQ, int BK, int D, int SG>
inline void bwd_dkv_mpp(device const T* Q, device const T* K, device const T* V, device const T* dO,
                        device const float* lse, device const float* delta,
                        device float* dK, device float* dV,
                        device const int* cu_q, device const int* cu_kv,
                        uint H, float scale, uint g, uint causal, uint window,
                        device const float* bias, uint bias_qs, uint bias_hs,
                        threadgroup float* Sb, threadgroup T* Pb,
                        threadgroup float* dKb, threadgroup float* dVb,
                        uint tid, uint ktile, uint seq, uint kvhead)
{
    using HT = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
    using TGSf = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
    using TGSh = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
    constexpr uint NT = SG * 32;
    const int qrs = int(H * D), krs = int((H / g) * D);     // row strides (contiguous)
    const int dk_rs = int((H / g) * D);                     // dK/dV row = Hkv*D
    const int q_start = cu_q[seq], q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq], kv_end = cu_kv[seq + 1];
    const int kv0 = kv_start + int(ktile) * BK;
    if (kv0 >= kv_end) return;
    const int k_cnt = min(BK, kv_end - kv0);
    const int kvbase = kv0 - kv_start;                      // seq-local first key
    const int coff = (kv_end - kv_start) - (q_end - q_start);
    thread array<int32_t, 2> st_q = {1, qrs}, st_k = {1, krs};

    for (uint i = tid; i < uint(BK * D); i += NT) { dKb[i] = 0.0f; dVb[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ho_kv = int(kvhead * D);
    HT tK((device T*)K + ulong(kv0) * krs + ho_kv, dextents<int32_t, 2>(D, k_cnt), st_k);
    HT tV((device T*)V + ulong(kv0) * krs + ho_kv, dextents<int32_t, 2>(D, k_cnt), st_k);
    TGSf tS(Sb, dextents<int32_t, 2>(BK, BQ));     // [BQ rows][BK cols]
    TGSh tP(Pb, dextents<int32_t, 2>(BK, BQ));
    TGSf tdK(dKb, dextents<int32_t, 2>(D, BK));    // [BK rows][D cols]
    TGSf tdV(dVb, dextents<int32_t, 2>(D, BK));

    for (uint hh = kvhead * g; hh < (kvhead + 1) * g; ++hh) {
        const int ho_q = int(hh * D);
        for (int q0 = q_start; q0 < q_end; q0 += BQ) {
            const int q_cnt = min(BQ, q_end - q0);
            const int qi0 = q0 - q_start;                  // seq-local first query
            // Causal: this key block needs only queries i with j<=i+coff for some
            // key j in [kvbase, kvbase+k_cnt). The earliest such query is
            // kvbase-coff; skip query tiles entirely below it.
            if (causal && (qi0 + q_cnt - 1) + coff < kvbase) continue;
            HT tQ((device T*)Q + ulong(q0) * qrs + ho_q, dextents<int32_t, 2>(D, q_cnt), st_q);
            HT tdO((device T*)dO + ulong(q0) * qrs + ho_q, dextents<int32_t, 2>(D, q_cnt), st_q);

            // S = Q @ K^T  -> Sb [BQ,BK]
            { constexpr auto d = matmul2d_descriptor(BQ, BK, static_cast<int>(dynamic_extent), false, true, false);
              matmul2d<d, execution_simdgroups<SG>> op; auto a=tQ.slice(0,0); auto b=tK.slice(0,0); auto c=tS.slice(0,0); op.run(a,b,c); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // P = exp(scale*S - lse), masked
            for (uint e = tid; e < uint(BQ * BK); e += NT) {
                uint r = e / BK, cc = e % BK;
                float w = 0.0f;
                if (int(r) < q_cnt && int(cc) < k_cnt) {
                    int il = qi0 + int(r), jl = kvbase + int(cc);
                    bool keep = true;
                    if (causal) keep = jl <= il + coff;
                    if (keep && window > 0) keep = jl > il + coff - int(window);
                    if (keep) {
                        float lg = scale * Sb[e] - lse[ulong(q0 + int(r)) * H + hh];
                        if (HAS_BIAS) lg += bias[ulong(q0 + int(r)) * bias_qs + ulong(hh) * bias_hs + ulong(jl)];
                        w = exp(lg);
                    }
                }
                Pb[e] = T(w);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dV += P^T @ dO
            { constexpr auto d = matmul2d_descriptor(BK, D, static_cast<int>(dynamic_extent), true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
              matmul2d<d, execution_simdgroups<SG>> op; auto a=tP.slice(0,0); auto b=tdO.slice(0,0); auto c=tdV.slice(0,0); op.run(a,b,c); }
            // dP = dO @ V^T -> Sb (reuse)
            { constexpr auto d = matmul2d_descriptor(BQ, BK, static_cast<int>(dynamic_extent), false, true, false);
              matmul2d<d, execution_simdgroups<SG>> op; auto a=tdO.slice(0,0); auto b=tV.slice(0,0); auto c=tS.slice(0,0); op.run(a,b,c); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dS = P * (dP - delta) -> Pb (reuse)
            for (uint e = tid; e < uint(BQ * BK); e += NT) {
                uint r = e / BK;
                float p = float(Pb[e]);
                float ds = (int(r) < q_cnt) ? p * (Sb[e] - delta[ulong(q0 + int(r)) * H + hh]) : 0.0f;
                Pb[e] = T(ds);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dK += dS^T @ Q   (scale applied at finalize)
            { constexpr auto d = matmul2d_descriptor(BK, D, static_cast<int>(dynamic_extent), true, false, false, matmul2d_descriptor::mode::multiply_accumulate);
              matmul2d<d, execution_simdgroups<SG>> op; auto a=tP.slice(0,0); auto b=tQ.slice(0,0); auto c=tdK.slice(0,0); op.run(a,b,c); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // Write dK (scaled) and dV.
    for (uint e = tid; e < uint(BK * D); e += NT) {
        uint r = e / D, dd = e % D;
        if (int(r) < k_cnt) {
            ulong base = ulong(kv0 + int(r)) * dk_rs + kvhead * D + dd;
            dK[base] = dKb[e] * scale;
            dV[base] = dVb[e];
        }
    }
}

#define INSTANTIATE_BWD_DKV(NAME, T, BQ, BK, D_)                                \
kernel void NAME(                                                               \
    device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]], device const T* V [[buffer(2)]], \
    device const T* dO [[buffer(3)]], device const float* lse [[buffer(4)]], device const float* delta [[buffer(5)]], \
    device float* dK [[buffer(6)]], device float* dV [[buffer(7)]],             \
    device const int* cu_q [[buffer(8)]], device const int* cu_kv [[buffer(9)]], \
    constant uint& H [[buffer(10)]], constant float& scale [[buffer(11)]],      \
    constant uint& g [[buffer(12)]], constant uint& causal [[buffer(13)]], constant uint& window [[buffer(14)]], \
    device const float* bias [[buffer(15)]],                                    \
    constant uint& bias_qs [[buffer(16)]], constant uint& bias_hs [[buffer(17)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]]) \
{                                                                               \
    constexpr int BQ_=BQ, BK_=BK, D=D_, SG=4;                                   \
    threadgroup float Sb[BQ_*BK_]; threadgroup T Pb[BQ_*BK_];                    \
    threadgroup float dKb[BK_*D]; threadgroup float dVb[BK_*D];                  \
    bwd_dkv_mpp<T,BQ_,BK_,D,SG>(Q,K,V,dO,lse,delta,dK,dV,cu_q,cu_kv,H,scale,g,causal,window,bias,bias_qs,bias_hs,Sb,Pb,dKb,dVb,tid,tgid.x,tgid.y,tgid.z); \
}
INSTANTIATE_BWD_DKV(attn_mpp_bwd_dkv_half,   half,   32, 16, 128)
INSTANTIATE_BWD_DKV(attn_mpp_bwd_dkv_bfloat, bfloat, 32, 16, 128)
INSTANTIATE_BWD_DKV(attn_mpp_bwd_dkv_half_d64,   half,   32, 16, 64)
INSTANTIATE_BWD_DKV(attn_mpp_bwd_dkv_bfloat_d64, bfloat, 32, 16, 64)
INSTANTIATE_BWD_DKV(attn_mpp_bwd_dkv_half_d96,   half,   32, 16, 96)
INSTANTIATE_BWD_DKV(attn_mpp_bwd_dkv_bfloat_d96, bfloat, 32, 16, 96)
INSTANTIATE_BWD_DKV(attn_mpp_bwd_dkv_half_d256,   half,   32, 8, 256)   // BK=8: [BK,256] dK/dV tight
INSTANTIATE_BWD_DKV(attn_mpp_bwd_dkv_bfloat_d256, bfloat, 32, 8, 256)


// Backward dQ on matmul2d. One threadgroup owns a BQ-query block of one
// (seq, query-head); it loops all BK-key tiles, recomputing S/P and
// accumulating dQ += scale*dS@K, dS = P∘(dP-delta), dP = dO@V^T.
template <typename T, int BQ, int BK, int D, int SG>
inline void bwd_dq_mpp(device const T* Q, device const T* K, device const T* V, device const T* dO,
                       device const float* lse, device const float* delta, device float* dQ,
                       device const int* cu_q, device const int* cu_kv,
                       uint H, float scale, uint g, uint causal, uint window,
                       device const float* bias, uint bias_qs, uint bias_hs,
                       threadgroup float* Sb, threadgroup T* Pb, threadgroup float* dQb,
                       uint tid, uint qtile, uint seq, uint head)
{
    using HT = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
    using TGSf = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
    using TGSh = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
    constexpr uint NT = SG * 32;
    const int qrs = int(H * D), krs = int((H / g) * D);
    const int q_start = cu_q[seq], q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq], kv_end = cu_kv[seq + 1];
    const int q0 = q_start + int(qtile) * BQ;
    if (q0 >= q_end) return;
    const int q_cnt = min(BQ, q_end - q0);
    const int qi0 = q0 - q_start;
    const int coff = (kv_end - kv_start) - (q_end - q_start);
    const int q_hi = qi0 + (q_cnt - 1) + coff;             // furthest key any query sees
    thread array<int32_t, 2> st_q = {1, qrs}, st_k = {1, krs};

    for (uint i = tid; i < uint(BQ * D); i += NT) dQb[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ho_q = int(head * D), ho_kv = int((head / g) * D);
    HT tQ((device T*)Q + ulong(q0) * qrs + ho_q, dextents<int32_t, 2>(D, q_cnt), st_q);
    HT tdO((device T*)dO + ulong(q0) * qrs + ho_q, dextents<int32_t, 2>(D, q_cnt), st_q);
    TGSf tS(Sb, dextents<int32_t, 2>(BK, BQ));
    TGSh tP(Pb, dextents<int32_t, 2>(BK, BQ));
    TGSf tdQ(dQb, dextents<int32_t, 2>(D, BQ));

    int kv0 = kv_start;
    if (window > 0) { int s = qi0 + coff - int(window) + 1; if (s > 0) kv0 = kv_start + (s / BK) * BK; }
    for (; kv0 < kv_end; kv0 += BK) {
        const int kvbase = kv0 - kv_start;
        if (causal && kvbase > q_hi) break;
        const int k_cnt = min(BK, kv_end - kv0);
        HT tK((device T*)K + ulong(kv0) * krs + ho_kv, dextents<int32_t, 2>(D, k_cnt), st_k);
        HT tV((device T*)V + ulong(kv0) * krs + ho_kv, dextents<int32_t, 2>(D, k_cnt), st_k);
        // S = Q @ K^T
        { constexpr auto d = matmul2d_descriptor(BQ, BK, static_cast<int>(dynamic_extent), false, true, false);
          matmul2d<d, execution_simdgroups<SG>> op; auto a=tQ.slice(0,0); auto b=tK.slice(0,0); auto c=tS.slice(0,0); op.run(a,b,c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < uint(BQ * BK); e += NT) {
            uint r = e / BK, cc = e % BK;
            float w = 0.0f;
            if (int(r) < q_cnt && int(cc) < k_cnt) {
                int il = qi0 + int(r), jl = kvbase + int(cc);
                bool keep = true;
                if (causal) keep = jl <= il + coff;
                if (keep && window > 0) keep = jl > il + coff - int(window);
                if (keep) {
                    float lg = scale * Sb[e] - lse[ulong(q0 + int(r)) * H + head];
                    if (HAS_BIAS) lg += bias[ulong(q0 + int(r)) * bias_qs + ulong(head) * bias_hs + ulong(jl)];
                    w = exp(lg);
                }
            }
            Pb[e] = T(w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dP = dO @ V^T -> Sb (reuse)
        { constexpr auto d = matmul2d_descriptor(BQ, BK, static_cast<int>(dynamic_extent), false, true, false);
          matmul2d<d, execution_simdgroups<SG>> op; auto a=tdO.slice(0,0); auto b=tV.slice(0,0); auto c=tS.slice(0,0); op.run(a,b,c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dS = P * (dP - delta) -> Pb
        for (uint e = tid; e < uint(BQ * BK); e += NT) {
            uint r = e / BK;
            float ds = (int(r) < q_cnt) ? float(Pb[e]) * (Sb[e] - delta[ulong(q0 + int(r)) * H + head]) : 0.0f;
            Pb[e] = T(ds);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dQ += dS @ K
        { constexpr auto d = matmul2d_descriptor(BQ, D, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
          matmul2d<d, execution_simdgroups<SG>> op; auto a=tP.slice(0,0); auto b=tK.slice(0,0); auto c=tdQ.slice(0,0); op.run(a,b,c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = tid; e < uint(BQ * D); e += NT) {
        uint r = e / D, dd = e % D;
        if (int(r) < q_cnt) dQ[ulong(q0 + int(r)) * qrs + head * D + dd] = dQb[e] * scale;
    }
}

#define INSTANTIATE_BWD_DQ(NAME, T, BQ, BK, D_)                                 \
kernel void NAME(                                                               \
    device const T* Q [[buffer(0)]], device const T* K [[buffer(1)]], device const T* V [[buffer(2)]], \
    device const T* dO [[buffer(3)]], device const float* lse [[buffer(4)]], device const float* delta [[buffer(5)]], \
    device float* dQ [[buffer(6)]],                                            \
    device const int* cu_q [[buffer(7)]], device const int* cu_kv [[buffer(8)]], \
    constant uint& H [[buffer(9)]], constant float& scale [[buffer(10)]],       \
    constant uint& g [[buffer(11)]], constant uint& causal [[buffer(12)]], constant uint& window [[buffer(13)]], \
    device const float* bias [[buffer(14)]],                                    \
    constant uint& bias_qs [[buffer(15)]], constant uint& bias_hs [[buffer(16)]], \
    uint tid [[thread_index_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]]) \
{                                                                               \
    constexpr int BQ_=BQ, BK_=BK, D=D_, SG=4;                                   \
    threadgroup float Sb[BQ_*BK_]; threadgroup T Pb[BQ_*BK_]; threadgroup float dQb[BQ_*D]; \
    bwd_dq_mpp<T,BQ_,BK_,D,SG>(Q,K,V,dO,lse,delta,dQ,cu_q,cu_kv,H,scale,g,causal,window,bias,bias_qs,bias_hs,Sb,Pb,dQb,tid,tgid.x,tgid.y,tgid.z); \
}
INSTANTIATE_BWD_DQ(attn_mpp_bwd_dq_half,   half,   32, 16, 128)
INSTANTIATE_BWD_DQ(attn_mpp_bwd_dq_bfloat, bfloat, 32, 16, 128)
INSTANTIATE_BWD_DQ(attn_mpp_bwd_dq_half_d64,   half,   32, 16, 64)
INSTANTIATE_BWD_DQ(attn_mpp_bwd_dq_bfloat_d64, bfloat, 32, 16, 64)
INSTANTIATE_BWD_DQ(attn_mpp_bwd_dq_half_d96,   half,   32, 16, 96)
INSTANTIATE_BWD_DQ(attn_mpp_bwd_dq_bfloat_d96, bfloat, 32, 16, 96)
INSTANTIATE_BWD_DQ(attn_mpp_bwd_dq_half_d256,   half,   16, 8, 256)   // BQ=16: [BQ,256] dQ tight
INSTANTIATE_BWD_DQ(attn_mpp_bwd_dq_bfloat_d256, bfloat, 16, 8, 256)

// ---- TM=64 variant with register-resident O (cooperative_tensor) ----
//
// NOT USED IN PRODUCTION — kept as a reference. The dispatcher (ext.mm) never
// selects this; only attn_mpp_varlen_half/bfloat (TM=16) are wired up.
//
// Motivation that looked promising: a TM=16 threadgroup re-streams all of K/V
// once per 16 queries, so a TM=64 tile would cut K/V re-reads 4x (M/64 passes
// instead of M/16) — the lever that matters for this memory-bound kernel. The
// blocker at TM=64 is that the [TM,D] O accumulator is 64*128*4 = 32KB, the
// entire threadgroup-memory budget on its own, so O cannot live in threadgroup
// memory as it does in attn_vl. This variant instead keeps O register-resident
// in a Metal-4 cooperative_tensor.
//
// It is numerically CORRECT (matches a CPU fp32 reference, cos=1.0, across the
// full size sweep M=64..2048). But it is ~10x SLOWER than the TM=16 path
// (measured M=L=4096, M5: TM=16 = 20ms, TM=64 here = 201ms — i.e. it collapses
// back to the simdgroup-path speed of ~200ms). Cause: two register-resident
// [64,128] fp32 cooperative tensors (the accumulator cAcc + each tile's PV
// result cPV) cost ~128 registers/thread, which tanks GPU occupancy so badly
// that memory latency can no longer be hidden — wiping out the 4x re-read
// saving and then some. TM=16 with O in threadgroup memory is the occupancy
// sweet spot and is faster at every sequence length tested, so there is no
// size regime where this variant wins and nothing to dispatch to dynamically.
//
// Retained because the cooperative_tensor accumulate/finalize mechanics are
// non-obvious and easy to want to re-attempt; this records that the idea was
// tried, made correct, and is a dead end for throughput on M5.
template <int TM, int TN, int D, int SG>
inline void attn_vl_coop(device const half* Q, device const half* K, device const half* V, device half* O,
                    device const int* cu_q, device const int* cu_kv, uint H, float scale,
                    uint q_rs, uint k_rs, uint v_rs,
                    threadgroup float* Sb, threadgroup half* Pb,
                    threadgroup float* mb, threadgroup float* lb, threadgroup float* cb,
                    uint tid, uint qtile, uint seq, uint head)
{
    using HT  = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using TGSf = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;
    using TGSh = tensor<threadgroup half,  dextents<int32_t, 2>, tensor_inline>;
    constexpr uint NT = SG * 32;
    const int qrs = int(q_rs), krs = int(k_rs), vrs = int(v_rs);
    const int ho = int(head * D);
    const int o_rs = int(H * D);
    const int q_start = cu_q[seq],  q_end = cu_q[seq + 1];
    const int kv_start = cu_kv[seq], kv_end = cu_kv[seq + 1];
    const int q0 = q_start + int(qtile) * TM;
    if (q0 >= q_end) return;
    const int q_cnt = min(TM, q_end - q0);
    thread array<int32_t, 2> stq = {1, qrs}, stk = {1, krs}, stv = {1, vrs};

    HT tQ((device half*)Q + ulong(q0) * qrs + ho, dextents<int32_t, 2>(D, q_cnt), stq);
    TGSf tS(Sb, dextents<int32_t, 2>(TN, TM));
    TGSh tP(Pb, dextents<int32_t, 2>(TN, TM));

    for (uint r = tid; r < TM; r += NT) { mb[r] = -INFINITY; lb[r] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Persistent O accumulator (register-resident), typed from the PV operands.
    constexpr auto pvd = matmul2d_descriptor(TM, D, static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<pvd, execution_simdgroups<SG>> pvop;
    HT tVrep((device half*)V + ulong(kv_start) * vrs + ho, dextents<int32_t, 2>(D, TN), stv);
    auto mPrep = tP.slice(0, 0);
    auto mVrep = tVrep.slice(0, 0);
    auto cAcc = pvop.template get_destination_cooperative_tensor<decltype(mPrep), decltype(mVrep), float>();
    for (uint16_t i = 0; i < cAcc.get_capacity(); ++i)
        if (cAcc.is_valid_element(i)) cAcc[i] = 0.0f;

    for (int kv = kv_start; kv < kv_end; kv += TN) {
        int tk = min(TN, kv_end - kv);
        HT tK((device half*)K + ulong(kv) * krs + ho, dextents<int32_t, 2>(D, tk), stk);
        HT tV((device half*)V + ulong(kv) * vrs + ho, dextents<int32_t, 2>(D, tk), stv);
        { constexpr auto d = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
          matmul2d<d, execution_simdgroups<SG>> op;
          auto a = tQ.slice(0, 0); auto b = tK.slice(0, 0); auto c = tS.slice(0, 0); op.run(a, b, c); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = tid; r < TM; r += NT) {
            float m_old = mb[r], tmax = m_old;
            for (int c = 0; c < tk; ++c) tmax = max(tmax, Sb[r*TN+c]*scale);
            float corr = exp(m_old - tmax), tsum = 0.0f;
            for (uint c = 0; c < TN; ++c) { float w=(int(c)<tk)?exp(Sb[r*TN+c]*scale-tmax):0.0f; tsum+=w; Pb[r*TN+c]=half(w); }
            mb[r] = tmax; lb[r] = lb[r]*corr + tsum; cb[r] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // matmul2d OVERWRITES its destination, so PV goes to a fresh cooperative
        // tensor and is folded into cAcc manually: O = O*corr[row] + PV.
        auto a = tP.slice(0, 0); auto b = tV.slice(0, 0);
        auto cPV = pvop.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
        pvop.run(a, b, cPV);
        for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {
            if (!cAcc.is_valid_element(i)) continue;
            auto idx = cAcc.get_multidimensional_index(i);   // idx[0]=row, idx[1]=col
            cAcc[i] = cAcc[i] * cb[uint(idx[0])] + cPV[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < cAcc.get_capacity(); ++i) {
        if (!cAcc.is_valid_element(i)) continue;
        auto idx = cAcc.get_multidimensional_index(i);
        uint row = uint(idx[0]), col = uint(idx[1]);
        if (int(row) >= q_cnt) continue;
        float inv = (lb[row] > 0.0f) ? 1.0f/lb[row] : 0.0f;
        O[ulong(q0 + int(row)) * o_rs + ho + col] = half(cAcc[i] * inv);
    }
}

// Reference entry point for attn_vl_coop. Compiled into the metallib so the
// variant stays build-checked, but intentionally never dispatched (see above).
kernel void attn_mpp_coop_half(
    device const half* Q [[buffer(0)]], device const half* K [[buffer(1)]], device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]], device const int* cu_q [[buffer(4)]], device const int* cu_kv [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant float& scale [[buffer(7)]],
    constant uint& q_rs [[buffer(8)]], constant uint& k_rs [[buffer(9)]], constant uint& v_rs [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM=16,TN=32,D=128,SG=4;
    threadgroup float Sb[TM*TN]; threadgroup half Pb[TM*TN];
    threadgroup float mb[TM], lb[TM], cb[TM];
    attn_vl_coop<TM,TN,D,SG>(Q,K,V,O,cu_q,cu_kv,H,scale,q_rs,k_rs,v_rs,Sb,Pb,mb,lb,cb,tid,tgid.x,tgid.y,tgid.z);
}
