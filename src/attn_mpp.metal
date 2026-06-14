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
                    uint q_rs, uint k_rs, uint v_rs, uint causal, uint g, uint window,
                    threadgroup float* Sb, threadgroup T* Pb, threadgroup float* PVb,
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
    TGSf tPV(PVb, dextents<int32_t, 2>(D, TM));

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
        for (uint r = tid; r < TM; r += NT) {
            int lim = tk;
            if (causal) { int h = (q0 - q_start) + int(r) + coff - kvbase + 1; lim = clamp(h, 0, tk); }
            int lo = 0;
            if (window > 0) { int dl = (q0 - q_start) + int(r) + coff - int(window) + 1 - kvbase; lo = clamp(dl, 0, lim); }
            float m_old = mb[r], tmax = m_old;
            for (int c = lo; c < lim; ++c) tmax = max(tmax, Sb[r*TN+c]*scale);
            // tmax stays -inf when this tile has no keys for the row (windowed
            // band skips it); rescale is then a no-op (1), not exp(-inf+inf)=NaN.
            float corr = (tmax == -INFINITY) ? 1.0f : exp(m_old - tmax), tsum = 0.0f;
            for (uint c = 0; c < TN; ++c) { float w=((int(c)>=lo)&&(int(c)<lim))?exp(Sb[r*TN+c]*scale-tmax):0.0f; tsum+=w; Pb[r*TN+c]=T(w); }
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
    constant uint& causal [[buffer(11)]], constant uint& g [[buffer(12)]],
    constant uint& window [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM=16,TN=64,D=128,SG=4;
    threadgroup float Sb[TM*TN]; threadgroup half Pb[TM*TN];
    threadgroup float PVb[TM*D]; threadgroup float Ob[TM*D];
    threadgroup float mb[TM], lb[TM], cb[TM];
    attn_vl<half,TM,TN,D,SG>(Q,K,V,O,cu_q,cu_kv,H,scale,q_rs,k_rs,v_rs,causal,g,window,Sb,Pb,PVb,Ob,mb,lb,cb,tid,tgid.x,tgid.y,tgid.z);
}

kernel void attn_mpp_varlen_bfloat(
    device const bfloat* Q [[buffer(0)]], device const bfloat* K [[buffer(1)]], device const bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]], device const int* cu_q [[buffer(4)]], device const int* cu_kv [[buffer(5)]],
    constant uint& H [[buffer(6)]], constant float& scale [[buffer(7)]],
    constant uint& q_rs [[buffer(8)]], constant uint& k_rs [[buffer(9)]], constant uint& v_rs [[buffer(10)]],
    constant uint& causal [[buffer(11)]], constant uint& g [[buffer(12)]],
    constant uint& window [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM=16,TN=64,D=128,SG=4;
    threadgroup float Sb[TM*TN]; threadgroup bfloat Pb[TM*TN];
    threadgroup float PVb[TM*D]; threadgroup float Ob[TM*D];
    threadgroup float mb[TM], lb[TM], cb[TM];
    attn_vl<bfloat,TM,TN,D,SG>(Q,K,V,O,cu_q,cu_kv,H,scale,q_rs,k_rs,v_rs,causal,g,window,Sb,Pb,PVb,Ob,mb,lb,cb,tid,tgid.x,tgid.y,tgid.z);
}

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
    constexpr int TM=64,TN=64,D=128,SG=4;
    threadgroup float Sb[TM*TN]; threadgroup half Pb[TM*TN];
    threadgroup float mb[TM], lb[TM], cb[TM];
    attn_vl_coop<TM,TN,D,SG>(Q,K,V,O,cu_q,cu_kv,H,scale,q_rs,k_rs,v_rs,Sb,Pb,mb,lb,cb,tid,tgid.x,tgid.y,tgid.z);
}
