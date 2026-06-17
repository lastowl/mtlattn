// mtlattn: fused varlen attention forward on Metal, for PyTorch MPS tensors.
//
// Integration pattern follows mtlgemm: MTLBuffers are obtained directly from
// MPS tensor storage (zero-copy), and the kernel is encoded into PyTorch's
// MPSStream so it sequences correctly with surrounding torch ops without a
// CPU sync per call.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <ATen/mps/MPSStream.h>
#include <torch/extension.h>
#include <dlfcn.h>
#include <exception>
#include <string>
#include <unordered_map>

namespace at { namespace native { namespace mps {
static inline id<MTLBuffer> getMTLBufferStorage(const at::TensorBase& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}
}}}

namespace {

constexpr uint32_t SG_PER_TG = 4;
constexpr uint32_t TG_SIZE = SG_PER_TG * 32;
constexpr uint32_t TILE_K = 16;
constexpr uint32_t HEAD_DIM_MAX = 128;

// Portable stand-in for at::mps::dispatch_sync_with_rethrow, which is a
// torch-internal symbol present only in some torch versions: run the block on
// the MPS stream's serial queue synchronously and rethrow any C++ exception
// (e.g. TORCH_CHECK) it raised. Keeps the extension buildable against any torch
// with MPS, not just the version that happens to export that helper.
static inline void mps_dispatch_sync(dispatch_queue_t q, void (^block)(void)) {
    __block std::exception_ptr eptr;
    dispatch_sync(q, ^() {
        try { block(); } catch (...) { eptr = std::current_exception(); }
    });
    if (eptr) std::rethrow_exception(eptr);
}

struct Params {
    uint32_t num_heads;
    uint32_t head_dim;
    float scale;
    uint32_t q_row_stride;
    uint32_t k_row_stride;
    uint32_t v_row_stride;
    uint32_t o_row_stride;
    uint32_t causal;      // 0 = full attention; 1 = causal mask
    uint32_t gqa_group;   // query heads per kv head (1 = standard MHA)
    uint32_t window;      // 0 = unlimited; >0 = sliding window (last `window` keys)
    uint32_t return_lse;  // 1 = also write per-query log-sum-exp (for backward)
    uint32_t num_seqs;    // batch size (backward kernels' sequence lookup)
};

struct Context {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLLibrary> mpp_library = nil;  // optional Metal 4 MPP path (any GPU on macOS 26.2+)
    bool na_capable = false;           // Apple10+ (M5+): has the per-core Neural Accelerator
    std::unordered_map<std::string, id<MTLComputePipelineState>> cache;

    static Context& instance() {
        static Context ctx;
        return ctx;
    }

    Context() {
        device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device != nil, "mtlattn: no Metal device");
        // GPU family Apple10 (numeric 1010) = M5 and newer (the Neural
        // Accelerator). Used to pick the MPP tile: M5's NA-fast matmul becomes
        // bandwidth-bound at long sequences (TM=32 wins there), but on M3/M4 the
        // regular matrix units stay compute-bound, so TM=16 wins at every size.
        na_capable = [device supportsFamily:(MTLGPUFamily)1010];

        NSString* dir = nil;
        @autoreleasepool {
            Dl_info info;
            if (dladdr((void*)&Context::instance, &info)) {
                NSString* soPath = [NSString stringWithUTF8String:info.dli_fname];
                dir = [soPath stringByDeletingLastPathComponent];
            }
        }
        TORCH_CHECK(dir != nil, "mtlattn: could not locate metallib");
        NSError* error = nil;
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"mtlattn.metallib"]] error:&error];
        TORCH_CHECK(library != nil, "mtlattn: failed to load metallib: ",
                    error ? [[error localizedDescription] UTF8String] : "?");
        // MPP metallib is optional — absent / fails to load before macOS 26.2
        // (or if the build SDK lacked MetalPerformancePrimitives).
        @autoreleasepool {
            NSError* e2 = nil;
            NSString* mp = [dir stringByAppendingPathComponent:@"mtlattn_mpp.metallib"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:mp])
                mpp_library = [device newLibraryWithURL:[NSURL fileURLWithPath:mp] error:&e2];
        }
    }

    id<MTLComputePipelineState> pipeline(const std::string& name) {
        auto it = cache.find(name);
        if (it != cache.end()) return it->second;
        NSString* fn = [NSString stringWithUTF8String:name.c_str()];
        id<MTLFunction> func = [library newFunctionWithName:fn];
        TORCH_CHECK(func != nil, "mtlattn: kernel not found: ", name);
        NSError* error = nil;
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:func error:&error];
        TORCH_CHECK(pso != nil, "mtlattn: pipeline failed for ", name);
        cache[name] = pso;
        return pso;
    }

    // Returns nil if MPP unavailable (caller falls back to the simdgroup path).
    id<MTLComputePipelineState> mpp_pipeline(const std::string& name) {
        if (mpp_library == nil) return nil;
        auto it = cache.find(name);
        if (it != cache.end()) return it->second;
        id<MTLFunction> func = [mpp_library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        if (func == nil) return nil;
        NSError* error = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
        if (pso == nil) return nil;
        cache[name] = pso;
        return pso;
    }

    // MPP attention kernels declare a HAS_BIAS function constant (additive
    // attn_mask). Specialize the pipeline at build time so the no-bias variant
    // is dead-branch-eliminated (no register/occupancy cost). The kernels have
    // no default for HAS_BIAS, so they MUST be built through this path, not the
    // single-arg mpp_pipeline above.
    id<MTLComputePipelineState> mpp_pipeline(const std::string& name, bool has_bias) {
        if (mpp_library == nil) return nil;
        std::string key = name + (has_bias ? "|b" : "");
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        id<MTLFunction> func = nil;
        @autoreleasepool {
            MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
            bool hb = has_bias;
            [cv setConstantValue:&hb type:MTLDataTypeBool atIndex:0];
            NSError* e = nil;
            func = [mpp_library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                                     constantValues:cv error:&e];
        }
        if (func == nil) return nil;
        NSError* error = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
        if (pso == nil) return nil;
        cache[key] = pso;
        return pso;
    }
};

// Metal 4 Performance-Primitives (matmul2d) varlen fast-path. Runs on any GPU
// with macOS 26.2+ (the path is OS-gated, not GPU-family-gated); on M5 matmul2d
// targets the per-core Neural Accelerator (~9 TFLOPS), on M3/M4 it runs on the
// regular GPU matrix units (faster than the hand-written simdgroup kernel; exact
// speed there is unconfirmed — no M3/M4 hardware on hand). Conditions: MPP
// metallib loaded, head_dim in {64,128}, dtype half/bfloat, head stride == D.
static bool try_mpp_varlen(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                           at::Tensor& out, const at::Tensor& cu_q, const at::Tensor& cu_kv,
                           int64_t H, int64_t D, int64_t max_seqlen_q, double scale,
                           uint32_t causal, uint32_t gqa_group, uint32_t window,
                           const at::Tensor& lse, uint32_t return_lse,
                           const at::Tensor& bias, uint32_t has_bias,
                           uint32_t bias_qs, uint32_t bias_hs) {
    if (D != 128 && D != 64 && D != 96 && D != 256) return false;
    auto st = q.scalar_type();
    if (st != at::kHalf && st != at::kBFloat16) return false;
    // MPP wins on large sequences; small windows are faster on the simdgroup
    // path. Gate on max_seqlen_q (override with MTLATTN_MPP_MIN). head_dim 256
    // has no simdgroup fallback, so MPP must take it at any length.
    int64_t min_seq = 1024;
    if (const char* e = std::getenv("MTLATTN_MPP_MIN")) min_seq = atoll(e);
    // With an additive bias, MPP is the ONLY path that supports it (the simdgroup
    // fallback refuses), so it must take any sequence length, not just large ones.
    if (max_seqlen_q < min_seq && D != 256 && has_bias == 0) return false;
    // Kernel assumes head stride == D (last dim contiguous, heads packed);
    // row stride is passed per-tensor so strided unbind views need no copy.
    if (q.stride(1) != D || k.stride(1) != D || v.stride(1) != D) return false;
    if (q.stride(2) != 1 || k.stride(2) != 1 || v.stride(2) != 1) return false;
    auto& ctx = Context::instance();
    // TM=16 maximizes occupancy mid-range; on M5 (NA) TM=32 halves K/V re-reads
    // and wins once the NA-fast matmul goes bandwidth-bound at very long sequences
    // (crossover ~14K with the TN=48 tile; below it TM=16 is faster). On M3/M4
    // (no NA) the matmul stays compute-bound and TM=16 wins at every size, so
    // TM=32 is gated to NA-capable GPUs. Override the crossover via MTLATTN_TM32_MIN.
    int64_t tm32_min = 14336;
    if (const char* e = std::getenv("MTLATTN_TM32_MIN")) tm32_min = atoll(e);
    // head_dim 256 only has a TM=16 kernel (its [32,256] O accumulator can't fit
    // threadgroup memory at TM=32).
    const bool tm32 = ctx.na_capable && max_seqlen_q >= tm32_min && D != 256;
    const uint32_t TM = tm32 ? 32u : 16u;
    // attn_mpp_varlen_{half|bfloat}[_d64|_d96|_d256][_tm32]  (D=128 has no suffix)
    std::string suffix = (D == 128) ? "" : ("_d" + std::to_string(D));
    std::string kname = std::string("attn_mpp_varlen_") + (st == at::kHalf ? "half" : "bfloat")
                      + suffix + (tm32 ? "_tm32" : "");
    auto pso = ctx.mpp_pipeline(kname, has_bias != 0);
    if (pso == nil) return false;

    const int64_t B = cu_q.numel() - 1;
    uint32_t Hh = (uint32_t)H; float sc = (float)scale;
    uint32_t qrs=(uint32_t)q.stride(0), krs=(uint32_t)k.stride(0), vrs=(uint32_t)v.stride(0);
    uint32_t qtiles = ((uint32_t)max_seqlen_q + TM - 1) / TM;   // q tiles of TM rows
    id<MTLBuffer> qb=at::native::mps::getMTLBufferStorage(q), kb=at::native::mps::getMTLBufferStorage(k),
                  vb=at::native::mps::getMTLBufferStorage(v), ob=at::native::mps::getMTLBufferStorage(out),
                  cqb=at::native::mps::getMTLBufferStorage(cu_q), ckb=at::native::mps::getMTLBufferStorage(cu_kv);
    NSUInteger es = q.element_size();
    NSUInteger qo=q.storage_offset()*es, ko=k.storage_offset()*es, vo=v.storage_offset()*es, oo=out.storage_offset()*es,
               cqo=cu_q.storage_offset()*4, cko=cu_kv.storage_offset()*4;
    id<MTLBuffer> lb=at::native::mps::getMTLBufferStorage(lse);
    NSUInteger lo=lse.storage_offset()*4; uint32_t rlse=return_lse;
    id<MTLBuffer> biasb=at::native::mps::getMTLBufferStorage(bias);
    NSUInteger biaso=bias.storage_offset()*bias.element_size();
    auto* stream = at::mps::getCurrentMPSStream();
    MTLSize tg = MTLSizeMake(qtiles, (NSUInteger)B, (NSUInteger)H);
    MTLSize tpt = MTLSizeMake(pso.threadExecutionWidth * 4, 1, 1);
    mps_dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            stream->endKernelCoalescing();
            id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:qb offset:qo atIndex:0]; [enc setBuffer:kb offset:ko atIndex:1]; [enc setBuffer:vb offset:vo atIndex:2];
            [enc setBuffer:ob offset:oo atIndex:3]; [enc setBuffer:cqb offset:cqo atIndex:4]; [enc setBuffer:ckb offset:cko atIndex:5];
            [enc setBytes:&Hh length:4 atIndex:6]; [enc setBytes:&sc length:4 atIndex:7];
            [enc setBytes:&qrs length:4 atIndex:8]; [enc setBytes:&krs length:4 atIndex:9]; [enc setBytes:&vrs length:4 atIndex:10];
            [enc setBytes:&causal length:4 atIndex:11];
            [enc setBytes:&gqa_group length:4 atIndex:12];
            [enc setBytes:&window length:4 atIndex:13];
            [enc setBuffer:lb offset:lo atIndex:14]; [enc setBytes:&rlse length:4 atIndex:15];
            [enc setBuffer:biasb offset:biaso atIndex:16];
            [enc setBytes:&bias_qs length:4 atIndex:17]; [enc setBytes:&bias_hs length:4 atIndex:18];
            [enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
            [enc endEncoding];
        }
    });
    return true;
}

struct KernelCfg {
    const char* name;
    uint32_t block_q;      // BQ: query rows per threadgroup
    uint32_t tg_threads;   // 32 * simdgroups
};

// block_q / tg_threads MUST match the kernel instantiation's BQ (= 8*SGS) and
// thread count (= 32*SGS) in attention.metal — the kernel's compile-time TGS is
// 32*SGS, so launching fewer threads silently leaves the upper simdgroups' query
// rows uncomputed. half/bf16 use the mixed-precision SGS=8 (BQ=64) kernels;
// fp32 stays SGS=4 (BQ=32) as its float fragments cost 2x the threadgroup memory.
KernelCfg kernel_for_dtype(at::ScalarType t) {
    switch (t) {
        case at::kHalf: return {"varlen_attn_half", 32, 128};
        case at::kBFloat16: return {"varlen_attn_bfloat", 32, 128};
        case at::kFloat: return {"varlen_attn_float", 32, 128};
        default: TORCH_CHECK(false, "mtlattn: unsupported dtype ", t);
    }
}

// q, k, v: [M, H, D] MPS tensors. May be views with a non-trivial stride on
// dim 0 only (e.g. unbind of packed [M, 3, H, D]); head stride must be D and
// the last dim contiguous. cu_seqlens_*: int32 [B+1] on MPS.
at::Tensor varlen_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_kv,
    int64_t max_seqlen_q,
    double scale,
    bool causal,
    int64_t window,
    c10::optional<at::Tensor> lse_out,  // optional [total_q, H] fp32; if given, emit LSE
    c10::optional<at::Tensor> attn_bias // optional [total_q, H or 1, max_kv] fp32 additive mask
) {
    TORCH_CHECK(q.device().is_mps() && k.device().is_mps() && v.device().is_mps(),
                "mtlattn: q/k/v must be MPS tensors");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "mtlattn: q/k/v must be [M, H, D]");
    const auto H = q.size(1);          // query heads (H_q)
    const auto H_kv = k.size(1);       // kv heads (<= H_q for GQA/MQA)
    const auto D = q.size(2);
    TORCH_CHECK(D <= HEAD_DIM_MAX || D == 256,
                "mtlattn: head_dim ", D, " unsupported (expected <= 128, or 256)");
    TORCH_CHECK(v.size(1) == H_kv && k.size(2) == D && v.size(2) == D,
                "mtlattn: k/v head/dim shape mismatch");
    TORCH_CHECK(H_kv > 0 && H % H_kv == 0,
                "mtlattn: query heads (", H, ") must be a multiple of kv heads (", H_kv, ")");
    const uint32_t gqa_group = (uint32_t)(H / H_kv);
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "mtlattn: dtype mismatch");
    for (const auto& t : {q, k, v}) {
        TORCH_CHECK(t.stride(2) == 1 && t.stride(1) == D,
                    "mtlattn: inner dims must be contiguous ([M, H, D] with row-only striding)");
    }
    TORCH_CHECK(cu_seqlens_q.device().is_mps() && cu_seqlens_kv.device().is_mps(),
                "mtlattn: cu_seqlens must be on MPS");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_kv.scalar_type() == at::kInt,
                "mtlattn: cu_seqlens must be int32");
    TORCH_CHECK(cu_seqlens_q.numel() == cu_seqlens_kv.numel() && cu_seqlens_q.numel() >= 2,
                "mtlattn: cu_seqlens_q/kv must both be [B+1]");

    const int64_t num_seqs = cu_seqlens_q.numel() - 1;
    auto cu_q = cu_seqlens_q.contiguous();
    auto cu_kv = cu_seqlens_kv.contiguous();

    auto out = at::empty({q.size(0), H, D}, q.options());

    const bool want_lse = lse_out.has_value() && lse_out->defined() && lse_out->numel() > 0;
    // The LSE buffer must always be bound; 1-elem dummy when not wanted.
    at::Tensor lse = want_lse ? *lse_out : at::empty({1}, q.options().dtype(at::kFloat));

    // Optional additive attention bias [total_q, H or 1, max_kv] fp32 (MPP-only).
    // Added to the logit before softmax: logit = scale*(Q·K) + bias[q, head, key],
    // where `key` is the seq-local key index. dim1==1 broadcasts across heads
    // (zero head stride). The buffer is always bound (1-elem dummy when absent);
    // the kernel's HAS_BIAS function constant gates the reads.
    const bool want_bias = attn_bias.has_value() && attn_bias->defined() && attn_bias->numel() > 0;
    uint32_t bias_qs = 0, bias_hs = 0;
    at::Tensor bias_t;
    if (want_bias) {
        bias_t = *attn_bias;
        TORCH_CHECK(bias_t.device().is_mps(), "mtlattn: attn_mask must be an MPS tensor");
        TORCH_CHECK(bias_t.scalar_type() == at::kFloat, "mtlattn: attn_mask must be float32");
        TORCH_CHECK(bias_t.dim() == 3, "mtlattn: attn_mask must be [total_q, H or 1, max_kv]");
        TORCH_CHECK(bias_t.size(0) == q.size(0),
                    "mtlattn: attn_mask dim0 (", bias_t.size(0), ") must equal total_q (", q.size(0), ")");
        TORCH_CHECK(bias_t.size(1) == H || bias_t.size(1) == 1,
                    "mtlattn: attn_mask dim1 must be num_heads (", H, ") or 1 (broadcast)");
        TORCH_CHECK(bias_t.stride(2) == 1, "mtlattn: attn_mask last dim must be contiguous");
        bias_qs = (uint32_t)bias_t.stride(0);
        bias_hs = (bias_t.size(1) == 1) ? 0u : (uint32_t)bias_t.stride(1);
    } else {
        bias_t = at::empty({1}, q.options().dtype(at::kFloat));
    }

    // Metal 4 MPP fast-path (matmul2d; M5 Neural Accelerator where present,
    // regular matrix units on other macOS-26.2 GPUs). Emits LSE on demand, so it
    // also serves the forward of a training step. Falls through on any
    // machine/shape it can't handle, to the portable simdgroup kernel below.
    if (q.size(0) > 0 && !std::getenv("MTLATTN_NO_MPP")) {
        if (try_mpp_varlen(q, k, v, out, cu_q, cu_kv, H, D, max_seqlen_q, scale,
                           causal ? 1u : 0u, gqa_group, (uint32_t)(window > 0 ? window : 0),
                           lse, want_lse ? 1u : 0u,
                           bias_t, want_bias ? 1u : 0u, bias_qs, bias_hs))
            return out;
    }
    // The portable simdgroup kernels are sized for head_dim <= 128; > 128 (256)
    // is MPP-only, so if we reach here with D > 128 the fast path was unavailable.
    TORCH_CHECK(D <= HEAD_DIM_MAX, "mtlattn: head_dim ", D,
                " (> 128) requires the MPP path (macOS 26.2+, fp16/bf16, no LSE)");
    // The additive attn_mask is implemented only on the MPP path; refuse rather
    // than silently dropping the bias on the simdgroup fallback.
    TORCH_CHECK(!(want_bias && q.size(0) > 0),
                "mtlattn: additive attn_mask requires the MPP path (macOS 26.2+, "
                "fp16/bf16, head_dim 64/96/128/256, head stride == D); not available here");

    Params p;
    p.num_heads = (uint32_t)H;
    p.head_dim = (uint32_t)D;
    p.scale = (float)scale;
    p.q_row_stride = (uint32_t)q.stride(0);
    p.k_row_stride = (uint32_t)k.stride(0);
    p.v_row_stride = (uint32_t)v.stride(0);
    p.o_row_stride = (uint32_t)(H * D);
    p.causal = causal ? 1u : 0u;
    p.gqa_group = gqa_group;
    p.window = (uint32_t)(window > 0 ? window : 0);
    p.return_lse = want_lse ? 1u : 0u;

    auto& ctx = Context::instance();
    KernelCfg cfg = kernel_for_dtype(q.scalar_type());
    // Register-resident v3 kernel: head_dim==128, half/bf16. Same dispatch
    // geometry (BQ=32, 128 threads) and buffer layout, so only the name swaps.
    // ~8% faster than v2 and architecturally lighter (no Ss/Ps/Diag, ~1 barrier
    // per tile). Default for its eligible shapes; opt out with MTLATTN_NO_REG.
    if (D == 128 && !std::getenv("MTLATTN_NO_REG") &&
        (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)) {
        cfg.name = (q.scalar_type() == at::kHalf) ? "varlen_attn_reg_half"
                                                  : "varlen_attn_reg_bfloat";
        cfg.block_q = 32; cfg.tg_threads = 128;
    }
    // Split-D v4 (head_dim==128, half/bf16): D split across simdgroups for lower
    // register pressure / higher occupancy. Gated by MTLATTN_SPLITD during A/B.
    if (D == 128 && std::getenv("MTLATTN_SPLITD") &&
        (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16)) {
        cfg.name = (q.scalar_type() == at::kHalf) ? "varlen_attn_splitd_half"
                                                  : "varlen_attn_splitd_bfloat";
        cfg.block_q = 24; cfg.tg_threads = 128;
    }
    auto pso = ctx.pipeline(cfg.name);

    const uint64_t q_tiles = ((uint64_t)max_seqlen_q + cfg.block_q - 1) / cfg.block_q;
    if (q_tiles == 0 || q.size(0) == 0) return out;

    id<MTLBuffer> qb = at::native::mps::getMTLBufferStorage(q);
    id<MTLBuffer> kb = at::native::mps::getMTLBufferStorage(k);
    id<MTLBuffer> vb = at::native::mps::getMTLBufferStorage(v);
    id<MTLBuffer> ob = at::native::mps::getMTLBufferStorage(out);
    id<MTLBuffer> cqb = at::native::mps::getMTLBufferStorage(cu_q);
    id<MTLBuffer> ckb = at::native::mps::getMTLBufferStorage(cu_kv);
    id<MTLBuffer> lb = at::native::mps::getMTLBufferStorage(lse);
    const NSUInteger lse_off = lse.storage_offset() * 4;

    const NSUInteger q_off = q.storage_offset() * q.element_size();
    const NSUInteger k_off = k.storage_offset() * k.element_size();
    const NSUInteger v_off = v.storage_offset() * v.element_size();
    const NSUInteger o_off = out.storage_offset() * out.element_size();
    const NSUInteger cq_off = cu_q.storage_offset() * cu_q.element_size();
    const NSUInteger ck_off = cu_kv.storage_offset() * cu_kv.element_size();

    auto* stream = at::mps::getCurrentMPSStream();
    mps_dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            stream->endKernelCoalescing();
            id<MTLCommandBuffer> cmdbuf = stream->commandBuffer();
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:qb offset:q_off atIndex:0];
            [enc setBuffer:kb offset:k_off atIndex:1];
            [enc setBuffer:vb offset:v_off atIndex:2];
            [enc setBuffer:ob offset:o_off atIndex:3];
            [enc setBuffer:cqb offset:cq_off atIndex:4];
            [enc setBuffer:ckb offset:ck_off atIndex:5];
            [enc setBytes:&p length:sizeof(Params) atIndex:6];
            [enc setBuffer:lb offset:lse_off atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(q_tiles, (NSUInteger)num_seqs, (NSUInteger)H)
                threadsPerThreadgroup:MTLSizeMake(cfg.tg_threads, 1, 1)];
            [enc endEncoding];
        }
    });

    return out;
}

// Backward: returns (dQ, dK, dV). Inputs q,k,v,o,dout,lse must be contiguous.
// dout = grad of out. lse = [total_q, H] fp32 from the forward (return_lse).
static std::tuple<at::Tensor, at::Tensor> mpp_bwd_dkv(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t, double, bool, int64_t, c10::optional<at::Tensor>);
static at::Tensor mpp_bwd_dq(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t, double, bool, int64_t, c10::optional<at::Tensor>);

std::tuple<at::Tensor, at::Tensor, at::Tensor> varlen_attention_bwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& o, const at::Tensor& dout, const at::Tensor& lse,
    const at::Tensor& cu_seqlens_q, const at::Tensor& cu_seqlens_kv,
    double scale, bool causal, int64_t window,
    c10::optional<at::Tensor> attn_bias) {
    const int64_t total_q = q.size(0), Hq = q.size(1), D = q.size(2);
    const int64_t total_kv = k.size(0), Hkv = k.size(1);
    auto cu_q = cu_seqlens_q.contiguous(), cu_kv = cu_seqlens_kv.contiguous();
    const int64_t num_seqs = cu_q.numel() - 1;

    // matmul2d backward (MPP) — ~12x the simdgroup-per-row path. head_dim 128,
    // half/bf16, MPP available. delta is a cheap torch reduction (ordered on the
    // MPS stream); the two matmul2d kernels do dQ and dK/dV.
    if ((D == 128 || D == 64 || D == 96 || D == 256) && (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
        Context::instance().mpp_library != nil && !std::getenv("MTLATTN_NO_MPP")) {
        auto delta = (dout.to(at::kFloat) * o.to(at::kFloat)).sum(-1).contiguous();  // [total_q, Hq]
        auto cqc = cu_q.to(at::kCPU), ckc = cu_kv.to(at::kCPU);
        const int* cqp = cqc.data_ptr<int>(); const int* ckp = ckc.data_ptr<int>();
        int64_t maxq = 1, maxk = 1;
        for (int64_t s = 0; s < num_seqs; ++s) {
            maxq = std::max(maxq, (int64_t)(cqp[s+1]-cqp[s]));
            maxk = std::max(maxk, (int64_t)(ckp[s+1]-ckp[s]));
        }
        auto dQf = mpp_bwd_dq(q, k, v, dout, lse, delta, cu_q, cu_kv, maxq, scale, causal, window, attn_bias);
        auto dkv = mpp_bwd_dkv(q, k, v, dout, lse, delta, cu_q, cu_kv, maxk, scale, causal, window, attn_bias);
        return {dQf.to(q.scalar_type()), std::get<0>(dkv).to(k.scalar_type()),
                std::get<1>(dkv).to(v.scalar_type())};
    }

    // The simdgroup fallback backward does not implement the additive mask.
    TORCH_CHECK(!(attn_bias.has_value() && attn_bias->defined() && attn_bias->numel() > 0),
                "mtlattn: additive attn_mask backward requires the MPP path "
                "(macOS 26.2+, fp16/bf16, head_dim 64/96/128/256)");

    auto dQ = at::empty_like(q), dK = at::empty_like(k), dV = at::empty_like(v);
    auto delta = at::empty({total_q, Hq}, q.options().dtype(at::kFloat));

    Params p{};
    p.num_heads = (uint32_t)Hq; p.head_dim = (uint32_t)D; p.scale = (float)scale;
    p.q_row_stride = (uint32_t)(Hq * D); p.k_row_stride = (uint32_t)(Hkv * D);
    p.v_row_stride = (uint32_t)(Hkv * D); p.o_row_stride = (uint32_t)(Hq * D);
    p.causal = causal ? 1u : 0u; p.gqa_group = (uint32_t)(Hq / Hkv);
    p.window = (uint32_t)(window > 0 ? window : 0); p.return_lse = 0u;
    p.num_seqs = (uint32_t)num_seqs;

    const char* suf = q.scalar_type() == at::kHalf ? "half"
                    : q.scalar_type() == at::kBFloat16 ? "bfloat" : "float";
    auto& ctx = Context::instance();
    id<MTLComputePipelineState> ps_delta = ctx.pipeline(std::string("bwd_delta_") + suf);
    id<MTLComputePipelineState> ps_dq = ctx.pipeline(std::string("bwd_dq_") + suf);
    id<MTLComputePipelineState> ps_dkv = ctx.pipeline(std::string("bwd_dkv_") + suf);

    auto buf = [](const at::Tensor& t) { return at::native::mps::getMTLBufferStorage(t); };
    id<MTLBuffer> qb=buf(q),kb=buf(k),vb=buf(v),ob=buf(o),dob=buf(dout),lb=buf(lse),
                  db=buf(delta),dqb=buf(dQ),dkb=buf(dK),dvb=buf(dV),cqb=buf(cu_q),ckb=buf(cu_kv);
    auto off = [](const at::Tensor& t) { return (NSUInteger)(t.storage_offset() * t.element_size()); };
    NSUInteger qo=off(q),ko=off(k),vo=off(v),oo=off(o),doo=off(dout),lo=off(lse),
               delo=off(delta),dqo=off(dQ),dko=off(dK),dvo=off(dV),cqo=off(cu_q),cko=off(cu_kv);

    auto* stream = at::mps::getCurrentMPSStream();
    mps_dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            stream->endKernelCoalescing();
            id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
            // delta = rowsum(dO * O)
            [enc setComputePipelineState:ps_delta];
            [enc setBuffer:ob offset:oo atIndex:0]; [enc setBuffer:dob offset:doo atIndex:1];
            [enc setBuffer:db offset:delo atIndex:2]; [enc setBytes:&p length:sizeof(Params) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(total_q, Hq, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // dQ
            [enc setComputePipelineState:ps_dq];
            [enc setBuffer:qb offset:qo atIndex:0]; [enc setBuffer:kb offset:ko atIndex:1];
            [enc setBuffer:vb offset:vo atIndex:2]; [enc setBuffer:dob offset:doo atIndex:3];
            [enc setBuffer:lb offset:lo atIndex:4]; [enc setBuffer:db offset:delo atIndex:5];
            [enc setBuffer:dqb offset:dqo atIndex:6]; [enc setBuffer:cqb offset:cqo atIndex:7];
            [enc setBuffer:ckb offset:cko atIndex:8]; [enc setBytes:&p length:sizeof(Params) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(total_q, Hq, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            // dK, dV
            [enc setComputePipelineState:ps_dkv];
            [enc setBuffer:qb offset:qo atIndex:0]; [enc setBuffer:kb offset:ko atIndex:1];
            [enc setBuffer:vb offset:vo atIndex:2]; [enc setBuffer:dob offset:doo atIndex:3];
            [enc setBuffer:lb offset:lo atIndex:4]; [enc setBuffer:db offset:delo atIndex:5];
            [enc setBuffer:dkb offset:dko atIndex:6]; [enc setBuffer:dvb offset:dvo atIndex:7];
            [enc setBuffer:cqb offset:cqo atIndex:8]; [enc setBuffer:ckb offset:cko atIndex:9];
            [enc setBytes:&p length:sizeof(Params) atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake(total_kv, Hkv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
        }
    });
    return {dQ, dK, dV};
}

// Test entry for the matmul2d dK/dV backward kernel (returns fp32 dK, dV).
static std::tuple<at::Tensor, at::Tensor> mpp_bwd_dkv(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& dO,
    const at::Tensor& lse, const at::Tensor& delta,
    const at::Tensor& cu_q, const at::Tensor& cu_kv,
    int64_t max_seqlen_kv, double scale, bool causal, int64_t window,
    c10::optional<at::Tensor> attn_bias) {
    const auto Hq = q.size(1), D = q.size(2), Hkv = k.size(1);
    const uint32_t g = (uint32_t)(Hq / Hkv);
    const int64_t num_seqs = cu_q.numel() - 1;
    auto dK = at::zeros({k.size(0), Hkv, D}, k.options().dtype(at::kFloat));
    auto dV = at::zeros_like(dK);
    const bool want_bias = attn_bias.has_value() && attn_bias->defined() && attn_bias->numel() > 0;
    uint32_t bias_qs = 0, bias_hs = 0;
    at::Tensor bias_t = want_bias ? *attn_bias : at::empty({1}, q.options().dtype(at::kFloat));
    if (want_bias) { bias_qs = (uint32_t)bias_t.stride(0); bias_hs = (bias_t.size(1) == 1) ? 0u : (uint32_t)bias_t.stride(1); }
    auto& ctx = Context::instance();
    auto pso = ctx.mpp_pipeline(std::string("attn_mpp_bwd_dkv_") + (q.scalar_type() == at::kHalf ? "half" : "bfloat") + (D == 128 ? "" : ("_d" + std::to_string(D))), want_bias);
    TORCH_CHECK(pso != nil, "mtlattn: mpp bwd dkv pipeline unavailable");
    auto cuq = cu_q.contiguous(), cukv = cu_kv.contiguous();
    auto dOc = dO.contiguous();
    uint32_t Hh = (uint32_t)Hq; float sc = (float)scale;
    uint32_t gg = g, caus = causal ? 1u : 0u, win = (uint32_t)(window > 0 ? window : 0);
    id<MTLBuffer> biasb = at::native::mps::getMTLBufferStorage(bias_t);
    NSUInteger biaso = bias_t.storage_offset() * bias_t.element_size();
    const uint32_t BK = (D == 256) ? 8 : 16;   // dK/dV accumulator is tight at 256
    uint32_t ktiles = ((uint32_t)max_seqlen_kv + BK - 1) / BK;
    NSUInteger es = q.element_size();
    id<MTLBuffer> qb=at::native::mps::getMTLBufferStorage(q), kb=at::native::mps::getMTLBufferStorage(k),
                  vb=at::native::mps::getMTLBufferStorage(v), dob=at::native::mps::getMTLBufferStorage(dOc),
                  lb=at::native::mps::getMTLBufferStorage(lse), deb=at::native::mps::getMTLBufferStorage(delta),
                  dkb=at::native::mps::getMTLBufferStorage(dK), dvb=at::native::mps::getMTLBufferStorage(dV),
                  cqb=at::native::mps::getMTLBufferStorage(cuq), ckb=at::native::mps::getMTLBufferStorage(cukv);
    NSUInteger qo=q.storage_offset()*es, ko=k.storage_offset()*es, vo=v.storage_offset()*es, doo=dOc.storage_offset()*es,
               lo=lse.storage_offset()*4, deo=delta.storage_offset()*4, dko=dK.storage_offset()*4, dvo=dV.storage_offset()*4,
               cqo=cuq.storage_offset()*4, cko=cukv.storage_offset()*4;
    auto* stream = at::mps::getCurrentMPSStream();
    MTLSize tg = MTLSizeMake(ktiles, (NSUInteger)num_seqs, (NSUInteger)Hkv);
    MTLSize tpt = MTLSizeMake(pso.threadExecutionWidth * 4, 1, 1);
    mps_dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            stream->endKernelCoalescing();
            id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:qb offset:qo atIndex:0]; [enc setBuffer:kb offset:ko atIndex:1]; [enc setBuffer:vb offset:vo atIndex:2];
            [enc setBuffer:dob offset:doo atIndex:3]; [enc setBuffer:lb offset:lo atIndex:4]; [enc setBuffer:deb offset:deo atIndex:5];
            [enc setBuffer:dkb offset:dko atIndex:6]; [enc setBuffer:dvb offset:dvo atIndex:7];
            [enc setBuffer:cqb offset:cqo atIndex:8]; [enc setBuffer:ckb offset:cko atIndex:9];
            [enc setBytes:&Hh length:4 atIndex:10]; [enc setBytes:&sc length:4 atIndex:11];
            [enc setBytes:&gg length:4 atIndex:12]; [enc setBytes:&caus length:4 atIndex:13]; [enc setBytes:&win length:4 atIndex:14];
            [enc setBuffer:biasb offset:biaso atIndex:15];
            [enc setBytes:&bias_qs length:4 atIndex:16]; [enc setBytes:&bias_hs length:4 atIndex:17];
            [enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
            [enc endEncoding];
        }
    });
    return {dK, dV};
}

// Test entry for the matmul2d dQ backward kernel (returns fp32 dQ).
static at::Tensor mpp_bwd_dq(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& dO,
    const at::Tensor& lse, const at::Tensor& delta,
    const at::Tensor& cu_q, const at::Tensor& cu_kv,
    int64_t max_seqlen_q, double scale, bool causal, int64_t window,
    c10::optional<at::Tensor> attn_bias) {
    const auto Hq = q.size(1), D = q.size(2), Hkv = k.size(1);
    const uint32_t g = (uint32_t)(Hq / Hkv);
    const int64_t num_seqs = cu_q.numel() - 1;
    auto dQ = at::zeros({q.size(0), Hq, D}, q.options().dtype(at::kFloat));
    const bool want_bias = attn_bias.has_value() && attn_bias->defined() && attn_bias->numel() > 0;
    uint32_t bias_qs = 0, bias_hs = 0;
    at::Tensor bias_t = want_bias ? *attn_bias : at::empty({1}, q.options().dtype(at::kFloat));
    if (want_bias) { bias_qs = (uint32_t)bias_t.stride(0); bias_hs = (bias_t.size(1) == 1) ? 0u : (uint32_t)bias_t.stride(1); }
    auto& ctx = Context::instance();
    auto pso = ctx.mpp_pipeline(std::string("attn_mpp_bwd_dq_") + (q.scalar_type() == at::kHalf ? "half" : "bfloat") + (D == 128 ? "" : ("_d" + std::to_string(D))), want_bias);
    TORCH_CHECK(pso != nil, "mtlattn: mpp bwd dq pipeline unavailable");
    auto cuq = cu_q.contiguous(), cukv = cu_kv.contiguous();
    auto dOc = dO.contiguous();
    uint32_t Hh = (uint32_t)Hq; float sc = (float)scale;
    uint32_t gg = g, caus = causal ? 1u : 0u, win = (uint32_t)(window > 0 ? window : 0);
    id<MTLBuffer> biasb = at::native::mps::getMTLBufferStorage(bias_t);
    NSUInteger biaso = bias_t.storage_offset() * bias_t.element_size();
    const uint32_t BQ = (D == 256) ? 16 : 32;   // dQ accumulator is tight at 256
    uint32_t qtiles = ((uint32_t)max_seqlen_q + BQ - 1) / BQ;
    NSUInteger es = q.element_size();
    id<MTLBuffer> qb=at::native::mps::getMTLBufferStorage(q), kb=at::native::mps::getMTLBufferStorage(k),
                  vb=at::native::mps::getMTLBufferStorage(v), dob=at::native::mps::getMTLBufferStorage(dOc),
                  lb=at::native::mps::getMTLBufferStorage(lse), deb=at::native::mps::getMTLBufferStorage(delta),
                  dqb=at::native::mps::getMTLBufferStorage(dQ),
                  cqb=at::native::mps::getMTLBufferStorage(cuq), ckb=at::native::mps::getMTLBufferStorage(cukv);
    NSUInteger qo=q.storage_offset()*es, ko=k.storage_offset()*es, vo=v.storage_offset()*es, doo=dOc.storage_offset()*es,
               lo=lse.storage_offset()*4, deo=delta.storage_offset()*4, dqo=dQ.storage_offset()*4,
               cqo=cuq.storage_offset()*4, cko=cukv.storage_offset()*4;
    auto* stream = at::mps::getCurrentMPSStream();
    MTLSize tg = MTLSizeMake(qtiles, (NSUInteger)num_seqs, (NSUInteger)Hq);
    MTLSize tpt = MTLSizeMake(pso.threadExecutionWidth * 4, 1, 1);
    mps_dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            stream->endKernelCoalescing();
            id<MTLComputeCommandEncoder> enc = [stream->commandBuffer() computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:qb offset:qo atIndex:0]; [enc setBuffer:kb offset:ko atIndex:1]; [enc setBuffer:vb offset:vo atIndex:2];
            [enc setBuffer:dob offset:doo atIndex:3]; [enc setBuffer:lb offset:lo atIndex:4]; [enc setBuffer:deb offset:deo atIndex:5];
            [enc setBuffer:dqb offset:dqo atIndex:6];
            [enc setBuffer:cqb offset:cqo atIndex:7]; [enc setBuffer:ckb offset:cko atIndex:8];
            [enc setBytes:&Hh length:4 atIndex:9]; [enc setBytes:&sc length:4 atIndex:10];
            [enc setBytes:&gg length:4 atIndex:11]; [enc setBytes:&caus length:4 atIndex:12]; [enc setBytes:&win length:4 atIndex:13];
            [enc setBuffer:biasb offset:biaso atIndex:14];
            [enc setBytes:&bias_qs length:4 atIndex:15]; [enc setBytes:&bias_hs length:4 atIndex:16];
            [enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
            [enc endEncoding];
        }
    });
    return dQ;
}

}  // namespace

// Diagnostic: is the Metal 4 MPP (matmul2d) fast path usable on this machine?
// True iff the mpp metallib loaded and a kernel compiles to a pipeline — i.e.
// macOS 26.2+ with MPP support. Used to confirm the path runs on M3/M4, not just M5.
static bool mpp_available() {
    return Context::instance().mpp_pipeline("attn_mpp_varlen_half", false) != nil;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mpp_available", &mpp_available, "True if the Metal 4 MPP fast path is usable here");
    m.def("mpp_bwd_dkv", &mpp_bwd_dkv, "matmul2d dK/dV backward (test entry)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"), py::arg("lse"), py::arg("delta"),
          py::arg("cu_q"), py::arg("cu_kv"), py::arg("max_seqlen_kv"), py::arg("scale"),
          py::arg("causal") = false, py::arg("window") = 0, py::arg("attn_bias") = c10::nullopt);
    m.def("mpp_bwd_dq", &mpp_bwd_dq, "matmul2d dQ backward (test entry)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"), py::arg("lse"), py::arg("delta"),
          py::arg("cu_q"), py::arg("cu_kv"), py::arg("max_seqlen_q"), py::arg("scale"),
          py::arg("causal") = false, py::arg("window") = 0, py::arg("attn_bias") = c10::nullopt);
    m.def("varlen_attention", &varlen_attention,
          "Fused varlen attention forward (MPS)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_kv"),
          py::arg("max_seqlen_q"), py::arg("scale"), py::arg("causal") = false,
          py::arg("window") = 0, py::arg("lse_out") = c10::nullopt,
          py::arg("attn_bias") = c10::nullopt);
    m.def("varlen_attention_bwd", &varlen_attention_bwd,
          "Varlen attention backward (dQ, dK, dV)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"), py::arg("dout"),
          py::arg("lse"), py::arg("cu_seqlens_q"), py::arg("cu_seqlens_kv"),
          py::arg("scale"), py::arg("causal") = false, py::arg("window") = 0,
          py::arg("attn_bias") = c10::nullopt);
}
