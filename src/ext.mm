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
};

struct Context {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLLibrary> mpp_library = nil;  // optional Metal-4 MPP path (M5 + macOS 26.2+)
    std::unordered_map<std::string, id<MTLComputePipelineState>> cache;

    static Context& instance() {
        static Context ctx;
        return ctx;
    }

    Context() {
        device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device != nil, "mtlattn: no Metal device");

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
        // MPP metallib is optional — absent / fails to load on pre-M5 or pre-26.2.
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
};

// MPP (M5 Neural Accelerator) varlen fast-path. Returns true if it handled the
// call. Conditions: MPP metallib present, head_dim==128, dtype half/bfloat,
// head stride == D (contiguous heads). ~4-7x the simdgroup path on M5.
static bool try_mpp_varlen(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                           at::Tensor& out, const at::Tensor& cu_q, const at::Tensor& cu_kv,
                           int64_t H, int64_t D, int64_t max_seqlen_q, double scale,
                           uint32_t causal, uint32_t gqa_group, uint32_t window) {
    if (D != 128) return false;
    auto st = q.scalar_type();
    if (st != at::kHalf && st != at::kBFloat16) return false;
    // MPP wins on large sequences; small windows are faster on the simdgroup
    // path. Gate on max_seqlen_q (override with MTLATTN_MPP_MIN).
    int64_t min_seq = 1024;
    if (const char* e = std::getenv("MTLATTN_MPP_MIN")) min_seq = atoll(e);
    if (max_seqlen_q < min_seq) return false;
    // Kernel assumes head stride == D (last dim contiguous, heads packed);
    // row stride is passed per-tensor so strided unbind views need no copy.
    if (q.stride(1) != D || k.stride(1) != D || v.stride(1) != D) return false;
    if (q.stride(2) != 1 || k.stride(2) != 1 || v.stride(2) != 1) return false;
    auto& ctx = Context::instance();
    auto pso = ctx.mpp_pipeline(st == at::kHalf ? "attn_mpp_varlen_half" : "attn_mpp_varlen_bfloat");
    if (pso == nil) return false;

    const int64_t B = cu_q.numel() - 1;
    uint32_t Hh = (uint32_t)H; float sc = (float)scale;
    uint32_t qrs=(uint32_t)q.stride(0), krs=(uint32_t)k.stride(0), vrs=(uint32_t)v.stride(0);
    uint32_t qtiles = ((uint32_t)max_seqlen_q + 15) / 16;
    id<MTLBuffer> qb=at::native::mps::getMTLBufferStorage(q), kb=at::native::mps::getMTLBufferStorage(k),
                  vb=at::native::mps::getMTLBufferStorage(v), ob=at::native::mps::getMTLBufferStorage(out),
                  cqb=at::native::mps::getMTLBufferStorage(cu_q), ckb=at::native::mps::getMTLBufferStorage(cu_kv);
    NSUInteger es = q.element_size();
    NSUInteger qo=q.storage_offset()*es, ko=k.storage_offset()*es, vo=v.storage_offset()*es, oo=out.storage_offset()*es,
               cqo=cu_q.storage_offset()*4, cko=cu_kv.storage_offset()*4;
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
    int64_t window
) {
    TORCH_CHECK(q.device().is_mps() && k.device().is_mps() && v.device().is_mps(),
                "mtlattn: q/k/v must be MPS tensors");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "mtlattn: q/k/v must be [M, H, D]");
    const auto H = q.size(1);          // query heads (H_q)
    const auto H_kv = k.size(1);       // kv heads (<= H_q for GQA/MQA)
    const auto D = q.size(2);
    TORCH_CHECK(D <= HEAD_DIM_MAX, "mtlattn: head_dim ", D, " > ", HEAD_DIM_MAX);
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

    // Metal-4 MPP fast-path (M5 Neural Accelerator). Falls through on any
    // machine/shape it can't handle, to the portable simdgroup kernel below.
    if (q.size(0) > 0 && !std::getenv("MTLATTN_NO_MPP")) {
        if (try_mpp_varlen(q, k, v, out, cu_q, cu_kv, H, D, max_seqlen_q, scale,
                           causal ? 1u : 0u, gqa_group, (uint32_t)(window > 0 ? window : 0)))
            return out;
    }

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

    auto& ctx = Context::instance();
    const KernelCfg cfg = kernel_for_dtype(q.scalar_type());
    auto pso = ctx.pipeline(cfg.name);

    const uint64_t q_tiles = ((uint64_t)max_seqlen_q + cfg.block_q - 1) / cfg.block_q;
    if (q_tiles == 0 || q.size(0) == 0) return out;

    id<MTLBuffer> qb = at::native::mps::getMTLBufferStorage(q);
    id<MTLBuffer> kb = at::native::mps::getMTLBufferStorage(k);
    id<MTLBuffer> vb = at::native::mps::getMTLBufferStorage(v);
    id<MTLBuffer> ob = at::native::mps::getMTLBufferStorage(out);
    id<MTLBuffer> cqb = at::native::mps::getMTLBufferStorage(cu_q);
    id<MTLBuffer> ckb = at::native::mps::getMTLBufferStorage(cu_kv);

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
            [enc dispatchThreadgroups:MTLSizeMake(q_tiles, (NSUInteger)num_seqs, (NSUInteger)H)
                threadsPerThreadgroup:MTLSizeMake(cfg.tg_threads, 1, 1)];
            [enc endEncoding];
        }
    });

    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("varlen_attention", &varlen_attention,
          "Fused varlen attention forward (MPS)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_kv"),
          py::arg("max_seqlen_q"), py::arg("scale"), py::arg("causal") = false,
          py::arg("window") = 0);
}
