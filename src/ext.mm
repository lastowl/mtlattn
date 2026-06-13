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

struct Params {
    uint32_t num_heads;
    uint32_t head_dim;
    float scale;
    uint32_t q_row_stride;
    uint32_t k_row_stride;
    uint32_t v_row_stride;
    uint32_t o_row_stride;
};

struct Context {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> cache;

    static Context& instance() {
        static Context ctx;
        return ctx;
    }

    Context() {
        device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device != nil, "mtlattn: no Metal device");

        NSString* libPath = nil;
        @autoreleasepool {
            Dl_info info;
            if (dladdr((void*)&Context::instance, &info)) {
                NSString* soPath = [NSString stringWithUTF8String:info.dli_fname];
                libPath = [[soPath stringByDeletingLastPathComponent]
                    stringByAppendingPathComponent:@"mtlattn.metallib"];
            }
        }
        TORCH_CHECK(libPath != nil, "mtlattn: could not locate metallib");
        NSError* error = nil;
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        TORCH_CHECK(library != nil, "mtlattn: failed to load metallib: ",
                    error ? [[error localizedDescription] UTF8String] : "?");
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
};

struct KernelCfg {
    const char* name;
    uint32_t block_q;      // BQ: query rows per threadgroup
    uint32_t tg_threads;   // 32 * simdgroups
};

KernelCfg kernel_for_dtype(at::ScalarType t) {
    switch (t) {
        case at::kHalf: return {"varlen_attn_half", 16, 64};
        case at::kBFloat16: return {"varlen_attn_bfloat", 16, 64};
        case at::kFloat: return {"varlen_attn_float", 16, 64};
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
    double scale
) {
    TORCH_CHECK(q.device().is_mps() && k.device().is_mps() && v.device().is_mps(),
                "mtlattn: q/k/v must be MPS tensors");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "mtlattn: q/k/v must be [M, H, D]");
    const auto H = q.size(1);
    const auto D = q.size(2);
    TORCH_CHECK(D <= HEAD_DIM_MAX, "mtlattn: head_dim ", D, " > ", HEAD_DIM_MAX);
    TORCH_CHECK(k.size(1) == H && v.size(1) == H && k.size(2) == D && v.size(2) == D,
                "mtlattn: q/k/v head shape mismatch");
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

    Params p;
    p.num_heads = (uint32_t)H;
    p.head_dim = (uint32_t)D;
    p.scale = (float)scale;
    p.q_row_stride = (uint32_t)q.stride(0);
    p.k_row_stride = (uint32_t)k.stride(0);
    p.v_row_stride = (uint32_t)v.stride(0);
    p.o_row_stride = (uint32_t)(H * D);

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
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
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
          py::arg("max_seqlen_q"), py::arg("scale"));
}
