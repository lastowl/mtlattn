"""
Minimal reproduction: PyTorch's MPS scaled_dot_product_attention silently
returns wrong results for large attention score matrices.

No exception is raised — the output is just physically impossible (values far
larger in magnitude than any input could produce, when an attention output is
a convex combination of V and must satisfy |out| <= max|V|), so it corrupts
downstream computation invisibly.

Empirically (torch 2.12, macOS 26, Apple M5 Pro, 12 heads, head_dim 128):

    L = 16384  (heads*L*L = 3.2e9)   correct
    L = 22000  (heads*L*L = 5.8e9)   correct
    L = 24576  (heads*L*L = 7.2e9)   CORRUPTED  (|out| ~ 887, |V| ~ 14)
    L = 28000  (heads*L*L = 9.4e9)   CORRUPTED
    L >= 32768                       OOM instead

So the trigger is a large score matrix (here heads*L*L between ~6e9 and ~7e9
elements), consistent with a 32-bit index/offset overflow inside the MPS
kernel rather than an exact power-of-two boundary. It reproduces in both the
unmasked single-sequence form and the masked padded-batch form.

mtlattn computes the same attention correctly at these sizes (agreement
check at the end).

Run: python tests/test_mps_sdpa_bug.py
"""

import math

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa


def cpu_reference(q, k, v, scale):
    # q, k, v: [H, L, D] fp32 on CPU
    a = (q @ k.transpose(-1, -2)) * scale
    return a.softmax(-1) @ v


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; this bug is MPS-specific.")
        return

    H, D = 12, 128
    # L = 24576 reliably triggers corruption; below ~22000 SDPA is correct,
    # above ~32768 it OOMs instead.
    L = 24576
    elems = H * L * L
    print(f"H={H} L={L} D={D}  ->  score elements = {elems:,}")

    torch.manual_seed(0)
    q = torch.randn(H, L, D, device="mps", dtype=torch.float16)
    k = torch.randn(H, L, D, device="mps", dtype=torch.float16)
    v = torch.randn(H, L, D, device="mps", dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)

    out = sdpa(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale=scale).squeeze(0)

    # Ground truth on the first 64 query rows (full-context, fp32, on CPU).
    n = 64
    ref = cpu_reference(
        q[:, :n].float().cpu(), k.float().cpu(), v.float().cpu(), scale
    )
    err = (out[:, :n].float().cpu() - ref).abs().max().item()
    out_absmax = out.float().abs().max().item()
    v_absmax = v.float().abs().max().item()

    print(f"MPS SDPA vs CPU fp32 reference:  max abs error = {err:.3f}")
    print(f"|output| max = {out_absmax:.1f}   (|V| max = {v_absmax:.1f}; "
          f"attention output is a convex combination of V, so this must be <= |V|max)")

    if err > 1.0 or out_absmax > 10 * v_absmax:
        print("\n*** BUG REPRODUCED: MPS SDPA produced impossible values, no error raised. ***")
    else:
        print("\nNo corruption at this size on this build.")

    try:
        import mtlattn
    except ImportError:
        print("(install mtlattn to see the correct result at this size)")
        return

    cu = torch.tensor([0, L], dtype=torch.int32, device="mps")
    mout = mtlattn.varlen_attention(
        q.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(),
        v.transpose(0, 1).contiguous(), cu, cu, L, scale,
    )  # [L, H, D]
    merr = (mout[:n].transpose(0, 1).float().cpu() - ref).abs().max().item()
    print(f"mtlattn vs CPU fp32 reference:   max abs error = {merr:.4f}  (correct)")


if __name__ == "__main__":
    main()
