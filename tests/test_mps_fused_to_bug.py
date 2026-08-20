"""
Reproduction of an upstream bug: PyTorch's fused device+dtype transfer from
MPS silently returns wrong values when the source is a view with a nonzero
storage offset.

    t = torch.randn(1 << 20, dtype=torch.float16, device="mps")
    t[4096:4160].to("cpu", torch.float32)     # WRONG, silently
    t[4096:4160].to("cpu").to(torch.float32)  # correct (two-step)
    t[4096:4160].clone().to("cpu", torch.float32)  # correct (fresh storage)

Found while benchmarking mtlattn's ComfyUI backend: the benchmark's CPU fp32
reference was built with the fused form and disagreed with every attention
backend at once — the reference, not the kernels, was corrupt.

Scope, established empirically (Apple M5 Pro, macOS 26.5.2):

  - ANY nonzero source storage offset triggers it. Offset 1 returns
    misaligned bit-reinterpretations (values like 3.7e19), small offsets
    return data read from the wrong address, offsets >= ~256 return zeros
    (unwritten memory). Deterministic across runs; no error is raised.
  - Needs the cast AND the cross-device copy together, MPS -> CPU:
      broken: fp16->fp32, bf16->fp32, fp32->fp16, int16->int32 (any pair)
      fine:   same-dtype MPS->CPU copy of the same view
      fine:   same-device cast on MPS (.float(), .to(dtype=...))
      fine:   fused cast+copy CPU->MPS
      fine:   strided view with storage offset 0
  - `.contiguous()` does NOT help: an offset slice like t[4096:4160] is
    contiguous by strides, so contiguous() is a no-op returning the same
    view. `.clone()` or the two-step transfer are the workarounds.
  - The SOURCE tensor is not modified — this is a read-side wrong-address
    bug, distinct from pytorch/pytorch#189563 (destination-offset D2H cast
    corrupting the source) and #189961 (equal gappy strides degrading to a
    raw blit).

Version matrix (this script, same machine):

    torch 2.9.1      BUG
    torch 2.12.1     BUG
    torch 2.13.0     BUG        (latest stable at time of writing)
    2.15.0.dev20260820 nightly  fixed

Same class as pytorch/pytorch#94980 — copy_cast_mps() ignoring
storage_offset — fixed for torch 2.0 by #95093 and evidently regressed
later. On main the D2H copy-cast path was reworked mid-2026 (#184740,
#189572, #189966), which is presumably what fixed this facet too; no stable
release has those changes yet. Kept here as a regression check and as the
reason mtlattn's benchmarks and tests use two-step host transfers when a
slice is involved.

Run: python tests/test_mps_fused_to_bug.py
"""

import torch

OFFSETS = (1, 2, 64, 256, 4096, 65536)
SLICE = 64


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; this bug is MPS-specific.")
        return

    print(f"torch {torch.__version__}")
    torch.manual_seed(0)
    base = torch.randn(1 << 20, dtype=torch.float16, device="mps")
    good = base.to("cpu").to(torch.float32)   # two-step: known correct

    bad = []
    for off in OFFSETS:
        want = good[off:off + SLICE]
        fused = base[off:off + SLICE].to("cpu", torch.float32)
        ok = torch.equal(fused, want)
        if not ok:
            bad.append(off)
            err = (fused - want).abs().max().item()
            zeros = (fused == 0).float().mean().item()
            print(f"  offset {off:>6}: WRONG  max abs err {err:9.3f}   "
                  f"{zeros:4.0%} zeros")
        else:
            print(f"  offset {off:>6}: ok")

        # the workarounds must always hold
        assert torch.equal(base[off:off + SLICE].to("cpu").to(torch.float32),
                           want), "two-step transfer broke: new bug"
        assert torch.equal(base[off:off + SLICE].clone().to("cpu", torch.float32),
                           want), "clone-first transfer broke: new bug"

    if bad:
        print(f"\n*** BUG REPRODUCED at offsets {bad}: fused "
              f".to('cpu', dtype) of an offset MPS view is silently wrong "
              f"(two-step and clone-first transfers are correct). ***")
    else:
        print("\nNo corruption on this build (fixed on nightly >= 2.15.0.dev20260820).")


if __name__ == "__main__":
    main()
