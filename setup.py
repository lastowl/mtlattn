"""
mtlattn: fused variable-length attention (forward) for Apple Silicon.

Build mirrors mtlgemm: .metal -> .air -> mtlattn.metallib via xcrun, and an
Objective-C++ pybind extension linked against torch, with the metallib
installed alongside the .so.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")


class MetalBuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".mm")
        original_spawn = self.compiler.spawn

        def patched_spawn(cmd, **kwargs):
            new_cmd = list(cmd)
            for i, arg in enumerate(new_cmd):
                if arg.endswith(".mm") and i > 0 and new_cmd[i - 1] == "-c":
                    new_cmd.insert(i, "objective-c++")
                    new_cmd.insert(i, "-x")
                    break
            return original_spawn(new_cmd, **kwargs)

        self.compiler.spawn = patched_spawn

        build_temp = os.path.join(self.build_temp, "metal")
        os.makedirs(build_temp, exist_ok=True)

        # Portable simdgroup kernel (metal3.1, runs on M1+). MPP kernel is
        # built separately at metal4.0 below.
        air_files = []
        for src in glob.glob(os.path.join(SRC_DIR, "*.metal")):
            if os.path.basename(src) == "attn_mpp.metal":
                continue
            air = os.path.join(build_temp, os.path.basename(src).replace(".metal", ".air"))
            subprocess.check_call(["xcrun", "-sdk", "macosx", "metal", "-c", src, "-o", air, "-std=metal3.1", "-O2"])
            air_files.append(air)
        metallib = os.path.join(build_temp, "mtlattn.metallib")
        subprocess.check_call(["xcrun", "-sdk", "macosx", "metallib"] + air_files + ["-o", metallib])

        # Optional Metal-4 MPP kernel (M5 Neural Accelerator, macOS 26.2+ at
        # runtime). Best-effort: skip if this toolchain can't build metal4.0.
        mpp_metallib = None
        mpp_src = os.path.join(SRC_DIR, "attn_mpp.metal")
        if os.path.exists(mpp_src):
            try:
                mpp_air = os.path.join(build_temp, "attn_mpp.air")
                subprocess.check_call(["xcrun", "-sdk", "macosx", "metal", "-c", mpp_src, "-o", mpp_air, "-std=metal4.0", "-O3"])
                mpp_metallib = os.path.join(build_temp, "mtlattn_mpp.metallib")
                subprocess.check_call(["xcrun", "-sdk", "macosx", "metallib", mpp_air, "-o", mpp_metallib])
            except subprocess.CalledProcessError:
                print("  mtlattn: metal4.0 MPP kernel not buildable on this toolchain; M5 fast-path disabled")
                mpp_metallib = None

        for ext in self.extensions:
            if hasattr(ext, "_resolve_torch"):
                ext._resolve_torch()
        super().build_extensions()

        for ext in self.extensions:
            ext_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
            os.makedirs(ext_dir, exist_ok=True)
            shutil.copy2(metallib, os.path.join(ext_dir, "mtlattn.metallib"))
            if mpp_metallib:
                shutil.copy2(mpp_metallib, os.path.join(ext_dir, "mtlattn_mpp.metallib"))


class LazyMetalExtension(Extension):
    def __init__(self, *args, **kwargs):
        self._torch_resolved = False
        super().__init__(*args, **kwargs)

    def _resolve_torch(self):
        if self._torch_resolved:
            return
        self._torch_resolved = True
        import torch.utils.cpp_extension as cpp_ext

        self.include_dirs.extend(cpp_ext.include_paths())
        lib_paths = cpp_ext.library_paths()
        self.library_dirs.extend(lib_paths)
        for p in lib_paths:
            self.extra_link_args.append("-Wl,-rpath," + p)


ext = LazyMetalExtension(
    name="mtlattn._C",
    sources=[os.path.join("src", "ext.mm")],
    depends=glob.glob(os.path.join(SRC_DIR, "*")),
    libraries=["c10", "torch", "torch_cpu", "torch_python"],
    extra_compile_args=[
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fobjc-arc",
        "-DTORCH_EXTENSION_NAME=_C",
    ],
    extra_link_args=[
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "MetalPerformancePrimitives",
        "-Wl,-rpath,@loader_path",
    ],
    language="objc++",
)

setup(
    name="mtlattn",
    version="0.1.0",
    packages=["mtlattn"],
    ext_modules=[ext],
    cmdclass={"build_ext": MetalBuildExt},
    install_requires=["torch"],
)
