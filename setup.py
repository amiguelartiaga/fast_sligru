from __future__ import annotations

import os

from setuptools import find_packages, setup


def _get_ext_modules_and_cmdclass():
    """Create CUDA extension if a CUDA toolchain is available.

    This project uses PyTorch's C++/CUDA extension build.
    With modern pip (PEP 517/660), setup.py can be evaluated in an isolated
    build env that may not have torch installed yet.

    We therefore:
    - import torch lazily
    - only define ext_modules when torch is importable and CUDA is usable
    """

    try:
        import torch
        from torch.utils.cpp_extension import (  # type: ignore
            BuildExtension,
            CUDAExtension,
            CUDA_HOME,
        )
    except Exception:
        return [], {}

    force_cuda = os.environ.get("FORCE_CUDA", "0") not in ("0", "false", "False", "")
    has_cuda_toolchain = CUDA_HOME is not None
    has_cuda_runtime = bool(getattr(torch.cuda, "is_available", lambda: False)())

    if not (has_cuda_toolchain and (has_cuda_runtime or force_cuda)):
        # Skip building the extension in CPU-only environments.
        return [], {}

    sources = [
        "fast_sligru/csrc/rnns.cpp",
        "fast_sligru/csrc/sligru_kernel.cu",
        "fast_sligru/csrc/ligru_kernel.cu",
    ]

    # Ensure runtime can resolve libtorch shared libraries (libc10.so, libtorch.so, ...)
    # especially when torch is coming from a non-standard prefix.
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

    extra_compile_args = {
        "cxx": ["-O3"],
        "nvcc": ["-O3"],
    }

    extra_link_args = [f"-Wl,-rpath,{torch_lib_dir}"]

    ext_modules = [
        CUDAExtension(
            name="fast_sligru_cpp",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            runtime_library_dirs=[torch_lib_dir],
        )
    ]

    cmdclass = {"build_ext": BuildExtension}
    return ext_modules, cmdclass


ext_modules, cmdclass = _get_ext_modules_and_cmdclass()

setup(
    name="fast_sligru",
    version="0.1.0",
    author="Adel Moumen",
    author_email="adel.moumen@univ-avignon.fr",
    description="A fast CUDA implementation of the SLiGRU model.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    python_requires=">=3.8",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)