from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root_dir = Path(__file__).parent.resolve()
cutlass_include = [
    root_dir / 'cutlass/include',
    root_dir / 'cutlass/tools/util/include',
    root_dir / 'cutlass/examples/common'
]

ext_modules = [
    CUDAExtension(
        "cuposit._CUDA",
        sources=["cusrc/bspgemm.cu"],
        include_dirs=cutlass_include,
        extra_compile_args={
            "cxx": [
                "-g",
                "-w",
                "-O3",
                "-DPy_LIMITED_API=0x03090000",
                "-DTORCH_TARGET_VERSION=0x020a000000000000",
            ],
            "nvcc": ["-O3", "-w", "--use_fast_math", "-lineinfo"],
        },
        extra_link_args=["-Wl,--no-as-needed", "-lcuda"],
    )
]

setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch>=2.10.0',
    ],
    python_requires='>=3.9',
    zip_safe=False,
)