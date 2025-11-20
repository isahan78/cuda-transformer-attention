"""
Setup script for CUDA Transformer Attention

Builds the CUDA extension using setuptools instead of JIT compilation.
This is more reliable in Google Colab environments.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Source files
cuda_sources = [
    "cuda/attention_qk.cu",
    "cuda/attention_softmax.cu",
    "cuda/attention_av.cu",
    "cuda/attention_fused.cu",
]

cpp_sources = [
    "cpp/attention_binding.cpp",
]

# Compiler flags - use C++14 for compatibility
extra_compile_args = {
    'cxx': ['-O2', '-std=c++14'],
    'nvcc': ['-O2', '-std=c++14', '--expt-relaxed-constexpr'],
}

# Build extension
setup(
    name='cuda_attention',
    version='1.0.0',
    author='Isahan Khan',
    description='CUDA-accelerated transformer attention kernels',
    ext_modules=[
        CUDAExtension(
            name='cuda_attn',
            sources=cuda_sources + cpp_sources,
            extra_compile_args=extra_compile_args,
            include_dirs=['cpp', 'cuda'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.12.0',
    ],
)
