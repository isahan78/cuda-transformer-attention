"""
CUDA Transformer Attention Package

This package provides CUDA-accelerated attention kernels with multiple
optimization levels:
- Naive CUDA implementation
- Tiled shared-memory optimization
- Fused FlashAttention-style kernel

Fallback to PyTorch reference implementation when CUDA extension is not available.
"""

from .reference_attention import (
    reference_attention,
    naive_attention_components,
    stable_softmax_reference
)

__version__ = "0.1.0"
__all__ = [
    "reference_attention",
    "naive_attention_components",
    "stable_softmax_reference",
]

# Try to import CUDA extension when available
try:
    from .cuda_attention import cuda_attention_forward
    __all__.append("cuda_attention_forward")
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    import warnings
    warnings.warn(
        "CUDA extension not available. Using reference implementation only. "
        "To build CUDA extension, compile with PyTorch cpp_extension.",
        ImportWarning
    )
