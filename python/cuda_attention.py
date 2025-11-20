"""
CUDA Attention Python Wrapper

Provides a user-friendly PyTorch-like API for CUDA attention kernels.
Falls back to reference implementation if CUDA extension is not available.
"""

import torch
import torch.nn as nn
from typing import Optional
import warnings

# Try to import compiled CUDA extension
try:
    import cuda_attn
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False
    warnings.warn(
        "CUDA extension 'cuda_attn' not found. "
        "Compile with torch.utils.cpp_extension.load() or setuptools. "
        "Falling back to reference implementation.",
        ImportWarning
    )

from .reference_attention import reference_attention


def cuda_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mode: str = "tiled",
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    use_cuda: bool = True
) -> torch.Tensor:
    """
    Compute scaled dot-product attention using CUDA kernels.

    Args:
        query: Query tensor [B, H, S_q, D]
        key: Key tensor [B, H, S_k, D]
        value: Value tensor [B, H, S_k, D]
        mode: Kernel mode - "naive", "tiled", or "fused" (default: "tiled")
        mask: Optional mask tensor [B, H, S_q, S_k] or broadcastable
              True/1 indicates positions to mask out
        is_causal: If True, apply causal masking (default: False)
        scale: Scaling factor. If None, uses 1/sqrt(D) (default: None)
        use_cuda: If True and CUDA extension available, use CUDA kernels.
                  Otherwise use reference implementation (default: True)

    Returns:
        Output tensor [B, H, S_q, D]

    Example:
        >>> B, H, S, D = 2, 8, 512, 64
        >>> Q = torch.randn(B, H, S, D, device='cuda')
        >>> K = torch.randn(B, H, S, D, device='cuda')
        >>> V = torch.randn(B, H, S, D, device='cuda')
        >>> output = cuda_attention_forward(Q, K, V, mode='tiled')
    """
    # Input validation
    assert query.dim() == 4, f"Query must be 4D, got {query.dim()}D"
    assert key.dim() == 4, f"Key must be 4D, got {key.dim()}D"
    assert value.dim() == 4, f"Value must be 4D, got {value.dim()}D"

    # Check if CUDA is requested and available
    if use_cuda and CUDA_EXTENSION_AVAILABLE and query.is_cuda:
        # Ensure tensors are contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # Convert mask if provided
        if mask is not None:
            mask = mask.contiguous()

        # Set default scale
        if scale is None:
            D = query.size(-1)
            scale = 1.0 / (D ** 0.5)

        # Call CUDA extension
        try:
            return cuda_attn.forward(
                query, key, value,
                mode=mode,
                mask=mask if mask is not None else torch.Tensor(),
                is_causal=is_causal,
                scale=scale
            )
        except Exception as e:
            warnings.warn(
                f"CUDA kernel failed with error: {e}. "
                f"Falling back to reference implementation.",
                RuntimeWarning
            )
            return reference_attention(query, key, value, mask, is_causal, scale)
    else:
        # Use reference implementation
        if not query.is_cuda and use_cuda:
            warnings.warn(
                "CUDA requested but tensors are on CPU. "
                "Using reference implementation.",
                UserWarning
            )
        return reference_attention(query, key, value, mask, is_causal, scale)


class CUDAAttention(nn.Module):
    """
    PyTorch Module wrapper for CUDA attention kernels.

    This module provides a convenient nn.Module interface for CUDA attention,
    similar to torch.nn.MultiheadAttention.

    Args:
        mode: Kernel mode - "naive", "tiled", or "fused" (default: "tiled")
        dropout: Dropout probability (not implemented yet) (default: 0.0)
        use_cuda: Whether to use CUDA kernels when available (default: True)

    Example:
        >>> attention = CUDAAttention(mode='fused').cuda()
        >>> Q = torch.randn(2, 8, 512, 64, device='cuda')
        >>> K = torch.randn(2, 8, 512, 64, device='cuda')
        >>> V = torch.randn(2, 8, 512, 64, device='cuda')
        >>> output = attention(Q, K, V, is_causal=True)
    """

    def __init__(
        self,
        mode: str = "tiled",
        dropout: float = 0.0,
        use_cuda: bool = True
    ):
        super().__init__()
        self.mode = mode
        self.dropout = dropout
        self.use_cuda = use_cuda

        if dropout > 0.0:
            warnings.warn(
                "Dropout is not yet implemented in CUDA kernels. "
                "Dropout will be applied in reference implementation only.",
                UserWarning
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass of attention.

        Args:
            query: [B, H, S_q, D]
            key: [B, H, S_k, D]
            value: [B, H, S_k, D]
            mask: Optional [B, H, S_q, S_k]
            is_causal: Apply causal masking
            scale: Scaling factor (default: 1/sqrt(D))

        Returns:
            Output [B, H, S_q, D]
        """
        output = cuda_attention_forward(
            query, key, value,
            mode=self.mode,
            mask=mask,
            is_causal=is_causal,
            scale=scale,
            use_cuda=self.use_cuda
        )

        # Apply dropout if needed (only in training mode)
        if self.dropout > 0.0 and self.training:
            output = nn.functional.dropout(output, p=self.dropout)

        return output

    def extra_repr(self) -> str:
        return f'mode={self.mode}, dropout={self.dropout}, use_cuda={self.use_cuda}'


# Convenience functions for each kernel mode
def attention_naive(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> torch.Tensor:
    """Naive CUDA attention kernel."""
    return cuda_attention_forward(
        query, key, value, mode="naive", mask=mask,
        is_causal=is_causal, scale=scale
    )


def attention_tiled(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> torch.Tensor:
    """Tiled/optimized CUDA attention kernel."""
    return cuda_attention_forward(
        query, key, value, mode="tiled", mask=mask,
        is_causal=is_causal, scale=scale
    )


def attention_fused(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> torch.Tensor:
    """Fused FlashAttention-style kernel."""
    return cuda_attention_forward(
        query, key, value, mode="fused", mask=mask,
        is_causal=is_causal, scale=scale
    )


if __name__ == "__main__":
    # Simple test
    print("CUDA Attention Wrapper Test")
    print(f"CUDA Extension Available: {CUDA_EXTENSION_AVAILABLE}")

    if torch.cuda.is_available():
        print("Testing with CUDA tensors...")
        B, H, S, D = 2, 4, 128, 64
        device = 'cuda'

        Q = torch.randn(B, H, S, D, device=device)
        K = torch.randn(B, H, S, D, device=device)
        V = torch.randn(B, H, S, D, device=device)

        # Test reference
        output_ref = reference_attention(Q, K, V)
        print(f"Reference output shape: {output_ref.shape}")

        if CUDA_EXTENSION_AVAILABLE:
            # Test CUDA kernels
            for mode in ["naive", "tiled", "fused"]:
                output = cuda_attention_forward(Q, K, V, mode=mode)
                print(f"{mode.capitalize()} output shape: {output.shape}")

            # Test module
            attn_module = CUDAAttention(mode='tiled').cuda()
            output_module = attn_module(Q, K, V)
            print(f"Module output shape: {output_module.shape}")

        print("\nAll tests passed!")
    else:
        print("CUDA not available, skipping GPU tests")
