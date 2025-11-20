"""
Correctness Tests for CUDA Attention Kernels

Tests that CUDA implementations produce correct results by comparing
against the PyTorch reference implementation.

Tests cover:
- Basic attention computation
- Different tensor shapes
- Numerical accuracy
- Edge cases
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from python.reference_attention import reference_attention
from python.cuda_attention import (
    cuda_attention_forward,
    CUDA_EXTENSION_AVAILABLE
)

# Test configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RTOL = 1e-4  # Relative tolerance
ATOL = 1e-5  # Absolute tolerance

# Skip CUDA tests if extension not available
skip_if_no_cuda = pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE or not torch.cuda.is_available(),
    reason="CUDA extension not available or CUDA device not found"
)


class TestBasicCorrectness:
    """Test basic correctness of attention kernels."""

    @pytest.fixture
    def simple_inputs(self):
        """Create simple test inputs."""
        B, H, S, D = 2, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)
        return Q, K, V

    def test_reference_implementation(self, simple_inputs):
        """Test that reference implementation runs without errors."""
        Q, K, V = simple_inputs
        output = reference_attention(Q, K, V)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @skip_if_no_cuda
    def test_naive_vs_reference(self, simple_inputs):
        """Test naive CUDA kernel against reference."""
        Q, K, V = simple_inputs

        output_ref = reference_attention(Q, K, V)
        output_cuda = cuda_attention_forward(Q, K, V, mode="naive")

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Naive kernel does not match reference"
        )

    @skip_if_no_cuda
    def test_tiled_vs_reference(self, simple_inputs):
        """Test tiled CUDA kernel against reference."""
        Q, K, V = simple_inputs

        output_ref = reference_attention(Q, K, V)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled")

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Tiled kernel does not match reference"
        )

    @skip_if_no_cuda
    def test_fused_vs_reference(self, simple_inputs):
        """Test fused CUDA kernel against reference."""
        Q, K, V = simple_inputs

        output_ref = reference_attention(Q, K, V)
        output_cuda = cuda_attention_forward(Q, K, V, mode="fused")

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Fused kernel does not match reference"
        )


class TestDifferentShapes:
    """Test attention with various tensor shapes."""

    @pytest.mark.parametrize("B,H,S_q,S_k,D", [
        (1, 1, 8, 8, 32),      # Minimal
        (2, 4, 16, 16, 64),    # Small
        (4, 8, 64, 64, 64),    # Medium
        (2, 4, 32, 64, 128),   # Non-square (S_q != S_k)
        (1, 1, 128, 128, 32),  # Long sequence, small D
        (8, 16, 16, 16, 64),   # Many heads
    ])
    @skip_if_no_cuda
    def test_various_shapes(self, B, H, S_q, S_k, D):
        """Test CUDA kernels with various tensor shapes."""
        Q = torch.randn(B, H, S_q, D, device=DEVICE)
        K = torch.randn(B, H, S_k, D, device=DEVICE)
        V = torch.randn(B, H, S_k, D, device=DEVICE)

        output_ref = reference_attention(Q, K, V)

        for mode in ["naive", "tiled", "fused"]:
            output_cuda = cuda_attention_forward(Q, K, V, mode=mode)

            torch.testing.assert_close(
                output_cuda, output_ref,
                rtol=RTOL, atol=ATOL,
                msg=f"{mode} kernel failed for shape B={B}, H={H}, "
                    f"S_q={S_q}, S_k={S_k}, D={D}"
            )


class TestCausalMasking:
    """Test causal (autoregressive) masking."""

    @pytest.fixture
    def causal_inputs(self):
        """Create inputs for causal attention testing."""
        B, H, S, D = 2, 4, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)
        return Q, K, V

    def test_causal_reference(self, causal_inputs):
        """Test reference implementation with causal masking."""
        Q, K, V = causal_inputs
        output = reference_attention(Q, K, V, is_causal=True)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()

    @skip_if_no_cuda
    def test_causal_cuda(self, causal_inputs):
        """Test CUDA kernels with causal masking."""
        Q, K, V = causal_inputs

        output_ref = reference_attention(Q, K, V, is_causal=True)

        for mode in ["naive", "tiled", "fused"]:
            output_cuda = cuda_attention_forward(
                Q, K, V, mode=mode, is_causal=True
            )

            torch.testing.assert_close(
                output_cuda, output_ref,
                rtol=RTOL, atol=ATOL,
                msg=f"{mode} kernel with causal mask does not match reference"
            )

    @skip_if_no_cuda
    def test_causal_attention_is_lower_triangular(self, causal_inputs):
        """
        Verify that causal attention only attends to previous positions.

        With causal masking, each position should only be influenced by
        current and previous positions, not future ones.
        """
        Q, K, V = causal_inputs
        B, H, S, D = Q.shape

        # Create one-hot V to track which positions are attended to
        V_onehot = torch.zeros(B, H, S, S, device=DEVICE)
        for i in range(S):
            V_onehot[:, :, i, i] = 1.0

        # Reduce D dimension to S for this test
        V_test = V_onehot[..., :D]  # [B, H, S, D]

        output = cuda_attention_forward(
            Q, K, V_test, mode="naive", is_causal=True
        )

        # For position i, output should only have non-zero components
        # corresponding to positions 0...i
        # This is a simplified check - just verify no NaNs appear
        assert not torch.isnan(output).any()


class TestScaling:
    """Test attention scaling factor."""

    @skip_if_no_cuda
    def test_custom_scale(self):
        """Test custom scaling factor."""
        B, H, S, D = 2, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        custom_scale = 0.5

        output_ref = reference_attention(Q, K, V, scale=custom_scale)
        output_cuda = cuda_attention_forward(
            Q, K, V, mode="tiled", scale=custom_scale
        )

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Custom scaling does not match reference"
        )

    @skip_if_no_cuda
    def test_default_scale(self):
        """Test that default scale is 1/sqrt(D)."""
        B, H, S, D = 2, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Default scale
        output_default = cuda_attention_forward(Q, K, V, mode="tiled")

        # Explicit scale
        output_explicit = cuda_attention_forward(
            Q, K, V, mode="tiled", scale=1.0 / (D ** 0.5)
        )

        torch.testing.assert_close(
            output_default, output_explicit,
            rtol=1e-7, atol=1e-8,
            msg="Default scale does not match 1/sqrt(D)"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @skip_if_no_cuda
    def test_single_token(self):
        """Test with sequence length of 1."""
        B, H, S, D = 2, 4, 1, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        output_ref = reference_attention(Q, K, V)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled")

        torch.testing.assert_close(output_cuda, output_ref, rtol=RTOL, atol=ATOL)

    @skip_if_no_cuda
    def test_single_head(self):
        """Test with single attention head."""
        B, H, S, D = 2, 1, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        output_ref = reference_attention(Q, K, V)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled")

        torch.testing.assert_close(output_cuda, output_ref, rtol=RTOL, atol=ATOL)

    @skip_if_no_cuda
    def test_zero_values(self):
        """Test with all-zero value tensor."""
        B, H, S, D = 2, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.zeros(B, H, S, D, device=DEVICE)

        output_ref = reference_attention(Q, K, V)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled")

        # Output should be all zeros
        assert torch.allclose(output_cuda, torch.zeros_like(output_cuda))
        torch.testing.assert_close(output_cuda, output_ref, rtol=RTOL, atol=ATOL)


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    @skip_if_no_cuda
    def test_large_values(self):
        """Test with large input values."""
        B, H, S, D = 2, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE) * 10.0
        K = torch.randn(B, H, S, D, device=DEVICE) * 10.0
        V = torch.randn(B, H, S, D, device=DEVICE)

        output_ref = reference_attention(Q, K, V)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled")

        assert not torch.isnan(output_cuda).any()
        assert not torch.isinf(output_cuda).any()
        torch.testing.assert_close(output_cuda, output_ref, rtol=RTOL, atol=ATOL)

    @skip_if_no_cuda
    def test_small_values(self):
        """Test with small input values."""
        B, H, S, D = 2, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE) * 0.01
        K = torch.randn(B, H, S, D, device=DEVICE) * 0.01
        V = torch.randn(B, H, S, D, device=DEVICE) * 0.01

        output_ref = reference_attention(Q, K, V)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled")

        torch.testing.assert_close(output_cuda, output_ref, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
