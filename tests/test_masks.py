"""
Mask Tests for CUDA Attention Kernels

Tests various masking scenarios:
- Causal masks (autoregressive)
- Padding masks
- Custom attention masks
- Mask broadcasting
- Combined masks
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RTOL = 1e-4
ATOL = 1e-5

skip_if_no_cuda = pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE or not torch.cuda.is_available(),
    reason="CUDA extension not available or CUDA device not found"
)


class TestCausalMask:
    """Test causal (autoregressive) masking."""

    @pytest.fixture
    def inputs(self):
        B, H, S, D = 2, 4, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)
        return Q, K, V

    def test_causal_mask_shape(self, inputs):
        """Test that causal masking produces correct output shape."""
        Q, K, V = inputs
        output = reference_attention(Q, K, V, is_causal=True)
        assert output.shape == Q.shape

    @skip_if_no_cuda
    def test_causal_consistency(self, inputs):
        """Test causal masking consistency across implementations."""
        Q, K, V = inputs

        output_ref = reference_attention(Q, K, V, is_causal=True)

        for mode in ["naive", "tiled", "fused"]:
            output_cuda = cuda_attention_forward(
                Q, K, V, mode=mode, is_causal=True
            )
            torch.testing.assert_close(
                output_cuda, output_ref,
                rtol=RTOL, atol=ATOL,
                msg=f"{mode} causal mask differs from reference"
            )

    @skip_if_no_cuda
    def test_causal_vs_manual_mask(self, inputs):
        """Test that is_causal matches manual causal mask."""
        Q, K, V = inputs
        B, H, S, D = Q.shape

        # Create manual causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=DEVICE),
            diagonal=1
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
        causal_mask = causal_mask.expand(B, H, S, S)

        output_builtin = cuda_attention_forward(
            Q, K, V, mode="tiled", is_causal=True
        )
        output_manual = cuda_attention_forward(
            Q, K, V, mode="tiled", mask=causal_mask
        )

        torch.testing.assert_close(
            output_builtin, output_manual,
            rtol=RTOL, atol=ATOL,
            msg="Built-in causal differs from manual mask"
        )


class TestPaddingMask:
    """Test padding masks for variable-length sequences."""

    @pytest.fixture
    def inputs_with_padding(self):
        """Create inputs with simulated padding."""
        B, H, S, D = 2, 4, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Create padding mask: last 8 positions are padding
        mask = torch.zeros(B, H, S, S, device=DEVICE)
        mask[:, :, :, -8:] = 1  # Mask out last 8 key positions

        return Q, K, V, mask

    def test_padding_mask_reference(self, inputs_with_padding):
        """Test reference implementation with padding mask."""
        Q, K, V, mask = inputs_with_padding
        output = reference_attention(Q, K, V, mask=mask)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()

    @skip_if_no_cuda
    def test_padding_mask_cuda(self, inputs_with_padding):
        """Test CUDA kernels with padding mask."""
        Q, K, V, mask = inputs_with_padding

        output_ref = reference_attention(Q, K, V, mask=mask)

        for mode in ["naive", "tiled", "fused"]:
            output_cuda = cuda_attention_forward(
                Q, K, V, mode=mode, mask=mask
            )
            torch.testing.assert_close(
                output_cuda, output_ref,
                rtol=RTOL, atol=ATOL,
                msg=f"{mode} with padding mask differs from reference"
            )

    @skip_if_no_cuda
    def test_full_padding(self):
        """Test with fully padded sequences (all positions masked)."""
        B, H, S, D = 2, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Mask all positions
        mask = torch.ones(B, H, S, S, device=DEVICE)

        output_ref = reference_attention(Q, K, V, mask=mask)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", mask=mask)

        # Output should be zeros (or very close to zero)
        assert torch.allclose(output_cuda, torch.zeros_like(output_cuda), atol=1e-6)
        torch.testing.assert_close(output_cuda, output_ref, rtol=RTOL, atol=ATOL)


class TestCustomMasks:
    """Test custom attention masks."""

    @skip_if_no_cuda
    def test_block_diagonal_mask(self):
        """Test block-diagonal attention pattern."""
        B, H, S, D = 2, 4, 64, 64
        block_size = 16

        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Create block-diagonal mask
        mask = torch.ones(S, S, device=DEVICE)
        for i in range(0, S, block_size):
            mask[i:i+block_size, i:i+block_size] = 0
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, S, S)

        output_ref = reference_attention(Q, K, V, mask=mask)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", mask=mask)

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Block-diagonal mask failed"
        )

    @skip_if_no_cuda
    def test_local_attention_mask(self):
        """Test local attention (only attend to nearby positions)."""
        B, H, S, D = 2, 4, 64, 64
        window_size = 8

        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Create local attention mask (band diagonal)
        mask = torch.ones(S, S, device=DEVICE)
        for i in range(S):
            start = max(0, i - window_size)
            end = min(S, i + window_size + 1)
            mask[i, start:end] = 0
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, S, S)

        output_ref = reference_attention(Q, K, V, mask=mask)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", mask=mask)

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Local attention mask failed"
        )

    @skip_if_no_cuda
    def test_random_mask(self):
        """Test with random attention mask."""
        B, H, S, D = 2, 4, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Random mask (50% masked)
        mask = (torch.rand(B, H, S, S, device=DEVICE) > 0.5).float()

        output_ref = reference_attention(Q, K, V, mask=mask)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", mask=mask)

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Random mask failed"
        )


class TestMaskBroadcasting:
    """Test mask broadcasting capabilities."""

    @skip_if_no_cuda
    def test_broadcast_batch_dimension(self):
        """Test mask broadcasting over batch dimension."""
        B, H, S, D = 4, 4, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Mask with batch dimension = 1 (should broadcast)
        mask = torch.randint(0, 2, (1, H, S, S), device=DEVICE).float()

        output_ref = reference_attention(Q, K, V, mask=mask)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", mask=mask)

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Batch broadcasting failed"
        )

    @skip_if_no_cuda
    def test_broadcast_head_dimension(self):
        """Test mask broadcasting over head dimension."""
        B, H, S, D = 2, 8, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Mask with head dimension = 1 (should broadcast)
        mask = torch.randint(0, 2, (B, 1, S, S), device=DEVICE).float()

        output_ref = reference_attention(Q, K, V, mask=mask)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", mask=mask)

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Head broadcasting failed"
        )

    @skip_if_no_cuda
    def test_broadcast_both_dimensions(self):
        """Test mask broadcasting over both batch and head dimensions."""
        B, H, S, D = 4, 8, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Mask with both batch and head dimensions = 1
        mask = torch.randint(0, 2, (1, 1, S, S), device=DEVICE).float()

        output_ref = reference_attention(Q, K, V, mask=mask)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", mask=mask)

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Broadcast both dimensions failed"
        )


class TestCombinedMasks:
    """Test combinations of different mask types."""

    @skip_if_no_cuda
    def test_causal_and_padding(self):
        """Test causal mask combined with padding mask."""
        B, H, S, D = 2, 4, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Padding mask for last 8 positions
        padding_mask = torch.zeros(B, H, S, S, device=DEVICE)
        padding_mask[:, :, :, -8:] = 1

        # Reference combines causal + padding
        output_ref = reference_attention(
            Q, K, V, mask=padding_mask, is_causal=True
        )

        # CUDA should do the same
        output_cuda = cuda_attention_forward(
            Q, K, V, mode="tiled", mask=padding_mask, is_causal=True
        )

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Causal + padding mask failed"
        )

    @skip_if_no_cuda
    def test_causal_different_lengths(self):
        """Test causal mask with different Q and K sequence lengths."""
        B, H, S_q, S_k, D = 2, 4, 16, 32, 64
        Q = torch.randn(B, H, S_q, D, device=DEVICE)
        K = torch.randn(B, H, S_k, D, device=DEVICE)
        V = torch.randn(B, H, S_k, D, device=DEVICE)

        # Causal mask for non-square attention
        output_ref = reference_attention(Q, K, V, is_causal=True)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", is_causal=True)

        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Causal with S_q != S_k failed"
        )


class TestMaskEdgeCases:
    """Test edge cases for masking."""

    @skip_if_no_cuda
    def test_no_mask(self):
        """Test that no mask gives same result as all-zeros mask."""
        B, H, S, D = 2, 4, 32, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        output_no_mask = cuda_attention_forward(Q, K, V, mode="tiled")

        # All-zeros mask (nothing masked)
        mask = torch.zeros(B, H, S, S, device=DEVICE)
        output_with_mask = cuda_attention_forward(
            Q, K, V, mode="tiled", mask=mask
        )

        torch.testing.assert_close(
            output_no_mask, output_with_mask,
            rtol=1e-6, atol=1e-7,
            msg="No mask differs from all-zeros mask"
        )

    @skip_if_no_cuda
    def test_single_unmasked_position(self):
        """Test with only one position unmasked per query."""
        B, H, S, D = 2, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE)
        K = torch.randn(B, H, S, D, device=DEVICE)
        V = torch.randn(B, H, S, D, device=DEVICE)

        # Mask all except diagonal
        mask = torch.ones(S, S, device=DEVICE)
        mask.fill_diagonal_(0)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, S, S)

        output_ref = reference_attention(Q, K, V, mask=mask)
        output_cuda = cuda_attention_forward(Q, K, V, mode="tiled", mask=mask)

        # With only diagonal unmasked, output should equal V (approximately)
        # (not exactly due to softmax, but close)
        torch.testing.assert_close(
            output_cuda, output_ref,
            rtol=RTOL, atol=ATOL,
            msg="Single unmasked position failed"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
