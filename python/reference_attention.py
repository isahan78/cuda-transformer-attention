"""
PyTorch Reference Implementation of Transformer Attention

This module provides a pure PyTorch implementation of the attention mechanism
used for correctness validation of CUDA kernels.

Supports:
- Scaled dot-product attention
- Causal masking
- Padding masks
- Numerically stable softmax
"""

import torch
import torch.nn.functional as F
import math


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    is_causal: bool = False,
    scale: float = None
) -> torch.Tensor:
    """
    Reference implementation of scaled dot-product attention.

    Args:
        query: Query tensor of shape [B, H, S_q, D]
        key: Key tensor of shape [B, H, S_k, D]
        value: Value tensor of shape [B, H, S_k, D]
        mask: Optional mask tensor of shape [B, H, S_q, S_k] or broadcastable
              True/1 indicates positions to mask out (set to -inf before softmax)
        is_causal: If True, apply causal (lower triangular) masking
        scale: Scaling factor. If None, uses 1/sqrt(D)

    Returns:
        Output tensor of shape [B, H, S_q, D]

    Algorithm:
        1. Compute attention scores: S = Q @ K^T
        2. Scale: S = S / sqrt(D)
        3. Apply mask (if provided)
        4. Apply softmax: A = softmax(S, dim=-1)
        5. Compute output: O = A @ V
    """
    B, H, S_q, D = query.shape
    _, _, S_k, _ = key.shape

    # Use default scale if not provided
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Step 1: Compute Q @ K^T
    # [B, H, S_q, D] @ [B, H, D, S_k] -> [B, H, S_q, S_k]
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale
    scores = scores * scale

    # Step 3: Apply masks
    if is_causal:
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    if mask is not None:
        # mask should have True/1 for positions to mask out
        scores = scores.masked_fill(mask.bool(), float('-inf'))

    # Step 4: Softmax (numerically stable - PyTorch handles this automatically)
    attn_weights = F.softmax(scores, dim=-1)

    # Handle NaN values that may arise from full rows of -inf
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Step 5: Compute output
    # [B, H, S_q, S_k] @ [B, H, S_k, D] -> [B, H, S_q, D]
    output = torch.matmul(attn_weights, value)

    return output


def naive_attention_components(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    is_causal: bool = False,
    scale: float = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference attention that returns intermediate results for testing.

    Returns:
        tuple: (scores, attn_weights, output)
            - scores: [B, H, S_q, S_k] attention scores after scaling and masking
            - attn_weights: [B, H, S_q, S_k] attention weights after softmax
            - output: [B, H, S_q, D] final output
    """
    B, H, S_q, D = query.shape
    _, _, S_k, _ = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Compute Q @ K^T and scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply masks
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    if mask is not None:
        scores = scores.masked_fill(mask.bool(), float('-inf'))

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Final output
    output = torch.matmul(attn_weights, value)

    return scores, attn_weights, output


def stable_softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax implementation for reference.

    Algorithm:
        1. m = max(x, dim)
        2. e = exp(x - m)
        3. s = sum(e, dim)
        4. return e / s

    Args:
        x: Input tensor
        dim: Dimension along which to compute softmax

    Returns:
        Softmax probabilities
    """
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


if __name__ == "__main__":
    # Simple test
    print("Testing reference attention implementation...")

    B, H, S, D = 2, 4, 8, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Q = torch.randn(B, H, S, D, device=device)
    K = torch.randn(B, H, S, D, device=device)
    V = torch.randn(B, H, S, D, device=device)

    # Test basic attention
    output = reference_attention(Q, K, V)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test causal attention
    output_causal = reference_attention(Q, K, V, is_causal=True)
    print(f"Causal output shape: {output_causal.shape}")

    # Test with components
    scores, weights, output = naive_attention_components(Q, K, V)
    print(f"Scores shape: {scores.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum (should be ~1.0): {weights.sum(dim=-1).mean():.4f}")

    print("\nReference implementation test passed!")
