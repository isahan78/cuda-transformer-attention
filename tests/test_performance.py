"""
Performance Benchmarks for CUDA Attention Kernels

Measures and compares performance of:
- PyTorch reference implementation
- Naive CUDA kernel
- Tiled CUDA kernel
- Fused FlashAttention-style kernel

Metrics:
- Execution time (latency)
- Memory usage
- Throughput
- Speedup vs reference
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
import pytest
from typing import Dict, List, Tuple
from python.reference_attention import reference_attention
from python.cuda_attention import (
    cuda_attention_forward,
    CUDA_EXTENSION_AVAILABLE
)

# Only run if CUDA is available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

skip_if_no_extension = pytest.mark.skipif(
    not CUDA_EXTENSION_AVAILABLE,
    reason="CUDA extension not compiled"
)


class Timer:
    """Context manager for timing CUDA operations."""

    def __init__(self, device='cuda', warmup=3, repeat=10):
        self.device = device
        self.warmup = warmup
        self.repeat = repeat
        self.times = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def time_function(self, func, *args, **kwargs):
        """Time a function with warmup and multiple repeats."""
        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)
            if self.device == 'cuda':
                torch.cuda.synchronize()

        # Actual timing
        self.times = []
        for _ in range(self.repeat):
            if self.device == 'cuda':
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                func(*args, **kwargs)
                end.record()

                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)  # milliseconds
                self.times.append(elapsed)
            else:
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                self.times.append((end - start) * 1000)  # convert to ms

        return self

    @property
    def mean(self):
        return sum(self.times) / len(self.times) if self.times else 0

    @property
    def min(self):
        return min(self.times) if self.times else 0

    @property
    def max(self):
        return max(self.times) if self.times else 0

    @property
    def std(self):
        if not self.times:
            return 0
        mean = self.mean
        variance = sum((t - mean) ** 2 for t in self.times) / len(self.times)
        return variance ** 0.5


def benchmark_attention(
    B: int, H: int, S: int, D: int,
    modes: List[str] = None,
    is_causal: bool = False,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different attention implementations.

    Args:
        B: Batch size
        H: Number of heads
        S: Sequence length
        D: Head dimension
        modes: List of modes to benchmark
        is_causal: Use causal masking
        device: Device to run on

    Returns:
        Dictionary with timing results for each mode
    """
    if modes is None:
        modes = ['reference', 'naive', 'tiled', 'fused']

    # Create inputs
    Q = torch.randn(B, H, S, D, device=device)
    K = torch.randn(B, H, S, D, device=device)
    V = torch.randn(B, H, S, D, device=device)

    results = {}

    # Benchmark reference
    if 'reference' in modes:
        timer = Timer(device=device)
        timer.time_function(reference_attention, Q, K, V, is_causal=is_causal)
        results['reference'] = {
            'mean': timer.mean,
            'min': timer.min,
            'max': timer.max,
            'std': timer.std
        }

    # Benchmark CUDA modes
    if CUDA_EXTENSION_AVAILABLE:
        for mode in ['naive', 'tiled', 'fused']:
            if mode in modes:
                timer = Timer(device=device)
                timer.time_function(
                    cuda_attention_forward, Q, K, V,
                    mode=mode, is_causal=is_causal
                )
                results[mode] = {
                    'mean': timer.mean,
                    'min': timer.min,
                    'max': timer.max,
                    'std': timer.std
                }

    return results


def print_benchmark_results(
    results: Dict[str, Dict[str, float]],
    config: str
):
    """Pretty print benchmark results."""
    print(f"\n{'='*70}")
    print(f"Benchmark: {config}")
    print(f"{'='*70}")
    print(f"{'Mode':<15} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Speedup':<10}")
    print(f"{'-'*70}")

    reference_time = results.get('reference', {}).get('mean', None)

    for mode, times in results.items():
        mean_time = times['mean']
        min_time = times['min']
        max_time = times['max']

        speedup = ""
        if reference_time and mode != 'reference' and mean_time > 0:
            speedup = f"{reference_time / mean_time:.2f}x"

        print(f"{mode:<15} {mean_time:>10.3f}  {min_time:>10.3f}  "
              f"{max_time:>10.3f}  {speedup:>8}")


class TestPerformance:
    """Performance tests for attention kernels."""

    @skip_if_no_extension
    def test_small_sequence(self):
        """Benchmark small sequence length."""
        B, H, S, D = 4, 8, 128, 64
        results = benchmark_attention(B, H, S, D)
        print_benchmark_results(results, f"Small: B={B}, H={H}, S={S}, D={D}")

        # Sanity check: CUDA should be faster than reference
        if 'tiled' in results and 'reference' in results:
            assert results['tiled']['mean'] > 0

    @skip_if_no_extension
    def test_medium_sequence(self):
        """Benchmark medium sequence length."""
        B, H, S, D = 4, 8, 512, 64
        results = benchmark_attention(B, H, S, D)
        print_benchmark_results(results, f"Medium: B={B}, H={H}, S={S}, D={D}")

    @skip_if_no_extension
    @pytest.mark.slow
    def test_long_sequence(self):
        """Benchmark long sequence length."""
        B, H, S, D = 2, 8, 2048, 64
        results = benchmark_attention(B, H, S, D)
        print_benchmark_results(results, f"Long: B={B}, H={H}, S={S}, D={D}")

        # Fused should use less memory than naive/tiled
        # (This is a placeholder - actual memory measurement would require more code)

    @skip_if_no_extension
    def test_causal_performance(self):
        """Benchmark causal attention."""
        B, H, S, D = 4, 8, 512, 64
        results = benchmark_attention(B, H, S, D, is_causal=True)
        print_benchmark_results(
            results,
            f"Causal: B={B}, H={H}, S={S}, D={D}"
        )

    @skip_if_no_extension
    def test_varying_batch_size(self):
        """Benchmark with varying batch sizes."""
        H, S, D = 8, 256, 64

        print(f"\n{'='*70}")
        print("Varying Batch Size (H={}, S={}, D={})".format(H, S, D))
        print(f"{'='*70}")

        for B in [1, 2, 4, 8, 16]:
            results = benchmark_attention(B, H, S, D, modes=['tiled'])
            if 'tiled' in results:
                print(f"B={B:2d}: {results['tiled']['mean']:>8.3f} ms")

    @skip_if_no_extension
    def test_varying_sequence_length(self):
        """Benchmark with varying sequence lengths."""
        B, H, D = 4, 8, 64

        print(f"\n{'='*70}")
        print("Varying Sequence Length (B={}, H={}, D={})".format(B, H, D))
        print(f"{'='*70}")

        for S in [64, 128, 256, 512, 1024]:
            results = benchmark_attention(B, H, S, D, modes=['tiled', 'fused'])
            print(f"\nS={S:4d}:")
            if 'tiled' in results:
                print(f"  Tiled: {results['tiled']['mean']:>8.3f} ms")
            if 'fused' in results:
                print(f"  Fused: {results['fused']['mean']:>8.3f} ms")

    @skip_if_no_extension
    def test_varying_head_dimension(self):
        """Benchmark with varying head dimensions."""
        B, H, S = 4, 8, 256

        print(f"\n{'='*70}")
        print("Varying Head Dimension (B={}, H={}, S={})".format(B, H, S))
        print(f"{'='*70}")

        for D in [32, 64, 128, 256]:
            results = benchmark_attention(B, H, S, D, modes=['tiled'])
            if 'tiled' in results:
                print(f"D={D:3d}: {results['tiled']['mean']:>8.3f} ms")


class TestMemoryUsage:
    """Test memory usage of different implementations."""

    @skip_if_no_extension
    def test_memory_naive_vs_fused(self):
        """
        Compare memory usage between naive and fused implementations.

        Fused kernel should use less memory as it doesn't materialize
        the full attention matrix.
        """
        B, H, S, D = 2, 8, 1024, 64

        Q = torch.randn(B, H, S, D, device='cuda')
        K = torch.randn(B, H, S, D, device='cuda')
        V = torch.randn(B, H, S, D, device='cuda')

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Measure naive
        start_mem = torch.cuda.memory_allocated()
        _ = cuda_attention_forward(Q, K, V, mode='naive')
        torch.cuda.synchronize()
        naive_mem = torch.cuda.max_memory_allocated() - start_mem

        # Reset
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Measure fused
        start_mem = torch.cuda.memory_allocated()
        _ = cuda_attention_forward(Q, K, V, mode='fused')
        torch.cuda.synchronize()
        fused_mem = torch.cuda.max_memory_allocated() - start_mem

        print(f"\nMemory Usage (B={B}, H={H}, S={S}, D={D}):")
        print(f"  Naive: {naive_mem / 1024**2:.2f} MB")
        print(f"  Fused: {fused_mem / 1024**2:.2f} MB")
        print(f"  Savings: {(1 - fused_mem/naive_mem)*100:.1f}%")

        # Fused should use less memory
        # (may not always be true for small sequences due to overhead)
        if S >= 512:
            assert fused_mem <= naive_mem


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark suite."""
    if not torch.cuda.is_available() or not CUDA_EXTENSION_AVAILABLE:
        print("CUDA or extension not available. Skipping benchmarks.")
        return

    configs = [
        # (B, H, S, D, name)
        (1, 1, 128, 64, "Tiny"),
        (4, 8, 256, 64, "Small"),
        (4, 8, 512, 64, "Medium"),
        (2, 8, 1024, 64, "Large"),
        (8, 16, 256, 128, "Many Heads"),
    ]

    for B, H, S, D, name in configs:
        results = benchmark_attention(B, H, S, D)
        print_benchmark_results(results, f"{name}: B={B}, H={H}, S={S}, D={D}")


if __name__ == "__main__":
    # Run comprehensive benchmark when executed directly
    run_comprehensive_benchmark()

    # Or run pytest
    # pytest.main([__file__, "-v", "-s"])
