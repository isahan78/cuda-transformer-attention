# CUDA Transformer Attention Kernel

A from-scratch CUDA implementation of Transformer attention with progressive optimizations, exposed as a PyTorch extension. This project demonstrates how to build, optimize, and integrate custom CUDA kernels into PyTorch.

## Overview

This project implements **three progressively optimized versions** of the scaled dot-product attention mechanism:

1. **Naive CUDA Kernel** - Straightforward implementation focusing on correctness
2. **Tiled Shared-Memory Kernel** - Optimized memory access patterns using shared memory
3. **Fused FlashAttention-Style Kernel** - Memory-efficient one-pass computation avoiding full attention matrix materialization

## Features

- Complete forward pass of attention: Q×K^T → Softmax → Attention×V
- Support for causal (autoregressive) masking
- Support for custom padding and attention masks
- Numerically stable softmax implementation
- Progressive optimizations demonstrating CUDA performance techniques
- Comprehensive test suite (correctness, performance, masking)
- PyTorch C++ extension integration
- Google Colab notebook for easy GPU testing
- Benchmarking tools comparing all implementations

## Project Structure

```
cuda-transformer-attention/
│
├── cuda/                              # CUDA kernel implementations
│   ├── attention_qk.cu               # Q×K^T computation (naive + tiled)
│   ├── attention_softmax.cu          # Numerically stable softmax
│   ├── attention_av.cu               # Attention×V multiplication
│   ├── attention_fused.cu            # FlashAttention-style fused kernel
│   └── utils.cuh                     # CUDA utilities and helpers
│
├── cpp/                               # C++ PyTorch bindings
│   ├── attention_binding.cpp         # PyBind11 interface
│   └── utils.h                       # C++ utilities
│
├── python/                            # Python interface
│   ├── __init__.py                   # Package initialization
│   ├── cuda_attention.py             # Main Python API
│   └── reference_attention.py        # PyTorch reference implementation
│
├── tests/                             # Test suite
│   ├── test_correctness.py           # Correctness validation
│   ├── test_performance.py           # Performance benchmarks
│   └── test_masks.py                 # Masking tests
│
├── notebooks/                         # Jupyter notebooks
│   └── colab_test.ipynb              # Google Colab execution notebook
│
├── README.md                          # This file
├── cuda_attention_project_plan.md    # Detailed project plan
└── .gitignore                        # Git ignore rules
```

## Requirements

- Python 3.8+
- PyTorch 1.12+ with CUDA support
- CUDA Toolkit 11.0+ (12.0+ recommended)
- C++ compiler with C++14 support
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)

## Installation

### Option 1: JIT Compilation (Recommended for Testing)

Compile the extension on-the-fly using PyTorch's JIT compilation:

```python
from torch.utils.cpp_extension import load

cuda_attn = load(
    name="cuda_attn",
    sources=[
        "cuda/attention_qk.cu",
        "cuda/attention_softmax.cu",
        "cuda/attention_av.cu",
        "cuda/attention_fused.cu",
        "cpp/attention_binding.cpp"
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True
)
```

### Option 2: Install as Package (TODO)

```bash
pip install -e .
```

### Option 3: Google Colab

Open `notebooks/colab_test.ipynb` in Google Colab and run all cells. The notebook handles compilation and testing automatically.

## Usage

### Basic Usage

```python
import torch
from python.cuda_attention import cuda_attention_forward

# Create input tensors [Batch, Heads, Sequence, Dimension]
B, H, S, D = 4, 8, 512, 64
Q = torch.randn(B, H, S, D, device='cuda')
K = torch.randn(B, H, S, D, device='cuda')
V = torch.randn(B, H, S, D, device='cuda')

# Compute attention with tiled kernel (default)
output = cuda_attention_forward(Q, K, V, mode="tiled")

# Or use specific kernel modes
output_naive = cuda_attention_forward(Q, K, V, mode="naive")
output_fused = cuda_attention_forward(Q, K, V, mode="fused")
```

### With Causal Masking

```python
# For autoregressive/causal attention (e.g., GPT-style)
output = cuda_attention_forward(
    Q, K, V,
    mode="fused",
    is_causal=True
)
```

### With Custom Masks

```python
# Create custom attention mask [B, H, S_q, S_k]
# 1/True = masked (set to -inf), 0/False = not masked
mask = torch.zeros(B, H, S, S, device='cuda')
mask[:, :, :, -10:] = 1  # Mask last 10 positions

output = cuda_attention_forward(Q, K, V, mask=mask)
```

### Using the Module Interface

```python
from python.cuda_attention import CUDAAttention

# Create attention module
attention = CUDAAttention(mode='fused').cuda()

# Use like any PyTorch module
output = attention(Q, K, V, is_causal=True)
```

### Using Reference Implementation

```python
from python.reference_attention import reference_attention

# Pure PyTorch implementation for validation
output_ref = reference_attention(Q, K, V, is_causal=True)
```

## Kernel Modes

### Naive (`mode="naive"`)
- **Purpose**: Correctness baseline
- **Strategy**: Each thread computes one output element
- **Pros**: Simple, easy to understand and debug
- **Cons**: Poor memory access patterns, many global memory accesses
- **Best for**: Debugging, understanding attention mechanics

### Tiled (`mode="tiled"`)
- **Purpose**: Production performance
- **Strategy**: Shared memory tiling to improve memory access patterns
- **Pros**: Significant speedup over naive, good balance of speed and complexity
- **Cons**: Still materializes full attention matrix
- **Best for**: Most production use cases with moderate sequence lengths

### Fused (`mode="fused"`)
- **Purpose**: Memory-efficient attention for long sequences
- **Strategy**: FlashAttention-style kernel avoiding full attention matrix
- **Pros**: O(N) memory instead of O(N²), excellent for long sequences
- **Cons**: More complex implementation, may be slower for short sequences
- **Best for**: Long sequences (S > 1024), memory-constrained scenarios

## Running Tests

```bash
# Run all correctness tests
pytest tests/test_correctness.py -v

# Run performance benchmarks
pytest tests/test_performance.py -v -s

# Run masking tests
pytest tests/test_masks.py -v

# Run all tests
pytest tests/ -v
```

## Performance Benchmarking

```python
from tests.test_performance import run_comprehensive_benchmark

# Run full benchmark suite
run_comprehensive_benchmark()
```

Expected speedups (compared to PyTorch reference):
- Naive: 1-2x
- Tiled: 2-5x (depending on hardware and sequence length)
- Fused: 2-8x for long sequences (memory-bound workloads)

## Algorithm Overview

### Standard Three-Stage Attention

```python
# Stage 1: Compute attention scores
scores = Q @ K.T / sqrt(d_k)  # [B, H, S_q, S_k]

# Stage 2: Apply softmax
attn_weights = softmax(scores, dim=-1)  # [B, H, S_q, S_k]

# Stage 3: Compute output
output = attn_weights @ V  # [B, H, S_q, D]
```

### Fused Attention (FlashAttention-style)

Avoids materializing the `[S_q, S_k]` attention matrix by:
1. Processing in blocks/tiles
2. Maintaining running statistics for stable softmax
3. Fusing softmax and attention×V into single pass
4. Memory complexity: O(S×D) instead of O(S²)

## Development

### Code Organization

- **CUDA kernels** (`cuda/`): Low-level GPU implementations
- **C++ bindings** (`cpp/`): PyTorch integration via PyBind11
- **Python API** (`python/`): High-level user interface
- **Tests** (`tests/`): Validation and benchmarking

### Building Locally

Ensure CUDA toolkit is installed and `nvcc` is in your PATH:

```bash
nvcc --version  # Should show CUDA 11.0+
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

Then compile using JIT or setuptools.

### Adding New Kernels

1. Implement CUDA kernel in `cuda/`
2. Add launch function with `extern "C"`
3. Bind in `cpp/attention_binding.cpp`
4. Expose via Python API in `python/cuda_attention.py`
5. Add tests in `tests/`

## Testing on Google Colab

1. Open `notebooks/colab_test.ipynb` in Google Colab
2. Ensure GPU is enabled: Runtime → Change runtime type → GPU
3. Run all cells
4. View compilation, tests, and benchmarks

The notebook includes:
- GPU verification
- Extension compilation
- Correctness tests
- Performance benchmarks
- Memory usage analysis

## Limitations and Future Work

**Current Limitations:**
- Forward pass only (no backward/autograd)
- Float32 only (no FP16/BF16)
- No dropout support in CUDA kernels
- Limited to self-attention (Q, K, V same sequence length support)

**Future Enhancements:**
- [ ] Backward pass implementation
- [ ] FP16/BF16 support for faster inference
- [ ] Multi-query attention (MQA) and grouped-query attention (GQA)
- [ ] Dropout in CUDA kernels
- [ ] Better auto-tuning of tile sizes
- [ ] Integration with PyTorch Autograd

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Further optimizations
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Inspired by the FlashAttention papers and implementation
- Built using PyTorch's excellent C++/CUDA extension API
- Developed as an educational project to understand CUDA optimization techniques

## Contributing

This is an educational project. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Use as a learning resource for CUDA programming

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ to understand CUDA and attention mechanisms**
