# PROJECT_PLAN.md — CUDA Transformer Attention Kernel

## Project Overview
This project implements a **CUDA-optimized Transformer attention kernel** from scratch and integrates it as a **PyTorch C++/CUDA extension**. It includes three progressively optimized versions:

1. Naive attention kernel
2. Tiled shared-memory optimized kernel
3. Fused FlashAttention-style kernel (no S×S matrix)

The code is developed locally using **Claude Code + GitHub** and executed in **Google Colab** for GPU support.

---

## Goals
- Implement the full forward pass of attention in CUDA:
  - QKᵀ
  - Softmax
  - Attention × V

- Build:
  - naive
  - optimized
  - fused (FlashAttention-like) versions

- Expose the kernels via PyTorch C++ extensions
- Provide tests, benchmarks, and documentation
- Run the entire project on Google Colab with an NVIDIA GPU

---

## Project Structure
```
cuda-transformer-attention/
│
├── cuda/
│   ├── attention_qk.cu
│   ├── attention_softmax.cu
│   ├── attention_av.cu
│   ├── attention_fused.cu
│   └── utils.cuh
│
├── cpp/
│   ├── attention_binding.cpp
│   └── utils.h
│
├── python/
│   ├── __init__.py
│   ├── cuda_attention.py
│   └── reference_attention.py
│
├── tests/
│   ├── test_correctness.py
│   ├── test_performance.py
│   └── test_masks.py
│
├── notebooks/
│   └── colab_test.ipynb
│
├── README.md
├── PROJECT_PLAN.md
└── .gitignore
```

---

## File Responsibilities
### cuda/attention_qk.cu
- Computes QKᵀ
- Implements naive & tiled versions
- Applies scale (1/√D)
- Handles optional causal/padding masks

### cuda/attention_softmax.cu
- Row-wise numerically stable softmax
- Shared memory reductions

### cuda/attention_av.cu
- Computes Attention × V
- Naive → tiled shared-memory versions

### cuda/attention_fused.cu
- FlashAttention-style fused kernel
- Streams over K/V blocks
- Maintains running softmax stats
- Avoids allocating the full [S × S] matrix

### cpp/attention_binding.cpp
- PyBind11 interface
- Defines `cuda_attention_forward()`

### python/cuda_attention.py
- Python API wrapper for CUDA kernels
- Fallback to reference implementation

### python/reference_attention.py
- Pure PyTorch baseline for correctness comparison

### tests/
Contains:
- correctness tests
- mask tests
- performance tests

### notebooks/colab_test.ipynb
- Google Colab execution notebook
- Installs CUDA toolkit
- Compiles PyTorch extension
- Runs tests & benchmarks

---

## Implementation Phases
### Phase 1 — Repo Initialization
Claude should:
- create folder structure
- add `.gitignore`
- write README shell
- write this project plan
- create empty source files

### Phase 2 — PyTorch Reference Implementation
In `reference_attention.py`:
- naive QKᵀ
- stable softmax
- attention × V
- support masks + causal

Used to validate correctness.

### Phase 3 — Naive CUDA Kernels
Implement three minimal CUDA kernels:
1. QKᵀ
2. softmax
3. Attn × V

Requirements:
- Each thread computes 1 output element
- No shared memory
- Correctness > performance

### Phase 4 — PyTorch Extension Integration
Add PyBind11 wrapper and compile.

### Phase 5 — Shared Memory Optimization
Add tiling for:
- QKᵀ
- Softmax
- Attn × V

Focus on global memory access patterns.

### Phase 6 — FlashAttention-Style Fused Kernel
Implement:
- Block-level K/V streaming
- Running max and sum for stable softmax
- No [S×S] allocations
- One-pass computation

### Phase 7 — Tests & Benchmarks
- Compare CUDA vs PyTorch
- Benchmark naive vs tiled vs fused
- Measure latency & memory usage

### Phase 8 — Final Documentation
Claude generates:
- README
- usage examples
- API descriptions
- performance charts (scripts only)

---

## Google Colab Execution Instructions
Notebook must:
```
!apt-get update
!apt-get install -y cuda-toolkit-12-1

!git clone https://github.com/<your_username>/cuda-transformer-attention
%cd cuda-transformer-attention

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
    verbose=True
)

!pytest tests/test_correctness.py
```

---

## Claude Code Execution Instructions
When loaded in Claude Code, it must:
1. Create all directories
2. Create all placeholder files
3. Implement each phase step-by-step
4. Keep commits atomic
5. Push code to GitHub when requested
6. Ensure Colab notebook runs successfully
7. Generate clean, maintainable CUDA + C++ code

---

**End of PROJECT_PLAN.md**

