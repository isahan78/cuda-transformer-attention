# Google Colab Instructions

## Quick Start (3 Steps)

### Step 1: Open the Notebook in Colab

**Option A: Direct Link**
1. Go to: https://colab.research.google.com/
2. Click **"GitHub"** tab
3. Enter: `isahan78/cuda-transformer-attention`
4. Click on: `notebooks/colab_test.ipynb`

**Option B: Manual Upload**
1. Download `notebooks/colab_test.ipynb` from GitHub
2. Go to https://colab.research.google.com/
3. Click **"Upload"** and select the notebook

### Step 2: Enable GPU

**IMPORTANT:** You MUST enable GPU for this to work!

1. In Colab, click **"Runtime"** → **"Change runtime type"**
2. Under "Hardware accelerator", select **"GPU"** (T4, L4, or V100)
3. Click **"Save"**

### Step 3: Run All Cells

1. Click **"Runtime"** → **"Run all"**
2. Wait for compilation (2-5 minutes)
3. View results!

## What the Notebook Does

The notebook will automatically:
1. ✅ Check GPU availability
2. ✅ Clone this repository
3. ✅ Compile CUDA kernels (takes ~2-5 minutes)
4. ✅ Run correctness tests
5. ✅ Run performance benchmarks
6. ✅ Compare memory usage
7. ✅ Show speedup comparisons

## Expected Output

You should see:
- **GPU detected**: e.g., "Tesla T4" or "Tesla V100"
- **Compilation successful**: After ~2-5 minutes
- **All tests passing**: ✓ marks for each test
- **Benchmark results**: Speedup comparisons (2-5x faster)

## Troubleshooting

### "CUDA not available"
- Make sure you enabled GPU in Runtime settings (Step 2)
- Try Runtime → Restart runtime and run all

### Compilation errors
- This is normal for the first run
- Colab installs necessary packages automatically
- If errors persist, try "Runtime → Restart and run all"

### Out of memory
- Reduce batch size or sequence length in test configurations
- Use T4 or V100 GPU (not CPU)

## Manual Testing (After Notebook Runs)

If you want to test manually, add a new cell:

```python
import torch
from python.cuda_attention import cuda_attention_forward

# Create test inputs
B, H, S, D = 2, 4, 256, 64
Q = torch.randn(B, H, S, D, device='cuda')
K = torch.randn(B, H, S, D, device='cuda')
V = torch.randn(B, H, S, D, device='cuda')

# Run attention
output = cuda_attention_forward(Q, K, V, mode='tiled')
print(f"Output shape: {output.shape}")
print("✅ Success!")
```

## Notes

- **First run**: Takes 2-5 minutes to compile CUDA kernels
- **Subsequent runs**: Much faster (kernels are cached)
- **Free Colab**: May have GPU time limits (~12 hours)
- **Colab Pro**: Faster GPUs and longer sessions

## Need Help?

- Check the notebook output for error messages
- Open an issue on GitHub: https://github.com/isahan78/cuda-transformer-attention/issues
- All kernels fallback to PyTorch reference if CUDA fails

---

**That's it! Just open the notebook in Colab, enable GPU, and run all cells.** 🚀
