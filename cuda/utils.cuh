/**
 * CUDA Utilities for Attention Kernels
 *
 * Common helper functions, macros, and constants used across
 * all CUDA attention kernel implementations.
 */

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel error checking
#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Common constants
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Tile sizes for shared memory optimizations
#define TILE_SIZE_16 16
#define TILE_SIZE_32 32
#define TILE_SIZE_64 64

/**
 * Device function: Warp-level reduction (sum)
 * Uses warp shuffle instructions for efficient reduction
 */
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Device function: Warp-level reduction (max)
 */
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Device function: Block-level reduction (sum)
 * Reduces all values in a block to a single sum
 */
template<int BLOCK_SIZE>
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Each warp performs partial reduction
    val = warpReduceSum(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;

    // Final reduce within first warp
    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

/**
 * Device function: Block-level reduction (max)
 */
template<int BLOCK_SIZE>
__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : -FLT_MAX;

    if (wid == 0) {
        val = warpReduceMax(val);
    }

    return val;
}

/**
 * Device function: Safe exponential (handles -inf gracefully)
 */
__device__ __forceinline__ float safe_exp(float x) {
    // exp(-inf) = 0, avoid NaN
    return (x == -INFINITY) ? 0.0f : expf(x);
}

/**
 * Device function: 2D index calculation
 * Converts 2D coordinates to 1D array index
 */
__device__ __forceinline__ int idx2d(int row, int col, int num_cols) {
    return row * num_cols + col;
}

/**
 * Device function: 4D index calculation for [B, H, S, D] tensors
 */
__device__ __forceinline__ int idx4d(int b, int h, int s, int d,
                                      int H, int S, int D) {
    return b * (H * S * D) + h * (S * D) + s * D + d;
}

/**
 * Device function: Check if index is within causal mask
 * Returns true if position (i, j) should be masked in causal attention
 */
__device__ __forceinline__ bool is_causal_masked(int row, int col) {
    return col > row;
}

/**
 * Host function: Calculate grid and block dimensions
 */
inline dim3 calculateGrid2D(int rows, int cols, int block_size) {
    dim3 grid(
        (cols + block_size - 1) / block_size,
        (rows + block_size - 1) / block_size
    );
    return grid;
}

/**
 * Host function: Get optimal block size for 1D kernel
 */
inline int getOptimalBlockSize1D(int size) {
    if (size >= 1024) return 1024;
    if (size >= 512) return 512;
    if (size >= 256) return 256;
    if (size >= 128) return 128;
    if (size >= 64) return 64;
    return 32;
}

#endif // CUDA_UTILS_CUH
