/**
 * CUDA Kernels for Q @ K^T Computation
 *
 * Implements:
 * 1. Naive kernel: each thread computes one output element
 * 2. Tiled kernel: uses shared memory for improved memory access patterns
 *
 * Input shapes:
 *   Q: [B, H, S_q, D]
 *   K: [B, H, S_k, D]
 * Output shape:
 *   scores: [B, H, S_q, S_k]
 *
 * Applies scaling by 1/sqrt(D) and optional causal masking.
 */

#include "utils.cuh"

/**
 * Naive QK^T Kernel
 *
 * Each thread computes one element of the output matrix.
 * Thread (i, j) computes scores[b, h, i, j] = Q[b,h,i,:] @ K[b,h,j,:]
 *
 * Grid: (S_k blocks in x, S_q blocks in y, B*H blocks in z)
 * Block: (block_size, block_size, 1) threads
 */
__global__ void attention_qk_naive_kernel(
    const float* Q,           // [B, H, S_q, D]
    const float* K,           // [B, H, S_k, D]
    float* scores,            // [B, H, S_q, S_k]
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D,
    const float scale,
    const bool is_causal,
    const float* mask         // [B, H, S_q, S_k] or nullptr
) {
    // Calculate global indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // S_k dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // S_q dimension
    int bh = blockIdx.z;                               // B*H dimension

    // Extract batch and head indices
    int b = bh / H;
    int h = bh % H;

    // Boundary check
    if (row >= S_q || col >= S_k || b >= B) {
        return;
    }

    // Check causal mask
    if (is_causal && col > row) {
        scores[idx4d(b, h, row, col, H, S_q, S_k)] = -INFINITY;
        return;
    }

    // Compute dot product: Q[b,h,row,:] @ K[b,h,col,:]
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float q_val = Q[idx4d(b, h, row, d, H, S_q, D)];
        float k_val = K[idx4d(b, h, col, d, H, S_k, D)];
        sum += q_val * k_val;
    }

    // Apply scaling
    sum *= scale;

    // Apply custom mask if provided
    if (mask != nullptr) {
        float mask_val = mask[idx4d(b, h, row, col, H, S_q, S_k)];
        if (mask_val != 0.0f) {
            sum = -INFINITY;
        }
    }

    // Write result
    scores[idx4d(b, h, row, col, H, S_q, S_k)] = sum;
}

/**
 * Tiled QK^T Kernel using Shared Memory
 *
 * Uses shared memory to cache tiles of Q and K, reducing global memory accesses.
 * Each block computes a TILE_SIZE x TILE_SIZE tile of the output.
 *
 * Algorithm:
 * 1. Load tile of Q and K into shared memory
 * 2. Compute partial dot products
 * 3. Accumulate across all tiles in D dimension
 */
template<int TILE_SIZE>
__global__ void attention_qk_tiled_kernel(
    const float* Q,
    const float* K,
    float* scores,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D,
    const float scale,
    const bool is_causal,
    const float* mask
) {
    // Shared memory for Q and K tiles
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int bh = blockIdx.z;

    int b = bh / H;
    int h = bh % H;

    if (b >= B) return;

    // Check causal mask early
    if (is_causal && col < S_k && row < S_q && col > row) {
        if (row < S_q && col < S_k) {
            scores[idx4d(b, h, row, col, H, S_q, S_k)] = -INFINITY;
        }
        return;
    }

    float sum = 0.0f;

    // Tile over D dimension
    int num_tiles = (D + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        // Load Q tile
        int d_q = tile * TILE_SIZE + threadIdx.x;
        if (row < S_q && d_q < D) {
            tile_Q[threadIdx.y][threadIdx.x] = Q[idx4d(b, h, row, d_q, H, S_q, D)];
        } else {
            tile_Q[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load K tile (note: we need K^T, so swap indices)
        int d_k = tile * TILE_SIZE + threadIdx.y;
        if (col < S_k && d_k < D) {
            tile_K[threadIdx.y][threadIdx.x] = K[idx4d(b, h, col, d_k, H, S_k, D)];
        } else {
            tile_K[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        if (row < S_q && col < S_k) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += tile_Q[threadIdx.y][k] * tile_K[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write result if within bounds
    if (row < S_q && col < S_k) {
        sum *= scale;

        // Apply custom mask
        if (mask != nullptr) {
            float mask_val = mask[idx4d(b, h, row, col, H, S_q, S_k)];
            if (mask_val != 0.0f) {
                sum = -INFINITY;
            }
        }

        scores[idx4d(b, h, row, col, H, S_q, S_k)] = sum;
    }
}

/**
 * Host function: Launch QK^T kernel (naive version)
 */
extern "C" void launch_attention_qk_naive(
    const float* Q,
    const float* K,
    float* scores,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D,
    const float scale,
    const bool is_causal,
    const float* mask,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (S_k + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (S_q + BLOCK_SIZE - 1) / BLOCK_SIZE,
        B * H
    );

    attention_qk_naive_kernel<<<grid, block, 0, stream>>>(
        Q, K, scores, B, H, S_q, S_k, D, scale, is_causal, mask
    );

    CUDA_CHECK_LAST_ERROR();
}

/**
 * Host function: Launch QK^T kernel (tiled version)
 */
extern "C" void launch_attention_qk_tiled(
    const float* Q,
    const float* K,
    float* scores,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D,
    const float scale,
    const bool is_causal,
    const float* mask,
    cudaStream_t stream
) {
    const int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (S_k + TILE_SIZE - 1) / TILE_SIZE,
        (S_q + TILE_SIZE - 1) / TILE_SIZE,
        B * H
    );

    attention_qk_tiled_kernel<TILE_SIZE><<<grid, block, 0, stream>>>(
        Q, K, scores, B, H, S_q, S_k, D, scale, is_causal, mask
    );

    CUDA_CHECK_LAST_ERROR();
}
