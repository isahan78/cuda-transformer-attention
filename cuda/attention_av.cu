/**
 * CUDA Kernels for Attention @ V Computation
 *
 * Computes the final output by multiplying attention weights with values.
 *
 * Input shapes:
 *   attn_weights: [B, H, S_q, S_k]
 *   V: [B, H, S_k, D]
 * Output shape:
 *   output: [B, H, S_q, D]
 *
 * Implements:
 * 1. Naive kernel: each thread computes one output element
 * 2. Tiled kernel: uses shared memory for better memory access patterns
 */

#include "utils.cuh"

/**
 * Naive Attention @ V Kernel
 *
 * Each thread computes one element of the output.
 * Thread computes output[b,h,row,d] = attn_weights[b,h,row,:] @ V[b,h,:,d]
 *
 * Grid: (D blocks in x, S_q blocks in y, B*H blocks in z)
 * Block: (block_size, block_size, 1) threads
 */
__global__ void attention_av_naive_kernel(
    const float* attn_weights,  // [B, H, S_q, S_k]
    const float* V,              // [B, H, S_k, D]
    float* output,               // [B, H, S_q, D]
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D
) {
    // Calculate global indices
    int d = blockIdx.x * blockDim.x + threadIdx.x;    // D dimension
    int s_q = blockIdx.y * blockDim.y + threadIdx.y;  // S_q dimension
    int bh = blockIdx.z;                               // B*H dimension

    int b = bh / H;
    int h = bh % H;

    // Boundary check
    if (s_q >= S_q || d >= D || b >= B) {
        return;
    }

    // Compute dot product: attn_weights[b,h,s_q,:] @ V[b,h,:,d]
    float sum = 0.0f;
    for (int s_k = 0; s_k < S_k; s_k++) {
        float attn_val = attn_weights[idx4d(b, h, s_q, s_k, H, S_q, S_k)];
        float v_val = V[idx4d(b, h, s_k, d, H, S_k, D)];
        sum += attn_val * v_val;
    }

    // Write result
    output[idx4d(b, h, s_q, d, H, S_q, D)] = sum;
}

/**
 * Tiled Attention @ V Kernel using Shared Memory
 *
 * Uses shared memory to cache tiles of attention weights and V.
 * Each block computes a TILE_SIZE x TILE_SIZE tile of the output.
 *
 * Algorithm:
 * 1. Load tile of attn_weights and V into shared memory
 * 2. Compute partial dot products
 * 3. Accumulate across all tiles in S_k dimension
 */
template<int TILE_SIZE>
__global__ void attention_av_tiled_kernel(
    const float* attn_weights,
    const float* V,
    float* output,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D
) {
    // Shared memory for tiles
    __shared__ float tile_attn[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];

    int d = blockIdx.x * TILE_SIZE + threadIdx.x;
    int s_q = blockIdx.y * TILE_SIZE + threadIdx.y;
    int bh = blockIdx.z;

    int b = bh / H;
    int h = bh % H;

    if (b >= B) return;

    float sum = 0.0f;

    // Tile over S_k dimension
    int num_tiles = (S_k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        // Load attention weights tile
        int s_k_attn = tile * TILE_SIZE + threadIdx.x;
        if (s_q < S_q && s_k_attn < S_k) {
            tile_attn[threadIdx.y][threadIdx.x] =
                attn_weights[idx4d(b, h, s_q, s_k_attn, H, S_q, S_k)];
        } else {
            tile_attn[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load V tile
        int s_k_v = tile * TILE_SIZE + threadIdx.y;
        if (s_k_v < S_k && d < D) {
            tile_V[threadIdx.y][threadIdx.x] =
                V[idx4d(b, h, s_k_v, d, H, S_k, D)];
        } else {
            tile_V[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        if (s_q < S_q && d < D) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += tile_attn[threadIdx.y][k] * tile_V[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write result if within bounds
    if (s_q < S_q && d < D) {
        output[idx4d(b, h, s_q, d, H, S_q, D)] = sum;
    }
}

/**
 * Optimized Attention @ V Kernel with Vectorized Loads
 *
 * Uses vectorized memory operations (float4) when possible.
 * This version works best when D is a multiple of 4.
 */
template<int TILE_SIZE>
__global__ void attention_av_vectorized_kernel(
    const float* attn_weights,
    const float* V,
    float* output,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D
) {
    // Similar to tiled kernel but uses float4 for loading V when D % 4 == 0
    // This is an optimization for future implementation
    // For now, we'll use the standard tiled kernel
    attention_av_tiled_kernel<TILE_SIZE><<<
        dim3((D + TILE_SIZE - 1) / TILE_SIZE,
             (S_q + TILE_SIZE - 1) / TILE_SIZE,
             B * H),
        dim3(TILE_SIZE, TILE_SIZE)
    >>>(attn_weights, V, output, B, H, S_q, S_k, D);
}

/**
 * Host function: Launch Attention @ V kernel (naive version)
 */
extern "C" void launch_attention_av_naive(
    const float* attn_weights,
    const float* V,
    float* output,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (D + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (S_q + BLOCK_SIZE - 1) / BLOCK_SIZE,
        B * H
    );

    attention_av_naive_kernel<<<grid, block, 0, stream>>>(
        attn_weights, V, output, B, H, S_q, S_k, D
    );

    CUDA_CHECK_LAST_ERROR();
}

/**
 * Host function: Launch Attention @ V kernel (tiled version)
 */
extern "C" void launch_attention_av_tiled(
    const float* attn_weights,
    const float* V,
    float* output,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D,
    cudaStream_t stream
) {
    const int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (D + TILE_SIZE - 1) / TILE_SIZE,
        (S_q + TILE_SIZE - 1) / TILE_SIZE,
        B * H
    );

    attention_av_tiled_kernel<TILE_SIZE><<<grid, block, 0, stream>>>(
        attn_weights, V, output, B, H, S_q, S_k, D
    );

    CUDA_CHECK_LAST_ERROR();
}
