/**
 * CUDA Kernels for Row-wise Softmax
 *
 * Implements numerically stable softmax over the last dimension.
 * Each row is processed independently.
 *
 * Input shape: [B, H, S_q, S_k]
 * Output shape: [B, H, S_q, S_k]
 *
 * Algorithm (numerically stable):
 * 1. Find row-wise max: m = max(x)
 * 2. Compute exp(x - m)
 * 3. Compute sum of exponentials: s = sum(exp(x - m))
 * 4. Normalize: softmax = exp(x - m) / s
 *
 * Versions:
 * 1. Naive: each block processes one row, simple reduction
 * 2. Optimized: uses shared memory and warp-level reductions
 */

#include "utils.cuh"

/**
 * Naive Softmax Kernel
 *
 * Each block processes one row of the scores matrix.
 * Uses simple parallel reduction for max and sum.
 *
 * Grid: (S_q * B * H blocks)
 * Block: (block_size threads)
 */
template<int BLOCK_SIZE>
__global__ void attention_softmax_naive_kernel(
    const float* scores,      // [B, H, S_q, S_k]
    float* attn_weights,      // [B, H, S_q, S_k]
    const int B,
    const int H,
    const int S_q,
    const int S_k
) {
    // Each block processes one row
    int row_idx = blockIdx.x;
    int total_rows = B * H * S_q;

    if (row_idx >= total_rows) return;

    // Decode row index into b, h, s_q
    int s_q = row_idx % S_q;
    int temp = row_idx / S_q;
    int h = temp % H;
    int b = temp / H;

    // Calculate base offset for this row
    int row_offset = idx4d(b, h, s_q, 0, H, S_q, S_k);

    // Step 1: Find maximum value in the row
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        float val = scores[row_offset + i];
        max_val = fmaxf(max_val, val);
    }

    // Reduce max across block
    max_val = blockReduceMax<BLOCK_SIZE>(max_val);

    // Broadcast max to all threads
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = max_val;
    }
    __syncthreads();
    max_val = shared_max;

    // Step 2 & 3: Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        float val = scores[row_offset + i];
        float exp_val = safe_exp(val - max_val);
        attn_weights[row_offset + i] = exp_val;  // Store temporarily
        sum_exp += exp_val;
    }

    // Reduce sum across block
    sum_exp = blockReduceSum<BLOCK_SIZE>(sum_exp);

    // Broadcast sum to all threads
    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = sum_exp;
    }
    __syncthreads();
    sum_exp = shared_sum;

    // Step 4: Normalize
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        attn_weights[row_offset + i] *= inv_sum;
    }
}

/**
 * Optimized Softmax Kernel with Shared Memory
 *
 * Uses shared memory to cache partial results and warp-level primitives
 * for faster reductions.
 */
template<int BLOCK_SIZE>
__global__ void attention_softmax_optimized_kernel(
    const float* scores,
    float* attn_weights,
    const int B,
    const int H,
    const int S_q,
    const int S_k
) {
    extern __shared__ float shared_mem[];

    int row_idx = blockIdx.x;
    int total_rows = B * H * S_q;

    if (row_idx >= total_rows) return;

    // Decode indices
    int s_q = row_idx % S_q;
    int temp = row_idx / S_q;
    int h = temp % H;
    int b = temp / H;

    int row_offset = idx4d(b, h, s_q, 0, H, S_q, S_k);

    // Load data into shared memory and find max
    float thread_max = -FLT_MAX;

    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        float val = scores[row_offset + i];
        if (i < S_k) {
            shared_mem[i] = val;
            thread_max = fmaxf(thread_max, val);
        }
    }

    // Reduce max
    float max_val = blockReduceMax<BLOCK_SIZE>(thread_max);

    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = max_val;
    }
    __syncthreads();

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        float exp_val = safe_exp(shared_mem[i] - row_max);
        shared_mem[i] = exp_val;
        thread_sum += exp_val;
    }

    // Reduce sum
    float sum_exp = blockReduceSum<BLOCK_SIZE>(thread_sum);

    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = sum_exp;
    }
    __syncthreads();

    // Normalize and write output
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        attn_weights[row_offset + i] = shared_mem[i] * inv_sum;
    }
}

/**
 * Online Softmax Kernel (Single Pass)
 *
 * Computes softmax in a single pass using online algorithm.
 * Maintains running max and sum, updating them as we go.
 *
 * This is useful for the fused kernel implementation.
 */
template<int BLOCK_SIZE>
__global__ void attention_softmax_online_kernel(
    const float* scores,
    float* attn_weights,
    const int B,
    const int H,
    const int S_q,
    const int S_k
) {
    int row_idx = blockIdx.x;
    if (row_idx >= B * H * S_q) return;

    int s_q = row_idx % S_q;
    int temp = row_idx / S_q;
    int h = temp % H;
    int b = temp / H;

    int row_offset = idx4d(b, h, s_q, 0, H, S_q, S_k);

    // Online algorithm: maintain running max and sum
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    // First pass: compute max
    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        float val = scores[row_offset + i];
        running_max = fmaxf(running_max, val);
    }
    running_max = blockReduceMax<BLOCK_SIZE>(running_max);

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = running_max;
    __syncthreads();
    running_max = shared_max;

    // Second pass: compute sum and store exp values
    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        float val = scores[row_offset + i];
        float exp_val = safe_exp(val - running_max);
        attn_weights[row_offset + i] = exp_val;
        running_sum += exp_val;
    }
    running_sum = blockReduceSum<BLOCK_SIZE>(running_sum);

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = running_sum;
    __syncthreads();
    running_sum = shared_sum;

    // Third pass: normalize
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (int i = threadIdx.x; i < S_k; i += blockDim.x) {
        attn_weights[row_offset + i] *= inv_sum;
    }
}

/**
 * Host function: Launch softmax kernel (naive version)
 */
extern "C" void launch_attention_softmax_naive(
    const float* scores,
    float* attn_weights,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 256;
    int num_rows = B * H * S_q;

    attention_softmax_naive_kernel<BLOCK_SIZE><<<num_rows, BLOCK_SIZE, 0, stream>>>(
        scores, attn_weights, B, H, S_q, S_k
    );

    CUDA_CHECK_LAST_ERROR();
}

/**
 * Host function: Launch softmax kernel (optimized version)
 */
extern "C" void launch_attention_softmax_optimized(
    const float* scores,
    float* attn_weights,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 256;
    int num_rows = B * H * S_q;
    int shared_mem_size = S_k * sizeof(float);

    // Fall back to naive if shared memory requirement is too large
    if (shared_mem_size > 48 * 1024) {  // 48KB limit for most GPUs
        launch_attention_softmax_naive(scores, attn_weights, B, H, S_q, S_k, stream);
        return;
    }

    attention_softmax_optimized_kernel<BLOCK_SIZE><<<num_rows, BLOCK_SIZE, shared_mem_size, stream>>>(
        scores, attn_weights, B, H, S_q, S_k
    );

    CUDA_CHECK_LAST_ERROR();
}
