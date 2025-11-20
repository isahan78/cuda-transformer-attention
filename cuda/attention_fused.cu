/**
 * FlashAttention-Style Fused Kernel
 *
 * Implements memory-efficient attention computation that avoids materializing
 * the full [S_q x S_k] attention matrix.
 *
 * Key ideas:
 * 1. Process attention in blocks/tiles
 * 2. Stream K and V blocks from HBM
 * 3. Maintain running statistics (max, sum) for numerically stable softmax
 * 4. Fuse softmax and attention@V into a single kernel
 * 5. Only materialize small blocks of attention scores in shared memory
 *
 * Algorithm (per query block):
 *   For each K/V block:
 *     1. Load Q block, K block, V block into SRAM
 *     2. Compute S = Q @ K^T (block-wise)
 *     3. Update running max and sum for stable softmax
 *     4. Compute attention weights and accumulate to output
 *     5. Rescale previous outputs based on new statistics
 *
 * This approach reduces memory usage from O(S_q * S_k) to O(S_q * D).
 */

#include "utils.cuh"

// Block size for fused kernel
#define BLOCK_M 64  // Query block size
#define BLOCK_N 64  // Key/Value block size
#define BLOCK_K 64  // Head dimension block size (if needed)

/**
 * FlashAttention-Style Fused Forward Kernel
 *
 * Computes attention output without materializing full attention matrix.
 * Each thread block processes a tile of queries.
 *
 * Grid: (ceil(S_q / BLOCK_M), B * H)
 * Block: Configured based on BLOCK_M and head dimension
 */
template<int BM, int BN, int BD>
__global__ void attention_fused_kernel(
    const float* Q,              // [B, H, S_q, D]
    const float* K,              // [B, H, S_k, D]
    const float* V,              // [B, H, S_k, D]
    float* O,                    // [B, H, S_q, D] output
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D,
    const float scale,
    const bool is_causal,
    const float* mask
) {
    // Batch and head indices
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;

    if (b >= B) return;

    // Query block index
    int q_block_idx = blockIdx.x;
    int q_start = q_block_idx * BM;
    int q_end = min(q_start + BM, S_q);
    int q_size = q_end - q_start;

    // Thread indices within block
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Shared memory for Q, K, V blocks and intermediate results
    __shared__ float smem_Q[BM * 64];   // Q block [BM, D]
    __shared__ float smem_K[BN * 64];   // K block [BN, D]
    __shared__ float smem_V[BN * 64];   // V block [BN, D]
    __shared__ float smem_S[BM * BN];   // Attention scores [BM, BN]

    // Per-thread output accumulator and statistics
    // Each thread handles multiple output elements
    float thread_out[64];  // Output values for this thread's queries
    float thread_max[64];  // Running max for each query
    float thread_sum[64];  // Running sum for each query

    // Initialize output and statistics
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        thread_out[i] = 0.0f;
        thread_max[i] = -FLT_MAX;
        thread_sum[i] = 0.0f;
    }

    // Load Q block into shared memory (done once per thread block)
    for (int idx = tid; idx < q_size * D; idx += num_threads) {
        int local_q = idx / D;
        int d = idx % D;
        int global_q = q_start + local_q;

        if (global_q < S_q && d < D) {
            smem_Q[local_q * D + d] = Q[idx4d(b, h, global_q, d, H, S_q, D)];
        }
    }
    __syncthreads();

    // Iterate over K/V blocks
    int num_kv_blocks = (S_k + BN - 1) / BN;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int k_start = kv_block * BN;
        int k_end = min(k_start + BN, S_k);
        int k_size = k_end - k_start;

        // Load K and V blocks into shared memory
        for (int idx = tid; idx < k_size * D; idx += num_threads) {
            int local_k = idx / D;
            int d = idx % D;
            int global_k = k_start + local_k;

            if (global_k < S_k && d < D) {
                smem_K[local_k * D + d] = K[idx4d(b, h, global_k, d, H, S_k, D)];
                smem_V[local_k * D + d] = V[idx4d(b, h, global_k, d, H, S_k, D)];
            }
        }
        __syncthreads();

        // Compute attention scores S = Q @ K^T for this block
        // Each thread computes a subset of [BM x BN] scores
        for (int idx = tid; idx < q_size * k_size; idx += num_threads) {
            int local_q = idx / k_size;
            int local_k = idx % k_size;
            int global_q = q_start + local_q;
            int global_k = k_start + local_k;

            // Skip if causal mask applies
            if (is_causal && global_k > global_q) {
                smem_S[local_q * BN + local_k] = -INFINITY;
                continue;
            }

            // Compute dot product
            float score = 0.0f;
            for (int d = 0; d < D; d++) {
                score += smem_Q[local_q * D + d] * smem_K[local_k * D + d];
            }
            score *= scale;

            // Apply custom mask if provided
            if (mask != nullptr) {
                float mask_val = mask[idx4d(b, h, global_q, global_k, H, S_q, S_k)];
                if (mask_val != 0.0f) {
                    score = -INFINITY;
                }
            }

            smem_S[local_q * BN + local_k] = score;
        }
        __syncthreads();

        // Online softmax update: for each query in this block
        // This is a simplified version - full FlashAttention does this more efficiently
        for (int local_q = 0; local_q < q_size; local_q++) {
            if (tid >= local_q && tid < local_q + 1) {  // Simplified: one thread per query
                // Find max in this block
                float block_max = -FLT_MAX;
                for (int local_k = 0; local_k < k_size; local_k++) {
                    block_max = fmaxf(block_max, smem_S[local_q * BN + local_k]);
                }

                // Update running max
                float old_max = thread_max[local_q];
                float new_max = fmaxf(old_max, block_max);
                thread_max[local_q] = new_max;

                // Compute exponentials and sum for this block
                float block_sum = 0.0f;
                for (int local_k = 0; local_k < k_size; local_k++) {
                    float exp_val = safe_exp(smem_S[local_q * BN + local_k] - new_max);
                    smem_S[local_q * BN + local_k] = exp_val;  // Overwrite with exp values
                    block_sum += exp_val;
                }

                // Rescale previous sum
                float old_sum_rescaled = thread_sum[local_q] * safe_exp(old_max - new_max);
                thread_sum[local_q] = old_sum_rescaled + block_sum;

                // Rescale previous output
                float rescale_factor = safe_exp(old_max - new_max);
                for (int d = 0; d < D; d++) {
                    thread_out[local_q * D + d] *= rescale_factor;
                }
            }
        }
        __syncthreads();

        // Accumulate to output: O += softmax(S) @ V
        for (int local_q = 0; local_q < q_size; local_q++) {
            for (int d = tid; d < D; d += num_threads) {
                float accum = 0.0f;
                for (int local_k = 0; local_k < k_size; local_k++) {
                    float attn_weight = smem_S[local_q * BN + local_k];
                    float v_val = smem_V[local_k * D + d];
                    accum += attn_weight * v_val;
                }
                thread_out[local_q * D + d] += accum;
            }
        }
        __syncthreads();
    }

    // Final normalization and write output
    for (int local_q = 0; local_q < q_size; local_q++) {
        float inv_sum = (thread_sum[local_q] > 0.0f) ? (1.0f / thread_sum[local_q]) : 0.0f;

        for (int d = tid; d < D; d += num_threads) {
            int global_q = q_start + local_q;
            if (global_q < S_q && d < D) {
                O[idx4d(b, h, global_q, d, H, S_q, D)] =
                    thread_out[local_q * D + d] * inv_sum;
            }
        }
    }
}

/**
 * Simplified Fused Kernel (Easier to Understand)
 *
 * This version is less optimized but clearer in its implementation.
 * Processes one query at a time per block.
 */
__global__ void attention_fused_simple_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B,
    const int H,
    const int S_q,
    const int S_k,
    const int D,
    const float scale,
    const bool is_causal,
    const float* mask
) {
    // Each block processes one query
    int query_idx = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;

    if (query_idx >= S_q || b >= B) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Running statistics for online softmax
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    // Output accumulator (one per head dimension)
    extern __shared__ float shared_output[];
    for (int d = tid; d < D; d += num_threads) {
        shared_output[d] = 0.0f;
    }
    __syncthreads();

    // Process K/V in blocks
    const int BLOCK_SIZE = 64;
    int num_blocks = (S_k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int block = 0; block < num_blocks; block++) {
        int k_start = block * BLOCK_SIZE;
        int k_end = min(k_start + BLOCK_SIZE, S_k);

        __shared__ float block_scores[BLOCK_SIZE];
        __shared__ float block_max;
        __shared__ float block_sum;

        // Compute scores for this block
        for (int k_idx = k_start + tid; k_idx < k_end; k_idx += num_threads) {
            if (is_causal && k_idx > query_idx) {
                block_scores[k_idx - k_start] = -INFINITY;
                continue;
            }

            float score = 0.0f;
            for (int d = 0; d < D; d++) {
                float q_val = Q[idx4d(b, h, query_idx, d, H, S_q, D)];
                float k_val = K[idx4d(b, h, k_idx, d, H, S_k, D)];
                score += q_val * k_val;
            }
            score *= scale;

            if (mask != nullptr) {
                float mask_val = mask[idx4d(b, h, query_idx, k_idx, H, S_q, S_k)];
                if (mask_val != 0.0f) score = -INFINITY;
            }

            block_scores[k_idx - k_start] = score;
        }
        __syncthreads();

        // Find max in block
        float local_max = -FLT_MAX;
        for (int i = tid; i < (k_end - k_start); i += num_threads) {
            local_max = fmaxf(local_max, block_scores[i]);
        }
        local_max = blockReduceMax<256>(local_max);
        if (tid == 0) block_max = local_max;
        __syncthreads();

        // Update global max and rescale
        float new_max = fmaxf(running_max, block_max);
        float exp_old_max = safe_exp(running_max - new_max);
        float exp_block_max = safe_exp(block_max - new_max);

        // Compute exp and sum for block
        float local_sum = 0.0f;
        for (int i = tid; i < (k_end - k_start); i += num_threads) {
            float exp_val = safe_exp(block_scores[i] - new_max);
            block_scores[i] = exp_val;
            local_sum += exp_val;
        }
        local_sum = blockReduceSum<256>(local_sum);
        if (tid == 0) block_sum = local_sum;
        __syncthreads();

        // Update running sum
        running_sum = running_sum * exp_old_max + block_sum;
        running_max = new_max;

        // Update output
        for (int d = tid; d < D; d += num_threads) {
            shared_output[d] *= exp_old_max;
            float accum = 0.0f;
            for (int i = 0; i < (k_end - k_start); i++) {
                float attn = block_scores[i];
                float v_val = V[idx4d(b, h, k_start + i, d, H, S_k, D)];
                accum += attn * v_val;
            }
            shared_output[d] += accum;
        }
        __syncthreads();
    }

    // Normalize and write output
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (int d = tid; d < D; d += num_threads) {
        O[idx4d(b, h, query_idx, d, H, S_q, D)] = shared_output[d] * inv_sum;
    }
}

/**
 * Host function: Launch fused attention kernel
 */
extern "C" void launch_attention_fused(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
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
    // Use simplified version for now
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(S_q, B * H);

    int shared_mem_size = D * sizeof(float);

    attention_fused_simple_kernel<<<grid, block, shared_mem_size, stream>>>(
        Q, K, V, O, B, H, S_q, S_k, D, scale, is_causal, mask
    );

    CUDA_CHECK_LAST_ERROR();
}
