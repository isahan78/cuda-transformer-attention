/**
 * PyTorch C++ Extension Binding for CUDA Attention Kernels
 *
 * Exposes CUDA kernels to Python via PyBind11.
 * Provides multiple attention implementations:
 * - Naive (correctness-focused)
 * - Tiled (shared memory optimization)
 * - Fused (FlashAttention-style)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "utils.h"

// Forward declarations of CUDA kernel launchers
extern "C" {
    void launch_attention_qk_naive(
        const float* Q, const float* K, float* scores,
        const int B, const int H, const int S_q, const int S_k, const int D,
        const float scale, const bool is_causal, const float* mask,
        cudaStream_t stream
    );

    void launch_attention_qk_tiled(
        const float* Q, const float* K, float* scores,
        const int B, const int H, const int S_q, const int S_k, const int D,
        const float scale, const bool is_causal, const float* mask,
        cudaStream_t stream
    );

    void launch_attention_softmax_naive(
        const float* scores, float* attn_weights,
        const int B, const int H, const int S_q, const int S_k,
        cudaStream_t stream
    );

    void launch_attention_softmax_optimized(
        const float* scores, float* attn_weights,
        const int B, const int H, const int S_q, const int S_k,
        cudaStream_t stream
    );

    void launch_attention_av_naive(
        const float* attn_weights, const float* V, float* output,
        const int B, const int H, const int S_q, const int S_k, const int D,
        cudaStream_t stream
    );

    void launch_attention_av_tiled(
        const float* attn_weights, const float* V, float* output,
        const int B, const int H, const int S_q, const int S_k, const int D,
        cudaStream_t stream
    );

    void launch_attention_fused(
        const float* Q, const float* K, const float* V, float* O,
        const int B, const int H, const int S_q, const int S_k, const int D,
        const float scale, const bool is_causal, const float* mask,
        cudaStream_t stream
    );
}

/**
 * Three-stage attention: QK -> Softmax -> AV (Naive version)
 */
torch::Tensor attention_forward_naive(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask,
    bool is_causal,
    float scale
) {
    // Validate inputs
    validate_attention_inputs(Q, K, V);

    // Extract dimensions
    int B = Q.size(0);
    int H = Q.size(1);
    int S_q = Q.size(2);
    int S_k = K.size(2);
    int D = Q.size(3);

    // Default scale
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    // Validate mask if provided
    const float* mask_ptr = nullptr;
    if (mask.defined()) {
        validate_mask(mask, B, H, S_q, S_k);
        mask_ptr = get_data_ptr(mask);
    }

    // Get CUDA stream
    cudaStream_t stream = get_current_stream();

    // Allocate intermediate tensors
    auto scores = create_output_tensor(Q, {B, H, S_q, S_k});
    auto attn_weights = create_output_tensor(Q, {B, H, S_q, S_k});
    auto output = create_output_tensor(Q, {B, H, S_q, D});

    // Stage 1: Compute Q @ K^T
    launch_attention_qk_naive(
        get_data_ptr(Q),
        get_data_ptr(K),
        get_data_ptr(scores),
        B, H, S_q, S_k, D,
        scale, is_causal, mask_ptr,
        stream
    );

    // Stage 2: Softmax
    launch_attention_softmax_naive(
        get_data_ptr(scores),
        get_data_ptr(attn_weights),
        B, H, S_q, S_k,
        stream
    );

    // Stage 3: Attention @ V
    launch_attention_av_naive(
        get_data_ptr(attn_weights),
        get_data_ptr(V),
        get_data_ptr(output),
        B, H, S_q, S_k, D,
        stream
    );

    return output;
}

/**
 * Three-stage attention: QK -> Softmax -> AV (Tiled/Optimized version)
 */
torch::Tensor attention_forward_tiled(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask,
    bool is_causal,
    float scale
) {
    validate_attention_inputs(Q, K, V);

    int B = Q.size(0);
    int H = Q.size(1);
    int S_q = Q.size(2);
    int S_k = K.size(2);
    int D = Q.size(3);

    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    const float* mask_ptr = nullptr;
    if (mask.defined()) {
        validate_mask(mask, B, H, S_q, S_k);
        mask_ptr = get_data_ptr(mask);
    }

    cudaStream_t stream = get_current_stream();

    auto scores = create_output_tensor(Q, {B, H, S_q, S_k});
    auto attn_weights = create_output_tensor(Q, {B, H, S_q, S_k});
    auto output = create_output_tensor(Q, {B, H, S_q, D});

    // Use tiled kernels
    launch_attention_qk_tiled(
        get_data_ptr(Q), get_data_ptr(K), get_data_ptr(scores),
        B, H, S_q, S_k, D, scale, is_causal, mask_ptr, stream
    );

    launch_attention_softmax_optimized(
        get_data_ptr(scores), get_data_ptr(attn_weights),
        B, H, S_q, S_k, stream
    );

    launch_attention_av_tiled(
        get_data_ptr(attn_weights), get_data_ptr(V), get_data_ptr(output),
        B, H, S_q, S_k, D, stream
    );

    return output;
}

/**
 * Fused FlashAttention-style kernel
 */
torch::Tensor attention_forward_fused(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask,
    bool is_causal,
    float scale
) {
    validate_attention_inputs(Q, K, V);

    int B = Q.size(0);
    int H = Q.size(1);
    int S_q = Q.size(2);
    int S_k = K.size(2);
    int D = Q.size(3);

    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    const float* mask_ptr = nullptr;
    if (mask.defined()) {
        validate_mask(mask, B, H, S_q, S_k);
        mask_ptr = get_data_ptr(mask);
    }

    cudaStream_t stream = get_current_stream();

    auto output = create_output_tensor(Q, {B, H, S_q, D});

    // Single fused kernel - no intermediate allocations
    launch_attention_fused(
        get_data_ptr(Q),
        get_data_ptr(K),
        get_data_ptr(V),
        get_data_ptr(output),
        B, H, S_q, S_k, D,
        scale, is_causal, mask_ptr,
        stream
    );

    return output;
}

/**
 * Main entry point - dispatches to appropriate kernel based on mode
 */
torch::Tensor cuda_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    std::string mode,
    torch::Tensor mask,
    bool is_causal,
    float scale
) {
    if (mode == "naive") {
        return attention_forward_naive(Q, K, V, mask, is_causal, scale);
    } else if (mode == "tiled" || mode == "optimized") {
        return attention_forward_tiled(Q, K, V, mask, is_causal, scale);
    } else if (mode == "fused" || mode == "flash") {
        return attention_forward_fused(Q, K, V, mask, is_causal, scale);
    } else {
        TORCH_CHECK(false, "Unknown mode: " + mode +
                    ". Valid modes: naive, tiled, fused");
    }
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA Transformer Attention Kernels";

    m.def("forward", &cuda_attention_forward,
          "CUDA Attention Forward Pass",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("mode") = "tiled",
          py::arg("mask") = torch::Tensor(),
          py::arg("is_causal") = false,
          py::arg("scale") = 0.0f
    );

    m.def("forward_naive", &attention_forward_naive,
          "Naive CUDA Attention",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("mask") = torch::Tensor(),
          py::arg("is_causal") = false,
          py::arg("scale") = 0.0f
    );

    m.def("forward_tiled", &attention_forward_tiled,
          "Tiled/Optimized CUDA Attention",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("mask") = torch::Tensor(),
          py::arg("is_causal") = false,
          py::arg("scale") = 0.0f
    );

    m.def("forward_fused", &attention_forward_fused,
          "Fused FlashAttention-style Attention",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("mask") = torch::Tensor(),
          py::arg("is_causal") = false,
          py::arg("scale") = 0.0f
    );
}
