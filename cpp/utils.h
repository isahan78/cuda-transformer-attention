/**
 * C++ Utilities for PyTorch CUDA Extension
 *
 * Helper functions for tensor validation, error checking,
 * and interfacing between PyTorch and CUDA.
 */

#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

// Macro for checking tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")

/**
 * Validate attention input tensors
 *
 * Ensures Q, K, V have correct shapes and properties:
 * - All are CUDA tensors
 * - All are contiguous
 * - All are float32
 * - Q, K, V have shape [B, H, S, D]
 * - Batch and head dimensions match
 * - Head dimension matches
 */
inline void validate_attention_inputs(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    // Check device and memory layout
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    // Check dtype
    CHECK_FLOAT(Q);
    CHECK_FLOAT(K);
    CHECK_FLOAT(V);

    // Check dimensions
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (B, H, S_q, D)");
    TORCH_CHECK(K.dim() == 4, "K must be 4D (B, H, S_k, D)");
    TORCH_CHECK(V.dim() == 4, "V must be 4D (B, H, S_k, D)");

    int B_q = Q.size(0), H_q = Q.size(1), D_q = Q.size(3);
    int B_k = K.size(0), H_k = K.size(1), S_k = K.size(2), D_k = K.size(3);
    int B_v = V.size(0), H_v = V.size(1), S_v = V.size(2), D_v = V.size(3);

    // Check batch and head dimensions match
    TORCH_CHECK(B_q == B_k && B_q == B_v,
                "Batch dimensions must match: Q=", B_q, " K=", B_k, " V=", B_v);
    TORCH_CHECK(H_q == H_k && H_q == H_v,
                "Head dimensions must match: Q=", H_q, " K=", H_k, " V=", H_v);

    // Check K and V have same sequence length
    TORCH_CHECK(S_k == S_v,
                "K and V sequence lengths must match: K=", S_k, " V=", S_v);

    // Check head dimensions match
    TORCH_CHECK(D_q == D_k && D_q == D_v,
                "Head dimensions must match: Q=", D_q, " K=", D_k, " V=", D_v);
}

/**
 * Validate optional mask tensor
 *
 * If provided, mask should be:
 * - CUDA tensor
 * - Contiguous
 * - Float32
 * - Shape [B, H, S_q, S_k] or broadcastable
 */
inline void validate_mask(
    const torch::Tensor& mask,
    int B, int H, int S_q, int S_k
) {
    if (!mask.defined()) {
        return;  // Mask is optional
    }

    CHECK_INPUT(mask);
    CHECK_FLOAT(mask);

    TORCH_CHECK(mask.dim() == 4, "Mask must be 4D (B, H, S_q, S_k)");

    // Allow broadcasting for some dimensions
    TORCH_CHECK(mask.size(0) == B || mask.size(0) == 1,
                "Mask batch dimension must be B or 1");
    TORCH_CHECK(mask.size(1) == H || mask.size(1) == 1,
                "Mask head dimension must be H or 1");
    TORCH_CHECK(mask.size(2) == S_q, "Mask S_q dimension must match");
    TORCH_CHECK(mask.size(3) == S_k, "Mask S_k dimension must match");
}

/**
 * Get tensor data pointer as float*
 */
inline float* get_data_ptr(torch::Tensor& tensor) {
    return tensor.data_ptr<float>();
}

inline const float* get_data_ptr(const torch::Tensor& tensor) {
    return tensor.data_ptr<float>();
}

/**
 * Create output tensor with same options as input
 */
inline torch::Tensor create_output_tensor(
    const torch::Tensor& reference,
    std::vector<int64_t> shape
) {
    return torch::zeros(
        shape,
        torch::TensorOptions()
            .dtype(reference.dtype())
            .device(reference.device())
    );
}

/**
 * Get current CUDA stream from PyTorch
 */
inline cudaStream_t get_current_stream() {
    return at::cuda::getCurrentCUDAStream();
}

/**
 * Format tensor shape for error messages
 */
inline std::string shape_string(const torch::Tensor& tensor) {
    std::string result = "[";
    for (int i = 0; i < tensor.dim(); i++) {
        if (i > 0) result += ", ";
        result += std::to_string(tensor.size(i));
    }
    result += "]";
    return result;
}

/**
 * Print tensor info for debugging
 */
inline void print_tensor_info(const std::string& name, const torch::Tensor& tensor) {
    std::cout << name << ": "
              << "shape=" << shape_string(tensor)
              << " dtype=" << tensor.dtype()
              << " device=" << tensor.device()
              << " contiguous=" << tensor.is_contiguous()
              << std::endl;
}

#endif // CPP_UTILS_H
