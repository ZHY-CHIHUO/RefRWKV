/******************************************************************************
 * Copyright (c) 2025 Shanghai AI Lab.
 ******************************************************************************/

#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor bi_wkv_cuda_forward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v);

std::vector<torch::Tensor> bi_wkv_cuda_backward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v, 
    torch::Tensor gy);

static void check_bi_wkv_inputs(const torch::Tensor& w, const torch::Tensor& u,
                                const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(k.dim() == 3, "bi_wkv: k must be 3D (B, T, C)");
    TORCH_CHECK(v.dim() == 3 && v.sizes() == k.sizes(), "bi_wkv: v must match k shape");
    TORCH_CHECK(w.dim() == 1 && w.size(0) == k.size(2), "bi_wkv: w must be 1D (C)");
    TORCH_CHECK(u.dim() == 1 && u.size(0) == k.size(2), "bi_wkv: u must be 1D (C)");
    TORCH_CHECK(w.scalar_type() == k.scalar_type(), "bi_wkv: w dtype mismatch");
    TORCH_CHECK(u.scalar_type() == k.scalar_type(), "bi_wkv: u dtype mismatch");
    TORCH_CHECK(v.scalar_type() == k.scalar_type(), "bi_wkv: v dtype mismatch");
    TORCH_CHECK(k.size(1) >= 1, "bi_wkv: T must be >= 1");
    TORCH_CHECK(k.size(2) >= 16, "bi_wkv: C must be >= 16");
    TORCH_CHECK((k.size(0) * k.size(2)) % 16 == 0, "bi_wkv: (B*C) must be divisible by 16");
}

torch::Tensor bi_wkv_forward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v) {
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    check_bi_wkv_inputs(w, u, k, v);
    return bi_wkv_cuda_forward(w, u, k, v);
}

std::vector<torch::Tensor> bi_wkv_backward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v, 
    torch::Tensor gy) {
    CHECK_INPUT(w);
    CHECK_INPUT(u);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(gy);
    TORCH_CHECK(gy.sizes() == k.sizes(), "bi_wkv: gy must match k shape");
    check_bi_wkv_inputs(w, u, k, v);
    return bi_wkv_cuda_backward(w, u, k, v, gy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bi_wkv_forward", &bi_wkv_forward, "Bi-WKV Forward(CUDA)");
    m.def("bi_wkv_backward", &bi_wkv_backward, "Bi-WKV Backward(CUDA)");
}

TORCH_LIBRARY(bi_wkv, m) {
    m.def("bi_wkv_forward", bi_wkv_forward);
    m.def("bi_wkv_backward", bi_wkv_backward);
}