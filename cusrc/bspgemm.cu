// Batched Strided Posit GEMM
// Expects tensors of form (N, H, W), where N is the batch dimension
// For NCHW, or other shapes, use im2col (https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html),
// unsqueeze/squeeze, etc.

// TODO: Completely get rid of CUPOSIT_ENABLED later.
// If the user needs float arithmetic, they should use built-in
// arithmetic
// It's here right now to enable debugging
// note that it's also in mma_sm50.h

#include "positclip.h"
#include <torch/extension.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>


void set_posit_constants(unsigned posit_n, unsigned posit_es, unsigned posit_rs) {
    unsigned const host_cuposit_nmantissa_max = posit_n - 3 - posit_es; // 1 bit for sign, 2 for regime, and posit_es bits for exponent
    unsigned const host_cuposit_exp_min = 127 - ((posit_rs - posit_es - 1)*(1 << posit_es) - 1);
    unsigned const host_cuposit_exp_max = 127 + ((posit_rs - posit_es - 1)*(1 << posit_es) - 1);

    TORCH_CHECK(((posit_rs - posit_es - 1) * (1 << posit_es) - 1) <= 127, "Unsupported Posit configuration: (n=", posit_n, ", es=", posit_es, ", rs=", posit_rs, "). ", "The exponent is outside Float32's range.");
    TORCH_CHECK(host_cuposit_nmantissa_max <= 23, "Unsupported Posit configuration: (n=", posit_n, ", es=", posit_es, ", rs=", posit_rs, "). ", "The mantissa length is outside Float32's range.");

    // cudaMemcpyToSymbol(CUPOSIT_ES, &posit_es, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_EXP_MIN, &host_cuposit_exp_min, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_EXP_MAX, &host_cuposit_exp_max, sizeof(unsigned));
    cudaMemcpyToSymbol(CUPOSIT_NMANTISSA_MAX, &host_cuposit_nmantissa_max, sizeof(unsigned));
}

void posit_clip(torch::Tensor A) {
    // Clips float32 to posit in-place
    // NOTE: CUPOSIT_* variables should be set and
    // copied to constant memory before this function is called
    int n = A.numel();
    float* data = A.data_ptr<float>();
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    kernel_posit_clip<<<blocks, threads>>>(data, data, n);
}

// This method will happily overwrite input matrices
// see python cuposit/bspgemm.py and only send clones of tensors
torch::Tensor bspgemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    float alpha,
    float beta,
    unsigned posit_n,
    unsigned posit_es,
    unsigned posit_rs
){
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && C.dtype() == torch::kFloat32,
                "Only float32 supported");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && C.dim() == 3, "Expected 3D tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(posit_es == 2, "Only posit_es == 2 is supported");


    int batch_count = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    TORCH_CHECK(B.size(0) == batch_count && B.size(1) == K, "B dimension mismatch");
    TORCH_CHECK(C.size(0) == batch_count && C.size(1) == M && C.size(2) == N, "C dimension mismatch");

    set_posit_constants(posit_n, posit_es, posit_rs);
    posit_clip(A);
    posit_clip(B);

    using Gemm = cutlass::gemm::device::GemmBatched<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm89>;

    Gemm::Arguments args{
        {M, N, K},
        {A.data_ptr<float>(), K},
        A.stride(0),
        {B.data_ptr<float>(), N},
        B.stride(0),
        {C.data_ptr<float>(), N},
        C.stride(0),
        {C.data_ptr<float>(), N},
        C.stride(0),
        {alpha, beta},
        batch_count};

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS cannot implement this GEMM configuration");

    status = gemm_op.initialize(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "Failed to initialize CUTLASS GEMM");

    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS GEMM kernel failed");

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bspgemm", &bspgemm, "Batched Strided Posit GEMM");
}