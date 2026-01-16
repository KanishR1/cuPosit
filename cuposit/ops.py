import torch
from torch import Tensor
from cuposit import bspgemm

# https://docs.pytorch.org/docs/stable/generated/torch.addmm.html
def mm(
    posit_config: dict[str, int],
    mat1: Tensor,
    mat2: Tensor,
    out_dtype: torch.dtype | None=None,
    out: Tensor | None = None,
):
    assert mat1.dim() == 2 and mat2.dim() == 2, "mat1 and mat2 should be 2D tensors"
    assert mat1.shape[1] == mat2.shape[0], "mat1 and mat2 have incompatible shapes for mm"

    if out is not None: 
        raise NotImplementedError("param out is not supported yet")

    return torch.squeeze(bspgemm(
        posit_config,
        torch.broadcast_to(mat1, (1, *mat1.shape)), # A
        torch.broadcast_to(mat2, (1, *mat2.shape)), # B
        torch.zeros((1, mat1.shape[0], mat2.shape[1]), dtype=torch.float32, device="cuda"), # C
        1.0,
        0.0
    ), dim=0)


# https://docs.pytorch.org/docs/stable/generated/torch.addmm.html
def addmm(
    posit_config: dict[str, int],
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    out_dtype: torch.dtype | None=None,
    beta: float = 1,
    alpha: float = 1,
    out: Tensor | None = None,
):
    assert mat1.dim() == 2 and mat2.dim() == 2, "mat1 and mat2 should be 2D tensors"
    assert mat1.shape[1] == mat2.shape[0], "mat1 and mat2 have incompatible shapes for mm"

    if out is not None: 
        raise NotImplementedError("param out is not supported yet")

    return torch.squeeze(bspgemm(
        posit_config,
        torch.broadcast_to(mat1, (1, *mat1.shape)), # A
        torch.broadcast_to(mat2, (1, *mat2.shape)), # B
        torch.broadcast_to(input, (1, mat1.shape[0], mat2.shape[1])), # C
        alpha,
        beta
    ), dim=0)


def convolution(
    posit_config: dict[str, int],
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride,
    padding: list[int],
    dilation,
    transposed,
    output_padding,
    groups,
):
    assert not transposed, "Transposed convolution not supported"
    assert groups == 1, "Grouped convolution not supported"

    # print('convolution', input.shape, weight.shape, bias.shape)

    # input: (N, C_in, H, W)
    # weight: (C_out, C_in, kH, kW)
    # bias: (C_out,) or None
    
    N, C_in, H, W = input.shape
    C_out, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
    
    # im2col: (N, C_in, H, W) -> (N, C_in*kH*kW, H_out*W_out)
    input_unfolded = torch.nn.functional.unfold(
        input, 
        kernel_size=(kH, kW),
        dilation=dilation,
        padding=padding,
        stride=stride
    )  # (N, C_in*kH*kW, H_out*W_out)
    
    # Reshape weight: (C_out, C_in*kH*kW)
    weight_flat = weight.view(C_out, -1)
    
    # Batch GEMM: (N, C_out, H_out*W_out) = (N, C_out, C_in*kH*kW) @ (N, C_in*kH*kW, H_out*W_out)
    # weight needs to be (N, C_out, C_in*kH*kW), input_unfolded is (N, C_in*kH*kW, H_out*W_out)
    
    weight_batched = weight_flat.unsqueeze(0).expand(N, -1, -1)  # (N, C_out, C_in*kH*kW)
    input_batched = input_unfolded.transpose(1, 2)  # (N, H_out*W_out, C_in*kH*kW)
    
    if bias is not None:
        bias_batched = bias.view(1, 1, C_out).expand(N, H_out * W_out, -1)
    else:
        bias_batched = torch.zeros(N, H_out * W_out, C_out, device=input.device, dtype=input.dtype)
    
    # bspgemm(A, B, C, alpha, beta) computes alpha*A@B + beta*C
    # We want: weight_batched @ input_batched.transpose(-2, -1) + bias
    # A: (N, C_out, C_in*kH*kW), B: (N, C_in*kH*kW, H_out*W_out), C: (N, C_out, H_out*W_out)
    
    output = bspgemm(
        posit_config,
        weight_batched,  # (N, C_out, C_in*kH*kW)
        input_unfolded,  # (N, C_in*kH*kW, H_out*W_out)
        bias.view(1, C_out, 1).expand(N, -1, H_out * W_out) if bias is not None 
            else torch.zeros(N, C_out, H_out * W_out, device=input.device, dtype=input.dtype),
        alpha=1.0,
        beta=1.0 if bias is not None else 0.0
    )  # (N, C_out, H_out*W_out)
    
    # Reshape to (N, C_out, H_out, W_out)
    return output.view(N, C_out, H_out, W_out)