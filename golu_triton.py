import triton
import triton.language as tl
import torch

@triton.jit
def golu_forward_kernel(
    input_ptr,
    output_ptr,
    alpha,
    beta,
    gamma,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_FP32_COMPUTE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    if USE_FP32_COMPUTE:
        x_compute = x.to(tl.float32)
    else:
        x_compute = x
    
    neg_gamma_x = -gamma * x_compute
    inner_exp = tl.exp(neg_gamma_x)
    outer_exp = tl.exp(-beta * inner_exp)
    golu = x_compute * alpha * outer_exp
    
    if USE_FP32_COMPUTE:
        golu = golu.to(x.dtype)
    
    tl.store(output_ptr + offsets, golu, mask=mask)

@triton.jit
def golu_backward_kernel(
    grad_output_ptr,
    x_ptr,
    grad_x_ptr,
    alpha,
    beta,
    gamma,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_FP32_COMPUTE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if USE_FP32_COMPUTE:
        grad_output_compute = grad_output.to(tl.float32)
        x_compute = x.to(tl.float32)
    else:
        grad_output_compute = grad_output
        x_compute = x
    
    neg_gamma_x = -gamma * x_compute
    inner_exp = tl.exp(neg_gamma_x)
    outer_exp = tl.exp(-beta * inner_exp)
    
    factor = 1.0 + beta * gamma * x_compute * inner_exp
    grad_x = grad_output_compute * alpha * outer_exp * factor
    
    grad_x = tl.where(grad_x != grad_x, 0.0, grad_x)
    
    if USE_FP32_COMPUTE:
        grad_x = grad_x.to(x.dtype)
    
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)

class GoLUTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, beta, gamma):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.gamma = gamma
        output = torch.empty_like(x)
        n_elements = x.numel()
        
        block_size = min(max(128, triton.next_power_of_2(n_elements // 8192)), 1024)
        grid = (triton.cdiv(n_elements, block_size),)
        
        use_fp32_compute = x.dtype in [torch.float16, torch.bfloat16]
        
        golu_forward_kernel[grid](
            x, output, alpha, beta, gamma, n_elements,
            BLOCK_SIZE=block_size,
            USE_FP32_COMPUTE=use_fp32_compute,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        n_elements = x.numel()
        
        block_size = min(max(128, triton.next_power_of_2(n_elements // 8192)), 1024)
        grid = (triton.cdiv(n_elements, block_size),)
        
        use_fp32_compute = x.dtype in [torch.float16, torch.bfloat16]
        
        golu_backward_kernel[grid](
            grad_output, x, grad_x, ctx.alpha, ctx.beta, ctx.gamma,
            n_elements, BLOCK_SIZE=block_size, 
            USE_FP32_COMPUTE=use_fp32_compute
        )
        return grad_x, None, None, None

class GoLUTriton(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        return GoLUTritonFunction.apply(x, self.alpha, self.beta, self.gamma)