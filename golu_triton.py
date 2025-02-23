import triton
import triton.language as tl
import torch

# Low precision uses fp32 upcasting as done in the og cuda kernels
@triton.jit
def golu_forward_kernel_low_precision(
    input_ptr,
    output_ptr,
    alpha,
    beta,
    gamma,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    golu = x * alpha * tl.exp(-beta * tl.exp(-gamma * x))
    
    tl.store(output_ptr + offsets, golu.to(tl.load(input_ptr + offsets, mask=mask).dtype), mask=mask)

# Low precision uses fp32 upcasting as done in the og cuda kernels
@triton.jit
def golu_backward_kernel_low_precision(
    grad_output_ptr,
    x_ptr,
    grad_x_ptr,
    alpha,
    beta,
    gamma,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    inner_exp = tl.exp(-gamma * x)
    grad_x = grad_output * alpha * tl.exp(-beta * inner_exp) * (1.0 + beta * gamma * x * inner_exp)
    grad_x = tl.where(grad_x == grad_x, grad_x, 0.0)
    
    tl.store(grad_x_ptr + offsets, grad_x.to(tl.load(x_ptr + offsets, mask=mask).dtype), mask=mask)

@triton.jit
def golu_forward_kernel_fp32(
    input_ptr,
    output_ptr,
    alpha,
    beta,
    gamma,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    golu = x * alpha * tl.exp(-beta * tl.exp(-gamma * x))
    tl.store(output_ptr + offsets, golu, mask=mask)

@triton.jit
def golu_backward_kernel_fp32(
    grad_output_ptr,
    x_ptr,
    grad_x_ptr,
    alpha,
    beta,
    gamma,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    inner_exp = tl.exp(-gamma * x)
    grad_x = grad_output * alpha * tl.exp(-beta * inner_exp) * (1.0 + beta * gamma * x * inner_exp)
    grad_x = tl.where(grad_x == grad_x, grad_x, 0.0)
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
        BLOCK_SIZE = min(max(32, triton.next_power_of_2(n_elements // 4096)), 1024)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        if x.dtype in [torch.float16, torch.bfloat16]:
            kernel = golu_forward_kernel_low_precision
        else:
            kernel = golu_forward_kernel_fp32

        kernel[grid](
            x, output, alpha, beta, gamma, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        if x.dtype in [torch.float16, torch.bfloat16]:
            kernel = golu_backward_kernel_low_precision
        else:
            kernel = golu_backward_kernel_fp32

        kernel[grid](
            grad_output, x, grad_x, ctx.alpha, ctx.beta, ctx.gamma,
            n_elements, BLOCK_SIZE=1024
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