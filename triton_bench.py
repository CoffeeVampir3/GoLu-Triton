import torch
import triton
import triton.testing
import numpy as np
from typing import List
from golu_triton import GoLUTriton

def run_benchmark(sizes: List[int], batch_sizes: List[int]):
    triton_golu = GoLUTriton().cuda()
    
    print("\nGoLU Triton Benchmark")
    print("-" * 80)
    print(f"{'Size':>10} {'Batch':>8} {'Fwd(μs)':>10} {'Bwd(μs)':>10}")
    print("-" * 80)
    
    for size in sizes:
        for batch in batch_sizes:
            x = torch.randn(batch, size).cuda().requires_grad_()
            grad = torch.randn_like(x)
            
            # fwd
            fwd_ms = triton.testing.do_bench(
                lambda: triton_golu(x),
                warmup=25,
                rep=100
            )
            
            # bwd
            def backward_pass():
                out = triton_golu(x)
                out.backward(grad, retain_graph=True)
            
            bwd_ms = triton.testing.do_bench(
                backward_pass,
                warmup=25,
                rep=100
            )
            
            fwd_us = fwd_ms * 1e3
            bwd_us = bwd_ms * 1e3
            print(f"{size:10d} {batch:8d} {fwd_us:10.2f} {bwd_us:10.2f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    
    sizes = [128, 256, 512, 1024, 2048, 4096]
    batch_sizes = [1, 32, 64, 128, 256, 512]
    
    run_benchmark(sizes, batch_sizes)