# GoLU Triton

Based on the paper [Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics](https://arxiv.org/abs/2502.03654)

Largely a port of the automl cuda kernel to triton [automl GoLu cuda kernel](https://github.com/automl/GoLU/tree/main)

## Overview

GoLU (Gompertz Linear Unit) is a novel self-gated activation function defined as:

```
GoLU(x) = x * Gompertz(x)
where Gompertz(x) = e^(-e^(-x))
```

This implementation provides a Triton-based kernel for efficient computation of GoLU and its gradients.

## Performance Benchmarks

Below are benchmarks comparing forward and backward pass times across different input sizes and batch sizes:

[benchmark table here]

## Installation

```bash
pip install triton
pip install torch
```

## Usage

### Basic Usage
```python
import torch
import torch.nn as nn
from golu_triton import GoLUTriton

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128)
        self.activation = GoLUTriton()  # default parameters (alpha=1.0, beta=1.0, gamma=1.0)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x

# Use the model
model = SimpleModel().cuda()
x = torch.randn(1, 128).cuda()
output = model(x)
```

### With Custom Parameters
```python
# Initialize with custom parameters
activation = GoLUTriton(alpha=0.8, beta=1.2, gamma=0.9)
```

## Implementation Details

The implementation consists of two main components:

1. Forward Pass Kernel:
   - Handles both full and low precision (fp16/bf16) computation
   - Automatically handles type conversion and memory management
   - Optimized for different input sizes through adaptive block sizing

2. Backward Pass Kernel:
   - Implements gradient computation for backpropagation
   - Includes numerical stability improvements
   - Handles edge cases and NaN prevention

### Key Features

- Automatic precision handling (fp32, fp16, bf16)
- Optimized memory access patterns
- Automatic block size selection
- Numerically stable gradient computation
- Efficient parallel execution
