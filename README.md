# GoLU Triton

Based on the paper [Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics](https://arxiv.org/abs/2502.03654)

Largely a port of the automl cuda kernel to triton [automl GoLu cuda kernel](https://github.com/automl/GoLU/tree/main)

Supports fp32 and automatic fp16/bf16 upcasting.

## Overview

GoLU (Gompertz Linear Unit) is a novel self-gated activation function defined as:

```
GoLU(x) = x * Gompertz(x)
where Gompertz(x) = e^(-e^(-x))
```

This implementation provides a Triton-based kernel for efficient computation of GoLU and its gradients.

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
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128)
        self.activation = GoLUTriton()  # default parameters (alpha=1.0, beta=1.0, gamma=1.0)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x

model = SimpleModel().cuda()
x = torch.randn(1, 128).cuda()
output = model(x)
```

### With Custom Parameters
```python
activation = GoLUTriton(alpha=0.8, beta=1.2, gamma=0.9)
```
