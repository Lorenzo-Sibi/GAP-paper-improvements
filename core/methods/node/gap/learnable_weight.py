import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union, Dict, Any
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from core.models import MLP
from abc import ABC, abstractmethod


class LearnableWeightFunction(ABC, nn.Module):
    """
    Abstract base class for learnable weight functions.
    
    Defines the interface for functions f: R^d × R^d → [a,b] that compute
    scalar weights for aggregation.
    """
    
    def __init__(self, input_dim: int, weight_range: tuple = (0.0, 1.0)):
        super().__init__()
        self.input_dim = input_dim
        self.weight_range = weight_range
        self.a, self.b = weight_range
    
    @abstractmethod
    def forward(self, x_v: Tensor, x_u: Tensor) -> Tensor:
        """
        Compute weights for aggregation.
        
        Args:
            x_v: Target node features [batch_size, input_dim]
            x_u: Source node features [batch_size, input_dim]
            
        Returns:
            weights: Scalar weights [batch_size, 1] in range [a, b]
        """
        pass
    
    def scale_weights(self, raw_weights: Tensor) -> Tensor:
        """Scale raw weights to the desired range [a, b]."""
        # Assume raw weights are in [0, 1] after sigmoid/softmax
        return self.a + (self.b - self.a) * raw_weights


class MLPWeightFunction(LearnableWeightFunction):
    """
    MLP-based weight function: f(x_v, x_u) = MLP(concat(x_v, x_u))
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0,
                 batch_norm: bool = True,
                 weight_range: tuple = (0.0, 1.0)):
        super().__init__(input_dim, weight_range)
        
        self.mlp = MLP(
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=True
        )
        
        if num_layers > 0:
            self.mlp.layers[0] = nn.Linear(2 * input_dim, 
                                          hidden_dim if num_layers > 1 else 1)
    
    def forward(self, x_v: Tensor, x_u: Tensor) -> Tensor:
        # Concatenate features
        combined = torch.cat([x_v, x_u], dim=-1)
        
        # Apply MLP and sigmoid activation
        raw_weights = torch.sigmoid(self.mlp(combined))
        
        # Scale to desired range
        return self.scale_weights(raw_weights)
    
    def reset_parameters(self):
        self.mlp.reset_parameters()


class AttentionWeightFunction(LearnableWeightFunction):
    """
    Attention-based weight function: f(x_v, x_u) = softmax(W_q*x_v^T * W_k*x_u)
    """
    
    def __init__(self,
                 input_dim: int,
                 attention_dim: int = 64,
                 num_heads: int = 1,
                 weight_range: tuple = (0.0, 1.0)):
        super().__init__(input_dim, weight_range)
        
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.temperature = nn.Parameter(torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        
    def forward(self, x_v: Tensor, x_u: Tensor) -> Tensor:
        batch_size = x_v.size(0)
        
        # Compute queries and keys
        q = self.W_q(x_v).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D/H]
        k = self.W_k(x_u).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D/H]
        
        # Compute attention scores
        scores = (q * k).sum(dim=-1) / self.temperature  # [B, H]
        
        # Average over heads and apply sigmoid
        weights = torch.sigmoid(scores.mean(dim=-1, keepdim=True))  # [B, 1]
        
        return self.scale_weights(weights)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.constant_(self.temperature, torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))