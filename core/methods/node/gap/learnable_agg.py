import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from core.models import MLP
from core.methods.node.gap.learnable_weight import *


class LearnableAggregation(nn.Module):
    """
    Generalized learnable aggregation module supporting multiple weight function strategies.
    
    Implements: AGG_f(A, X)_v = Σ_{u: A_vu=1} f(X_v, X_u) · X_u
    
    where f is any learnable weight function.
    """
    
    # Registry of available weight functions
    WEIGHT_FUNCTIONS = {
        'mlp': MLPWeightFunction,
        'attention': AttentionWeightFunction,
    }
    
    def __init__(self,
                 input_dim: int,
                 weight_function: Union[str, LearnableWeightFunction] = 'mlp',
                 weight_range: tuple = (0.0, 1.0),
                 use_efficient_computation: bool = True,
                 chunk_size: int = 10000,
                 **weight_function_kwargs):
        """
        Args:
            input_dim: Dimension of node features
            weight_function: Either string name or LearnableWeightFunction instance
            weight_range: Range [a, b] for output weights
            use_efficient_computation: Whether to use chunked computation for large graphs
            chunk_size: Size of chunks for efficient computation
            **weight_function_kwargs: Arguments passed to weight function constructor
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.weight_range = weight_range
        self.use_efficient_computation = use_efficient_computation
        self.chunk_size = chunk_size
        
        # Initialize weight function
        if isinstance(weight_function, str):
            if weight_function not in self.WEIGHT_FUNCTIONS:
                raise ValueError(f"Unknown weight function: {weight_function}. "
                               f"Available: {list(self.WEIGHT_FUNCTIONS.keys())}")
            
            WeightFunctionClass = self.WEIGHT_FUNCTIONS[weight_function]
            self.weight_function = WeightFunctionClass(
                input_dim=input_dim,
                weight_range=weight_range,
                **weight_function_kwargs
            )
            self.weight_function_name = weight_function
        else:
            self.weight_function = weight_function
            self.weight_function_name = weight_function.__class__.__name__
        
    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        """
        Forward pass of learnable aggregation.
        
        Args:
            x: Node features [N, input_dim]
            adj_t: Sparse adjacency matrix [N, N]
            
        Returns:
            Aggregated features [N, input_dim]
        """
        if self.use_efficient_computation and adj_t.nnz() > self.chunk_size:
            return self._forward_efficient(x, adj_t)
        else:
            return self._forward_direct(x, adj_t)
    
    def _forward_direct(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        """Direct computation for smaller graphs."""
        row, col, _ = adj_t.coo()
        
        # Get node features for each edge
        x_v = x[row]  # Target nodes
        x_u = x[col]  # Source nodes
        
        # Compute weights using the learnable function
        weights = self.weight_function(x_v, x_u)  # [num_edges, 1]
        
        # Apply weights to source features
        weighted_features = weights * x_u  # [num_edges, input_dim]
        
        # Aggregate for each target node
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), weighted_features)
        
        return aggregated
    
    def _forward_efficient(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        """Efficient chunked computation for larger graphs."""
        N, D = x.shape
        aggregated = torch.zeros_like(x)
        
        row, col, _ = adj_t.coo()
        
        # Process edges in chunks
        for i in range(0, len(row), self.chunk_size):
            end_idx = min(i + self.chunk_size, len(row))
            
            row_chunk = row[i:end_idx]
            col_chunk = col[i:end_idx]
            
            x_v_chunk = x[row_chunk]
            x_u_chunk = x[col_chunk]
            
            # Compute weights for this chunk
            weights_chunk = self.weight_function(x_v_chunk, x_u_chunk)
            weighted_features_chunk = weights_chunk * x_u_chunk
            
            # Aggregate this chunk
            aggregated.scatter_add_(0, row_chunk.unsqueeze(1).expand(-1, D), weighted_features_chunk)
        
        return aggregated
    
    def reset_parameters(self):
        """Reset parameters of the weight function."""
        if hasattr(self.weight_function, 'reset_parameters'):
            self.weight_function.reset_parameters()
    
    def get_weight_function_info(self) -> Dict[str, Any]:
        """Get information about the current weight function."""
        return {
            'name': self.weight_function_name,
            'input_dim': self.input_dim,
            'weight_range': self.weight_range,
            'parameters': sum(p.numel() for p in self.weight_function.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.weight_function.parameters() if p.requires_grad)
        }
    
    @classmethod
    def create_weight_function(cls, 
                              function_name: str, 
                              input_dim: int, 
                              **kwargs) -> LearnableWeightFunction:
        """Factory method to create weight functions."""
        if function_name not in cls.WEIGHT_FUNCTIONS:
            raise ValueError(f"Unknown weight function: {function_name}")
        
        WeightFunctionClass = cls.WEIGHT_FUNCTIONS[function_name]
        return WeightFunctionClass(input_dim=input_dim, **kwargs)