import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from core.models import MLP


class LearnableAggregation(nn.Module):
    """
    Learnable aggregation function based on MLP that can replace standard aggregation.
    
    This module learns how to combine node features and neighbor features instead of 
    using simple sum/mean aggregation.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0,
                 batch_norm: bool = True,
                 aggregation_type: str = 'concat'  # 'concat', 'sum', 'attention'
                 ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for MLP
            num_layers: Number of MLP layers
            activation_fn: Activation function
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            aggregation_type: How to combine self and neighbor features
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggregation_type = aggregation_type
        
        # Define input dimension based on aggregation type
        if aggregation_type == 'concat':
            mlp_input_dim = input_dim * 2  # self + neighbor features
        elif aggregation_type == 'sum':
            mlp_input_dim = input_dim
        elif aggregation_type == 'attention':
            raise NotImplementedError("Attention aggregation is not implemented yet.")
        else:
            raise ValueError(f"Unknown aggregation_type: {aggregation_type}")
        
        # Main aggregation MLP
        # TODO: change output_dim (remember constraind between [a, b] - we use just sigmoid so [0, 1])
        self.aggregation_mlp = nn.Sequential(
            MLP(
            output_dim=input_dim,  # Output same dimension as input
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=True),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
        
        # Set input dimension for the MLP
        self.aggregation_mlp.layers[0] = nn.Linear(mlp_input_dim, hidden_dim)
        
    def forward(self, 
                self_features: Tensor, 
                neighbor_features: Tensor,
                adj_t: Optional[SparseTensor] = None) -> Tensor:
        """
        Forward pass for learnable aggregation.
        
        Args:
            self_features: Features of the nodes themselves [N, D]
            neighbor_features: Aggregated features from neighbors [N, D]
            adj_t: Adjacency matrix (used for attention if needed)
            
        Returns:
            Aggregated features [N, D]
        """
        if self.aggregation_type == 'concat':
            # Concatenate self and neighbor features
            combined_features = torch.cat([self_features, neighbor_features], dim=-1)
            return self.aggregation_mlp(combined_features)
            
        elif self.aggregation_type == 'sum':
            # Sum self and neighbor features, then apply MLP
            combined_features = self_features + neighbor_features
            return self.aggregation_mlp(combined_features)
            
        elif self.aggregation_type == 'attention':
            raise NotImplementedError("Attention aggregation is not implemented yet.")
            # Use attention to weight the combination
            attention_input = torch.cat([self_features, neighbor_features], dim=-1)
            attention_weights = torch.sigmoid(self.attention_mlp(attention_input))
            
            # Weighted combination
            combined_features = attention_weights * self_features + (1 - attention_weights) * neighbor_features
            return self.aggregation_mlp(combined_features)
        
    def reset_parameters(self):
        """Reset all parameters."""
        def _reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.aggregation_mlp.apply(_reset)
        if hasattr(self, 'attention_mlp'):
            raise NotImplementedError("Attention aggregation is not implemented yet.")
            self.attention_mlp.reset_parameters()


class LearnableAggregationLayer(nn.Module):
    """
    A complete layer that performs message passing with learnable aggregation.
    This can be used as a drop-in replacement for standard GNN layers.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 aggregation_layers: int = 2,
                 projection_layers: int = 1,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0,
                 batch_norm: bool = True,
                 aggregation_type: str = 'concat',
                 normalize: bool = True):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dim: Hidden dimension for aggregation MLP
            aggregation_layers: Number of layers in aggregation MLP
            projection_layers: Number of layers in final projection MLP
            activation_fn: Activation function
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            aggregation_type: Type of aggregation ('concat', 'sum', 'attention')
            normalize: Whether to normalize features
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize
        
        # Learnable aggregation function
        self.learnable_aggregation = LearnableAggregation(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=aggregation_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            aggregation_type=aggregation_type
        )
        
        # Final projection to output dimension
        if projection_layers > 0:
            self.projection = MLP(
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=projection_layers,
                dropout=dropout,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
                plain_last=True
            )
            self.projection.layers[0] = nn.Linear(input_dim, hidden_dim if projection_layers > 1 else output_dim)
        else:
            self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        """
        Forward pass of the learnable aggregation layer.
        
        Args:
            x: Node features [N, input_dim]
            adj_t: Sparse adjacency tensor
            
        Returns:
            Updated node features [N, output_dim]
        """
        # Normalize input features if required
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        # Standard neighbor aggregation (sum of neighbors)
        neighbor_features = matmul(adj_t, x)
        
        # Apply learnable aggregation
        aggregated_features = self.learnable_aggregation(x, neighbor_features, adj_t)
        
        # Project to output dimension
        output = self.projection(aggregated_features)
        
        return output
    
    def reset_parameters(self):
        """Reset all parameters."""
        self.learnable_aggregation.reset_parameters()
        if hasattr(self.projection, 'reset_parameters'):
            self.projection.reset_parameters()
        else:
            # For simple Linear layer
            nn.init.xavier_uniform_(self.projection.weight)
            if self.projection.bias is not None:
                nn.init.zeros_(self.projection.bias)