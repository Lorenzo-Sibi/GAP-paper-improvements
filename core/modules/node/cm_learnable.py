# Crea un nuovo file: core/modules/node/cm_learnable.py

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from core.models.multi_mlp import MultiMLP
from core.modules.node.mlp import MLPNodeClassifier
from core.methods.node.gap.learnable_agg import LearnableAggregation
from core.modules.base import Stage, Metrics
from typing import Optional, List, Callable


class LearnableClassificationModule(MLPNodeClassifier):
    """
    Classification module che calcola le aggregazioni learnable al volo
    durante il forward pass invece di pre-calcolarle.
    """
    
    def __init__(self, *, 
                 num_classes: int,
                 hops: int,
                 hidden_dim: int = 16,  
                 base_layers: int = 2, 
                 head_layers: int = 1, 
                 combination: MultiMLP.CombType = 'cat',
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 # Parametri per aggregazione learnable
                 learnable_agg_layers: Optional[List[LearnableAggregation]] = None,
                 use_learnable_agg: bool = True,
                 ):

        super().__init__(num_classes=num_classes)  # Dummy initialization

        self.hops = hops
        self.hidden_dim = hidden_dim
        self.use_learnable_agg = use_learnable_agg
        self.learnable_agg_layers = learnable_agg_layers or []
        
        # MultiMLP che combina le rappresentazioni multi-hop
        self.model = MultiMLP(
            num_channels=hops + 1,  # +1 per le features originali (0-hop)
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combination,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=True,
        )

    def _compute_aggregations_on_the_fly(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        """
        Calcola le aggregazioni al volo durante il forward pass.
        Questo permette ai gradienti di fluire correttamente.
        """
        x = F.normalize(x, p=2, dim=-1)
        x_list = [x]  # Features originali (0-hop)
        
        current_x = x
        for hop_idx in range(self.hops):
            if self.use_learnable_agg and hop_idx < len(self.learnable_agg_layers):
                # Aggregazione learnable - i gradienti fluiscono!
                current_x = self.learnable_agg_layers[hop_idx](current_x, adj_t)
            else:
                # Aggregazione standard come fallback
                from torch_sparse import matmul
                current_x = matmul(adj_t, current_x)
                current_x = F.normalize(current_x, p=2, dim=-1)
            
            x_list.append(current_x)
        
        # Stack per MultiMLP: [N, D, hops+1]
        return torch.stack(x_list, dim=-1)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        """
        Forward pass che calcola aggregazioni e classificazione in un colpo solo.
        """
        # Calcola aggregazioni multi-hop al volo
        multi_hop_features = self._compute_aggregations_on_the_fly(x, adj_t)
        
        # Applica MultiMLP per la classificazione
        return self.model(multi_hop_features)

    def step(self, data: Data, stage: Stage) -> tuple[Optional[Tensor], Metrics]:
        """
        Step modificato per gestire le aggregazioni al volo.
        """
        mask = data[f'{stage}_mask']
        
        # Forward pass con aggregazioni al volo
        output = self.forward(data.x, data.adj_t)
        predictions = output[mask]
        targets = data.y[mask]
        
        # Calcola accuracy
        preds = F.log_softmax(predictions, dim=-1)
        acc = preds.argmax(dim=1).eq(targets).float().mean() * 100
        metrics = {'acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=targets)
            metrics['loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        """
        Predict modificato per gestire le aggregazioni al volo.
        """
        self.eval()
        output = self.forward(data.x, data.adj_t)
        return torch.softmax(output, dim=-1)