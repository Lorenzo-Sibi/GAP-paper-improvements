import numpy as np
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.data.loader import NodeDataLoader
from core.methods.node import GAP
from core.methods.node.gap.gap_learnable import GAPLearnable
from core.privacy.algorithms import PMA
from core.modules.base import Metrics, Stage


class NodePrivGAPLearn (GAPLearnable):
    """node-private GAP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[GAP], exclude=['batch_norm'])]
                 ):

        super().__init__(num_classes, 
            batch_norm=False, 
            batch_size=batch_size, 
            **kwargs
        )
        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm

        self.num_train_nodes = None  # will be used to set delta if it is 'auto'

        if hasattr(self, 'agg_weight_range'):
            try:
                weight_range_parts = kwargs.get('agg_weight_range', '0,1').split(',')
                a, b = float(weight_range_parts[0]), float(weight_range_parts[1])
                self.C = max(abs(a), abs(b))
            except:
                self.C = 1.0
        else:
            self.C = 1.0
            
        console.info(f"Edge-DP GAP Learnable with C = {self.C}")

    def calibrate(self):
        """Calibra usando AutoDP standard e poi scala per C"""
        
        # Usa AutoDP normale con sensitivity = 1
        
        with console.status('Calibrating noise using AutoDP + C scaling'):
            # Auto-set delta
            if self.delta == 'auto':
                if np.isinf(self.epsilon):
                    delta = 0.0
                else:
                    delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info(f'Auto-set δ = {delta:.2e}')
                self.delta = delta
            else:
                delta = self.delta
            
            # Calibra con AutoDP (sensitivity = 1)
            self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)
            self.noise_scale = self.C * self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)
            self.pma_mechanism.update(self.noise_scale)

            console.info(f'AutoDP noise (C=1): σ_base = {self.noise_scale:.6f}')
            console.info(f'Scaled noise (C={self.C}): σ = {self.noise_scale:.6f}')
            console.info(f'Privacy guarantee: (ε={self.epsilon}, δ={delta:.2e})')

    def _add_noise_to_aggregation_layers(self):
        """Apply C scaled noise"""
        
        if not self._classifier.learnable_agg_layers:
            return
        console.info(f'Adding noise scaled by C={self.C} to aggregation layers')
        
        for i, agg_layer in enumerate(self._classifier.learnable_agg_layers):
            original_forward = agg_layer.forward
            
            def create_noisy_forward(orig_forward):
                def noisy_forward(x, adj_t):
                    result = orig_forward(x, adj_t)
                    
                    if self.pma_mechanism and self.noise_scale > 0.0:
                        result = self.pma_mechanism(result, sensitivity=np.sqrt(self.max_degree))
                    return result
                return noisy_forward
            
            agg_layer.forward = create_noisy_forward(original_forward)


    def fit(self, data: Data, prefix: str = '') -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

        self._add_noise_to_aggregation_layers()

        return super().fit(data, prefix=prefix)


    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        dataloader = super().data_loader(data, stage)
        if stage == 'train':
            dataloader.poisson_sampling = True
        return dataloader