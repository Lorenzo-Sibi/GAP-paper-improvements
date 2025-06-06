import numpy as np
import torch
from typing import Annotated, Literal, Union, Optional
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.methods.node.gap.gap_learnable import GAPLearnable
from core.privacy.algorithms.graph.pma import PMA
from core.modules.base import Metrics


class EdgePrivGAPLearn(GAPLearnable):
    """GAP Learnable with Edge-DP using AutoDP scaled by C"""

    def __init__(self,
                 num_classes,
                 epsilon: Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta: Annotated[Union[Literal['auto'], float], ArgInfo(help='DP delta parameter')] = 'auto',
                 **kwargs: Annotated[dict, ArgInfo(help='extra options passed to base class', bases=[GAPLearnable])]
                 ):

        super().__init__(num_classes, **kwargs)        
        self.epsilon = epsilon
        self.delta = delta
        self.num_edges = None
        self.pma_mechanism = None
        
        # Calcola C dal range dei pesi
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

    def calibrate_noise(self):
        """Calibra usando AutoDP standard e poi scala per C"""
        
        # Usa AutoDP normale con sensitivity = 1
        
        with console.status('Calibrating noise using AutoDP + C scaling'):
            # Auto-set delta
            if self.delta == 'auto':
                if np.isinf(self.epsilon):
                    delta = 0.0
                else:
                    delta = 1.0 / (10 ** len(str(self.num_edges)))
                console.info(f'Auto-set δ = {delta:.2e}')
                self.delta = delta
            else:
                delta = self.delta
            
            # Calibra con AutoDP (sensitivity = 1)
            autodp_noise = PMA(noise_scale=0.0, hops=self.hops).calibrate(eps=self.epsilon, delta=delta)
            
            self.noise_scale = self.C * autodp_noise
            self.pma_mechanism = PMA(noise_scale=self.noise_scale, hops=self.hops)
            
            console.info(f'AutoDP noise (C=1): σ_base = {autodp_noise:.6f}')
            console.info(f'Scaled noise (C={self.C}): σ = {self.noise_scale:.6f}')
            console.info(f'Privacy guarantee: (ε={self.epsilon}, δ={delta:.2e})')

    def _add_noise_to_aggregation_layers(self):
        """Applica rumore con il scaling C"""
        
        if not self.use_learnable_agg or not hasattr(self._classifier, 'learnable_agg_layers'):
            return
            
        if not self._classifier.learnable_agg_layers:
            return
            
        console.info(f'Adding noise scaled by C={self.C} to aggregation layers')
        
        for i, agg_layer in enumerate(self._classifier.learnable_agg_layers):
            original_forward = agg_layer.forward
            
            def create_noisy_forward(orig_forward):
                def noisy_forward(x, adj_t):
                    result = orig_forward(x, adj_t)
                    
                    if self.pma_mechanism and self.noise_scale > 0.0:
                        # Usa AutoDP ma con noise_scale già scalato per C
                        result = self.pma_mechanism(result, sensitivity=1)
                    
                    return result
                return noisy_forward
            
            agg_layer.forward = create_noisy_forward(original_forward)

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges
            self.calibrate_noise()
            
            if self.use_learnable_agg:
                self._add_noise_to_aggregation_layers()

        return super().fit(data, prefix=prefix)

    def get_privacy_accountant(self) -> dict:
        return {
            'method': 'GAP Learnable EDP (AutoDP + C scaling)',
            'epsilon': self.epsilon,
            'delta': self.delta,
            'noise_scale': getattr(self, 'noise_scale', 0.0),
            'sensitivity_C': self.C,
            'autodp_base_noise': getattr(self, 'noise_scale', 0.0) / self.C if self.C > 0 else 0.0,
            'scaling_formula': 'σ_final = C × σ_autodp',
            'hops': self.hops,
            'num_edges': self.num_edges
        }