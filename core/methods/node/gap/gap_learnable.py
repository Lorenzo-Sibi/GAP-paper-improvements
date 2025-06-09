# Sostituisci il contenuto di gap_learnable.py con questo:

import torch
from typing import Annotated, Iterable, Optional
from torch.nn import LazyLinear
import torch.nn.functional as F
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.methods.node.base import NodeClassification
from core.models.multi_mlp import MultiMLP
from core.modules.base import Metrics
from core.modules.node.em import EncoderModule
from core.methods.node.gap.learnable_agg import LearnableAggregation
from core.modules.node.cm_learnable import LearnableClassificationModule


class GAPLearnable(NodeClassification):
    """GAP method with Learnable Aggregation using MLP-based aggregation functions"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hops: Annotated[int, ArgInfo(help='number of hops', option='-k')] = 2,
                 hidden_dim: Annotated[int, ArgInfo(help='dimension of the hidden layers')] = 16,
                 encoder_layers: Annotated[int, ArgInfo(help='number of encoder MLP layers')] = 2,
                 base_layers: Annotated[int, ArgInfo(help='number of base MLP layers')] = 1,
                 head_layers: Annotated[int, ArgInfo(help='number of head MLP layers')] = 1,
                 combine: Annotated[str, ArgInfo(help='combination type of transformed hops', choices=MultiMLP.supported_combinations)] = 'cat',
                 activation: Annotated[str, ArgInfo(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout: Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm: Annotated[bool, ArgInfo(help='if true, then model uses batch normalization')] = True,
                 encoder_epochs: Annotated[int, ArgInfo(help='number of epochs for encoder pre-training (ignored if encoder_layers=0)')] = 100,
                 
                 # New parameters for learnable aggregation
                 use_learnable_agg: Annotated[bool, ArgInfo(help='whether to use learnable aggregation')] = True,
                 agg_function: Annotated[str, ArgInfo(help='type of learnable weight function', choices=['mlp', 'attention'])] = 'mlp',
                 agg_hidden_dim: Annotated[int, ArgInfo(help='hidden dimension for aggregation function')] = 16,
                 agg_layers: Annotated[int, ArgInfo(help='number of layers in aggregation function (for MLP)')] = 3,
                 agg_weight_range: Annotated[str, ArgInfo(help='weight range as "a,b" (e.g., "0,1")')] = "0,1",
                 **kwargs: Annotated[dict, ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        super().__init__(num_classes, **kwargs)

        if encoder_layers == 0 and encoder_epochs > 0:
            console.warning('encoder_layers is 0, setting encoder_epochs to 0')
            encoder_epochs = 0

        self.hops = hops
        self.encoder_layers = encoder_layers
        self.encoder_epochs = encoder_epochs
        self.use_learnable_agg = use_learnable_agg
        self.agg_function = agg_function
        activation_fn = self.supported_activations[activation]
        
        # Parse weight range
        try:
            weight_range_parts = agg_weight_range.split(',')
            weight_range = (float(weight_range_parts[0]), float(weight_range_parts[1]))
        except (ValueError, IndexError):
            console.warning(f'Invalid weight range "{agg_weight_range}", using default (0,1)')
            weight_range = (0.0, 1.0)
        finally:
            self.weight_range = weight_range

        # Encoder module (unchanged from original GAP)
        self._encoder = EncoderModule(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            head_layers=1,
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        # Initialize learnable aggregation layers for each hop
        learnable_agg_layers = None
        if self.use_learnable_agg:
            # Prepare weight function kwargs based on function type
            weight_function_kwargs = {}
            
            if agg_function == 'mlp':
                weight_function_kwargs.update({
                    'hidden_dim': agg_hidden_dim,
                    'num_layers': agg_layers,
                    'activation_fn': activation_fn,
                    'dropout': dropout,
                    'batch_norm': batch_norm
                })
            elif agg_function == 'attention':
                weight_function_kwargs.update({
                    'attention_dim': agg_hidden_dim,
                    'num_heads': max(1, agg_hidden_dim // 16)
                })
            
            learnable_agg_layers = torch.nn.ModuleList([
                LearnableAggregation(
                    input_dim=hidden_dim,
                    weight_function=agg_function,
                    weight_range=weight_range,
                    use_efficient_computation=True,
                    **weight_function_kwargs
                ) for _ in range(hops)
            ])
            
            console.info(f'Using learnable aggregation: {agg_function} with weight range {weight_range}')

        # Classifier module che gestisce aggregazioni al volo
        self._classifier = LearnableClassificationModule(
            num_classes=num_classes,
            hops=hops,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combine,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            learnable_agg_layers=learnable_agg_layers,
            use_learnable_agg=use_learnable_agg,
            include_lazy_linear=encoder_layers == 0
        )

    @property
    def classifier(self) -> LearnableClassificationModule:
        return self._classifier

    def reset_parameters(self):
        self._encoder.reset_parameters()
        self._classifier.reset_parameters()

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        self.data = data.to(self.device, non_blocking=True)
        
        # Pre-train encoder (same as original GAP)
        if self.encoder_layers > 0:
            self.data = self.pretrain_encoder(self.data, prefix=prefix)
        
        # Train classifier
        return super().fit(self.data, prefix=prefix)

    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        if data is None or data == self.data:
            data = self.data
        else:
            data = data.to(self.device, non_blocking=True)
            data.x = self._encoder.predict(data)

        return super().test(data, prefix=prefix)

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if data is None or data == self.data:
            data = self.data
        else:
            data.x = self._encoder.predict(data)

        return super().predict(data)

    def pretrain_encoder(self, data: Data, prefix: str) -> Data:
        """Pre-train encoder (same as original GAP)"""
        console.info('pretraining encoder')
        self._encoder.to(self.device)
        
        self.trainer.fit(
            model=self._encoder,
            epochs=self.encoder_epochs,
            optimizer=self.configure_encoder_optimizer(), 
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=None,
            checkpoint=True,
            prefix=f'{prefix}encoder/',
        )

        self.trainer.reset()
        data.x = self._encoder.predict(data)
        return data

    def configure_encoder_optimizer(self) -> Optimizer:
        """Configure optimizer for encoder training"""
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        params = list(self._encoder.parameters())
        
        # Add learnable aggregation parameters to encoder optimizer
        if self.use_learnable_agg and self._classifier.learnable_agg_layers:
            for layer in self._classifier.learnable_agg_layers:
                params.extend(layer.parameters())
        
        return Optim(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def configure_optimizer(self) -> Optimizer:
        """Configure optimizer for classifier training"""
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        params = list(self._classifier.parameters())
        
        # Include learnable aggregation parameters if encoder wasn't pre-trained
        if self.use_learnable_agg and self.encoder_epochs == 0 and self._classifier.learnable_agg_layers:
            for layer in self._classifier.learnable_agg_layers:
                params.extend(layer.parameters())
        
        return Optim(params, lr=self.learning_rate, weight_decay=self.weight_decay)