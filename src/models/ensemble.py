"""
Ensemble Model: Linear Weighted Combination of iTransformer and TimeCMA

This module provides an ensemble model that combines predictions from
iTransformer and TimeCMA models using linear weighted averaging.
"""

import torch
import torch.nn as nn
from typing import Optional
from .itransformer import iTransformer, iTransformerSimple
from .timecma import TimeCMA


class EnsembleModel(nn.Module):
    """
    Ensemble model that linearly combines iTransformer and TimeCMA predictions
    
    The final prediction is: 
    output = alpha * itransformer_output + (1 - alpha) * timecma_output
    
    where alpha is a learnable or fixed weight parameter.
    """
    def __init__(self,
                 itransformer_model: nn.Module,
                 timecma_model: nn.Module,
                 alpha: Optional[float] = None,
                 learnable_weight: bool = True):
        """
        Args:
            itransformer_model: Pre-initialized iTransformer model
            timecma_model: Pre-initialized TimeCMA model
            alpha: Fixed weight for iTransformer (if None and learnable_weight=False, uses 0.5)
            learnable_weight: If True, alpha is a learnable parameter
        """
        super().__init__()
        
        self.itransformer = itransformer_model
        self.timecma = timecma_model
        
        if learnable_weight:
            # Learnable weight parameter (initialized to 0.5 for equal weighting)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            # Fixed weight
            if alpha is None:
                alpha = 0.5
            self.register_buffer('alpha', torch.tensor(alpha))
            self.alpha = torch.tensor(alpha, requires_grad=False)
        
        self.learnable_weight = learnable_weight
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, lookback, n_variables)
        
        Returns:
            predictions: (batch_size, 1)
        """
        # Get predictions from both models
        itransformer_pred = self.itransformer(x)  # (batch, 1)
        timecma_pred = self.timecma(x)  # (batch, 1)
        
        # Apply weight constraint: alpha should be in [0, 1]
        if self.learnable_weight:
            # Use sigmoid to ensure alpha is in [0, 1]
            alpha_constrained = torch.sigmoid(self.alpha)
        else:
            alpha_constrained = self.alpha
        
        # Linear weighted combination
        ensemble_pred = alpha_constrained * itransformer_pred + (1 - alpha_constrained) * timecma_pred
        
        return ensemble_pred
    
    def get_weight(self):
        """Get current weight value (constrained to [0, 1])"""
        if self.learnable_weight:
            return torch.sigmoid(self.alpha).item()
        else:
            return self.alpha.item()


def create_ensemble_model(n_variables: int,
                          lookback: int,
                          itransformer_type: str = 'simple',
                          itransformer_config: Optional[dict] = None,
                          timecma_config: Optional[dict] = None,
                          alpha: Optional[float] = None,
                          learnable_weight: bool = True) -> EnsembleModel:
    """
    Factory function to create an ensemble model with default configurations
    
    Args:
        n_variables: Number of input variables/features
        lookback: Number of historical time steps
        itransformer_type: 'simple' or 'full'
        itransformer_config: Custom config for iTransformer (optional)
        timecma_config: Custom config for TimeCMA (optional)
        alpha: Fixed weight for iTransformer (optional)
        learnable_weight: Whether weight is learnable
    
    Returns:
        EnsembleModel instance
    """
    # Default iTransformer config
    if itransformer_config is None:
        if itransformer_type == 'simple':
            itransformer_config = {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 3,
                'dim_feedforward': 1024,
                'dropout': 0.1
            }
        else:
            itransformer_config = {
                'forecast_horizon': 1,
                'd_model': 256,
                'nhead': 8,
                'num_encoder_layers': 3,
                'num_decoder_layers': 1,
                'dim_feedforward': 1024,
                'dropout': 0.1
            }
    
    # Default TimeCMA config
    if timecma_config is None:
        timecma_config = {
            'd_model': 256,
            'nhead': 8,
            'num_ts_layers': 2,
            'num_prompt_layers': 3,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'alignment_temperature': 0.07
        }
    
    # Create iTransformer model
    if itransformer_type == 'simple':
        itransformer = iTransformerSimple(
            n_variables=n_variables,
            lookback=lookback,
            **itransformer_config
        )
    else:
        itransformer = iTransformer(
            n_variables=n_variables,
            lookback=lookback,
            **itransformer_config
        )
    
    # Create TimeCMA model
    timecma = TimeCMA(
        n_variables=n_variables,
        lookback=lookback,
        **timecma_config
    )
    
    # Create ensemble
    ensemble = EnsembleModel(
        itransformer_model=itransformer,
        timecma_model=timecma,
        alpha=alpha,
        learnable_weight=learnable_weight
    )
    
    return ensemble


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the ensemble model
    batch_size = 32
    lookback = 50
    n_variables = 100
    
    # Test with learnable weight
    print("Testing Ensemble Model with learnable weight...")
    ensemble = create_ensemble_model(
        n_variables=n_variables,
        lookback=lookback,
        itransformer_type='simple',
        learnable_weight=True
    )
    
    print(f"Ensemble parameters: {count_parameters(ensemble):,}")
    print(f"Initial weight (alpha): {ensemble.get_weight():.4f}")
    
    x = torch.randn(batch_size, lookback, n_variables)
    y = ensemble(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test with fixed weight
    print("\nTesting Ensemble Model with fixed weight...")
    ensemble_fixed = create_ensemble_model(
        n_variables=n_variables,
        lookback=lookback,
        itransformer_type='simple',
        alpha=0.6,
        learnable_weight=False
    )
    
    print(f"Fixed weight (alpha): {ensemble_fixed.get_weight():.4f}")
    y_fixed = ensemble_fixed(x)
    print(f"Output shape: {y_fixed.shape}")

