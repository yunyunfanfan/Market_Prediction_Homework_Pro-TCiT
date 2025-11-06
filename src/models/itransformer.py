"""
iTransformer: Inverted Transformers for Time Series Forecasting

Based on the paper: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
https://arxiv.org/abs/2310.06625
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for transformer
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class InvertedTransformerEncoder(nn.Module):
    """
    Inverted Transformer Encoder
    Instead of treating time steps as tokens, we treat variables as tokens
    """
    def __init__(self, 
                 n_variables: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        
        self.n_variables = n_variables
        self.d_model = d_model
        
        # Project each variable's time series to d_model dimension
        self.variable_embedding = nn.Linear(1, d_model)
        
        # Positional encoding for variables
        self.pos_encoder = nn.Parameter(torch.randn(1, n_variables, d_model))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, n_variables)
            
        Returns:
            Encoded features of shape (batch_size, n_variables, d_model)
        """
        batch_size, seq_len, n_variables = x.shape
        
        # Reshape: (batch, seq_len, n_vars) -> (batch, n_vars, seq_len)
        x = x.transpose(1, 2)
        
        # Process each time step of each variable
        # (batch, n_vars, seq_len) -> (batch * n_vars, seq_len, 1)
        x = x.reshape(batch_size * n_variables, seq_len, 1)
        
        # Embed: (batch * n_vars, seq_len, 1) -> (batch * n_vars, seq_len, d_model)
        x = self.variable_embedding(x)
        
        # Average pooling over time: (batch * n_vars, seq_len, d_model) -> (batch * n_vars, d_model)
        x = x.mean(dim=1)
        
        # Reshape back: (batch * n_vars, d_model) -> (batch, n_vars, d_model)
        x = x.reshape(batch_size, n_variables, self.d_model)
        
        # Add positional encoding for variables
        x = x + self.pos_encoder
        x = self.dropout(x)
        
        # Apply transformer encoder
        # Now each variable is a token
        x = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        return x


class iTransformer(nn.Module):
    """
    iTransformer model for multivariate time series forecasting
    
    Key idea: Treat each variable as a token, capture multivariate correlations through attention
    """
    def __init__(self,
                 n_variables: int,
                 lookback: int,
                 forecast_horizon: int = 1,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 1,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        """
        Args:
            n_variables: Number of input variables/features
            lookback: Number of historical time steps
            forecast_horizon: Number of future steps to predict
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.n_variables = n_variables
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        
        # Encoder: treats variables as tokens
        self.encoder = InvertedTransformerEncoder(
            n_variables=n_variables,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # Decoder: projects from d_model back to forecast horizon
        # Each variable's representation is projected to its forecast
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, dim_feedforward // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, forecast_horizon)
            ) for _ in range(num_decoder_layers)
        ])
        
        # Final projection for single target prediction
        self.final_projection = nn.Sequential(
            nn.Linear(n_variables * forecast_horizon, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, lookback, n_variables)
            
        Returns:
            Predictions of shape (batch_size, 1) for single-step forecasting
        """
        # Encode: (batch, lookback, n_vars) -> (batch, n_vars, d_model)
        encoded = self.encoder(x)
        
        # Decode each variable
        # (batch, n_vars, d_model) -> (batch, n_vars, forecast_horizon)
        decoded = encoded
        for decoder in self.decoder_layers:
            # Process each variable's representation
            batch_size, n_vars, d_model = decoded.shape
            decoded_flat = decoded.reshape(batch_size * n_vars, d_model)
            decoded_flat = decoder[0](decoded_flat) if len(decoder) == 1 else decoder(decoded_flat)
            decoded = decoded_flat.reshape(batch_size, n_vars, -1)
        
        # Flatten and project to single output
        # (batch, n_vars, forecast_horizon) -> (batch, n_vars * forecast_horizon)
        batch_size = decoded.shape[0]
        decoded_flat = decoded.reshape(batch_size, -1)
        
        # Final prediction: (batch, n_vars * forecast_horizon) -> (batch, 1)
        output = self.final_projection(decoded_flat)
        
        return output


class iTransformerSimple(nn.Module):
    """
    Simplified iTransformer for direct single-step prediction
    """
    def __init__(self,
                 n_variables: int,
                 lookback: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_variables = n_variables
        self.lookback = lookback
        self.d_model = d_model
        
        # Embedding: project each time step to d_model
        self.time_embedding = nn.Linear(n_variables, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=lookback, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * lookback, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, lookback, n_variables)
        Returns:
            (batch_size, 1)
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Embed: (batch, seq_len, n_vars) -> (batch, seq_len, d_model)
        x = self.time_embedding(x)
        
        # Add positional encoding: (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer: (batch, seq_len, d_model)
        x = self.transformer(x)
        
        # Flatten and project
        x = x.reshape(batch_size, -1)  # (batch, seq_len * d_model)
        x = self.output_projection(x)  # (batch, 1)
        
        return x


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    batch_size = 32
    lookback = 96
    n_variables = 100
    
    # Test iTransformer
    model = iTransformer(
        n_variables=n_variables,
        lookback=lookback,
        forecast_horizon=1,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=1,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"iTransformer parameters: {count_parameters(model):,}")
    
    # Test input
    x = torch.randn(batch_size, lookback, n_variables)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test simplified version
    model_simple = iTransformerSimple(
        n_variables=n_variables,
        lookback=lookback,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"\niTransformerSimple parameters: {count_parameters(model_simple):,}")
    
    y_simple = model_simple(x)
    print(f"Output shape (simple): {y_simple.shape}")

