"""
TimeCMA: Cross-Modality Alignment for Multivariate Time Series Forecasting

Based on the paper: "TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting 
via Cross-Modality Alignment" (https://arxiv.org/pdf/2406.01638)

This is a simplified implementation that captures the core ideas:
- Dual-modality encoding (time series branch + text/prompt branch)
- Cross-modality alignment via similarity-based retrieval
- Last token embedding for efficient prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TimeSeriesEncoder(nn.Module):
    """
    Time Series Encoding Branch
    Extracts disentangled yet weak time series embeddings
    """
    def __init__(self,
                 n_variables: int,
                 lookback: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_variables = n_variables
        self.lookback = lookback
        self.d_model = d_model
        
        # Embed each time step
        self.time_embedding = nn.Linear(n_variables, d_model)
        
        # Positional encoding for time steps
        self.pos_encoder = nn.Parameter(torch.randn(1, lookback, d_model))
        
        # Transformer encoder for time series
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, lookback, n_variables)
        Returns:
            embeddings: (batch_size, lookback, d_model)
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Embed time steps
        x = self.time_embedding(x)  # (batch, lookback, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)  # (batch, lookback, d_model)
        
        return x


class PromptEncoder(nn.Module):
    """
    LLM-Empowered Encoding Branch (Simplified)
    Wraps time series with text prompts to obtain entangled yet robust prompt embeddings
    In the full implementation, this would use a pre-trained LLM.
    Here we use a simplified transformer-based text encoder.
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
        
        # Convert time series to "text-like" representation
        # In full implementation, this would be actual text prompts
        # Here we create a learned embedding that simulates text encoding
        self.prompt_embedding = nn.Sequential(
            nn.Linear(n_variables, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Positional encoding for prompts
        self.pos_encoder = nn.Parameter(torch.randn(1, lookback, d_model))
        
        # Transformer encoder (simulating LLM processing)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, lookback, n_variables)
        Returns:
            embeddings: (batch_size, lookback, d_model)
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Convert time series to prompt-like embeddings
        # Reshape to process each time step
        x_flat = x.reshape(batch_size * seq_len, n_vars)
        x_embedded = self.prompt_embedding(x_flat)  # (batch*seq, d_model)
        x_embedded = x_embedded.reshape(batch_size, seq_len, self.d_model)
        
        # Add positional encoding
        x_embedded = x_embedded + self.pos_encoder
        x_embedded = self.dropout(x_embedded)
        
        # Apply transformer (simulating LLM)
        x_embedded = self.transformer_encoder(x_embedded)  # (batch, lookback, d_model)
        
        return x_embedded


class CrossModalityAlignment(nn.Module):
    """
    Cross-Modality Alignment Module
    Retrieves disentangled and robust time series embeddings via similarity-based retrieval
    """
    def __init__(self, d_model: int = 256, temperature: float = 0.07):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        
        # Projection layers for similarity computation
        self.ts_proj = nn.Linear(d_model, d_model)
        self.prompt_proj = nn.Linear(d_model, d_model)
        
    def forward(self, ts_embeddings, prompt_embeddings):
        """
        Args:
            ts_embeddings: (batch, lookback, d_model) - disentangled but weak
            prompt_embeddings: (batch, lookback, d_model) - entangled but robust
        
        Returns:
            aligned_embeddings: (batch, lookback, d_model) - disentangled and robust
        """
        batch_size, lookback, d_model = ts_embeddings.shape
        
        # Project embeddings
        ts_proj = self.ts_proj(ts_embeddings)  # (batch, lookback, d_model)
        prompt_proj = self.prompt_proj(prompt_embeddings)  # (batch, lookback, d_model)
        
        # Normalize for cosine similarity
        ts_proj_norm = F.normalize(ts_proj, p=2, dim=-1)  # (batch, lookback, d_model)
        prompt_proj_norm = F.normalize(prompt_proj, p=2, dim=-1)  # (batch, lookback, d_model)
        
        # Compute similarity matrix: (batch, lookback, lookback)
        # Each row i: similarity between ts_emb[i] and all prompt_emb[j]
        similarity = torch.bmm(ts_proj_norm, prompt_proj_norm.transpose(1, 2))  # (batch, lookback, lookback)
        similarity = similarity / self.temperature
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarity, dim=-1)  # (batch, lookback, lookback)
        
        # Retrieve aligned embeddings: weighted combination of prompt embeddings
        aligned_embeddings = torch.bmm(attention_weights, prompt_embeddings)  # (batch, lookback, d_model)
        
        # Combine with original TS embeddings (residual connection)
        aligned_embeddings = aligned_embeddings + ts_embeddings
        
        return aligned_embeddings


class TimeCMA(nn.Module):
    """
    TimeCMA Model for Multivariate Time Series Forecasting
    
    Key components:
    1. Time Series Encoding Branch: extracts disentangled embeddings
    2. Prompt Encoding Branch: extracts robust embeddings (simulating LLM)
    3. Cross-Modality Alignment: retrieves best of both worlds
    4. Last Token Prediction: uses last token embedding for efficiency
    """
    def __init__(self,
                 n_variables: int,
                 lookback: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_ts_layers: int = 2,
                 num_prompt_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 alignment_temperature: float = 0.07):
        super().__init__()
        
        self.n_variables = n_variables
        self.lookback = lookback
        self.d_model = d_model
        
        # Time series encoding branch
        self.ts_encoder = TimeSeriesEncoder(
            n_variables=n_variables,
            lookback=lookback,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_ts_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Prompt encoding branch (simulating LLM)
        self.prompt_encoder = PromptEncoder(
            n_variables=n_variables,
            lookback=lookback,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_prompt_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Cross-modality alignment
        self.alignment = CrossModalityAlignment(
            d_model=d_model,
            temperature=alignment_temperature
        )
        
        # Output projection (using last token embedding)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, lookback, n_variables)
        
        Returns:
            predictions: (batch_size, 1)
        """
        # Time series encoding branch
        ts_embeddings = self.ts_encoder(x)  # (batch, lookback, d_model)
        
        # Prompt encoding branch
        prompt_embeddings = self.prompt_encoder(x)  # (batch, lookback, d_model)
        
        # Cross-modality alignment
        aligned_embeddings = self.alignment(ts_embeddings, prompt_embeddings)  # (batch, lookback, d_model)
        
        # Use last token embedding for prediction (as per paper)
        last_token_embedding = aligned_embeddings[:, -1, :]  # (batch, d_model)
        
        # Project to output
        output = self.output_projection(last_token_embedding)  # (batch, 1)
        
        return output


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    batch_size = 32
    lookback = 50
    n_variables = 100
    
    model = TimeCMA(
        n_variables=n_variables,
        lookback=lookback,
        d_model=256,
        nhead=8,
        num_ts_layers=2,
        num_prompt_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"TimeCMA parameters: {count_parameters(model):,}")
    
    # Test input
    x = torch.randn(batch_size, lookback, n_variables)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

