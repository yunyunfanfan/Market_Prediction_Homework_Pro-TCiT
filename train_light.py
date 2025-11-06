"""
Lightweight training script for iTransformer with minimal features
Optimized for low memory usage
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import json
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from src.data.preprocessing import prepare_data
from src.models.itransformer import iTransformerSimple
from src.utils.trainer import Trainer, create_dataloaders


def main():
    print("\n" + "="*70)
    print("iTransformer Lightweight Training (Optimized for Low Memory)")
    print("="*70)
    
    # Fixed configuration for low memory
    config = {
        'train_path': 'train.csv',
        'lookback': 30,  # Reduced lookback
        'forecast_horizon': 1,
        'val_split': 0.2,
        
        # Minimal feature engineering
        'include_lagged': False,  # Disable lagged features
        'include_rolling': False,  # Disable rolling features
        
        # Small model
        'd_model': 128,  # Reduced from 256
        'nhead': 4,  # Reduced from 8
        'num_layers': 2,  # Reduced from 3
        'dim_feedforward': 512,  # Reduced from 1024
        'dropout': 0.1,
        
        # Training
        'batch_size': 32,  # Reduced from 64
        'num_epochs': 50,  # Reduced for testing
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'scheduler': 'cosine',
        'early_stopping_patience': 10,
        
        # Other
        'seed': 42,
        'device': 'cpu',  # Force CPU to avoid GPU memory issues
        'checkpoint_dir': 'checkpoints',
        'experiment_name': 'itransformer_light'
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n" + "="*70)
    print("STEP 1: Data Preparation (Minimal Features)")
    print("="*70)
    
    print("Loading training data with minimal feature engineering...")
    data_dict = prepare_data(
        train_path=config['train_path'],
        test_path=config['train_path'],  # Use train for both
        lookback=config['lookback'],
        forecast_horizon=config['forecast_horizon'],
        include_lagged=config['include_lagged'],
        include_rolling=config['include_rolling']
    )
    
    X_train_full = data_dict['X_train']
    y_train_full = data_dict['y_train']
    n_features = data_dict['n_features']
    
    print(f"\nData shapes:")
    print(f"  X_train_full: {X_train_full.shape}")
    print(f"  y_train_full: {y_train_full.shape}")
    print(f"  Number of features: {n_features}")
    
    # Check if features are reasonable
    if n_features > 500:
        print(f"\n⚠ Warning: {n_features} features is still too many!")
        print("Consider reducing features further or using original features only.")
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config['val_split'],
        shuffle=False
    )
    
    print(f"\nAfter train/val split:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=config['batch_size'],
        num_workers=0
    )
    
    # Create model
    print("\n" + "="*70)
    print("STEP 2: Model Creation (Small Model)")
    print("="*70)
    
    model = iTransformerSimple(
        n_variables=n_features,
        lookback=config['lookback'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: iTransformerSimple (Lightweight)")
    print(f"Total trainable parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler_type=config['scheduler']
    )
    
    # Train model
    print("\n" + "="*70)
    print("STEP 3: Training")
    print("="*70)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_dir=config['checkpoint_dir'],
        model_name=config['experiment_name']
    )
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("STEP 4: Final Validation Set Evaluation")
    print("="*70)
    
    val_metrics = trainer.validate(val_loader)
    
    print("\nFinal Validation Set Results:")
    print(f"  MSE:  {val_metrics['mse']:.6f}")
    print(f"  MAE:  {val_metrics['mae']:.6f}")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    
    # Make predictions on validation set
    predictions, targets = trainer.predict(val_loader)
    
    # Save predictions
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'target': targets.flatten(),
        'prediction': predictions.flatten()
    })
    results_path = os.path.join(results_dir, f"{config['experiment_name']}_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Predictions saved to {results_path}")
    
    # Save training history
    history_path = os.path.join(results_dir, f"{config['experiment_name']}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    # Save configuration
    config['val_metrics'] = val_metrics
    config['num_parameters'] = num_params
    config_path = os.path.join(results_dir, f"{config['experiment_name']}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to {config_path}")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print(f"  python visualize.py --experiment_name {config['experiment_name']}")


if __name__ == '__main__':
    main()

