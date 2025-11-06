"""
Main training script for iTransformer on Hull Tactical Market Prediction
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
from src.models.itransformer import iTransformer, iTransformerSimple
from src.utils.trainer import Trainer, create_dataloaders


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train iTransformer for market prediction')
    
    # Data arguments
    parser.add_argument('--train_path', type=str, default='train.csv',
                       help='Path to training data')
    parser.add_argument('--test_path', type=str, default='test.csv',
                       help='Path to test data')
    parser.add_argument('--lookback', type=int, default=50,
                       help='Number of time steps to look back')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                       help='Number of steps ahead to predict')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Feature engineering arguments
    parser.add_argument('--include_lagged', action='store_true', default=True,
                       help='Include lagged features')
    parser.add_argument('--include_rolling', action='store_true', default=True,
                       help='Include rolling features')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'full'],
                       help='Type of iTransformer model')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Dimension of model embeddings')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau'],
                       help='Learning rate scheduler type')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Patience for early stopping')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default='itransformer',
                       help='Name of the experiment')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n" + "="*70)
    print("STEP 1: Data Preparation")
    print("="*70)
    
    # Only load training data
    print("Loading training data only (test set will not be used)...")
    data_dict = prepare_data(
        train_path=args.train_path,
        test_path=args.train_path,  # Use train data for both (we won't use the "test" output)
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon,
        include_lagged=args.include_lagged,
        include_rolling=args.include_rolling
    )
    
    X_train_full = data_dict['X_train']
    y_train_full = data_dict['y_train']
    n_features = data_dict['n_features']
    
    print(f"\nData shapes:")
    print(f"  X_train_full: {X_train_full.shape}")
    print(f"  y_train_full: {y_train_full.shape}")
    print(f"  Number of features: {n_features}")
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_split,
        shuffle=False  # Keep temporal order
    )
    
    print(f"\nAfter train/val split:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Create model
    print("\n" + "="*70)
    print("STEP 2: Model Creation")
    print("="*70)
    
    if args.model_type == 'simple':
        model = iTransformerSimple(
            n_variables=n_features,
            lookback=args.lookback,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
    else:
        model = iTransformer(
            n_variables=n_features,
            lookback=args.lookback,
            forecast_horizon=args.forecast_horizon,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=2,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model_type}")
    print(f"Total trainable parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler
    )
    
    # Train model
    print("\n" + "="*70)
    print("STEP 3: Training")
    print("="*70)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.experiment_name
    )
    
    # Evaluate on validation set (use as final evaluation)
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
    results_path = os.path.join(results_dir, f'{args.experiment_name}_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Predictions saved to {results_path}")
    
    # Save training history
    history_path = os.path.join(results_dir, f'{args.experiment_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    # Save configuration
    config = vars(args)
    config['val_metrics'] = val_metrics
    config['num_parameters'] = num_params
    config_path = os.path.join(results_dir, f'{args.experiment_name}_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to {config_path}")
    
    print("\n" + "="*70)
    print("Training and evaluation completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()

