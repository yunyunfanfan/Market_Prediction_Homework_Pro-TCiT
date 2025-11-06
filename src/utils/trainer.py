"""
Training utilities for iTransformer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, Optional, Tuple
import os


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience: int = 10, min_delta: float = 0, 
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Args:
            score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            is_better = score < (self.best_score - self.min_delta)
        else:
            is_better = score > (self.best_score + self.min_delta)
        
        if is_better:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'EarlyStopping: Improvement found at epoch {epoch}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: No improvement for {self.counter} epochs')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'EarlyStopping: Stopping training. Best epoch was {self.best_epoch}')
                return True
        
        return False


class Trainer:
    """
    Trainer class for iTransformer models
    """
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 scheduler_type: str = 'cosine',
                 criterion: Optional[nn.Module] = None):
        """
        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            scheduler_type: Type of learning rate scheduler ('cosine' or 'plateau')
            criterion: Loss function (default: MSELoss)
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        
        # Scheduler
        self.scheduler_type = scheduler_type
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'learning_rates': []
        }
        
    def create_scheduler(self, num_epochs: int, steps_per_epoch: int):
        """Create learning rate scheduler"""
        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs * steps_per_epoch,
                eta_min=1e-7
            )
        elif self.scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        total_mse = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)  # (batch, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            
            # Compute loss
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update scheduler (for cosine annealing)
            if self.scheduler is not None and self.scheduler_type == 'cosine':
                self.scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            with torch.no_grad():
                mse = F.mse_loss(predictions, batch_y).item()
                total_mse += mse
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.6f}',
                'mse': f'{total_mse/num_batches:.6f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_mse = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)
            
            # Forward pass
            predictions = self.model(batch_X)
            
            # Compute loss
            loss = self.criterion(predictions, batch_y)
            
            total_loss += loss.item()
            mse = F.mse_loss(predictions, batch_y).item()
            total_mse += mse
            
            num_batches += 1
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute additional metrics
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()
        
        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'mae': mae,
            'rmse': rmse
        }
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 100,
            early_stopping_patience: int = 15,
            checkpoint_dir: str = 'checkpoints',
            model_name: str = 'itransformer'):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            model_name: Name for saved model
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create scheduler
        self.create_scheduler(num_epochs, len(train_loader))
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        
        best_val_loss = float('inf')
        best_model_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Update scheduler (for plateau)
            if self.scheduler is not None and self.scheduler_type == 'plateau':
                self.scheduler.step(val_metrics['loss'])
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
            print(f"Train MSE: {train_metrics['mse']:.6f} | Val MSE: {val_metrics['mse']:.6f}")
            print(f"Val MAE: {val_metrics['mae']:.6f} | Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 70)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'history': self.history
                }, best_model_path)
                print(f"✓ Saved best model to {best_model_path}")
            
            # Early stopping
            if early_stopping(val_metrics['loss'], epoch):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best model saved to: {best_model_path}")
        
        # Load best model
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.history
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc='Predicting'):
                batch_X = batch_X.to(self.device)
                predictions = self.model(batch_X)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return predictions, targets


def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       batch_size: int = 64,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Import F for mse_loss in train_epoch and validate methods
import torch.nn.functional as F

