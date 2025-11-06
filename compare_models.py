"""
比较不同模型在市场预测任务上的性能
包括传统机器学习方法、深度学习方法和iTransformer
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.data.preprocessing import prepare_data
from src.models.baseline_models import get_model, is_sklearn_model, is_pytorch_model
from src.models.itransformer import iTransformerSimple
from src.utils.trainer import Trainer, create_dataloaders


def train_sklearn_model(model, X_train, y_train, X_val, y_val):
    """训练sklearn模型"""
    print(f"\nTraining {model.name}...")
    start_time = time.time()
    
    # 训练
    model.fit(X_train, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_time = time.time() - start_time
    
    # 计算指标
    metrics = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
        'val_mse': mean_squared_error(y_val, y_pred_val),
        'val_mae': mean_absolute_error(y_val, y_pred_val),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'val_r2': r2_score(y_val, y_pred_val),
        'train_time': train_time
    }
    
    return metrics, y_pred_val


def train_pytorch_model(model, X_train, y_train, X_val, y_val, 
                       device, num_epochs=50, batch_size=64, learning_rate=1e-3):
    """训练PyTorch模型"""
    print(f"\nTraining {model.name}...")
    start_time = time.time()
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=batch_size,
        num_workers=0
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        scheduler_type='cosine'
    )
    
    # 训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=10,
        checkpoint_dir='checkpoints',
        model_name=f'baseline_{model.name.lower().replace(" ", "_")}'
    )
    
    # 评估
    train_metrics = trainer.validate(train_loader)
    val_metrics = trainer.validate(val_loader)
    
    # 预测
    y_pred_val, _ = trainer.predict(val_loader)
    
    train_time = time.time() - start_time
    
    metrics = {
        'train_mse': train_metrics['mse'],
        'train_mae': train_metrics['mae'],
        'train_rmse': train_metrics['rmse'],
        'train_r2': 1 - train_metrics['mse'] / np.var(y_train),
        'val_mse': val_metrics['mse'],
        'val_mae': val_metrics['mae'],
        'val_rmse': val_metrics['rmse'],
        'val_r2': val_metrics.get('r2', 0),
        'train_time': train_time
    }
    
    return metrics, y_pred_val.flatten()


def compare_all_models(X_train, y_train, X_val, y_val, input_shape, device='cpu'):
    """
    比较所有模型
    
    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        input_shape: (lookback, n_features)
        device: 计算设备
        
    Returns:
        results: 所有模型的结果
    """
    results = {}
    
    # 定义要比较的模型
    sklearn_models = [
        'linear',
        'random_forest',
        'gradient_boosting',
    ]
    
    # 检查是否有XGBoost
    try:
        import xgboost
        sklearn_models.append('xgboost')
    except ImportError:
        print("XGBoost not available, skipping...")
    
    pytorch_models = [
        'mlp',
        'lstm',
        'gru'
    ]
    
    print("\n" + "="*70)
    print("MODEL COMPARISON EXPERIMENT")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Input shape: {input_shape}")
    print("="*70)
    
    # 训练sklearn模型
    print("\n" + "="*70)
    print("TRADITIONAL MACHINE LEARNING MODELS")
    print("="*70)
    
    for model_name in sklearn_models:
        try:
            model = get_model(model_name, input_shape, device)
            metrics, predictions = train_sklearn_model(
                model, X_train, y_train, X_val, y_val
            )
            
            results[model.name] = {
                'metrics': metrics,
                'predictions': predictions,
                'model_type': 'sklearn'
            }
            
            print(f"\n{model.name} Results:")
            print(f"  Val MSE:  {metrics['val_mse']:.6f}")
            print(f"  Val MAE:  {metrics['val_mae']:.6f}")
            print(f"  Val RMSE: {metrics['val_rmse']:.6f}")
            print(f"  Val R²:   {metrics['val_r2']:.6f}")
            print(f"  Time:     {metrics['train_time']:.2f}s")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # 训练PyTorch模型
    print("\n" + "="*70)
    print("DEEP LEARNING MODELS")
    print("="*70)
    
    for model_name in pytorch_models:
        try:
            model = get_model(model_name, input_shape, device)
            metrics, predictions = train_pytorch_model(
                model, X_train, y_train, X_val, y_val,
                device=device,
                num_epochs=30,  # 较少的epochs用于快速比较
                batch_size=64,
                learning_rate=1e-3
            )
            
            results[model.name] = {
                'metrics': metrics,
                'predictions': predictions,
                'model_type': 'pytorch'
            }
            
            print(f"\n{model.name} Results:")
            print(f"  Val MSE:  {metrics['val_mse']:.6f}")
            print(f"  Val MAE:  {metrics['val_mae']:.6f}")
            print(f"  Val RMSE: {metrics['val_rmse']:.6f}")
            print(f"  Val R²:   {metrics['val_r2']:.6f}")
            print(f"  Time:     {metrics['train_time']:.2f}s")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # 训练iTransformer
    print("\n" + "="*70)
    print("iTransformer (Our Method)")
    print("="*70)
    
    try:
        lookback, n_features = input_shape
        model = iTransformerSimple(
            n_variables=n_features,
            lookback=lookback,
            d_model=192,
            nhead=6,
            num_layers=3,
            dim_feedforward=768,
            dropout=0.1
        ).to(device)
        
        metrics, predictions = train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            device=device,
            num_epochs=50,
            batch_size=48,
            learning_rate=1e-4
        )
        
        results['iTransformer'] = {
            'metrics': metrics,
            'predictions': predictions,
            'model_type': 'pytorch'
        }
        
        print(f"\niTransformer Results:")
        print(f"  Val MSE:  {metrics['val_mse']:.6f}")
        print(f"  Val MAE:  {metrics['val_mae']:.6f}")
        print(f"  Val RMSE: {metrics['val_rmse']:.6f}")
        print(f"  Val R²:   {metrics['val_r2']:.6f}")
        print(f"  Time:     {metrics['train_time']:.2f}s")
        
    except Exception as e:
        print(f"Error training iTransformer: {e}")
    
    return results


def create_comparison_table(results):
    """创建对比表格"""
    data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        data.append({
            'Model': model_name,
            'Val MSE': metrics['val_mse'],
            'Val MAE': metrics['val_mae'],
            'Val RMSE': metrics['val_rmse'],
            'Val R²': metrics['val_r2'],
            'Train Time (s)': metrics['train_time']
        })
    
    df = pd.DataFrame(data)
    
    # 按照Val MSE排序
    df = df.sort_values('Val MSE')
    
    return df


def save_results(results, output_dir='results'):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建对比表格
    comparison_df = create_comparison_table(results)
    
    # 保存CSV
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Comparison table saved to {comparison_path}")
    
    # 保存详细结果
    detailed_results = {}
    for model_name, result in results.items():
        detailed_results[model_name] = {
            'metrics': result['metrics'],
            'model_type': result['model_type']
        }
    
    json_path = os.path.join(output_dir, 'model_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✓ Detailed results saved to {json_path}")
    
    return comparison_df


def print_summary(comparison_df):
    """打印摘要"""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print("\n" + comparison_df.to_string(index=False))
    print("\n" + "="*70)
    
    # 找出最佳模型
    best_model = comparison_df.iloc[0]['Model']
    best_mse = comparison_df.iloc[0]['Val MSE']
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Val MSE: {best_mse:.6f}")
    
    # 如果iTransformer是最佳模型
    if best_model == 'iTransformer':
        print("\n✨ iTransformer achieves the best performance!")
        
        # 计算改进幅度
        if len(comparison_df) > 1:
            second_best_mse = comparison_df.iloc[1]['Val MSE']
            improvement = ((second_best_mse - best_mse) / second_best_mse) * 100
            print(f"   Improvement over 2nd best: {improvement:.2f}%")
    
    print("="*70)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("MULTI-MODEL COMPARISON FOR MARKET PREDICTION")
    print("="*70)
    
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 准备数据（使用较小的lookback以加快比较）
    print("\nPreparing data...")
    data_dict = prepare_data(
        train_path='train.csv',
        test_path='train.csv',
        lookback=30,  # 较小的lookback加快训练
        forecast_horizon=1,
        include_lagged=True,
        include_rolling=True
    )
    
    X_train_full = data_dict['X_train']
    y_train_full = data_dict['y_train']
    n_features = data_dict['n_features']
    lookback = data_dict['lookback']
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        shuffle=False
    )
    
    print(f"\nData prepared:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  Features: {n_features}")
    print(f"  Lookback: {lookback}")
    
    # 比较所有模型
    results = compare_all_models(
        X_train, y_train,
        X_val, y_val,
        input_shape=(lookback, n_features),
        device=device
    )
    
    # 保存并显示结果
    comparison_df = save_results(results)
    print_summary(comparison_df)


if __name__ == '__main__':
    main()

