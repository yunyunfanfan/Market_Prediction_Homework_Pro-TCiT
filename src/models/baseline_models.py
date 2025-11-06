"""
Baseline models for comparison with iTransformer
包括传统机器学习和深度学习方法
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class LinearRegressionModel:
    """线性回归模型 (Ridge正则化)"""
    
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.name = "Linear Regression (Ridge)"
    
    def fit(self, X, y):
        """训练模型"""
        # 将3D数据展平为2D
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
        return self
    
    def predict(self, X):
        """预测"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class RandomForestModel:
    """随机森林回归"""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        self.name = "Random Forest"
    
    def fit(self, X, y):
        """训练模型"""
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
        return self
    
    def predict(self, X):
        """预测"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class GradientBoostingModel:
    """梯度提升树"""
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=0
        )
        self.name = "Gradient Boosting"
    
    def fit(self, X, y):
        """训练模型"""
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
        return self
    
    def predict(self, X):
        """预测"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class XGBoostModel:
    """XGBoost模型"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            tree_method='hist',
            verbosity=0
        )
        self.name = "XGBoost"
    
    def fit(self, X, y):
        """训练模型"""
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y, verbose=False)
        return self
    
    def predict(self, X):
        """预测"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class SimpleMLP(nn.Module):
    """简单的多层感知机"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        self.name = "MLP"
    
    def forward(self, x):
        # 展平输入
        x = x.reshape(x.shape[0], -1)
        return self.network(x)


class SimpleLSTM(nn.Module):
    """简单的LSTM模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.name = "LSTM"
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        
        return output


class SimpleGRU(nn.Module):
    """简单的GRU模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.name = "GRU"
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        gru_out, hidden = self.gru(x)
        
        # 使用最后一个时间步的输出
        last_output = gru_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        
        return output


def get_model(model_name, input_shape, device='cpu'):
    """
    获取指定的模型
    
    Args:
        model_name: 模型名称
        input_shape: 输入形状 (lookback, n_features)
        device: 设备
        
    Returns:
        模型实例
    """
    lookback, n_features = input_shape
    
    if model_name == 'linear':
        return LinearRegressionModel()
    
    elif model_name == 'random_forest':
        return RandomForestModel(n_estimators=100, max_depth=10)
    
    elif model_name == 'gradient_boosting':
        return GradientBoostingModel(n_estimators=100, max_depth=5)
    
    elif model_name == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        return XGBoostModel(n_estimators=100, max_depth=6)
    
    elif model_name == 'mlp':
        input_size = lookback * n_features
        model = SimpleMLP(input_size, hidden_sizes=[512, 256, 128])
        return model.to(device)
    
    elif model_name == 'lstm':
        model = SimpleLSTM(n_features, hidden_size=128, num_layers=2)
        return model.to(device)
    
    elif model_name == 'gru':
        model = SimpleGRU(n_features, hidden_size=128, num_layers=2)
        return model.to(device)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def is_sklearn_model(model):
    """判断是否是sklearn模型"""
    return hasattr(model, 'fit') and hasattr(model, 'predict') and not isinstance(model, nn.Module)


def is_pytorch_model(model):
    """判断是否是PyTorch模型"""
    return isinstance(model, nn.Module)

