"""
Data preprocessing and feature engineering for Hull Tactical Market Prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for multivariate time series market prediction
    """
    
    def __init__(self):
        self.feature_groups = {
            'D': [f'D{i}' for i in range(1, 10)],  # Discrete features
            'E': [f'E{i}' for i in range(1, 21)],  # Economic indicators
            'I': [f'I{i}' for i in range(1, 10)],  # Interest rate features
            'M': [f'M{i}' for i in range(1, 19)],  # Market features
            'P': [f'P{i}' for i in range(1, 14)],  # Price features
            'S': [f'S{i}' for i in range(1, 13)],  # Sentiment features
            'V': [f'V{i}' for i in range(1, 14)]   # Volatility features
        }
        
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 5]) -> pd.DataFrame:
        """
        Create lagged features for time series prediction
        
        Args:
            df: Input dataframe
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        df_lagged = df.copy()
        
        # Get numeric columns (excluding date_id and target columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns']
        
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for lag in lags:
            for col in feature_cols:
                if col in df.columns:
                    df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               windows: List[int] = [10]) -> pd.DataFrame:
        """
        Create rolling statistical features
        
        Args:
            df: Input dataframe
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df_rolling = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns']
        
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for window in windows:
            for col in feature_cols:
                if col in df.columns:
                    # Rolling mean only (reduce features)
                    df_rolling[f'{col}_ma{window}'] = df[col].rolling(window=window).mean()
        
        return df_rolling
    
    def create_group_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated statistics for feature groups
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with group statistics
        """
        df_stats = df.copy()
        
        for group_name, features in self.feature_groups.items():
            # Get available features in the group
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) > 0:
                # Mean of the group
                df_stats[f'{group_name}_mean'] = df[available_features].mean(axis=1)
                
                # Std of the group
                df_stats[f'{group_name}_std'] = df[available_features].std(axis=1)
                
                # Max-min range
                df_stats[f'{group_name}_range'] = (df[available_features].max(axis=1) - 
                                                   df[available_features].min(axis=1))
        
        return df_stats
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important groups
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with interaction features
        """
        df_interact = df.copy()
        
        # Price * Volatility interactions
        for p_feat in self.feature_groups['P']:
            for v_feat in self.feature_groups['V']:
                if p_feat in df.columns and v_feat in df.columns:
                    df_interact[f'{p_feat}_{v_feat}_interact'] = df[p_feat] * df[v_feat]
        
        # Market * Sentiment interactions
        for m_feat in self.feature_groups['M']:
            for s_feat in self.feature_groups['S']:
                if m_feat in df.columns and s_feat in df.columns:
                    df_interact[f'{m_feat}_{s_feat}_interact'] = df[m_feat] * df[s_feat]
        
        return df_interact
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input dataframe
            strategy: Strategy for handling missing values
            
        Returns:
            DataFrame with handled missing values
        """
        df_filled = df.copy()
        
        if strategy == 'forward_fill':
            # Forward fill for time series
            df_filled = df_filled.fillna(method='ffill')
            # Backward fill for remaining NaNs at the beginning
            df_filled = df_filled.fillna(method='bfill')
        elif strategy == 'interpolate':
            # Linear interpolation
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear')
        elif strategy == 'zero':
            # Fill with zeros
            df_filled = df_filled.fillna(0)
        
        # Fill any remaining NaNs with 0
        df_filled = df_filled.fillna(0)
        
        return df_filled
    
    def engineer_all_features(self, df: pd.DataFrame, 
                             include_lagged: bool = True,
                             include_rolling: bool = True,
                             include_groups: bool = True,
                             include_interactions: bool = False) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Input dataframe
            include_lagged: Whether to include lagged features
            include_rolling: Whether to include rolling features
            include_groups: Whether to include group statistics
            include_interactions: Whether to include interaction features
            
        Returns:
            Engineered dataframe
        """
        df_eng = df.copy()
        
        print("Starting feature engineering...")
        print(f"Initial shape: {df_eng.shape}")
        
        # Handle missing values first
        df_eng = self.handle_missing_values(df_eng)
        print(f"After handling missing values: {df_eng.shape}")
        
        # Create group statistics
        if include_groups:
            df_eng = self.create_group_statistics(df_eng)
            print(f"After group statistics: {df_eng.shape}")
        
        # Create lagged features (reduced to save memory)
        if include_lagged:
            df_eng = self.create_lagged_features(df_eng, lags=[1, 5])
            print(f"After lagged features: {df_eng.shape}")
        
        # Create rolling features (reduced to save memory)
        if include_rolling:
            df_eng = self.create_rolling_features(df_eng, windows=[10])
            print(f"After rolling features: {df_eng.shape}")
        
        # Create interaction features (optional, can create many features)
        if include_interactions:
            df_eng = self.create_interaction_features(df_eng)
            print(f"After interaction features: {df_eng.shape}")
        
        # Final missing value handling after feature engineering
        df_eng = self.handle_missing_values(df_eng)
        
        print(f"Final shape: {df_eng.shape}")
        
        return df_eng


class DataPreprocessor:
    """
    Data preprocessor for iTransformer model
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Args:
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()
        self.feature_names = None
        
    def fit_transform(self, df: pd.DataFrame, 
                     target_col: str = 'market_forward_excess_returns') -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler and transform data
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            
        Returns:
            Tuple of (features, targets)
        """
        # Separate features and target
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored',
                       'lagged_forward_returns', 'lagged_risk_free_rate',
                       'lagged_market_forward_excess_returns']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        
        # Handle target
        if target_col in df.columns:
            y = df[target_col].values
        else:
            y = None
        
        # Fit and transform features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def transform(self, df: pd.DataFrame, 
                  target_col: str = 'market_forward_excess_returns') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted scaler
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            
        Returns:
            Tuple of (features, targets)
        """
        if self.feature_names is None:
            raise ValueError("Scaler has not been fitted yet. Call fit_transform first.")
        
        X = df[self.feature_names].values
        
        if target_col in df.columns:
            y = df[target_col].values
        else:
            y = None
        
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def inverse_transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features
        
        Args:
            X: Scaled features
            
        Returns:
            Original scale features
        """
        return self.scaler.inverse_transform(X)


def create_sequences(X: np.ndarray, y: np.ndarray, 
                    lookback: int = 96, 
                    forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        lookback: Number of time steps to look back
        forecast_horizon: Number of steps ahead to predict
        
    Returns:
        Tuple of (X_seq, y_seq)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - lookback - forecast_horizon + 1):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback+forecast_horizon-1])
    
    return np.array(X_seq), np.array(y_seq)


def prepare_data(train_path: str, test_path: str, 
                lookback: int = 96,
                forecast_horizon: int = 1,
                include_lagged: bool = True,
                include_rolling: bool = True) -> dict:
    """
    Complete data preparation pipeline
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        lookback: Number of time steps to look back
        forecast_horizon: Number of steps ahead to predict
        include_lagged: Whether to include lagged features
        include_rolling: Whether to include rolling features
        
    Returns:
        Dictionary containing processed data and metadata
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    
    print("\nEngineering training features...")
    train_eng = feature_engineer.engineer_all_features(
        train_df, 
        include_lagged=include_lagged,
        include_rolling=include_rolling,
        include_groups=True,
        include_interactions=False
    )
    
    print("\nEngineering test features...")
    test_eng = feature_engineer.engineer_all_features(
        test_df,
        include_lagged=include_lagged,
        include_rolling=include_rolling,
        include_groups=True,
        include_interactions=False
    )
    
    # Preprocessing
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    print("\nScaling features...")
    X_train, y_train = preprocessor.fit_transform(train_eng)
    X_test, y_test = preprocessor.transform(test_eng)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Create sequences
    print(f"\nCreating sequences with lookback={lookback}, forecast_horizon={forecast_horizon}...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback, forecast_horizon)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback, forecast_horizon)
    
    print(f"X_train_seq shape: {X_train_seq.shape}")
    print(f"y_train_seq shape: {y_train_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}")
    print(f"y_test_seq shape: {y_test_seq.shape}")
    
    return {
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_test': X_test_seq,
        'y_test': y_test_seq,
        'preprocessor': preprocessor,
        'feature_engineer': feature_engineer,
        'n_features': X_train.shape[1],
        'lookback': lookback,
        'forecast_horizon': forecast_horizon
    }

