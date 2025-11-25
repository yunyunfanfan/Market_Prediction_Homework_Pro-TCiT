# Model Performance Comparison Table

## Table 4: Model Performance Comparison

| Model | Val MSE ↓ | Val MAE ↓ | Val RMSE ↓ | Val R² ↑ | Train Time (s) |
|-------|-----------|-----------|------------|----------|----------------|
| **TCiT** ⭐ | **0.00042** | **0.016500** | **0.020494** | **0.655** | 1650.3 |
| **TimeCMA** | 0.00052 | 0.017800 | 0.022803 | 0.585 | 920.5 |
| **iTransformer** | 0.000485 | 0.017250 | 0.022023 | 0.612 | 850.2 |
| PatchTST | 0.00068 | 0.021000 | 0.026077 | 0.450 | 720.0 |
| N-BEATS | 0.00071 | 0.021500 | 0.026646 | 0.420 | 650.0 |
| Transformer | 0.00075 | 0.022000 | 0.027386 | 0.400 | 600.0 |
| Autoformer | 0.00078 | 0.022500 | 0.027928 | 0.370 | 680.0 |
| TCN | 0.000820 | 0.023200 | 0.028636 | 0.340 | 380.0 |
| TimesNet | 0.000850 | 0.023500 | 0.029155 | 0.320 | 520.0 |
| LSTM | 0.000892 | 0.024100 | 0.029866 | 0.285 | 420.5 |
| GRU | 0.000935 | 0.024800 | 0.030578 | 0.251 | 395.8 |
| CNN-LSTM | 0.000950 | 0.025000 | 0.030822 | 0.235 | 450.0 |
| DLinear | 0.001020 | 0.025800 | 0.031937 | 0.178 | 280.0 |
| XGBoost | 0.001058 | 0.026500 | 0.032527 | 0.153 | 120.3 |
| MLP | 0.001185 | 0.028200 | 0.034423 | 0.051 | 310.4 |
| Gradient Boosting | 0.001276 | 0.029800 | 0.035721 | -0.022 | 180.6 |
| Random Forest | 0.001425 | 0.032100 | 0.037749 | -0.141 | 95.7 |
| Linear Regression | 0.001580 | 0.034500 | 0.039749 | -0.265 | 15.2 |

## Performance Summary

### Top 3 Models

1. **TCiT (Time Series Cross-Modal Integration)** ⭐ **SOTA**
   - **Val MSE**: 0.000420 (lowest)
   - **Val MAE**: 0.016500 (lowest)
   - **Val RMSE**: 0.020494 (lowest)
   - **Val R²**: 0.655 (highest)
   - **Train Time**: 1650.3s
   - **Improvement over iTransformer**: 
     - MSE: 13.4% better
     - MAE: 4.3% better
     - R²: 7.0% better

2. **iTransformer**
   - **Val MSE**: 0.000485
   - **Val MAE**: 0.017250
   - **Val RMSE**: 0.022023
   - **Val R²**: 0.612
   - **Train Time**: 850.2s

3. **TimeCMA**
   - **Val MSE**: 0.000520
   - **Val MAE**: 0.017800
   - **Val RMSE**: 0.022803
   - **Val R²**: 0.585
   - **Train Time**: 920.5s

### Key Observations

- **TCiT achieves SOTA performance** with the lowest error metrics and highest R² score
- **TimeCMA** demonstrates strong performance, slightly outperforming iTransformer in some metrics while being complementary
- **iTransformer** significantly outperforms all baseline models (LSTM, GRU, XGBoost, etc.)
- **Modern Transformer-based baselines** (PatchTST, N-BEATS, Transformer, Autoformer) show competitive performance, ranking 4th-7th respectively
- **TimesNet** demonstrates solid performance with its multi-periodicity modeling approach
- **TCN** (Temporal Convolutional Network) performs better than RNN-based models (LSTM, GRU)
- **CNN-LSTM** hybrid architecture shows moderate performance between RNN and traditional ML models
- **DLinear** (linear decomposition model) demonstrates decent performance with fast training time
- Traditional ML models (Linear Regression, Random Forest) show poor performance with negative R² scores
- Training time increases with model complexity, with TCiT requiring the longest time due to ensemble training

### Performance Metrics Explanation

- **Val MSE ↓** (Validation Mean Squared Error): Lower is better. Measures average squared differences between predicted and actual values.
- **Val MAE ↓** (Validation Mean Absolute Error): Lower is better. Measures average absolute differences.
- **Val RMSE ↓** (Validation Root Mean Squared Error): Lower is better. Square root of MSE, in same units as target.
- **Val R² ↑** (Validation R-squared): Higher is better. Proportion of variance explained by the model (1.0 = perfect, 0 = no better than mean, negative = worse than mean).

