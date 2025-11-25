"""
Generate visualizations for all tables in the project report.
This script creates comprehensive visualizations for:
- Table 3: Baseline Models Configuration
- Table 4: Model Performance Comparison
- Table 7: Fusion Weight Ablation
- Table 8: Feature Engineering Impact
- Table 9: Model Size Ablation
- Table 10: Lookback Window Impact
- Table 11: Residual Statistics
- Table 12: Normality Tests
- Table 13: Market Regime Performance
- Table 14: Tail Events Performance
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Create figures directory
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)


def create_table3_visualization():
    """Table 3: Model Performance Comparison - All Metrics"""
    # Data from the provided image description
    models = ['iTransformer', 'LSTM', 'GRU', 'XGBoost', 'MLP', 'GB', 'RF', 'Linear']
    
    mse = [0.000485, 0.000892, 0.000935, 0.001058, 0.001185, 0.001276, 0.001425, 0.001580]
    mae = [0.01725, 0.02410, 0.02480, 0.02650, 0.02820, 0.02980, 0.03210, 0.03450]
    rmse = [0.02202, 0.02987, 0.03058, 0.03253, 0.03442, 0.03572, 0.03775, 0.03975]
    r2 = [0.612, 0.285, 0.251, 0.153, 0.051, -0.022, -0.141, -0.265]
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color scheme: highlight iTransformer
    colors = ['#d62728' if m == 'iTransformer' else '#1f77b4' for m in models]
    
    # 1. MSE Comparison (Lower is Better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, mse, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('MSE ↓', fontsize=12, fontweight='bold')
    ax1.set_title('MSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    # Add value labels on bars
    for bar, val in zip(bars1, mse):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    # 2. MAE Comparison (Lower is Better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, mae, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('MAE ↓', fontsize=12, fontweight='bold')
    ax2.set_title('MAE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    # Add value labels on bars
    for bar, val in zip(bars2, mae):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    
    # 3. RMSE Comparison (Lower is Better)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, rmse, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_ylabel('RMSE ↓', fontsize=12, fontweight='bold')
    ax3.set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    # Add value labels on bars
    for bar, val in zip(bars3, rmse):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.5f}', ha='center', va='bottom', fontsize=9)
    
    # 4. R² Comparison (Higher is Better)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, r2, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_ylabel('R² ↑', fontsize=12, fontweight='bold')
    ax4.set_title('R² Comparison (Higher is Better)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.tick_params(axis='x', rotation=45, labelsize=10)
    # Add horizontal reference lines
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Baseline (R²=0)')
    ax4.axhline(y=0.6, color='green', linestyle='--', linewidth=1.5, label='R²=0.6')
    # Add value labels on bars
    for bar, val in zip(bars4, r2):
        height = bar.get_height()
        y_pos = height + 0.02 if height >= 0 else height - 0.05
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    ax4.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Model Performance Comparison - All Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table3_baseline_configs.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table3_baseline_configs.pdf', bbox_inches='tight')
    print("✓ Created Table 3 visualization")
    plt.close()


def create_table4_visualization():
    """Table 4: Model Performance Comparison - All Models"""
    # Data for all models from MODEL_PERFORMANCE_TABLE.md (18 models)
    models = ['TCiT', 'iTransformer', 'TimeCMA', 'PatchTST', 'N-BEATS', 'Transformer', 
              'Autoformer', 'TCN', 'TimesNet', 'LSTM', 'GRU', 'CNN-LSTM', 
              'DLinear', 'XGBoost', 'MLP', 'Gradient Boosting', 'Random Forest', 'Linear Regression']
    
    mse = [0.00042, 0.000485, 0.00052, 0.00068, 0.00071, 0.00075, 0.00078, 0.000820, 
           0.000850, 0.000892, 0.000935, 0.000950, 0.001020, 0.001058, 0.001185, 
           0.001276, 0.001425, 0.001580]
    mae = [0.016500, 0.017250, 0.017800, 0.021000, 0.021500, 0.022000, 0.022500, 0.023200, 
           0.023500, 0.024100, 0.024800, 0.025000, 0.025800, 0.026500, 0.028200, 
           0.029800, 0.032100, 0.034500]
    rmse = [0.020494, 0.022023, 0.022803, 0.026077, 0.026646, 0.027386, 0.027928, 0.028636, 
            0.029155, 0.029866, 0.030578, 0.030822, 0.031937, 0.032527, 0.034423, 
            0.035721, 0.037749, 0.039749]
    r2 = [0.655, 0.612, 0.585, 0.450, 0.420, 0.400, 0.370, 0.340, 
          0.320, 0.285, 0.251, 0.235, 0.178, 0.153, 0.051, 
          -0.022, -0.141, -0.265]
    train_time = [1650.3, 850.2, 920.5, 720.0, 650.0, 600.0, 680.0, 380.0, 
                  520.0, 420.5, 395.8, 450.0, 280.0, 120.3, 310.4, 
                  180.6, 95.7, 15.2]
    
    # Create 2x3 subplot layout to accommodate more models
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Color scheme: 使用提供的配色方案 - 循环使用
    custom_colors = [
        # 第一组配色 (5种颜色)
        '#A5AEB7',  # 浅蓝灰色 (Light Blue-Gray)
        '#925EB0',  # 中紫色 (Medium Purple)
        '#7E99F4',  # 长春花蓝 (Periwinkle Blue)
        '#CC7C71',  # 哑光红/陶土色 (Muted Red/Terracotta)
        '#7AB656',  # 橄榄绿/青柠绿 (Olive Green/Lime Green)
        # 第二组配色 (5种颜色)
        '#B7B7EB',  # 浅紫色 (Light Purple)
        '#9D9EA3',  # 灰色 (Gray)
        '#EAB883',  # 浅橙色/桃色 (Light Orange/Peach)
        '#9BBBE1',  # 浅蓝色 (Light Blue)
        '#F09BA0'   # 浅粉色 (Light Pink)
    ]
    # 循环使用配色为18个模型分配颜色
    colors = [custom_colors[i % len(custom_colors)] for i in range(len(models))]
    
    # 1. MSE Comparison (Lower is Better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(models)), mse, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('MSE ↓', fontsize=12, fontweight='bold')
    ax1.set_title('MSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on top bars (only for top 3 to avoid clutter)
    for i, (bar, val) in enumerate(zip(bars1, mse)):
        if i < 3:  # Only label top 3
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.6f}', ha='center', va='bottom', fontsize=7)
    
    # 2. MAE Comparison (Lower is Better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(models)), mae, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('MAE ↓', fontsize=12, fontweight='bold')
    ax2.set_title('MAE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on top bars (only for top 3)
    for i, (bar, val) in enumerate(zip(bars2, mae)):
        if i < 3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=7)
    
    # 3. RMSE Comparison (Lower is Better)
    ax3 = axes[0, 2]
    bars3 = ax3.bar(range(len(models)), rmse, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_ylabel('RMSE ↓', fontsize=12, fontweight='bold')
    ax3.set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on top bars (only for top 3)
    for i, (bar, val) in enumerate(zip(bars3, rmse)):
        if i < 3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=7)
    
    # 4. R² Comparison (Higher is Better)
    ax4 = axes[1, 0]
    bars4 = ax4.bar(range(len(models)), r2, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_ylabel('R² ↑', fontsize=12, fontweight='bold')
    ax4.set_title('R² Comparison (Higher is Better)', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    # Add horizontal reference lines
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Baseline (R²=0)')
    ax4.axhline(y=0.6, color='green', linestyle='--', linewidth=1.5, label='R²=0.6')
    # Add value labels on top bars (only for top 3)
    for i, (bar, val) in enumerate(zip(bars4, r2)):
        if i < 3:
            height = bar.get_height()
            y_pos = height + 0.02 if height >= 0 else height - 0.05
            ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=7)
    ax4.legend(loc='upper right', fontsize=8)
    
    # 5. Training Time Comparison
    ax5 = axes[1, 1]
    bars5 = ax5.bar(range(len(models)), train_time, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax5.set_ylabel('Training Time (s)', fontsize=12, fontweight='bold')
    ax5.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(len(models)))
    ax5.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 6. Combined Performance Score (normalized metrics)
    ax6 = axes[1, 2]
    # Create a composite score: normalized R² (higher is better) - normalized MSE (lower is better)
    normalized_r2 = [(r - min(r2)) / (max(r2) - min(r2)) if max(r2) != min(r2) else 0 for r in r2]
    normalized_mse = [(max(mse) - m) / (max(mse) - min(mse)) if max(mse) != min(mse) else 0 for m in mse]
    composite_score = [0.6 * nr2 + 0.4 * nm for nr2, nm in zip(normalized_r2, normalized_mse)]
    bars6 = ax6.bar(range(len(models)), composite_score, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax6.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
    ax6.set_title('Composite Performance Score', fontsize=13, fontweight='bold')
    ax6.set_xticks(range(len(models)))
    ax6.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Model Performance Comparison - All Models', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table4_model_comparison.png', dpi=300, bbox_inches='tight')
    try:
        plt.savefig(figures_dir / 'table4_model_comparison.pdf', bbox_inches='tight')
    except PermissionError:
        print("  (PDF file is locked, skipping PDF save)")
    print("✓ Created Table 4 visualization")
    plt.close()


def create_table7_visualization():
    """Table 7: Fusion Weight α Ablation Study"""
    alpha_values = np.arange(0, 1.1, 0.1)
    mse = [0.000520, 0.000493, 0.000468, 0.000452, 0.000435, 0.000420, 
           0.000432, 0.000445, 0.000462, 0.000475, 0.000485]
    mae = [0.01780, 0.01745, 0.01720, 0.01695, 0.01670, 0.01650,
           0.01665, 0.01675, 0.01690, 0.01705, 0.01725]
    rmse = [0.022803, 0.022205, 0.021640, 0.021320, 0.020900, 0.020494,
            0.020820, 0.021000, 0.021350, 0.021670, 0.022023]
    r2 = [0.585, 0.603, 0.622, 0.636, 0.648, 0.655, 0.646, 0.641, 0.633, 0.622, 0.612]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MSE
    axes[0, 0].plot(alpha_values, mse, 'o-', linewidth=2, markersize=8, color='#d62728')
    axes[0, 0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Optimal α=0.5')
    axes[0, 0].scatter([0.5], [0.000420], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    axes[0, 0].set_xlabel('Fusion Weight α', fontweight='bold')
    axes[0, 0].set_ylabel('Validation MSE ↓', fontweight='bold')
    axes[0, 0].set_title('(a) MSE vs Fusion Weight', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    
    # MAE
    axes[0, 1].plot(alpha_values, mae, 'o-', linewidth=2, markersize=8, color='#ff7f0e')
    axes[0, 1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Optimal α=0.5')
    axes[0, 1].scatter([0.5], [0.01650], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    axes[0, 1].set_xlabel('Fusion Weight α', fontweight='bold')
    axes[0, 1].set_ylabel('Validation MAE ↓', fontweight='bold')
    axes[0, 1].set_title('(b) MAE vs Fusion Weight', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    
    # RMSE
    axes[1, 0].plot(alpha_values, rmse, 'o-', linewidth=2, markersize=8, color='#2ca02c')
    axes[1, 0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Optimal α=0.5')
    axes[1, 0].scatter([0.5], [0.020494], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    axes[1, 0].set_xlabel('Fusion Weight α', fontweight='bold')
    axes[1, 0].set_ylabel('Validation RMSE ↓', fontweight='bold')
    axes[1, 0].set_title('(c) RMSE vs Fusion Weight', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()
    
    # R²
    axes[1, 1].plot(alpha_values, r2, 'o-', linewidth=2, markersize=8, color='#9467bd')
    axes[1, 1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Optimal α=0.5')
    axes[1, 1].scatter([0.5], [0.655], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    axes[1, 1].set_xlabel('Fusion Weight α', fontweight='bold')
    axes[1, 1].set_ylabel('Validation R² ↑', fontweight='bold')
    axes[1, 1].set_title('(d) R² vs Fusion Weight', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle('Table 7: Effect of Fusion Weight α on TCiT Performance', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table7_fusion_weight_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table7_fusion_weight_ablation.pdf', bbox_inches='tight')
    print("✓ Created Table 7 visualization")
    plt.close()


def create_table8_visualization():
    """Table 8: Feature Engineering Impact"""
    feature_sets = ['Original only\n(base 94)', '+ Lagged\nfeatures', 
                    '+ Rolling\nstatistics', '+ Group Stats\n(Full)']
    dims = [94, 282, 376, 397]
    mse = [0.001120, 0.000770, 0.000600, 0.000420]
    mae = [0.0285, 0.0229, 0.0189, 0.0165]
    rmse = [0.033466, 0.027748, 0.024495, 0.020494]
    r2 = [0.104, 0.381, 0.528, 0.655]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(feature_sets))
    width = 0.6
    
    # MSE
    bars1 = axes[0, 0].bar(x, mse, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[0, 0].set_ylabel('Validation MSE ↓', fontweight='bold')
    axes[0, 0].set_title('(a) Mean Squared Error', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(feature_sets, rotation=0, ha='center')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, mse)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
                       f'{val:.6f}', ha='center', va='bottom', fontsize=8)
    
    # MAE
    bars2 = axes[0, 1].bar(x, mae, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[0, 1].set_ylabel('Validation MAE ↓', fontweight='bold')
    axes[0, 1].set_title('(b) Mean Absolute Error', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(feature_sets, rotation=0, ha='center')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, mae)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # R²
    bars3 = axes[1, 0].bar(x, r2, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    axes[1, 0].set_ylabel('Validation R² ↑', fontweight='bold')
    axes[1, 0].set_title('(c) R-squared Score', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(feature_sets, rotation=0, ha='center')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 0.7])
    for i, (bar, val) in enumerate(zip(bars3, r2)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Dimension vs Performance
    ax2 = axes[1, 1]
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(dims, mse, 'o-', color='#d62728', linewidth=2, markersize=10, label='MSE')
    line2 = ax2_twin.plot(dims, r2, 's-', color='#2ca02c', linewidth=2, markersize=10, label='R²')
    ax2.set_xlabel('Feature Dimension', fontweight='bold')
    ax2.set_ylabel('Validation MSE ↓', fontweight='bold', color='#d62728')
    ax2_twin.set_ylabel('Validation R² ↑', fontweight='bold', color='#2ca02c')
    ax2.set_title('(d) Feature Dimension vs Performance', fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2_twin.tick_params(axis='y', labelcolor='#2ca02c')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    plt.suptitle('Table 8: Impact of Feature Engineering on TCiT Performance', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table8_feature_engineering.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table8_feature_engineering.pdf', bbox_inches='tight')
    print("✓ Created Table 8 visualization")
    plt.close()


def create_table9_visualization():
    """Table 9: Model Size Ablation"""
    d_models = [64, 128, 192, 256, 512]
    params = [0.5, 1.2, 2.5, 4.2, 12.8]
    mse = [0.000780, 0.000610, 0.000420, 0.000442, 0.000471]
    mae = [0.0208, 0.0196, 0.0165, 0.0168, 0.0172]
    r2 = [0.372, 0.521, 0.655, 0.639, 0.617]
    train_time = [310, 520, 850, 1270, 2920]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MSE and R² vs d_model
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(d_models, mse, 'o-', color='#d62728', linewidth=2, markersize=10, label='MSE')
    line2 = ax1_twin.plot(d_models, r2, 's-', color='#2ca02c', linewidth=2, markersize=10, label='R²')
    ax1.axvline(x=192, color='green', linestyle='--', linewidth=2, label='Optimal d_model=192')
    ax1.scatter([192], [0.000420], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    ax1.set_xlabel('d_model', fontweight='bold')
    ax1.set_ylabel('Validation MSE ↓', fontweight='bold', color='#d62728')
    ax1_twin.set_ylabel('Validation R² ↑', fontweight='bold', color='#2ca02c')
    ax1.set_title('(a) Model Dimension vs Performance', fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1_twin.tick_params(axis='y', labelcolor='#2ca02c')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines + [mpatches.Patch(color='green', label='Optimal d_model=192')], 
              labels + ['Optimal d_model=192'], loc='center right')
    
    # Parameters vs Performance
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    line3 = ax2.plot(params, mse, 'o-', color='#d62728', linewidth=2, markersize=10, label='MSE')
    line4 = ax2_twin.plot(params, r2, 's-', color='#2ca02c', linewidth=2, markersize=10, label='R²')
    ax2.scatter([2.5], [0.000420], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    ax2.set_xlabel('Parameters (M)', fontweight='bold')
    ax2.set_ylabel('Validation MSE ↓', fontweight='bold', color='#d62728')
    ax2_twin.set_ylabel('Validation R² ↑', fontweight='bold', color='#2ca02c')
    ax2.set_title('(b) Model Parameters vs Performance', fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2_twin.tick_params(axis='y', labelcolor='#2ca02c')
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    # Training Time vs Performance
    ax3 = axes[1, 0]
    scatter = ax3.scatter(train_time, mse, s=[r*500 for r in r2], c=d_models, 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
    ax3.scatter([850], [0.000420], s=300, color='green', zorder=5, edgecolors='black', linewidth=2)
    for i, d in enumerate(d_models):
        ax3.annotate(f'd={d}', (train_time[i], mse[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Training Time (s)', fontweight='bold')
    ax3.set_ylabel('Validation MSE ↓', fontweight='bold')
    ax3.set_title('(c) Training Time vs Performance (bubble size = R²)', fontweight='bold')
    ax3.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('d_model', fontweight='bold')
    
    # MAE comparison
    bars = axes[1, 1].bar(range(len(d_models)), mae, color=['#1f77b4', '#ff7f0e', '#d62728', 
                                                             '#2ca02c', '#9467bd'], alpha=0.8)
    bars[2].set_color('#d62728')  # Highlight optimal
    axes[1, 1].set_xlabel('d_model', fontweight='bold')
    axes[1, 1].set_ylabel('Validation MAE ↓', fontweight='bold')
    axes[1, 1].set_title('(d) Mean Absolute Error by Model Size', fontweight='bold')
    axes[1, 1].set_xticks(range(len(d_models)))
    axes[1, 1].set_xticklabels(d_models)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, mae)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Table 9: Model Size Ablation for iTransformer Branch within TCiT', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table9_model_size_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table9_model_size_ablation.pdf', bbox_inches='tight')
    print("✓ Created Table 9 visualization")
    plt.close()


def create_table10_visualization():
    """Table 10: Lookback Window Impact"""
    lookbacks = [20, 30, 40, 50, 60]
    mse = [0.000965, 0.000690, 0.000420, 0.000456, 0.000508]
    mae = [0.0246, 0.0209, 0.0165, 0.0170, 0.0174]
    rmse = [0.031064, 0.026266, 0.020494, 0.021354, 0.022536]
    r2 = [0.225, 0.441, 0.655, 0.629, 0.583]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MSE
    axes[0, 0].plot(lookbacks, mse, 'o-', linewidth=2, markersize=10, color='#d62728')
    axes[0, 0].axvline(x=40, color='green', linestyle='--', linewidth=2, label='Optimal Lookback=40')
    axes[0, 0].scatter([40], [0.000420], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    axes[0, 0].set_xlabel('Lookback Window Size', fontweight='bold')
    axes[0, 0].set_ylabel('Validation MSE ↓', fontweight='bold')
    axes[0, 0].set_title('(a) Mean Squared Error', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    
    # MAE
    axes[0, 1].plot(lookbacks, mae, 'o-', linewidth=2, markersize=10, color='#ff7f0e')
    axes[0, 1].axvline(x=40, color='green', linestyle='--', linewidth=2, label='Optimal Lookback=40')
    axes[0, 1].scatter([40], [0.0165], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    axes[0, 1].set_xlabel('Lookback Window Size', fontweight='bold')
    axes[0, 1].set_ylabel('Validation MAE ↓', fontweight='bold')
    axes[0, 1].set_title('(b) Mean Absolute Error', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    
    # R²
    axes[1, 0].plot(lookbacks, r2, 'o-', linewidth=2, markersize=10, color='#2ca02c')
    axes[1, 0].axvline(x=40, color='green', linestyle='--', linewidth=2, label='Optimal Lookback=40')
    axes[1, 0].scatter([40], [0.655], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    axes[1, 0].set_xlabel('Lookback Window Size', fontweight='bold')
    axes[1, 0].set_ylabel('Validation R² ↑', fontweight='bold')
    axes[1, 0].set_title('(c) R-squared Score', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()
    
    # All metrics together
    ax = axes[1, 1]
    ax_twin = ax.twinx()
    line1 = ax.plot(lookbacks, mse, 'o-', color='#d62728', linewidth=2, markersize=8, label='MSE')
    line2 = ax_twin.plot(lookbacks, r2, 's-', color='#2ca02c', linewidth=2, markersize=8, label='R²')
    ax.axvline(x=40, color='green', linestyle='--', linewidth=2)
    ax.scatter([40], [0.000420], s=200, color='green', zorder=5, edgecolors='black', linewidth=2)
    ax.set_xlabel('Lookback Window Size', fontweight='bold')
    ax.set_ylabel('Validation MSE ↓', fontweight='bold', color='#d62728')
    ax_twin.set_ylabel('Validation R² ↑', fontweight='bold', color='#2ca02c')
    ax.set_title('(d) Combined Metrics View', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='y', labelcolor='#d62728')
    ax_twin.tick_params(axis='y', labelcolor='#2ca02c')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    plt.suptitle('Table 10: Effect of Lookback Window Size on TCiT Performance', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table10_lookback_impact.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table10_lookback_impact.pdf', bbox_inches='tight')
    print("✓ Created Table 10 visualization")
    plt.close()


def create_table11_12_visualization():
    """Table 11 & 12: Residual Statistics and Normality Tests"""
    models = ['TCiT', 'iTransformer', 'TimeCMA', 'LSTM']
    mean = [-0.000001, -0.000000, 0.000004, 0.000032]
    std = [0.02049, 0.02202, 0.02280, 0.02987]
    skewness = [-0.012, -0.045, -0.031, -0.118]
    kurtosis = [2.986, 3.012, 2.941, 3.214]
    dw_stat = [2.03, 2.01, 1.97, 1.85]
    sw_pvalue = [0.442, 0.385, 0.361, 0.128]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Mean residuals
    bars1 = axes[0, 0].bar(models, mean, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 0].set_ylabel('Mean Residual', fontweight='bold')
    axes[0, 0].set_title('(a) Mean Residuals', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Standard deviation
    bars2 = axes[0, 1].bar(models, std, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[0, 1].set_ylabel('Standard Deviation', fontweight='bold')
    axes[0, 1].set_title('(b) Residual Standard Deviation', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Skewness
    bars3 = axes[0, 2].bar(models, skewness, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 2].set_ylabel('Skewness', fontweight='bold')
    axes[0, 2].set_title('(c) Residual Skewness', fontweight='bold')
    axes[0, 2].grid(axis='y', alpha=0.3)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Kurtosis
    bars4 = axes[1, 0].bar(models, kurtosis, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[1, 0].axhline(y=3, color='red', linestyle='--', linewidth=1, label='Normal kurtosis=3')
    axes[1, 0].set_ylabel('Kurtosis', fontweight='bold')
    axes[1, 0].set_title('(d) Residual Kurtosis', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Durbin-Watson statistic
    bars5 = axes[1, 1].bar(models, dw_stat, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[1, 1].axhline(y=2, color='green', linestyle='--', linewidth=1, label='Ideal DW=2')
    axes[1, 1].set_ylabel('Durbin-Watson Statistic', fontweight='bold')
    axes[1, 1].set_title('(e) Durbin-Watson Statistic', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Shapiro-Wilk p-value
    bars6 = axes[1, 2].bar(models, sw_pvalue, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[1, 2].axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='α=0.05 threshold')
    axes[1, 2].set_ylabel('Shapiro-Wilk p-value', fontweight='bold')
    axes[1, 2].set_title('(f) Normality Test (p-value)', fontweight='bold')
    axes[1, 2].grid(axis='y', alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].set_ylim([0, 0.5])
    
    plt.suptitle('Table 11 & 12: Residual Statistics and Normality Tests', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table11_12_residual_stats.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table11_12_residual_stats.pdf', bbox_inches='tight')
    print("✓ Created Table 11 & 12 visualization")
    plt.close()


def create_table13_visualization():
    """Table 13: Market Regime Performance"""
    models = ['TCiT', 'iTransformer', 'TimeCMA', 'LSTM']
    volatile_mse = [0.000760, 0.000985, 0.001150, 0.001850]
    normal_mse = [0.000280, 0.000325, 0.000395, 0.000685]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Volatile markets
    bars1 = axes[0].bar(x - width/2, volatile_mse, width, label='Volatile Markets', 
                       color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[0].set_ylabel('MSE', fontweight='bold')
    axes[0].set_title('(a) Volatile Markets Performance', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()
    for bar, val in zip(bars1, volatile_mse):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00002,
                    f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    # Normal markets
    bars2 = axes[1].bar(x - width/2, normal_mse, width, label='Normal Markets',
                       color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[1].set_ylabel('MSE', fontweight='bold')
    axes[1].set_title('(b) Normal Markets Performance', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()
    for bar, val in zip(bars2, normal_mse):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
                    f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Table 13: MSE Across Market Regimes', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table13_market_regimes.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table13_market_regimes.pdf', bbox_inches='tight')
    print("✓ Created Table 13 visualization")
    plt.close()
    
    # Additional comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, volatile_mse, width, label='Volatile Markets', 
                   color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, normal_mse, width, label='Normal Markets', 
                   color='#2ca02c', alpha=0.8)
    ax.set_ylabel('MSE', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Table 13: Market Regime Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table13_market_regimes_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table13_market_regimes_comparison.pdf', bbox_inches='tight')
    plt.close()


def create_table14_visualization():
    """Table 14: Tail Events Performance"""
    models = ['TCiT', 'iTransformer', 'TimeCMA', 'LSTM']
    pos_tail_mse = [0.00192, 0.00245, 0.00278, 0.00420]
    neg_tail_mse = [0.00204, 0.00266, 0.00301, 0.00465]
    avg_tail_mae = [0.0308, 0.0342, 0.0369, 0.0458]
    tail_r2 = [0.411, 0.355, 0.330, 0.241]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(models))
    width = 0.6
    
    # Positive tail MSE
    bars1 = axes[0, 0].bar(x, pos_tail_mse, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[0, 0].set_ylabel('Positive Tail MSE ↓', fontweight='bold')
    axes[0, 0].set_title('(a) Positive Tail Events MSE', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, pos_tail_mse):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Negative tail MSE
    bars2 = axes[0, 1].bar(x, neg_tail_mse, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[0, 1].set_ylabel('Negative Tail MSE ↓', fontweight='bold')
    axes[0, 1].set_title('(b) Negative Tail Events MSE', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, neg_tail_mse):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Average tail MAE
    bars3 = axes[1, 0].bar(x, avg_tail_mae, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[1, 0].set_ylabel('Average Tail MAE ↓', fontweight='bold')
    axes[1, 0].set_title('(c) Average Tail Events MAE', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, avg_tail_mae):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Tail R²
    bars4 = axes[1, 1].bar(x, tail_r2, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd'], alpha=0.8)
    axes[1, 1].set_ylabel('Tail R² ↑', fontweight='bold')
    axes[1, 1].set_title('(d) Tail Events R² Score', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, tail_r2):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Table 14: Error in Tail Events (5% Largest Absolute Returns)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'table14_tail_events.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'table14_tail_events.pdf', bbox_inches='tight')
    print("✓ Created Table 14 visualization")
    plt.close()


def main():
    """Generate all table visualizations"""
    print("Generating visualizations for all tables...")
    print("=" * 60)
    
    create_table3_visualization()
    create_table4_visualization()
    create_table7_visualization()
    create_table8_visualization()
    create_table9_visualization()
    create_table10_visualization()
    create_table11_12_visualization()
    create_table13_visualization()
    create_table14_visualization()
    
    print("=" * 60)
    print(f"✓ All visualizations saved to '{figures_dir}/' directory")
    print("Generated files:")
    print("  - table3_baseline_configs.png/pdf")
    print("  - table4_model_comparison.png/pdf")
    print("  - table7_fusion_weight_ablation.png/pdf")
    print("  - table8_feature_engineering.png/pdf")
    print("  - table9_model_size_ablation.png/pdf")
    print("  - table10_lookback_impact.png/pdf")
    print("  - table11_12_residual_stats.png/pdf")
    print("  - table13_market_regimes.png/pdf")
    print("  - table13_market_regimes_comparison.png/pdf")
    print("  - table14_tail_events.png/pdf")


if __name__ == '__main__':
    main()

