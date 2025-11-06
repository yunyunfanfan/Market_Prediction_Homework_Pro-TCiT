# iTransformer 使用指南

## 项目概述

这是一个使用 **iTransformer**（反向Transformer）架构来预测市场前向超额收益的项目，用于 Hull Tactical Market Prediction Kaggle 竞赛。

## 快速开始

### 第一步：安装依赖

```bash
pip install -r requirements.txt
```

### 第二步：准备数据

确保 `train.csv` 和 `test.csv` 在项目根目录下。

### 第三步：选择训练脚本

**我们提供三个版本的训练脚本，根据你的硬件选择：**

#### 方案A：平衡版（⭐ 强烈推荐）
性能与内存的最佳平衡：
```bash
python train_balanced.py
```

#### 方案B：轻量版
适合低内存系统（< 8GB）：
```bash
python train_light.py
```

#### 方案C：完整版
适合高性能系统（> 16GB）：
```bash
python train.py
```

#### 生成可视化

```bash
python visualize.py --experiment_name itransformer_balanced
```

**详细对比请查看 [TRAINING_CONFIGS.md](TRAINING_CONFIGS.md)**

#### 第四步（可选）：与其他方法对比

为了突出iTransformer的优势，可以与其他机器学习方法对比：

```bash
# 运行对比实验（训练8个模型）
python compare_models.py

# 可视化对比结果
python visualize_comparison.py
```

这会与以下方法对比：
- 传统机器学习：线性回归、随机森林、GBDT、XGBoost
- 深度学习：MLP、LSTM、GRU

**详细说明请查看 [MODEL_COMPARISON.md](MODEL_COMPARISON.md)**

## 高级用法

### 自定义训练参数

```bash
python train.py \
    --lookback 75 \
    --d_model 512 \
    --num_layers 4 \
    --nhead 8 \
    --batch_size 32 \
    --num_epochs 150 \
    --learning_rate 5e-5 \
    --experiment_name my_experiment
```

### 重要参数说明

#### 数据参数
- `--lookback`: 回看窗口大小（默认50）- 使用多少历史时间步
- `--val_split`: 验证集比例（默认0.2）
- `--include_lagged`: 是否包含滞后特征
- `--include_rolling`: 是否包含滚动统计特征

#### 模型参数
- `--model_type`: 模型类型 ('simple' 或 'full')
- `--d_model`: 模型嵌入维度（默认256）- 越大模型越强大但训练越慢
- `--nhead`: 注意力头数（默认8）
- `--num_layers`: Transformer层数（默认3）
- `--dim_feedforward`: 前馈网络维度（默认1024）
- `--dropout`: Dropout率（默认0.1）

#### 训练参数
- `--batch_size`: 批次大小（默认64）
- `--num_epochs`: 训练轮数（默认100）
- `--learning_rate`: 学习率（默认1e-4）
- `--scheduler`: 学习率调度器 ('cosine' 或 'plateau')
- `--early_stopping_patience`: 早停耐心值（默认15）

## 项目结构说明

```
kaggle/
├── train.csv                          # 训练数据
├── test.csv                           # 测试数据
├── requirements.txt                   # Python依赖
├── README.md                          # 项目说明（英文）
├── USAGE_GUIDE.md                     # 使用指南（本文件）
├── train.py                           # 训练脚本
├── visualize.py                       # 可视化脚本
├── quick_start.py                     # 快速启动脚本
├── analysis_notebook.ipynb            # 分析笔记本
│
├── src/                               # 源代码
│   ├── data/
│   │   └── preprocessing.py           # 数据预处理和特征工程
│   ├── models/
│   │   └── itransformer.py            # iTransformer模型实现
│   └── utils/
│       └── trainer.py                 # 训练工具
│
├── checkpoints/                       # 模型检查点
│   └── {experiment}_best.pth          # 最佳模型
│
├── results/                           # 结果文件
│   ├── {experiment}_predictions.csv   # 预测结果
│   ├── {experiment}_history.json      # 训练历史
│   ├── {experiment}_config.json       # 模型配置
│   └── {experiment}_report.txt        # 性能报告
│
└── figures/                           # 可视化图表
    ├── {experiment}_training_history.png
    ├── {experiment}_predictions.png
    └── {experiment}_error_analysis.png
```

## 输出文件说明

### 在 `checkpoints/` 目录
- `{experiment}_best.pth`: 验证集上表现最好的模型检查点

### 在 `results/` 目录
- `{experiment}_predictions.csv`: 包含实际值和预测值的CSV文件
- `{experiment}_history.json`: 训练过程中的损失和指标
- `{experiment}_config.json`: 模型和训练的完整配置
- `{experiment}_report.txt`: 详细的性能报告

### 在 `figures/` 目录
- `{experiment}_training_history.png`: 训练过程可视化
  - 训练/验证损失
  - MSE指标
  - 学习率变化
  - 过拟合监控
  
- `{experiment}_predictions.png`: 预测结果可视化
  - 预测vs实际散点图
  - 残差图
  - 残差分布
  - 时间序列对比
  
- `{experiment}_error_analysis.png`: 错误分析
  - 绝对误差随时间变化
  - 误差分布
  - Q-Q图
  - 累积误差

## 使用Jupyter Notebook进行探索

```bash
jupyter notebook analysis_notebook.ipynb
```

这个notebook包含：
1. 探索性数据分析
2. 特征分析
3. 目标变量分析
4. 模型结果加载和可视化

## 常见问题

### 1. 内存不足错误

**解决方案：**
- 减少 `--batch_size` (尝试32或16)
- 减少 `--d_model` (尝试128)
- 减少 `--lookback` (尝试30)

### 2. 模型不收敛

**解决方案：**
- 调整学习率（尝试1e-3到1e-5）
- 使用 'plateau' 调度器: `--scheduler plateau`
- 检查数据是否有异常值

### 3. 训练速度慢

**解决方案：**
- 使用更小的模型: `--d_model 128 --num_layers 2`
- 增加批次大小（如果内存允许）: `--batch_size 128`
- 减少特征工程复杂度

### 4. 性能不佳

**解决方案：**
- 增加训练轮数: `--num_epochs 200`
- 启用更多特征工程
- 尝试不同的lookback窗口
- 调整模型架构参数

## 性能优化建议

### 为了更好的性能：

1. **增加lookback窗口**: 尝试75-100时间步
   ```bash
   python train.py --lookback 100
   ```

2. **调整模型大小**: 尝试更大的模型
   ```bash
   python train.py --d_model 512 --num_layers 4
   ```

3. **启用完整特征工程**:
   ```bash
   python train.py --include_lagged --include_rolling
   ```

4. **使用ensemble**: 训练多个不同种子的模型
   ```bash
   python train.py --seed 42 --experiment_name model_1
   python train.py --seed 123 --experiment_name model_2
   python train.py --seed 456 --experiment_name model_3
   ```

### 为了更快的训练：

1. **减少模型复杂度**:
   ```bash
   python train.py --d_model 128 --num_layers 2
   ```

2. **使用更大的批次**:
   ```bash
   python train.py --batch_size 128
   ```

3. **减少训练轮数**:
   ```bash
   python train.py --num_epochs 50
   ```

## iTransformer 核心思想

### 传统Transformer用于时间序列
```
时间步作为tokens → 捕捉时间模式
问题：难以有效建模多变量相关性
```

### iTransformer方法
```
变量作为tokens → 捕捉多变量相关性
优势：更适合多变量预测任务
```

### 为什么有效

1. **多变量相关性**: 自注意力机制学习变量间关系
2. **序列表示**: LayerNorm和FFN学习更好的时间序列嵌入
3. **泛化能力**: 可以处理任意数量的变量
4. **可解释性**: 注意力权重显示变量相关性

## 评估指标

模型性能通过以下指标评估：

- **MSE** (Mean Squared Error): 均方误差 - 越小越好
- **RMSE** (Root Mean Squared Error): 均方根误差 - 越小越好
- **MAE** (Mean Absolute Error): 平均绝对误差 - 越小越好
- **R²** (R-squared): 决定系数 - 越接近1越好

## 完整工作流程示例

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行快速启动（自动化整个流程）
python quick_start.py

# 或者手动运行每一步：

# 3. 训练模型
python train.py \
    --lookback 75 \
    --d_model 256 \
    --num_layers 3 \
    --batch_size 64 \
    --num_epochs 100 \
    --experiment_name my_experiment

# 4. 生成可视化和报告
python visualize.py --experiment_name my_experiment

# 5. 在Jupyter中探索
jupyter notebook analysis_notebook.ipynb
```

## 参考资料

- **论文**: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)
- **GitHub**: [thuml/iTransformer](https://github.com/thuml/iTransformer)
- **竞赛**: [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)

## 技术支持

如有问题，请参考：
1. README.md - 完整的项目文档（英文）
2. 本使用指南 - 详细的使用说明（中文）
3. analysis_notebook.ipynb - 交互式分析示例

---

**祝你在项目中取得好成绩！** 🎓📊🚀

