# 数学推理评估脚本使用说明

## 快速开始

### 方式一：自动路径检测（推荐）

脚本会自动检测 `data/math_datasets` 目录下的数据集文件：

```bash
cd examples/ToolBench

# 只需要设置预测文件路径
export SVAMP_PREDICTIONS="predictions/svamp_predictions.json"
export GSM8K_PREDICTIONS="predictions/gsm8k_predictions.json"
export NUMGLUE_PREDICTIONS="predictions/numglue_predictions.json"
export MATHEMATICA_PREDICTIONS="predictions/mathematica_predictions.json"

# 运行评估（数据集路径会自动检测）
bash scripts/eval_math_reasoning.sh
```

### 方式二：使用配置加载脚本

```bash
cd examples/ToolBench

# 加载默认配置
source scripts/load_math_eval_config.sh

# 设置预测文件路径（如果需要覆盖默认路径）
export SVAMP_PREDICTIONS="path/to/your/predictions.json"

# 运行评估
bash scripts/eval_math_reasoning.sh
```

### 方式三：手动设置所有路径

```bash
cd examples/ToolBench

# 设置数据集路径
export SVAMP_DATASET="data/math_datasets/svamp/test.json"
export GSM8K_DATASET="data/math_datasets/gsm8k/test.jsonl"
export NUMGLUE_DATASET="data/math_datasets/numglue/test.json"
export MATHEMATICA_DATASET="data/math_datasets/mathematica/test.json"

# 设置预测文件路径
export SVAMP_PREDICTIONS="predictions/svamp_predictions.json"
export GSM8K_PREDICTIONS="predictions/gsm8k_predictions.json"
export NUMGLUE_PREDICTIONS="predictions/numglue_predictions.json"
export MATHEMATICA_PREDICTIONS="predictions/mathematica_predictions.json"

# 运行评估
bash scripts/eval_math_reasoning.sh
```

## 默认路径结构

脚本默认查找以下路径结构：

```
examples/ToolBench/
├── data/
│   └── math_datasets/
│       ├── svamp/
│       │   └── test.json
│       ├── gsm8k/
│       │   └── test.jsonl
│       ├── numglue/
│       │   └── test.json
│       └── mathematica/
│           └── test.json
└── predictions/
    └── math_reasoning/
        ├── svamp_predictions.json
        ├── gsm8k_predictions.json
        ├── numglue_predictions.json
        └── mathematica_predictions.json
```

## 自定义路径

可以通过环境变量自定义基础路径：

```bash
# 自定义数据集目录
export DATA_BASE_DIR="/path/to/your/datasets"

# 自定义预测文件目录
export PREDICTIONS_BASE_DIR="/path/to/your/predictions"

# 然后运行脚本
bash scripts/eval_math_reasoning.sh
```

## 只评估部分数据集

脚本会自动跳过未设置的数据集。例如，只评估 SVAMP：

```bash
export SVAMP_PREDICTIONS="predictions/svamp_predictions.json"
# 数据集路径会自动检测，或者手动设置：
# export SVAMP_DATASET="data/math_datasets/svamp/test.json"

bash scripts/eval_math_reasoning.sh
```

## 输出结果

结果保存在 `math_reasoning_eval_results/{TIMESTAMP}/` 目录下：

```
math_reasoning_eval_results/
└── 20241207_143022/
    ├── svamp/
    │   ├── svamp_results.json
    │   └── svamp_summary.txt
    ├── gsm8k/
    │   ├── gsm8k_results.json
    │   └── gsm8k_summary.txt
    └── ...
```

## 环境变量说明

### 数据集路径（可选，会自动检测）
- `SVAMP_DATASET`: SVAMP 测试数据集路径
- `GSM8K_DATASET`: GSM8K 测试数据集路径
- `NUMGLUE_DATASET`: NumGLUE 测试数据集路径
- `MATHEMATICA_DATASET`: Mathematica 测试数据集路径

### 预测文件路径（必需）
- `SVAMP_PREDICTIONS`: SVAMP 预测结果文件
- `GSM8K_PREDICTIONS`: GSM8K 预测结果文件
- `NUMGLUE_PREDICTIONS`: NumGLUE 预测结果文件
- `MATHEMATICA_PREDICTIONS`: Mathematica 预测结果文件

### 基础路径（可选）
- `DATA_BASE_DIR`: 数据集基础目录（默认：`scripts/../data/math_datasets`）
- `PREDICTIONS_BASE_DIR`: 预测文件基础目录（默认：`scripts/../predictions/math_reasoning`）

## 示例工作流

```bash
cd examples/ToolBench

# 1. 下载数据集（如果还没有）
python scripts/download_math_datasets.py --data_dir data/math_datasets

# 2. 生成模型预测（使用你的模型）
# ... 你的推理代码生成预测文件 ...

# 3. 运行评估（最简单的方式）
export SVAMP_PREDICTIONS="predictions/my_model_svamp.json"
export GSM8K_PREDICTIONS="predictions/my_model_gsm8k.json"

bash scripts/eval_math_reasoning.sh
```

## 故障排除

1. **数据集路径未找到**：检查 `data/math_datasets` 目录结构是否正确
2. **预测文件未找到**：确保设置了正确的 `*_PREDICTIONS` 环境变量
3. **路径问题**：使用绝对路径或确保从 `examples/ToolBench` 目录运行脚本

