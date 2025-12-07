# 数学推理推理脚本使用说明

## 概述

这些脚本用于使用训练好的模型在数学推理数据集上生成预测，是连接训练和评估的中间步骤。

## 脚本说明

### 1. `inference_math_reasoning.sh` - 单个数据集推理

对单个数学推理数据集运行推理。

**用法：**
```bash
bash scripts/inference_math_reasoning.sh --dataset DATASET [OPTIONS]
```

**必需参数：**
- `--dataset`: 数据集名称（svamp, gsm8k, numglue, mathematica）

**可选参数：**
- `--model_path PATH`: 模型路径（默认：`toolllama`）
- `--lora`: 使用 LoRA 模型
- `--lora_path PATH`: LoRA 适配器路径（使用 `--lora` 时必须指定）
- `--device DEVICE`: 设备（cuda, cpu, mps，默认：cuda）
- `--max_new_tokens N`: 最大生成 token 数（默认：512）
- `--max_seq_length N`: 最大序列长度（默认：2048）
- `--template TEMPLATE`: 对话模板（默认：tool-llama-single-round）
- `--data_dir DIR`: 数据集基础目录（默认：`data/math_datasets`）
- `--output_dir DIR`: 预测输出目录（默认：`predictions/math_reasoning`）

**示例：**
```bash
# 基本使用
bash scripts/inference_math_reasoning.sh --dataset svamp --model_path toolllama/

# 使用 LoRA 模型
bash scripts/inference_math_reasoning.sh \
    --dataset gsm8k \
    --model_path base_model/ \
    --lora \
    --lora_path lora_model/

# 自定义路径
bash scripts/inference_math_reasoning.sh \
    --dataset numglue \
    --model_path /path/to/model \
    --data_dir /path/to/datasets \
    --output_dir /path/to/output
```

### 2. `inference_all_math_datasets.sh` - 批量推理

对所有数学推理数据集运行推理。

**用法：**
```bash
bash scripts/inference_all_math_datasets.sh [OPTIONS]
```

**参数：**（与 `inference_math_reasoning.sh` 相同，但不需要 `--dataset`）

**特殊选项：**
- `--skip_existing`: 跳过已有预测文件的数据集

**示例：**
```bash
# 推理所有数据集
bash scripts/inference_all_math_datasets.sh --model_path toolllama/

# 跳过已有预测
bash scripts/inference_all_math_datasets.sh \
    --model_path toolllama/ \
    --skip_existing
```

### 3. `inference_math_reasoning.py` - Python 推理脚本

Python 底层推理脚本，可直接调用。

**用法：**
```bash
python toolbench/inference/inference_math_reasoning.py \
    --dataset DATASET \
    --dataset_path PATH \
    --model_path PATH \
    --output_path PATH \
    [OPTIONS]
```

## 完整工作流程

### 步骤 1: 训练模型
```bash
cd examples/ToolBench
bash scripts/train_toolllama_bf16.sh
# 输出: toolllama/ 目录
```

### 步骤 2: 下载数据集
```bash
python scripts/download_math_datasets.py --data_dir data/math_datasets
```

### 步骤 3: 生成预测
```bash
# 方式1: 单个数据集
bash scripts/inference_math_reasoning.sh \
    --dataset svamp \
    --model_path toolllama/

# 方式2: 所有数据集
bash scripts/inference_all_math_datasets.sh \
    --model_path toolllama/
```

### 步骤 4: 评估预测
```bash
# 设置预测文件路径
export SVAMP_PREDICTIONS="predictions/math_reasoning/svamp_predictions.json"
export GSM8K_PREDICTIONS="predictions/math_reasoning/gsm8k_predictions.json"
export NUMGLUE_PREDICTIONS="predictions/math_reasoning/numglue_predictions.json"
export MATHEMATICA_PREDICTIONS="predictions/math_reasoning/mathematica_predictions.json"

# 运行评估
bash scripts/eval_math_reasoning.sh
```

## 输出格式

预测文件保存为 JSON 格式，键为问题 ID，值为模型响应：

```json
{
    "question_id_1": "模型的响应文本",
    "question_id_2": "模型的响应文本",
    ...
}
```

此格式可直接用于 `eval_math_reasoning.sh` 评估脚本。

## 支持的数据集

- **SVAMP**: Simple Variations on Arithmetic Math Problems
- **GSM8K**: Grade School Math 8K
- **NumGLUE**: Numerical GLUE benchmark
- **Mathematica**: Mathematica-style math problems

## 模型支持

- **标准模型**: 使用 `--model_path` 指定模型路径
- **LoRA 模型**: 使用 `--lora` 和 `--lora_path` 指定基础模型和 LoRA 适配器
- **ToolLLaMA**: 使用 `--use_toolllama` 标志（在 Python 脚本中）

## 故障排除

### 1. 模型加载失败
- 检查模型路径是否正确
- 确保模型目录包含必要的文件（`config.json`, `pytorch_model.bin` 等）
- 如果使用 LoRA，确保基础模型和 LoRA 路径都正确

### 2. 内存不足
- 减少 `--max_sequence_length`
- 使用 CPU 模式（`--device cpu`）
- 减少 `--max_new_tokens`

### 3. 数据集路径错误
- 使用 `--data_dir` 指定正确的数据集目录
- 确保数据集文件存在且格式正确

### 4. 输出格式问题
- 检查输出目录是否有写入权限
- 确保有足够的磁盘空间

## 性能优化

1. **批量处理**: 使用 `inference_all_math_datasets.sh` 批量处理所有数据集
2. **跳过已存在**: 使用 `--skip_existing` 避免重复推理
3. **GPU 加速**: 使用 `--device cuda`（默认）以获得最佳性能
4. **Token 限制**: 根据问题复杂度调整 `--max_new_tokens`

## 与评估脚本集成

推理脚本的输出格式与评估脚本完全兼容：

```bash
# 生成预测
bash scripts/inference_math_reasoning.sh --dataset svamp --model_path toolllama/

# 直接评估（预测路径会自动检测）
export SVAMP_PREDICTIONS="predictions/math_reasoning/svamp_predictions.json"
bash scripts/eval_math_reasoning.sh
```

