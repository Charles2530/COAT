# COAT 在 OLMo 中的适配说明

本文档详细说明 COAT 如何适配 OLMo 模型，包括激活量化、权重量化和优化器量化的集成方式。

## 目录

1. [整体架构](#整体架构)
2. [模型适配](#模型适配)
3. [激活量化](#激活量化)
4. [权重量化](#权重量化)
5. [优化器量化](#优化器量化)
6. [配置文件](#配置文件)
7. [关键代码路径](#关键代码路径)

---

## 整体架构

COAT 通过以下方式适配 OLMo：

1. **模型替换**: 将 `OLMo` 替换为 `CoatOLMo`
2. **模块替换**: 将 `OLMoBlock` 替换为 `CoatOLMoBlock`
3. **优化器替换**: 将 `AdamW` 替换为 `CoatOLMoAdamW` (基于 `CoatAdamW`)
4. **自定义 Autograd Functions**: 使用 `torch.autograd.Function` 实现量化前向和反向传播

---

## 模型适配

### 1. 模型初始化 (`examples/OLMo/scripts/train.py`)

```python
if cfg.quantize_model.use_quantize_model == "coat_real":
    olmo_model = CoatOLMo(cfg.model, cfg.quantize_model)
    for name, module in olmo_model.named_modules():
        module.layer_name = name
```

**关键点**:
- 传入 `ModelConfig` 和 `QuantActivationConfig`
- 为每个模块设置 `layer_name`，用于量化时的标识

### 2. CoatOLMo 类结构 (`coat/models/coat_olmo.py`)

```python
class CoatOLMo(nn.Module):
    def __init__(self, config: ModelConfig, qargs: QuantActivationConfig, init_params: bool = True):
        # ... 初始化代码 ...
        
        # 关键：输入输出量化模块
        self.quantize_input_before_block = Coat_quantize_bgn(qargs)
        self.quantize_output_after_block = Coat_quantize_end(qargs)
```

**关键组件**:
- `quantize_input_before_block`: 在进入 decoder layers 前量化输入
- `quantize_output_after_block`: 在离开 decoder layers 后量化输出

### 3. Forward 流程

```python
def forward(self, input_ids, ...):
    # ... embedding 等 ...
    
    # 1. 进入 decoder 前量化
    x, qx, sx = self.quantize_input_before_block(x)
    
    # 2. 逐层处理
    for block in self.transformer.blocks:
        x, qx, sx, cache = block(x, qx, sx, ...)
    
    # 3. 离开 decoder 后量化
    x = self.quantize_output_after_block(x, qx, sx)
    
    # ... 输出层 ...
```

---

## 激活量化

### 1. 输入量化 (`Coat_quantize_bgn`)

**位置**: `coat/activation/real_quantization/func_quantize.py`

```python
class Coat_quantize_bgn(nn.Module):
    def forward(self, input):
        if self.training:
            return Coat_quantize_bgn_func.apply(input, self.args.group_size, self.fp8type)
        else:
            return input, None, None
```

**功能**:
- 前向：量化输入为 FP8，返回 `(原始输入, 量化输入, scale)`
- 反向：直接返回梯度（straight-through estimator）

### 2. 输出量化 (`Coat_quantize_end`)

```python
class Coat_quantize_end(nn.Module):
    def forward(self, input, Qinput, Iscale):
        if self.training:
            return Coat_quantize_end_func.apply(input, Qinput, Iscale, ...)
        else:
            return input
```

**功能**:
- 前向：直接返回原始输入
- 反向：量化梯度为 FP8

### 3. Block 内的激活量化

在 `CoatOLMoBlock` 中，每个子模块都进行量化：

#### BeforeAttention (`CoatOLMoBeforeAttentionResidual`)

```python
def forward(self, re_x, x, s):
    # 1. 量化权重
    weight1_s = self.prepare_weight(self.att_proj.weight, ...)
    
    # 2. 使用自定义 Function 进行量化前向传播
    return _CoatOLMoBeforeAttentionResidual.apply(
        re_x, x, s, self.att_proj.weight, ..., weight1_s, ...
    )
```

**量化点**:
- 输入激活 (`x`)
- 权重 (`att_proj.weight`)
- 输出激活

#### AfterAttention (`CoatOLMoAfterAttentionResidual`)

```python
def forward(self, re_x, flash_x):
    # 量化 FlashAttention 输出
    flash_qx, flash_s, _ = fp8_quantize_pertensor(flash_x, ...)
    
    # FP8 线性层
    fc2_x = fp8_linear_forward(flash_qx, flash_s, weight2, weight2_s, ...)
    
    # FP8 加法
    fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, fc2_x, ...)
```

#### MLPResidual (`CoatOLMoMLPResidual`)

类似地，MLP 模块也进行量化：
- 输入激活
- 权重 (`ff_proj.weight`, `ff_out.weight`)
- 中间激活 (GELU/SiLU)
- 输出激活

---

## 权重量化

### 1. 权重缓存机制 (`FP8CacheWeightModule`)

所有需要量化权重的模块继承自 `FP8CacheWeightModule`:

```python
class CoatOLMoBeforeAttentionResidual(FP8CacheWeightModule):
    def prepare_weight(self, weight, name, is_first_microbatch):
        if self.qargs.weight_memory_efficient:
            # 内存高效模式：只缓存 scale
            weight_s = self.get_weight_scale(weight, name, is_first_microbatch)
            return weight_s
        else:
            # 标准模式：缓存量化权重和 scale
            weight_q, weight_t, weight_s = self.get_weight_quantized(weight, name, is_first_microbatch)
            return weight_q, weight_t, weight_s
```

**两种模式**:
- **内存高效模式** (`weight_memory_efficient=True`): 只缓存 scale，每次重新量化权重
- **标准模式**: 缓存量化权重，减少重复计算

### 2. 权重量化格式

根据配置使用不同的 FP8 格式：
- `fabit`: 前向激活位宽 (如 `E4M3`)
- `fwbit`: 前向权重点宽 (如 `E4M3`)
- `fobit`: 前向输出位宽 (如 `E4M3`)
- `bwbit`: 反向权重点宽 (如 `E5M2`)
- `babit`: 反向激活位宽 (如 `E5M2`)
- `bobit`: 反向输出位宽 (如 `E5M2`)

---

## 优化器量化

### 1. 优化器类 (`CoatOLMoAdamW`)

**位置**: `examples/OLMo/olmo/optim.py`

```python
class CoatOLMoAdamW(CoatAdamW, Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, qargs=None, ...):
        CoatAdamW.__init__(self, params, lr, betas, eps, weight_decay, amsgrad, qargs=qargs)
    
    step = CoatAdamW.step
```

**关键点**:
- 继承自 `CoatAdamW` (COAT 的 FP8 AdamW 实现)
- 同时继承 `Optimizer` (OLMo 的优化器基类，提供指标收集等功能)

### 2. 优化器构建 (`build_optimizer`)

```python
if cfg.quantize_optimizer.use_quantize_optimizer == OptimizerType.coat_fp8_adamw:
    return CoatOLMoAdamW(
        param_groups,
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
        qargs=cfg.quantize_optimizer,  # 传入量化配置
    )
```

### 3. 优化器量化配置

```yaml
quantize_optimizer:
  use_quantize_optimizer: coat_fp8_adamw
  qgroup_size: 128                    # 量化组大小
  first_order_bit: E4M3              # 一阶动量 (exp_avg) 位宽
  first_order_expansion: expansion     # 一阶动量扩展策略
  second_order_bit: E4M3              # 二阶动量 (exp_avg_sq) 位宽
  second_order_expansion: expansion    # 二阶动量扩展策略
```

---

## 配置文件

### 完整配置示例 (`OLMo-1B-COAT-Both.yaml`)

```yaml
# 模型配置
model:
  d_model: 2048
  n_heads: 16
  n_layers: 16
  # ... 其他模型参数 ...

# 激活和权重量化配置
quantize_model:
  use_quantize_model: coat_real
  fabit: E4M3          # 前向激活
  fwbit: E4M3          # 前向权重
  fobit: E4M3          # 前向输出
  bwbit: E5M2          # 反向权重
  babit: E5M2          # 反向激活
  bobit: E5M2          # 反向输出
  weight_memory_efficient: true  # 权重内存高效模式
  group_size: 16       # 量化组大小

# 优化器量化配置
quantize_optimizer:
  use_quantize_optimizer: coat_fp8_adamw
  qgroup_size: 128
  first_order_bit: E4M3
  first_order_expansion: expansion
  second_order_bit: E4M3
  second_order_expansion: expansion

# 标准优化器配置
optimizer:
  name: adamw
  learning_rate: 4.0e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
```

### 配置选项说明

#### `quantize_model` 选项

- `use_quantize_model`: 
  - `"coat_real"`: 使用 COAT 真实量化
  - `"fp8deepseek"`: 使用 DeepSeek FP8 Linear
  - `"fp8linear"`: 使用 PerTensor FP8 Linear
  - `null` 或其他: 不使用量化

- `fabit/fwbit/fobit/bwbit/babit/bobit`: FP8 格式
  - `E4M3`: 4位指数，3位尾数 (最大 448)
  - `E5M2`: 5位指数，2位尾数 (最大 57344)

- `weight_memory_efficient`: 
  - `true`: 只缓存 scale，节省内存
  - `false`: 缓存量化权重，节省计算

- `group_size`: 量化组大小 (如 16, 32, 64, 128)

#### `quantize_optimizer` 选项

- `use_quantize_optimizer`: 
  - `"coat_fp8_adamw"`: 使用 COAT FP8 AdamW
  - `null` 或其他: 使用标准优化器

- `qgroup_size`: 优化器状态量化组大小

- `first_order_bit/second_order_bit`: 优化器状态位宽

- `first_order_expansion/second_order_expansion`: 扩展策略
  - `"expansion"`: 使用扩展策略
  - 其他: 不使用扩展

---

## 关键代码路径

### 1. 模型相关

```
coat/models/coat_olmo.py          # CoatOLMo 主类
  ├── CoatOLMo                     # 主模型类
  ├── CoatOLMoBlock                # Block 基类
  ├── CoatOLMoSequentialBlock      # Sequential Block 实现
  ├── CoatOLMoBeforeAttentionResidual  # Attention 前残差模块
  ├── CoatOLMoAfterAttentionResidual   # Attention 后残差模块
  └── CoatOLMoMLPResidual          # MLP 残差模块
```

### 2. 激活量化相关

```
coat/activation/real_quantization/
  ├── func_quantize.py             # Coat_quantize_bgn/end
  ├── _quantize.py                 # FP8 量化核心实现
  ├── _quantize_pertensor.py      # Per-tensor 量化
  ├── linear.py                    # FP8 线性层
  ├── add_fwd.py                   # FP8 前向加法
  ├── add_bwd.py                   # FP8 反向加法
  ├── gelu_fwd.py / gelu_bwd.py   # FP8 GELU
  ├── silu_fwd.py / silu_bwd.py   # FP8 SiLU
  └── func_rmsnorm.py             # FP8 RMSNorm
```

### 3. 优化器相关

```
coat/optimizer/
  └── fp8_adamw.py                 # CoatAdamW 实现

examples/OLMo/olmo/optim.py
  └── CoatOLMoAdamW                # OLMo 适配的优化器
```

### 4. 工具类

```
coat/utils/
  ├── _fp8manager.py               # FP8Manager (管理量化状态)
  └── _fp8_weightcache.py          # FP8CacheWeightModule (权重缓存)
```

---

## 量化流程总结

### 前向传播流程

```
输入 (BF16)
  ↓
quantize_input_before_block (量化输入)
  ↓
CoatOLMoBlock:
  ├─ BeforeAttention: 量化输入 + 权重 → QKV
  ├─ Attention (FlashAttention, 保持 BF16)
  ├─ AfterAttention: 量化 FlashAttention 输出 + 权重 → 输出
  └─ MLPResidual: 量化输入 + 权重 → 激活 → 输出
  ↓
quantize_output_after_block (量化输出)
  ↓
输出层 (BF16)
```

### 反向传播流程

```
梯度 (BF16)
  ↓
quantize_output_after_block.backward (量化梯度)
  ↓
CoatOLMoBlock.backward:
  ├─ MLPResidual.backward: FP8 反向传播
  ├─ AfterAttention.backward: FP8 反向传播
  └─ BeforeAttention.backward: FP8 反向传播
  ↓
quantize_input_before_block.backward (straight-through)
  ↓
梯度 (BF16)
```

### 优化器更新流程

```
参数梯度 (BF16)
  ↓
CoatOLMoAdamW.step:
  ├─ 量化一阶动量 (exp_avg) 为 FP8
  ├─ 量化二阶动量 (exp_avg_sq) 为 FP8
  ├─ 计算更新 (FP8 计算)
  └─ 更新参数 (BF16)
```

---

## 关键设计决策

1. **模块化设计**: 每个 Transformer 组件都有对应的量化版本
2. **Autograd Function**: 使用 `torch.autograd.Function` 确保梯度正确传播
3. **内存高效模式**: 支持只缓存 scale，减少内存占用
4. **格式选择**: 前向使用 E4M3 (精度优先)，反向使用 E5M2 (范围优先)
5. **组量化**: 使用 group_size 进行分组量化，平衡精度和效率

---

## 使用示例

### 训练命令

```bash
cd examples/OLMo
python scripts/train.py configs/coat/OLMo-1B-COAT-Both.yaml
```

### 只使用激活量化

使用 `OLMo-1B-COAT-Activation.yaml` 配置文件

### 只使用优化器量化

使用 `OLMo-1B-COAT-Optimizer.yaml` 配置文件

### 同时使用激活和优化器量化

使用 `OLMo-1B-COAT-Both.yaml` 配置文件

---

## 注意事项

1. **FlashAttention**: Attention 计算保持 BF16，只有输入输出进行量化
2. **LayerNorm**: 使用 FP8 版本的 LayerNorm/RMSNorm
3. **激活函数**: GELU/SiLU 使用 FP8 版本
4. **内存管理**: `FP8Manager.is_first_microbatch` 用于管理第一个 micro-batch 的特殊处理
5. **兼容性**: 需要 PyTorch 2.0+ 和 CUDA 11.8+

---

## 参考

- COAT 主代码: `coat/`
- OLMo 适配代码: `examples/OLMo/`
- 配置文件: `examples/OLMo/configs/coat/`

