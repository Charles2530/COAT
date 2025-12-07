# COAT Fake Quantization with fake_quant_ops

## 概述

`coat_fake` 是 COAT 项目新增的模拟量化选项，使用 `fake_quant_ops` 模块实现。与 `coat_real`（真实量化）不同，`coat_fake` 提供模拟量化功能，主要用于研究和评估量化效果，不会加速训练。

## 使用方法

### 1. 配置文件设置

在训练配置文件中，设置 `use_quantize_model: coat_fake`：

```yaml
quantize_model:
  use_quantize_model: coat_fake
  # 量化格式设置
  fabit: E4M3  # 前向激活格式 (E4M3 或 E5M2)
  fwbit: E4M3  # 前向权重格式
  fobit: E4M3  # 前向输出格式
  
  # 可选的 Attention QKV 模拟量化
  attn_quantize: true
  attn_quantize_bit: mxfp8_e4m3  # 或 mxfp8_e5m2, bf16 等
```

### 2. 支持的量化格式

- **E4M3**: 转换为 `fp8_e4m3` (4位指数，3位尾数)
- **E5M2**: 转换为 `fp8_e5m2` (5位指数，2位尾数)
- **Attention QKV 量化**: `mxfp8_e4m3`, `mxfp8_e5m2`, `bf16` 等

### 3. 实现细节

`CoatOLMoFake` 类继承自标准 `OLMo` 模型，并在以下位置应用模拟量化：

1. **Linear 层包装**: 使用 `FakeQuantizedLinear` 包装所有线性层
   - 输入量化
   - 权重量化
   - 输出量化

2. **Attention QKV 量化**: 在 Attention 计算前对 Q、K、V 张量进行模拟量化（如果启用 `attn_quantize`）

3. **Straight-Through Estimator (STE)**: 使用 STE 保持梯度流

### 4. 与 coat_real 的区别

| 特性 | coat_fake | coat_real |
|------|-----------|-----------|
| 量化类型 | 模拟量化 | 真实量化 |
| 训练加速 | ❌ 不加速 | ✅ 加速 |
| 数据类型 | 保持 BF16/FP32 | 转换为 FP8 |
| 用途 | 研究、评估 | 生产训练 |
| 实现 | fake_quant_ops | Triton kernels |

### 5. 示例配置文件

参考 `examples/OLMo/configs/coat/OLMo-1B-COAT-Fake.yaml` 获取完整配置示例。

### 6. 运行训练

```bash
cd examples/OLMo
torchrun --nproc_per_node=8 scripts/train.py configs/coat/OLMo-1B-COAT-Fake.yaml
```

## 注意事项

1. **性能**: `coat_fake` 不会加速训练，仅用于研究量化效果
2. **内存**: 模拟量化会增加一些计算开销，但不会显著改变内存使用
3. **兼容性**: 需要确保 `fake_quant_ops` 模块在 Python 路径中可用
4. **优化器量化**: 当前版本不支持优化器状态的模拟量化

## 技术实现

- **模块位置**: `coat/models/coat_olmo_fake.py`
- **核心类**: 
  - `CoatOLMoFake`: 主模型类
  - `CoatOLMoFakeSequentialBlock`: 带模拟量化的 Transformer Block
  - `FakeQuantizedLinear`: 线性层包装器

## 未来改进

- [ ] 支持更多量化格式（FP4, FP6 等）
- [ ] 优化器状态模拟量化
- [ ] 更细粒度的量化控制选项

