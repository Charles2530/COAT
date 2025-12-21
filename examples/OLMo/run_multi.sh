#!/bin/bash
# 在两个节点都运行，修改node_rank

# 网络配置
export NCCL_SOCKET_IFNAME=ib0  # 或你的高速网络接口
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN  # 或INFO用于调试

# 性能优化
export OMP_NUM_THREADS=8
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# FSDP优化（如果使用FULL_SHARD，确保limit_all_gathers=True已经在代码中）
# 这已经在train.py:228中设置了

torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node_rank=$1 \
  --master_addr=10.119.21.251 \
  --master_port=29500 \
  --rdzv_backend=c10d \
  scripts/train.py examples/OLMo/configs/fake_quant/OLMo-7B-COAT-Activation-Mxfp-4-Minus-auto.yaml \
    --fsdp.sharding_strategy=HYBRID_SHARD \
    --fsdp.hybrid_sharding_num_model_replicas=2