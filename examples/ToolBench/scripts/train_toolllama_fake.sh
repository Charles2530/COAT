#!/bin/bash

# ================= 配置区域 =================
# 1. 定义关键路径（使用服务器绝对路径）
COAT_ROOT="/mnt/lm_data_afs/wangzining/charles/COAT"
TOOLBENCH_ROOT="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench"

# 2. 核心修正：将 COAT 和 ToolBench 根目录加入 PYTHONPATH
# 这样 train.py 里的 "from coat.models ..." 才能生效
export PYTHONPATH=$PYTHONPATH:$COAT_ROOT:$TOOLBENCH_ROOT

# 3. 环境变量设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_NAME="/mnt/lm_data_afs/wangzining/charles/models/Llama-2-7b-hf"
export SAVE_DIR="toolllama/fake_quant_mxfp4"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-toolllama-fake-mxfp4-$(date +%Y%m%d-%H%M%S)}"

# 确保输出目录存在
mkdir -p $SAVE_DIR

# 4. 切换到 ToolBench 目录，确保 data/ 相对路径有效
cd $TOOLBENCH_ROOT

echo "Starting training with CoatLlamaFake..."
echo "Model Path: $MODEL_NAME"

# ================= 启动命令 =================
torchrun --nproc_per_node=8 --master_port=20001 $TOOLBENCH_ROOT/toolbench/train/train.py \
    --model_name_or_path $MODEL_NAME \
    --data_path data/toolllama_G123_dfs_train.json \
    --eval_data_path data/toolllama_G123_dfs_eval.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --eval_strategy "steps" \
    --eval_steps 5 \
    --prediction_loss_only \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    # optimizer states: disable FP8 quant and expansion
    --first_order_expansion false \
    --second_order_expansion false \
    --first_order_bit BF16 \
    --second_order_bit BF16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config '{"transformer_layer_cls_to_wrap": ["CoatLlamaFakeDecoderLayer"]}' \
    --tf32 True \
    --source_model_max_length 4096 \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --run_name $WANDB_RUN_NAME \
    --report_to wandb \
    --fabit nvfp4_e2m1 \
    --babit nvfp8_e5m2 \
    --attn_quantize False

    # Below are the default value for FP8 training
    # --quantize_model true \
    # --fabit E4M3 \
    # --fwbit E4M3 \
    # --fobit E4M3 \
    # --bwbit E5M2 \
    # --babit E5M2 \
    # --bobit E5M2 \
    # --group_size 16 \
    # --first_order_expansion true \
    # --second_order_expansion true \
    # --first_order_bit E4M3 \
    # --second_order_bit E4M3 \
    # --qgroup_size 128 \
    # --expand_min 16

# Training will automatically run inference and evaluation on math reasoning datasets after training completes
# Results will be logged to the same wandb run
# Optional: Configure paths for math reasoning pipeline test
# export DATA_BASE_DIR="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench/data/math_datasets"
# export PREDICTIONS_BASE_DIR="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench/predictions/math_reasoning"
# export SKIP_EXISTING="true"