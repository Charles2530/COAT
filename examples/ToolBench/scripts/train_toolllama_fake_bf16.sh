export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_NAME="/mnt/lm_data_afs/wangzining/charles/models/Llama-2-7b-hf"
# 注意改路径
export SAVE_DIR="toolllama"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-toolllama-fake-bf16-$(date +%Y%m%d-%H%M%S)}"

# BF16 fake-quantized training
torchrun --nproc_per_node=8 --master_port=20001 toolbench/train/train.py \
    --model_name_or_path $MODEL_NAME \
    --use_mxfp4_fake True \
    --fabit bf16 \
    --babit bf16 \
    --backward_quantize True \
    --minus_exp None \
    --auto_reverse False \
    --data_path MathInstruct/MathInstruct_toolbench_format.json \
    --eval_data_path MathInstruct/MathInstruct_toolbench_format.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 6 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayerFake' \
    --gradient_checkpointing True \
    --tf32 True \
    --source_model_max_length 4096 \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --optim adamw_torch \
    --run_name $WANDB_RUN_NAME \
    --report_to wandb

# Training will automatically run inference and evaluation on math reasoning datasets after training completes
# Results will be logged to the same wandb run
# Optional: Configure paths for math reasoning pipeline test
# export DATA_BASE_DIR="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench/data/math_datasets"
# export PREDICTIONS_BASE_DIR="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench/predictions/math_reasoning"
# export SKIP_EXISTING="true"
