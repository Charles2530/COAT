export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_NAME="/mnt/lm_data_afs/wangzining/charles/models/Llama-2-7b-hf"
export SAVE_DIR="toolllama/bf16"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-toolllama-bf16-$(date +%Y%m%d-%H%M%S)}"

torchrun --nproc_per_node=8 --master_port=20001 toolbench/train/train.py \
    --model_name_or_path $MODEL_NAME  \
    --data_path  data/toolllama_G123_dfs_train.json \
    --eval_data_path  data/toolllama_G123_dfs_eval.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --source_model_max_length 4096 \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --run_name $WANDB_RUN_NAME \
    --report_to wandb
# # Run inference on the math reasoning datasets
# bash scripts/inference_all_math_datasets.sh \
#     --model_path toolllama/bf16/ \
#     --skip_existing

# # Evaluate the model on the math reasoning datasets
# export SVAMP_PREDICTIONS="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench/predictions/math_reasoning/svamp_predictions.json"
# export GSM8K_PREDICTIONS="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench/predictions/math_reasoning/gsm8k_predictions.json"
# export NUMGLUE_PREDICTIONS="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench/predictions/math_reasoning/numglue_predictions.json"
# export MATHEMATICA_PREDICTIONS="/mnt/lm_data_afs/wangzining/charles/COAT/examples/ToolBench/predictions/math_reasoning/mathematica_predictions.json"
# bash scripts/eval_math_reasoning.sh
