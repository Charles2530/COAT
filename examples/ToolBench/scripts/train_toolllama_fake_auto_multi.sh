torchrun --nnodes=4 \
    --nproc-per-node=8 \
    --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    toolbench/train/train.py \
    --model_name_or_path $MODEL_NAME \
    --use_mxfp4_fake True \
    --fabit mxfp4_e2m1 \
    --babit mxfp4_e2m1 \
    --backward_quantize True \
    --minus_exp "auto" \
    --auto_reverse False \
    --data_path MathInstruct/MathInstruct_toolbench_format.json \
    --eval_data_path MathInstruct/MathInstruct_toolbench_format.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir $SAVE_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --prediction_loss_only \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
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
# sleep infinity
# torchrun --nproc_per_node=8 scripts/train.py configs/coat/OLMo-7B-COAT-Activation.yaml

