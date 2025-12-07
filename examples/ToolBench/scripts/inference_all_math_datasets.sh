#!/bin/bash

# Batch inference script for all math reasoning datasets
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLBENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-toolllama}"
DATA_BASE_DIR="${DATA_BASE_DIR:-${TOOLBENCH_DIR}/data/math_datasets}"
PREDICTIONS_BASE_DIR="${PREDICTIONS_BASE_DIR:-${TOOLBENCH_DIR}/predictions/math_reasoning}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --data_dir) DATA_BASE_DIR="$2"; shift 2 ;;
        --output_dir) PREDICTIONS_BASE_DIR="$2"; shift 2 ;;
        --skip_existing) SKIP_EXISTING=true; shift ;;
        *) shift ;;
    esac
done

mkdir -p ${PREDICTIONS_BASE_DIR}

for dataset in svamp gsm8k numglue mathematica; do
    case $dataset in
        svamp) dataset_path="${DATA_BASE_DIR}/svamp/test.json" ;;
        gsm8k) dataset_path="${DATA_BASE_DIR}/gsm8k/test.jsonl" ;;
        numglue) dataset_path="${DATA_BASE_DIR}/numglue/test.json" ;;
        mathematica) dataset_path="${DATA_BASE_DIR}/mathematica/test.json" ;;
    esac
    
    output_path="${PREDICTIONS_BASE_DIR}/${dataset}_predictions.json"
    
    [ "$SKIP_EXISTING" = true ] && [ -f "$output_path" ] && continue
    [ ! -f "$dataset_path" ] && continue
    
    bash ${SCRIPT_DIR}/inference_math_reasoning.sh \
        --dataset ${dataset} \
        --model_path ${MODEL_PATH} \
        --data_dir ${DATA_BASE_DIR} \
        --output_dir ${PREDICTIONS_BASE_DIR}
done

