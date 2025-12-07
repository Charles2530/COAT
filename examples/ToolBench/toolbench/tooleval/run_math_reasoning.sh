#!/bin/bash

# Evaluation script for math reasoning datasets
# Usage: bash run_math_reasoning.sh <dataset_name> <predictions_path> <dataset_path> <output_path>

DATASET=${1:-"svamp"}  # svamp, gsm8k, numglue, mathematica
PREDICTIONS_PATH=${2:-""}
DATASET_PATH=${3:-""}
OUTPUT_PATH=${4:-"math_reasoning_results"}

if [ -z "$PREDICTIONS_PATH" ] || [ -z "$DATASET_PATH" ]; then
    echo "Usage: bash run_math_reasoning.sh <dataset_name> <predictions_path> <dataset_path> [output_path]"
    echo "Example: bash run_math_reasoning.sh svamp predictions/svamp_results.json data/svamp_test.json results/"
    exit 1
fi

export PYTHONPATH=../..

python eval_math_reasoning.py \
    --predictions_path ${PREDICTIONS_PATH} \
    --dataset ${DATASET} \
    --dataset_path ${DATASET_PATH} \
    --output_path ${OUTPUT_PATH} \
    --extract_answer_from_response

echo "Evaluation completed. Results saved to ${OUTPUT_PATH}"

