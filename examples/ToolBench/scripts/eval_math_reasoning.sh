#!/bin/bash

# Comprehensive evaluation script for all math reasoning datasets
# This script evaluates on SVAMP, GSM8K, NumGLUE, and Mathematica

# Configuration
PYTHONPATH=.
OUTPUT_BASE_DIR="math_reasoning_eval_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TIMESTAMP}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "========================================="
echo "Math Reasoning Evaluation"
echo "========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Function to evaluate a dataset
evaluate_dataset() {
    local dataset=$1
    local predictions_path=$2
    local dataset_path=$3
    
    if [ ! -f "$predictions_path" ]; then
        echo "Warning: Predictions file not found: $predictions_path"
        return
    fi
    
    if [ ! -f "$dataset_path" ]; then
        echo "Warning: Dataset file not found: $dataset_path"
        return
    fi
    
    echo "Evaluating $dataset..."
    echo "  Predictions: $predictions_path"
    echo "  Dataset: $dataset_path"
    
    python toolbench/tooleval/eval_math_reasoning.py \
        --predictions_path ${predictions_path} \
        --dataset ${dataset} \
        --dataset_path ${dataset_path} \
        --output_path ${OUTPUT_DIR}/${dataset} \
        --extract_answer_from_response
    
    echo ""
}

# Evaluate SVAMP
if [ ! -z "${SVAMP_PREDICTIONS}" ] && [ ! -z "${SVAMP_DATASET}" ]; then
    evaluate_dataset "svamp" "${SVAMP_PREDICTIONS}" "${SVAMP_DATASET}"
fi

# Evaluate GSM8K
if [ ! -z "${GSM8K_PREDICTIONS}" ] && [ ! -z "${GSM8K_DATASET}" ]; then
    evaluate_dataset "gsm8k" "${GSM8K_PREDICTIONS}" "${GSM8K_DATASET}"
fi

# Evaluate NumGLUE
if [ ! -z "${NUMGLUE_PREDICTIONS}" ] && [ ! -z "${NUMGLUE_DATASET}" ]; then
    evaluate_dataset "numglue" "${NUMGLUE_PREDICTIONS}" "${NUMGLUE_DATASET}"
fi

# Evaluate Mathematica
if [ ! -z "${MATHEMATICA_PREDICTIONS}" ] && [ ! -z "${MATHEMATICA_DATASET}" ]; then
    evaluate_dataset "mathematica" "${MATHEMATICA_PREDICTIONS}" "${MATHEMATICA_DATASET}"
fi

# Generate summary
echo "========================================="
echo "Summary"
echo "========================================="

for dataset in svamp gsm8k numglue mathematica; do
    summary_file="${OUTPUT_DIR}/${dataset}/${dataset}_summary.txt"
    if [ -f "$summary_file" ]; then
        echo ""
        echo "--- $dataset ---"
        cat "$summary_file"
    fi
done

echo ""
echo "All results saved to: ${OUTPUT_DIR}"

