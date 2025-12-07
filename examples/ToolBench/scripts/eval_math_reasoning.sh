#!/bin/bash

# Comprehensive evaluation script for all math reasoning datasets
# This script evaluates on SVAMP, GSM8K, NumGLUE, and Mathematica

# Configuration
PYTHONPATH=.
OUTPUT_BASE_DIR="math_reasoning_eval_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TIMESTAMP}"

# Default dataset base directory (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_DIR="${SCRIPT_DIR}/../data/math_datasets"
DEFAULT_PREDICTIONS_DIR="${SCRIPT_DIR}/../predictions/math_reasoning"

# Use environment variable if set, otherwise use default
DATA_BASE_DIR="${DATA_BASE_DIR:-${DEFAULT_DATA_DIR}}"
PREDICTIONS_BASE_DIR="${PREDICTIONS_BASE_DIR:-${DEFAULT_PREDICTIONS_DIR}}"

# Auto-detect dataset paths if not explicitly set
if [ -z "${SVAMP_DATASET}" ] && [ -f "${DATA_BASE_DIR}/svamp/test.json" ]; then
    export SVAMP_DATASET="${DATA_BASE_DIR}/svamp/test.json"
fi

if [ -z "${GSM8K_DATASET}" ] && [ -f "${DATA_BASE_DIR}/gsm8k/test.jsonl" ]; then
    export GSM8K_DATASET="${DATA_BASE_DIR}/gsm8k/test.jsonl"
fi

if [ -z "${NUMGLUE_DATASET}" ] && [ -f "${DATA_BASE_DIR}/numglue/test.json" ]; then
    export NUMGLUE_DATASET="${DATA_BASE_DIR}/numglue/test.json"
fi

if [ -z "${MATHEMATICA_DATASET}" ] && [ -f "${DATA_BASE_DIR}/mathematica/test.json" ]; then
    export MATHEMATICA_DATASET="${DATA_BASE_DIR}/mathematica/test.json"
fi

# Auto-detect prediction paths if not explicitly set (optional, only if file exists)
if [ -z "${SVAMP_PREDICTIONS}" ] && [ -f "${PREDICTIONS_BASE_DIR}/svamp_predictions.json" ]; then
    export SVAMP_PREDICTIONS="${PREDICTIONS_BASE_DIR}/svamp_predictions.json"
fi

if [ -z "${GSM8K_PREDICTIONS}" ] && [ -f "${PREDICTIONS_BASE_DIR}/gsm8k_predictions.json" ]; then
    export GSM8K_PREDICTIONS="${PREDICTIONS_BASE_DIR}/gsm8k_predictions.json"
fi

if [ -z "${NUMGLUE_PREDICTIONS}" ] && [ -f "${PREDICTIONS_BASE_DIR}/numglue_predictions.json" ]; then
    export NUMGLUE_PREDICTIONS="${PREDICTIONS_BASE_DIR}/numglue_predictions.json"
fi

if [ -z "${MATHEMATICA_PREDICTIONS}" ] && [ -f "${PREDICTIONS_BASE_DIR}/mathematica_predictions.json" ]; then
    export MATHEMATICA_PREDICTIONS="${PREDICTIONS_BASE_DIR}/mathematica_predictions.json"
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "========================================="
echo "Math Reasoning Evaluation"
echo "========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Data directory: ${DATA_BASE_DIR}"
echo "Predictions directory: ${PREDICTIONS_BASE_DIR}"
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
elif [ ! -z "${SVAMP_DATASET}" ]; then
    echo "Warning: SVAMP dataset found but predictions file not set:"
    echo "  Dataset: ${SVAMP_DATASET}"
    echo "  Set SVAMP_PREDICTIONS environment variable to evaluate"
    echo ""
fi

# Evaluate GSM8K
if [ ! -z "${GSM8K_PREDICTIONS}" ] && [ ! -z "${GSM8K_DATASET}" ]; then
    evaluate_dataset "gsm8k" "${GSM8K_PREDICTIONS}" "${GSM8K_DATASET}"
elif [ ! -z "${GSM8K_DATASET}" ]; then
    echo "Warning: GSM8K dataset found but predictions file not set:"
    echo "  Dataset: ${GSM8K_DATASET}"
    echo "  Set GSM8K_PREDICTIONS environment variable to evaluate"
    echo ""
fi

# Evaluate NumGLUE
if [ ! -z "${NUMGLUE_PREDICTIONS}" ] && [ ! -z "${NUMGLUE_DATASET}" ]; then
    evaluate_dataset "numglue" "${NUMGLUE_PREDICTIONS}" "${NUMGLUE_DATASET}"
elif [ ! -z "${NUMGLUE_DATASET}" ]; then
    echo "Warning: NumGLUE dataset found but predictions file not set:"
    echo "  Dataset: ${NUMGLUE_DATASET}"
    echo "  Set NUMGLUE_PREDICTIONS environment variable to evaluate"
    echo ""
fi

# Evaluate Mathematica
if [ ! -z "${MATHEMATICA_PREDICTIONS}" ] && [ ! -z "${MATHEMATICA_DATASET}" ]; then
    evaluate_dataset "mathematica" "${MATHEMATICA_PREDICTIONS}" "${MATHEMATICA_DATASET}"
elif [ ! -z "${MATHEMATICA_DATASET}" ]; then
    echo "Warning: Mathematica dataset found but predictions file not set:"
    echo "  Dataset: ${MATHEMATICA_DATASET}"
    echo "  Set MATHEMATICA_PREDICTIONS environment variable to evaluate"
    echo ""
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

