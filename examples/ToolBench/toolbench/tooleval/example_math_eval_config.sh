#!/bin/bash

# Example configuration for math reasoning evaluation
# Copy this file and modify the paths to your dataset and prediction files
# Usage: source example_math_eval_config.sh

# Get script directory and ToolBench root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLBENCH_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Base paths - modify these to match your setup
# These are relative to ToolBench directory
DATA_BASE_DIR="${DATA_BASE_DIR:-${TOOLBENCH_DIR}/data/math_datasets}"
PREDICTIONS_BASE_DIR="${PREDICTIONS_BASE_DIR:-${TOOLBENCH_DIR}/predictions/math_reasoning}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-${TOOLBENCH_DIR}/results/math_reasoning}"

# SVAMP dataset
export SVAMP_PREDICTIONS="${PREDICTIONS_BASE_DIR}/svamp_predictions.json"
export SVAMP_DATASET="${DATA_BASE_DIR}/svamp/test.json"

# GSM8K dataset
export GSM8K_PREDICTIONS="${PREDICTIONS_BASE_DIR}/gsm8k_predictions.json"
export GSM8K_DATASET="${DATA_BASE_DIR}/gsm8k/test.jsonl"

# NumGLUE dataset
export NUMGLUE_PREDICTIONS="${PREDICTIONS_BASE_DIR}/numglue_predictions.json"
export NUMGLUE_DATASET="${DATA_BASE_DIR}/numglue/test.json"

# Mathematica dataset
export MATHEMATICA_PREDICTIONS="${PREDICTIONS_BASE_DIR}/mathematica_predictions.json"
export MATHEMATICA_DATASET="${DATA_BASE_DIR}/mathematica/test.json"

# Display configuration
echo "========================================="
echo "Math Reasoning Evaluation Configuration"
echo "========================================="
echo "Data base directory: ${DATA_BASE_DIR}"
echo "Predictions base directory: ${PREDICTIONS_BASE_DIR}"
echo ""
echo "Dataset paths:"
[ ! -z "${SVAMP_DATASET}" ] && echo "  SVAMP: ${SVAMP_DATASET}" || echo "  SVAMP: (not set)"
[ ! -z "${GSM8K_DATASET}" ] && echo "  GSM8K: ${GSM8K_DATASET}" || echo "  GSM8K: (not set)"
[ ! -z "${NUMGLUE_DATASET}" ] && echo "  NumGLUE: ${NUMGLUE_DATASET}" || echo "  NumGLUE: (not set)"
[ ! -z "${MATHEMATICA_DATASET}" ] && echo "  Mathematica: ${MATHEMATICA_DATASET}" || echo "  Mathematica: (not set)"
echo ""
echo "Prediction paths:"
[ ! -z "${SVAMP_PREDICTIONS}" ] && echo "  SVAMP: ${SVAMP_PREDICTIONS}" || echo "  SVAMP: (not set)"
[ ! -z "${GSM8K_PREDICTIONS}" ] && echo "  GSM8K: ${GSM8K_PREDICTIONS}" || echo "  GSM8K: (not set)"
[ ! -z "${NUMGLUE_PREDICTIONS}" ] && echo "  NumGLUE: ${NUMGLUE_PREDICTIONS}" || echo "  NumGLUE: (not set)"
[ ! -z "${MATHEMATICA_PREDICTIONS}" ] && echo "  Mathematica: ${MATHEMATICA_PREDICTIONS}" || echo "  Mathematica: (not set)"
echo ""
echo "To run evaluation, execute:"
echo "  cd ${TOOLBENCH_DIR}"
echo "  bash scripts/eval_math_reasoning.sh"
echo ""
echo "Note: This script only sets environment variables. It does NOT run the evaluation."

