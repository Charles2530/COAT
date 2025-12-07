#!/bin/bash

# Example configuration for math reasoning evaluation
# Copy this file and modify the paths to your dataset and prediction files

# Base paths - modify these to match your setup
DATA_BASE_DIR="data/math_datasets"
PREDICTIONS_BASE_DIR="predictions/math_reasoning"
RESULTS_BASE_DIR="results/math_reasoning"

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

# Run evaluation
echo "Starting math reasoning evaluation..."
echo "SVAMP: ${SVAMP_DATASET}"
echo "GSM8K: ${GSM8K_DATASET}"
echo "NumGLUE: ${NUMGLUE_DATASET}"
echo "Mathematica: ${MATHEMATICA_DATASET}"
echo ""

cd ../..
bash scripts/eval_math_reasoning.sh

