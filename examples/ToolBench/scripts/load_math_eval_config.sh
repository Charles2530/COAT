#!/bin/bash

# Convenience script to load default configuration for math reasoning evaluation
# Usage: source scripts/load_math_eval_config.sh
# Or: . scripts/load_math_eval_config.sh

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLBENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default paths (relative to ToolBench directory)
DATA_BASE_DIR="${DATA_BASE_DIR:-${TOOLBENCH_DIR}/data/math_datasets}"
PREDICTIONS_BASE_DIR="${PREDICTIONS_BASE_DIR:-${TOOLBENCH_DIR}/predictions/math_reasoning}"

# Export dataset paths (only if files exist)
if [ -f "${DATA_BASE_DIR}/svamp/test.json" ]; then
    export SVAMP_DATASET="${DATA_BASE_DIR}/svamp/test.json"
    echo "✓ SVAMP dataset: ${SVAMP_DATASET}"
fi

if [ -f "${DATA_BASE_DIR}/gsm8k/test.jsonl" ]; then
    export GSM8K_DATASET="${DATA_BASE_DIR}/gsm8k/test.jsonl"
    echo "✓ GSM8K dataset: ${GSM8K_DATASET}"
fi

if [ -f "${DATA_BASE_DIR}/numglue/test.json" ]; then
    export NUMGLUE_DATASET="${DATA_BASE_DIR}/numglue/test.json"
    echo "✓ NumGLUE dataset: ${NUMGLUE_DATASET}"
fi

if [ -f "${DATA_BASE_DIR}/mathematica/test.json" ]; then
    export MATHEMATICA_DATASET="${DATA_BASE_DIR}/mathematica/test.json"
    echo "✓ Mathematica dataset: ${MATHEMATICA_DATASET}"
fi

# Export prediction paths (if set, or use defaults if files exist)
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

echo ""
echo "Configuration loaded. You can now run:"
echo "  bash scripts/eval_math_reasoning.sh"
echo ""
echo "To override paths, set environment variables:"
echo "  export SVAMP_PREDICTIONS=\"path/to/your/predictions.json\""
echo "  export GSM8K_PREDICTIONS=\"path/to/your/predictions.json\""
echo "  # ... etc"

