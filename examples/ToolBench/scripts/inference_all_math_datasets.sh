#!/bin/bash

# Batch inference script for all math reasoning datasets
# Runs inference on SVAMP, GSM8K, NumGLUE, and Mathematica

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLBENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default paths
MODEL_PATH="${MODEL_PATH:-toolllama}"
DATA_BASE_DIR="${DATA_BASE_DIR:-${TOOLBENCH_DIR}/data/math_datasets}"
PREDICTIONS_BASE_DIR="${PREDICTIONS_BASE_DIR:-${TOOLBENCH_DIR}/predictions/math_reasoning}"

# Parse arguments
USE_LORA=false
LORA_PATH=""
DEVICE="cuda"
MAX_NEW_TOKENS=512
MAX_SEQ_LENGTH=2048
TEMPLATE="tool-llama-single-round"
SKIP_EXISTING=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Optional arguments:"
    echo "  --model_path PATH          Path to model directory (default: ${MODEL_PATH})"
    echo "  --lora                     Use LoRA model"
    echo "  --lora_path PATH           Path to LoRA adapter (required if --lora is set)"
    echo "  --device DEVICE            Device: cuda, cpu, or mps (default: cuda)"
    echo "  --max_new_tokens N         Maximum new tokens to generate (default: 512)"
    echo "  --max_seq_length N         Maximum sequence length (default: 2048)"
    echo "  --template TEMPLATE        Conversation template (default: tool-llama-single-round)"
    echo "  --data_dir DIR             Dataset base directory (default: ${DATA_BASE_DIR})"
    echo "  --output_dir DIR           Output directory for predictions (default: ${PREDICTIONS_BASE_DIR})"
    echo "  --skip_existing            Skip datasets that already have predictions"
    echo ""
    echo "Examples:"
    echo "  # Run inference on all datasets"
    echo "  $0 --model_path toolllama/"
    echo ""
    echo "  # Run inference with LoRA, skip existing predictions"
    echo "  $0 --model_path base_model/ --lora --lora_path lora_model/ --skip_existing"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora)
            USE_LORA=true
            shift
            ;;
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_BASE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            PREDICTIONS_BASE_DIR="$2"
            shift 2
            ;;
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if model exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Please specify correct model path with --model_path"
    exit 1
fi

# Check LoRA path if LoRA is enabled
if [ "$USE_LORA" = true ]; then
    if [ -z "$LORA_PATH" ]; then
        echo "Error: --lora_path must be specified when --lora is set"
        exit 1
    fi
    if [ ! -d "$LORA_PATH" ]; then
        echo "Error: LoRA path not found: $LORA_PATH"
        exit 1
    fi
fi

# Create output directory
mkdir -p ${PREDICTIONS_BASE_DIR}

# Print configuration
echo "========================================="
echo "Batch Math Reasoning Inference"
echo "========================================="
echo "Model path: ${MODEL_PATH}"
[ "$USE_LORA" = true ] && echo "LoRA path: ${LORA_PATH}"
echo "Device: ${DEVICE}"
echo "Data directory: ${DATA_BASE_DIR}"
echo "Output directory: ${PREDICTIONS_BASE_DIR}"
echo ""

# Define datasets
declare -A DATASETS
DATASETS[svamp]="${DATA_BASE_DIR}/svamp/test.json"
DATASETS[gsm8k]="${DATA_BASE_DIR}/gsm8k/test.jsonl"
DATASETS[numglue]="${DATA_BASE_DIR}/numglue/test.json"
DATASETS[mathematica]="${DATA_BASE_DIR}/mathematica/test.json"

# Build base command
BASE_CMD="bash ${SCRIPT_DIR}/inference_math_reasoning.sh \
    --model_path ${MODEL_PATH} \
    --device ${DEVICE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --template ${TEMPLATE} \
    --data_dir ${DATA_BASE_DIR} \
    --output_dir ${PREDICTIONS_BASE_DIR}"

if [ "$USE_LORA" = true ]; then
    BASE_CMD="${BASE_CMD} --lora --lora_path ${LORA_PATH}"
fi

# Run inference for each dataset
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for dataset in "${!DATASETS[@]}"; do
    dataset_path="${DATASETS[$dataset]}"
    output_path="${PREDICTIONS_BASE_DIR}/${dataset}_predictions.json"
    
    echo "----------------------------------------"
    echo "Processing: ${dataset}"
    echo "----------------------------------------"
    
    # Check if dataset file exists
    if [ ! -f "$dataset_path" ]; then
        echo "Warning: Dataset file not found: $dataset_path"
        echo "Skipping ${dataset}..."
        ((SKIP_COUNT++))
        continue
    fi
    
    # Check if prediction already exists
    if [ "$SKIP_EXISTING" = true ] && [ -f "$output_path" ]; then
        echo "Prediction file already exists: $output_path"
        echo "Skipping ${dataset}..."
        ((SKIP_COUNT++))
        continue
    fi
    
    # Run inference
    CMD="${BASE_CMD} --dataset ${dataset}"
    echo "Running: $CMD"
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed ${dataset}"
        ((SUCCESS_COUNT++))
    else
        echo "✗ Failed to process ${dataset}"
        ((FAIL_COUNT++))
    fi
    
    echo ""
done

# Print summary
echo "========================================="
echo "Summary"
echo "========================================="
echo "Successfully processed: ${SUCCESS_COUNT} datasets"
echo "Failed: ${FAIL_COUNT} datasets"
echo "Skipped: ${SKIP_COUNT} datasets"
echo ""
echo "Predictions saved to: ${PREDICTIONS_BASE_DIR}"
echo ""
echo "To evaluate all predictions, run:"
echo "  export SVAMP_PREDICTIONS=\"${PREDICTIONS_BASE_DIR}/svamp_predictions.json\""
echo "  export GSM8K_PREDICTIONS=\"${PREDICTIONS_BASE_DIR}/gsm8k_predictions.json\""
echo "  export NUMGLUE_PREDICTIONS=\"${PREDICTIONS_BASE_DIR}/numglue_predictions.json\""
echo "  export MATHEMATICA_PREDICTIONS=\"${PREDICTIONS_BASE_DIR}/mathematica_predictions.json\""
echo "  bash scripts/eval_math_reasoning.sh"

