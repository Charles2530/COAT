#!/bin/bash

# Inference script for math reasoning datasets
# Uses trained model to generate predictions on SVAMP, GSM8K, NumGLUE, and Mathematica

# Configuration
export PYTHONPATH=./

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLBENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODEL_PATH="${MODEL_PATH:-toolllama}"
DEFAULT_DATA_DIR="${DATA_BASE_DIR:-${TOOLBENCH_DIR}/data/math_datasets}"
DEFAULT_OUTPUT_DIR="${PREDICTIONS_BASE_DIR:-${TOOLBENCH_DIR}/predictions/math_reasoning}"

# Create output directory
mkdir -p ${DEFAULT_OUTPUT_DIR}

# Parse arguments
DATASET=""
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
USE_LORA=false
LORA_PATH=""
DEVICE="cuda"
MAX_NEW_TOKENS=512
MAX_SEQ_LENGTH=2048
TEMPLATE="tool-llama-single-round"

# Function to show usage
usage() {
    echo "Usage: $0 --dataset DATASET [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --dataset DATASET          Dataset name: svamp, gsm8k, numglue, or mathematica"
    echo ""
    echo "Optional arguments:"
    echo "  --model_path PATH          Path to model directory (default: ${DEFAULT_MODEL_PATH})"
    echo "  --lora                     Use LoRA model"
    echo "  --lora_path PATH           Path to LoRA adapter (required if --lora is set)"
    echo "  --device DEVICE            Device: cuda, cpu, or mps (default: cuda)"
    echo "  --max_new_tokens N         Maximum new tokens to generate (default: 512)"
    echo "  --max_seq_length N         Maximum sequence length (default: 2048)"
    echo "  --template TEMPLATE        Conversation template (default: tool-llama-single-round)"
    echo "  --data_dir DIR             Dataset base directory (default: ${DEFAULT_DATA_DIR})"
    echo "  --output_dir DIR           Output directory for predictions (default: ${DEFAULT_OUTPUT_DIR})"
    echo ""
    echo "Examples:"
    echo "  # Run inference on SVAMP"
    echo "  $0 --dataset svamp --model_path toolllama/"
    echo ""
    echo "  # Run inference with LoRA"
    echo "  $0 --dataset gsm8k --model_path base_model/ --lora --lora_path lora_model/"
    echo ""
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
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
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    usage
fi

# Validate dataset name
if [[ ! "$DATASET" =~ ^(svamp|gsm8k|numglue|mathematica)$ ]]; then
    echo "Error: Invalid dataset name: $DATASET"
    echo "Valid datasets: svamp, gsm8k, numglue, mathematica"
    exit 1
fi

# Determine dataset path
DATA_BASE_DIR="${DATA_BASE_DIR:-${DEFAULT_DATA_DIR}}"
case $DATASET in
    svamp)
        DATASET_PATH="${DATA_BASE_DIR}/svamp/test.json"
        ;;
    gsm8k)
        DATASET_PATH="${DATA_BASE_DIR}/gsm8k/test.jsonl"
        ;;
    numglue)
        DATASET_PATH="${DATA_BASE_DIR}/numglue/test.json"
        ;;
    mathematica)
        DATASET_PATH="${DATA_BASE_DIR}/mathematica/test.json"
        ;;
esac

# Check if dataset file exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    echo "Please download the dataset first or specify correct path with --data_dir"
    exit 1
fi

# Determine output path
PREDICTIONS_BASE_DIR="${PREDICTIONS_BASE_DIR:-${DEFAULT_OUTPUT_DIR}}"
OUTPUT_PATH="${PREDICTIONS_BASE_DIR}/${DATASET}_predictions.json"

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

# Print configuration
echo "========================================="
echo "Math Reasoning Inference"
echo "========================================="
echo "Dataset: ${DATASET}"
echo "Dataset path: ${DATASET_PATH}"
echo "Model path: ${MODEL_PATH}"
[ "$USE_LORA" = true ] && echo "LoRA path: ${LORA_PATH}"
echo "Device: ${DEVICE}"
echo "Output path: ${OUTPUT_PATH}"
echo ""

# Build command
cd ${TOOLBENCH_DIR}

CMD="python toolbench/inference/inference_math_reasoning.py \
    --dataset ${DATASET} \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH} \
    --device ${DEVICE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --max_sequence_length ${MAX_SEQ_LENGTH} \
    --template ${TEMPLATE}"

if [ "$USE_LORA" = true ]; then
    CMD="${CMD} --lora --lora_path ${LORA_PATH}"
fi

# Run inference
echo "Running inference..."
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Inference completed successfully!"
    echo "========================================="
    echo "Predictions saved to: ${OUTPUT_PATH}"
    echo ""
    echo "To evaluate the predictions, run:"
    echo "  export ${DATASET^^}_PREDICTIONS=\"${OUTPUT_PATH}\""
    echo "  bash scripts/eval_math_reasoning.sh"
else
    echo ""
    echo "Error: Inference failed"
    exit 1
fi

