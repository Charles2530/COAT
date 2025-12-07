#!/bin/bash

# Script to download math reasoning datasets: SVAMP, GSM8K, NumGLUE, and MATH (Mathematica)
# This script downloads datasets from various sources including HuggingFace, GitHub, etc.

set -e

# Configuration
DATA_DIR="${DATA_DIR:-data/math_datasets}"
mkdir -p ${DATA_DIR}

echo "========================================="
echo "Downloading Math Reasoning Datasets"
echo "========================================="
echo "Output directory: ${DATA_DIR}"
echo ""

# Function to download from URL
download_file() {
    local url=$1
    local output_file=$2
    local description=$3
    
    echo "Downloading ${description}..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "${output_file}" "${url}"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "${output_file}" "${url}"
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    echo "  ✓ Downloaded to ${output_file}"
}

# Function to download from HuggingFace
download_hf_dataset() {
    local dataset_name=$1
    local output_dir=$2
    local split=$3
    local config_name=${4:-""}  # Optional config name
    
    echo "Downloading ${dataset_name} from HuggingFace..."
    python3 << EOF
import os
from datasets import load_dataset

os.makedirs("${output_dir}", exist_ok=True)
if "${config_name}":
    dataset = load_dataset("${dataset_name}", "${config_name}", split="${split}")
else:
    dataset = load_dataset("${dataset_name}", split="${split}")
output_file = os.path.join("${output_dir}", "${split}.json")
dataset.to_json(output_file)
print(f"  ✓ Downloaded to {output_file}")
EOF
}

# 1. Download SVAMP
echo "--- Downloading SVAMP ---"
SVAMP_DIR="${DATA_DIR}/svamp"
mkdir -p ${SVAMP_DIR}

# Try downloading from GitHub raw file first
SVAMP_URL="https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
download_file "${SVAMP_URL}" "${SVAMP_DIR}/test.json" "SVAMP dataset" || {
    echo "  Trying alternative source..."
    # Alternative: Download from HuggingFace if available
    if python3 -c "from datasets import load_dataset" 2>/dev/null; then
        python3 << EOF
from datasets import load_dataset
import json
import os

os.makedirs("${SVAMP_DIR}", exist_ok=True)
try:
    dataset = load_dataset("svamp", split="test")
    dataset.to_json("${SVAMP_DIR}/test.json")
    print(f"  ✓ Downloaded SVAMP to ${SVAMP_DIR}/test.json")
except:
    print("  ✗ SVAMP download failed. Please download manually from: https://github.com/arkilpatel/SVAMP")
EOF
    fi
}
echo ""

# 2. Download GSM8K
echo "--- Downloading GSM8K ---"
GSM8K_DIR="${DATA_DIR}/gsm8k"
mkdir -p ${GSM8K_DIR}

if python3 -c "from datasets import load_dataset" 2>/dev/null; then
    echo "Downloading GSM8K test split..."
    python3 << EOF
from datasets import load_dataset
import json
import os

os.makedirs("${GSM8K_DIR}", exist_ok=True)
try:
    # Download test split
    test_dataset = load_dataset("gsm8k", "main", split="test")
    test_dataset.to_json("${GSM8K_DIR}/test.jsonl")
    print(f"  ✓ Downloaded GSM8K test to ${GSM8K_DIR}/test.jsonl")
    
    # Also download train for reference
    train_dataset = load_dataset("gsm8k", "main", split="train")
    train_dataset.to_json("${GSM8K_DIR}/train.jsonl")
    print(f"  ✓ Downloaded GSM8K train to ${GSM8K_DIR}/train.jsonl")
except Exception as e:
    print(f"  ✗ GSM8K download failed: {e}")
    print("  Please download manually from: https://huggingface.co/datasets/gsm8k")
EOF
else
    echo "  Installing datasets library..."
    pip install datasets --quiet || echo "  Failed to install datasets. Please install manually: pip install datasets"
    # Try downloading with explicit config
    python3 << EOF
from datasets import load_dataset
import os
os.makedirs("${GSM8K_DIR}", exist_ok=True)
try:
    test_dataset = load_dataset("gsm8k", "main", split="test")
    test_dataset.to_json("${GSM8K_DIR}/test.jsonl")
    print(f"  ✓ Downloaded GSM8K test to ${GSM8K_DIR}/test.jsonl")
except Exception as e:
    print(f"  ✗ Failed: {e}")
EOF
fi
echo ""

# 3. Download NumGLUE
echo "--- Downloading NumGLUE ---"
NUMGLUE_DIR="${DATA_DIR}/numglue"
mkdir -p ${NUMGLUE_DIR}

# Use raw GitHub URL for direct file download
NUMGLUE_URL="https://raw.githubusercontent.com/allenai/numglue/main/data/NumGLUE_test.json"
download_file "${NUMGLUE_URL}" "${NUMGLUE_DIR}/test.json" "NumGLUE dataset" || {
    echo "  ✗ NumGLUE download failed. Please download manually from:"
    echo "    https://github.com/allenai/numglue"
}
echo ""

# 4. Download MATH (Mathematica/Math Competition Problems)
echo "--- Downloading MATH ---"
MATH_DIR="${DATA_DIR}/mathematica"
mkdir -p ${MATH_DIR}

if python3 -c "from datasets import load_dataset" 2>/dev/null; then
    python3 << EOF
from datasets import load_dataset
import json
import os

os.makedirs("${MATH_DIR}", exist_ok=True)
try:
    # Download MATH dataset from qwedsacf/competition_math
    dataset = load_dataset("qwedsacf/competition_math", split="train")
    dataset.to_json("${MATH_DIR}/test.json")
    print(f"  ✓ Downloaded MATH to ${MATH_DIR}/test.json")
except Exception as e:
    print(f"  ✗ MATH download failed: {e}")
    print("  Please download manually from:")
    print("    - https://huggingface.co/datasets/qwedsacf/competition_math")
    print("    - https://github.com/hendrycks/math")
EOF
else
    echo "  Please install datasets library: pip install datasets"
    echo "  Then download MATH from: https://huggingface.co/datasets/qwedsacf/competition_math"
fi
echo ""

echo "========================================="
echo "Download Summary"
echo "========================================="
echo "Data saved to: ${DATA_DIR}"
echo ""
echo "Downloaded datasets:"
[ -f "${DATA_DIR}/svamp/test.json" ] && echo "  ✓ SVAMP: ${DATA_DIR}/svamp/test.json"
[ -f "${DATA_DIR}/gsm8k/test.jsonl" ] && echo "  ✓ GSM8K: ${DATA_DIR}/gsm8k/test.jsonl"
[ -f "${DATA_DIR}/numglue/test.json" ] && echo "  ✓ NumGLUE: ${DATA_DIR}/numglue/test.json"
[ -f "${DATA_DIR}/mathematica/test.json" ] && echo "  ✓ MATH: ${DATA_DIR}/mathematica/test.json"
echo ""
echo "To install required dependencies:"
echo "  pip install datasets"
echo ""

