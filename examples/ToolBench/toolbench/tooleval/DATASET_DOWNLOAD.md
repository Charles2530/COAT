# Math Reasoning Datasets Download Guide

This guide provides instructions for downloading the math reasoning datasets used in evaluation: **SVAMP**, **GSM8K**, **NumGLUE**, and **MATH (Mathematica)**.

## Quick Start

### Option 1: Using Python Script (Recommended)

```bash
cd examples/ToolBench

# Install dependencies
pip install datasets requests

# Download all datasets
python scripts/download_math_datasets.py --data_dir data/math_datasets

# Download specific datasets
python scripts/download_math_datasets.py --datasets svamp gsm8k --data_dir data/math_datasets
```

### Option 2: Using Shell Script

```bash
cd examples/ToolBench

# Install dependencies
pip install datasets

# Download all datasets
bash scripts/download_math_datasets.sh
```

## Manual Download

If automatic download fails, you can download datasets manually:

### 1. SVAMP

**Source**: GitHub
- **Repository**: https://github.com/arkilpatel/SVAMP
- **Direct download**: https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json
- **Save to**: `data/math_datasets/svamp/test.json`

```bash
mkdir -p data/math_datasets/svamp
wget https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json \
     -O data/math_datasets/svamp/test.json
```

### 2. GSM8K

**Source**: HuggingFace Datasets
- **HuggingFace**: https://huggingface.co/datasets/gsm8k
- **Download via Python**:

```python
from datasets import load_dataset

# Download test split
test = load_dataset("gsm8k", "main", split="test")
test.to_json("data/math_datasets/gsm8k/test.jsonl")

# Download train split (optional)
train = load_dataset("gsm8k", "main", split="train")
train.to_json("data/math_datasets/gsm8k/train.jsonl")
```

**Or via command line**:
```bash
mkdir -p data/math_datasets/gsm8k
python -c "from datasets import load_dataset; load_dataset('gsm8k', 'main', split='test').to_json('data/math_datasets/gsm8k/test.jsonl')"
```

### 3. NumGLUE

**Source**: GitHub / HuggingFace
- **GitHub**: https://github.com/allenai/numglue
- **HuggingFace**: https://huggingface.co/datasets/allenai/numglue (if available)

**Download from GitHub**:
```bash
mkdir -p data/math_datasets/numglue
git clone https://github.com/allenai/numglue.git temp_numglue
# Copy test data from the repository
# Check the repository structure for exact file locations
```

**Or via HuggingFace** (if available):
```python
from datasets import load_dataset
dataset = load_dataset("allenai/numglue", split="test")
dataset.to_json("data/math_datasets/numglue/test.json")
```

### 4. MATH (Mathematica)

**Source**: HuggingFace / GitHub
- **HuggingFace**: https://huggingface.co/datasets/hendrycks/competition_math
- **GitHub**: https://github.com/hendrycks/math

**Download via HuggingFace**:
```python
from datasets import load_dataset
dataset = load_dataset("hendrycks/competition_math", split="test")
dataset.to_json("data/math_datasets/mathematica/test.json")
```

**Or via command line**:
```bash
mkdir -p data/math_datasets/mathematica
python -c "from datasets import load_dataset; load_dataset('hendrycks/competition_math', split='test').to_json('data/math_datasets/mathematica/test.json')"
```

## Dataset Structure

After downloading, your directory structure should look like:

```
data/math_datasets/
├── svamp/
│   └── test.json
├── gsm8k/
│   ├── test.jsonl
│   └── train.jsonl
├── numglue/
│   └── test.json
└── mathematica/
    └── test.json
```

## Dataset Formats

### SVAMP
- Format: JSON
- Fields: `Body`, `Question`, `Answer`, `ID`

### GSM8K
- Format: JSONL (one JSON object per line)
- Fields: `question`, `answer`

### NumGLUE
- Format: JSON
- Structure: List of items with `question` and `answer` fields

### MATH
- Format: JSON
- Fields: `problem`, `level`, `type`, `solution`

## Verification

After downloading, verify the datasets:

```bash
# Check file sizes (should not be empty)
ls -lh data/math_datasets/*/*.json*

# Check JSON validity
python -c "import json; json.load(open('data/math_datasets/svamp/test.json'))"
```

## Troubleshooting

1. **"datasets library not found"**
   ```bash
   pip install datasets
   ```

2. **"Connection timeout"**
   - Check internet connection
   - Try using a VPN if in certain regions
   - Download manually using the provided links

3. **"File not found"**
   - Verify the dataset is publicly available
   - Check if the repository structure has changed
   - Try alternative download methods

4. **"Permission denied"**
   - Make sure you have write permissions to the data directory
   - Use `sudo` if necessary (not recommended)

## Alternative Sources

If primary sources are unavailable:
- **Papers with Code**: https://paperswithcode.com/dataset/gsm8k
- **Open Data Portal**: Search for dataset names
- **Contact authors**: For restricted datasets

## License Information

Before using these datasets, please check their licenses:
- **SVAMP**: Check repository LICENSE file
- **GSM8K**: MIT License (OpenAI)
- **NumGLUE**: Apache 2.0 (AllenAI)
- **MATH**: Check repository LICENSE file

