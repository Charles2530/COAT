# Math Reasoning Evaluation

This module provides evaluation support for math reasoning datasets: **SVAMP**, **GSM8K**, **NumGLUE**, and **MATH (Mathematica)**.

## Downloading Datasets

Before evaluation, you need to download the datasets. We provide automated download scripts:

### Quick Download

```bash
# Install dependencies
pip install datasets requests

# Download all datasets using Python script (recommended)
python scripts/download_math_datasets.py --data_dir data/math_datasets

# Or use shell script
bash scripts/download_math_datasets.sh
```

### Manual Download

See [DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md) for detailed download instructions and alternative sources.

## Overview

The math reasoning evaluation uses direct answer comparison instead of the complex evaluator system used for tool-use tasks. It extracts numerical answers from model responses and compares them with ground truth answers.

## Datasets Supported

1. **SVAMP** - Simple Variations on Arithmetic Math Problems
2. **GSM8K** - Grade School Math 8K
3. **NumGLUE** - Numerical GLUE benchmark
4. **Mathematica** - Mathematica-style math problems

## Usage

### Single Dataset Evaluation

Evaluate a single dataset:

```bash
cd toolbench/tooleval
python eval_math_reasoning.py \
    --predictions_path path/to/predictions.json \
    --dataset svamp \
    --dataset_path path/to/svamp_test.json \
    --output_path results/svamp \
    --extract_answer_from_response
```

### Batch Evaluation

Use the comprehensive evaluation script:

```bash
cd examples/ToolBench

# Set environment variables for dataset paths
export SVAMP_PREDICTIONS="path/to/svamp_predictions.json"
export SVAMP_DATASET="path/to/svamp_test.json"

export GSM8K_PREDICTIONS="path/to/gsm8k_predictions.json"
export GSM8K_DATASET="path/to/gsm8k_test.json"

export NUMGLUE_PREDICTIONS="path/to/numglue_predictions.json"
export NUMGLUE_DATASET="path/to/numglue_test.json"

export MATHEMATICA_PREDICTIONS="path/to/mathematica_predictions.json"
export MATHEMATICA_DATASET="path/to/mathematica_test.json"

# Run evaluation
bash scripts/eval_math_reasoning.sh
```

## Predictions Format

The predictions file should be a JSON file with one of the following formats:

### Format 1: ID-based mapping
```json
{
    "question_id_1": "model response text 1",
    "question_id_2": "model response text 2",
    ...
}
```

### Format 2: ID-based with structured response
```json
{
    "question_id_1": {
        "answer": "model response text 1",
        "response": "full response",
        "output": "extracted answer"
    },
    ...
}
```

### Format 3: List format
```json
[
    {"id": "question_id_1", "answer": "model response text 1"},
    {"id": "question_id_2", "answer": "model response text 2"},
    ...
]
```

## Answer Extraction

The evaluation script automatically extracts numerical answers from model responses using:
- Pattern matching for common answer formats ("The answer is X", "Answer: X", etc.)
- LaTeX boxed format detection (`\boxed{X}`)
- Last number extraction as fallback

## Output

The evaluation produces:
- `{dataset}_results.json`: Detailed results for each question
- `{dataset}_summary.txt`: Summary statistics

Example output:
```
Dataset: svamp
Accuracy: 0.7500 (75.00%)
Correct: 750
Total: 1000
```

## Integration with ToolBench Inference

To use with ToolBench inference pipeline:

1. Run inference on math datasets using `qa_pipeline.py`
2. Convert predictions to the required format
3. Run evaluation using `eval_math_reasoning.py`

## Notes

- Answer normalization handles decimal precision and formatting differences
- The evaluation is case-insensitive and handles various number formats
- Empty or invalid answers are marked as incorrect

