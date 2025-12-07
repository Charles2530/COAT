# Quick Start Guide: Math Reasoning Evaluation

This guide helps you quickly set up and run math reasoning evaluation on ToolBench.

## Step 1: Install Dependencies

```bash
pip install datasets requests
```

## Step 2: Download Datasets

```bash
cd examples/ToolBench

# Download all math reasoning datasets
python scripts/download_math_datasets.py --data_dir data/math_datasets

# Or download specific datasets
python scripts/download_math_datasets.py --datasets svamp gsm8k --data_dir data/math_datasets
```

Expected output:
```
data/math_datasets/
├── svamp/test.json
├── gsm8k/test.jsonl
├── numglue/test.json
└── mathematica/test.json
```

## Step 3: Run Model Inference

Generate predictions using your model. The predictions should be in JSON format:

```json
{
    "question_id_1": "model response text with answer",
    "question_id_2": "model response text with answer",
    ...
}
```

Or structured format:
```json
{
    "question_id_1": {
        "answer": "extracted answer",
        "response": "full model response"
    },
    ...
}
```

## Step 4: Evaluate

### Single Dataset

```bash
cd toolbench/tooleval

python eval_math_reasoning.py \
    --predictions_path ../predictions/svamp_predictions.json \
    --dataset svamp \
    --dataset_path ../../data/math_datasets/svamp/test.json \
    --output_path results/svamp \
    --extract_answer_from_response
```

### All Datasets

```bash
cd examples/ToolBench

# Set paths
export SVAMP_PREDICTIONS="predictions/svamp_predictions.json"
export SVAMP_DATASET="data/math_datasets/svamp/test.json"

export GSM8K_PREDICTIONS="predictions/gsm8k_predictions.json"
export GSM8K_DATASET="data/math_datasets/gsm8k/test.jsonl"

export NUMGLUE_PREDICTIONS="predictions/numglue_predictions.json"
export NUMGLUE_DATASET="data/math_datasets/numglue/test.json"

export MATHEMATICA_PREDICTIONS="predictions/math_predictions.json"
export MATHEMATICA_DATASET="data/math_datasets/mathematica/test.json"

# Run evaluation
bash scripts/eval_math_reasoning.sh
```

## Step 5: View Results

Results will be saved in the output directory:

```
results/
├── svamp/
│   ├── svamp_results.json       # Detailed results
│   └── svamp_summary.txt        # Summary statistics
├── gsm8k/
│   ├── gsm8k_results.json
│   └── gsm8k_summary.txt
└── ...
```

Each summary file contains:
```
Dataset: svamp
Accuracy: 0.7500 (75.00%)
Correct: 750
Total: 1000
```

## Troubleshooting

1. **Dataset download fails**: See [DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md) for manual download instructions

2. **"datasets library not found"**: 
   ```bash
   pip install datasets
   ```

3. **Answer extraction fails**: The script automatically extracts numbers from responses. Make sure your model outputs contain numerical answers.

4. **Format errors**: Check that your predictions JSON file matches the expected format (see MATH_REASONING_README.md)

## Dataset Links Reference

- **SVAMP**: https://github.com/arkilpatel/SVAMP
- **GSM8K**: https://huggingface.co/datasets/gsm8k
- **NumGLUE**: https://github.com/allenai/numglue
- **MATH**: https://huggingface.co/datasets/hendrycks/competition_math

