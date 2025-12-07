"""
Evaluation script for math reasoning datasets: SVAMP, GSM8K, NumGLUE, Mathematica
"""
import argparse
import json
import os
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import sys

# Add dataset module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
try:
    from dataset.math_datasets import (
        load_svamp, load_gsm8k, load_numglue, load_mathematica,
        format_question_for_model, get_ground_truth, get_item_id
    )
except ImportError:
    # Fallback if import fails
    def load_svamp(path): return json.load(open(path))
    def load_gsm8k(path): return json.load(open(path))
    def load_numglue(path): return json.load(open(path))
    def load_mathematica(path): return json.load(open(path))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate math reasoning tasks')
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to model predictions file (JSON format)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['svamp', 'gsm8k', 'numglue', 'mathematica'],
                        help='Dataset name to evaluate')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the test dataset file')
    parser.add_argument('--output_path', type=str, default='math_reasoning_results',
                        help='Output directory for results')
    parser.add_argument('--extract_answer_from_response', action='store_true',
                        help='Extract numerical answer from model response text')
    return parser.parse_args()


def extract_number(text: str) -> str:
    """Extract the last number from text, handling various formats."""
    # Remove commas
    text = text.replace(',', '')
    
    # Try to find the last number in the text
    # Match integers and decimals
    patterns = [
        r'[-+]?\d+\.?\d*',  # Basic number
        r'\$?\d+\.?\d*',    # With dollar sign
        r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # With commas
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            num_str = matches[-1].replace('$', '').replace(',', '')
            try:
                # Try to convert to float first to handle decimals
                num = float(num_str)
                # Return as integer if it's a whole number, otherwise return decimal
                if num.is_integer():
                    return str(int(num))
                return str(num)
            except:
                continue
    
    return None


def extract_answer_from_response(response: str, dataset: str) -> str:
    """Extract the final answer from model response."""
    # Try to find "The answer is" or similar patterns
    answer_patterns = [
        r'(?:The answer is|Answer:|Final answer:|Therefore,?|So,?)\s*:?\s*([^\n\.]+)',
        r'boxed\{([^\}]+)\}',
        r'\$\$([^\$]+)\$\$',
        r'\\boxed\{([^\}]+)\}',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            num = extract_number(answer)
            if num:
                return num
    
    # If no pattern found, try to extract the last number
    return extract_number(response)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace, commas, dollar signs
    answer = str(answer).strip().lower()
    answer = answer.replace(',', '').replace('$', '').replace(' ', '')
    
    # Remove trailing zeros for decimal numbers
    if '.' in answer:
        answer = answer.rstrip('0').rstrip('.')
    
    return answer


def evaluate_svamp(predictions: Dict, dataset_path: str) -> Tuple[float, List[Dict]]:
    """Evaluate on SVAMP dataset."""
    try:
        test_data = load_svamp(dataset_path)
    except:
        with open(dataset_path, 'r') as f:
            test_data = json.load(f)
    
    correct = 0
    total = 0
    results = []
    
    for idx, item in enumerate(tqdm(test_data, desc='Evaluating SVAMP')):
        question_id = get_item_id(item, 'svamp', idx)
        ground_truth = get_ground_truth(item, 'svamp')
        
        # Get prediction
        pred_text = predictions.get(str(question_id), predictions.get(question_id, predictions.get(idx, "")))
        if isinstance(pred_text, dict):
            pred_text = pred_text.get('answer', pred_text.get('response', pred_text.get('output', '')))
        
        # Extract answer
        pred_answer = extract_answer_from_response(str(pred_text), 'svamp')
        
        # Normalize and compare
        pred_norm = normalize_answer(pred_answer) if pred_answer else ""
        gt_norm = normalize_answer(ground_truth) if ground_truth else ""
        
        is_correct = pred_norm == gt_norm and pred_norm != ""
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'id': question_id,
            'question': format_question_for_model(item, 'svamp'),
            'ground_truth': ground_truth,
            'prediction': pred_answer or "",
            'correct': is_correct
        })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def evaluate_gsm8k(predictions: Dict, dataset_path: str) -> Tuple[float, List[Dict]]:
    """Evaluate on GSM8K dataset."""
    try:
        test_data = load_gsm8k(dataset_path)
    except:
        with open(dataset_path, 'r') as f:
            test_data = json.load(f) if dataset_path.endswith('.json') else [json.loads(line) for line in f]
    
    correct = 0
    total = 0
    results = []
    
    for idx, item in enumerate(tqdm(test_data, desc='Evaluating GSM8K')):
        question_id = get_item_id(item, 'gsm8k', idx)
        ground_truth = get_ground_truth(item, 'gsm8k')
        
        # Extract ground truth number
        gt_answer = extract_number(ground_truth)
        
        # Get prediction
        pred_text = predictions.get(str(question_id), predictions.get(question_id, predictions.get(idx, "")))
        if isinstance(pred_text, dict):
            pred_text = pred_text.get('answer', pred_text.get('response', pred_text.get('output', '')))
        
        # Extract answer
        pred_answer = extract_answer_from_response(str(pred_text), 'gsm8k')
        
        # Normalize and compare
        pred_norm = normalize_answer(pred_answer) if pred_answer else ""
        gt_norm = normalize_answer(gt_answer) if gt_answer else ""
        
        is_correct = pred_norm == gt_norm and pred_norm != ""
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'id': question_id,
            'question': format_question_for_model(item, 'gsm8k'),
            'ground_truth': gt_answer or ground_truth,
            'prediction': pred_answer or "",
            'correct': is_correct
        })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def evaluate_numglue(predictions: Dict, dataset_path: str) -> Tuple[float, List[Dict]]:
    """Evaluate on NumGLUE dataset."""
    try:
        test_data = load_numglue(dataset_path)
    except:
        with open(dataset_path, 'r') as f:
            test_data = json.load(f)
    
    correct = 0
    total = 0
    results = []
    
    for idx, item in enumerate(tqdm(test_data, desc='Evaluating NumGLUE')):
        question_id = get_item_id(item, 'numglue', idx)
        ground_truth = get_ground_truth(item, 'numglue')
        
        # Get prediction
        pred_text = predictions.get(str(question_id), predictions.get(question_id, predictions.get(idx, "")))
        if isinstance(pred_text, dict):
            pred_text = pred_text.get('answer', pred_text.get('response', pred_text.get('output', '')))
        
        # Extract answer
        pred_answer = extract_answer_from_response(str(pred_text), 'numglue')
        
        # Normalize and compare
        pred_norm = normalize_answer(pred_answer) if pred_answer else ""
        gt_norm = normalize_answer(ground_truth) if ground_truth else ""
        
        is_correct = pred_norm == gt_norm and pred_norm != ""
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'id': question_id,
            'question': format_question_for_model(item, 'numglue'),
            'ground_truth': ground_truth,
            'prediction': pred_answer or "",
            'correct': is_correct
        })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def evaluate_mathematica(predictions: Dict, dataset_path: str) -> Tuple[float, List[Dict]]:
    """Evaluate on Mathematica dataset."""
    try:
        test_data = load_mathematica(dataset_path)
    except:
        with open(dataset_path, 'r') as f:
            test_data = json.load(f)
    
    correct = 0
    total = 0
    results = []
    
    for idx, item in enumerate(tqdm(test_data, desc='Evaluating Mathematica')):
        question_id = get_item_id(item, 'mathematica', idx)
        ground_truth = get_ground_truth(item, 'mathematica')
        
        # Get prediction
        pred_text = predictions.get(str(question_id), predictions.get(question_id, predictions.get(idx, "")))
        if isinstance(pred_text, dict):
            pred_text = pred_text.get('answer', pred_text.get('response', pred_text.get('output', '')))
        
        # Extract answer
        pred_answer = extract_answer_from_response(str(pred_text), 'mathematica')
        
        # Normalize and compare
        pred_norm = normalize_answer(pred_answer) if pred_answer else ""
        gt_norm = normalize_answer(ground_truth) if ground_truth else ""
        
        is_correct = pred_norm == gt_norm and pred_norm != ""
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'id': question_id,
            'question': format_question_for_model(item, 'mathematica'),
            'ground_truth': ground_truth,
            'prediction': pred_answer or "",
            'correct': is_correct
        })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def main():
    args = parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions_path}")
    with open(args.predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Evaluate based on dataset
    dataset_lower = args.dataset.lower()
    if dataset_lower == 'svamp':
        accuracy, results = evaluate_svamp(predictions, args.dataset_path)
    elif dataset_lower == 'gsm8k':
        accuracy, results = evaluate_gsm8k(predictions, args.dataset_path)
    elif dataset_lower == 'numglue':
        accuracy, results = evaluate_numglue(predictions, args.dataset_path)
    elif dataset_lower == 'mathematica':
        accuracy, results = evaluate_mathematica(predictions, args.dataset_path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save detailed results
    output_file = os.path.join(args.output_path, f"{args.dataset}_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'accuracy': accuracy,
            'total': len(results),
            'correct': sum(1 for r in results if r['correct']),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary_file = os.path.join(args.output_path, f"{args.dataset}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Correct: {sum(1 for r in results if r['correct'])}\n")
        f.write(f"Total: {len(results)}\n")
    
    print(f"\n{'='*50}")
    print(f"Dataset: {args.dataset}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {sum(1 for r in results if r['correct'])} / {len(results)}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

