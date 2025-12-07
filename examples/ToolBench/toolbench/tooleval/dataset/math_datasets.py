"""
Dataset loaders for math reasoning datasets: SVAMP, GSM8K, NumGLUE, Mathematica
"""
import json
from typing import List, Dict, Any


def load_svamp(dataset_path: str) -> List[Dict[str, Any]]:
    """Load SVAMP dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        # If it's a dict with 'problems' key
        if 'problems' in data:
            data = data['problems']
        # If it's a dict mapping IDs to problems
        elif all(isinstance(k, str) for k in data.keys()):
            data = [{'id': k, **v} for k, v in data.items()]
        else:
            data = list(data.values())
    
    return data


def load_gsm8k(dataset_path: str) -> List[Dict[str, Any]]:
    """Load GSM8K dataset from JSON or JSONL file."""
    data = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.endswith('.jsonl'):
            # JSONL format
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        else:
            # JSON format
            data = json.load(f)
            if isinstance(data, dict):
                if 'test' in data:
                    data = data['test']
                elif 'problems' in data:
                    data = data['problems']
    
    return data


def load_numglue(dataset_path: str) -> List[Dict[str, Any]]:
    """Load NumGLUE dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        if 'test' in data:
            data = data['test']
        elif 'data' in data:
            data = data['data']
        elif 'instances' in data:
            data = data['instances']
    
    return data


def load_mathematica(dataset_path: str) -> List[Dict[str, Any]]:
    """Load Mathematica dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        if 'test' in data:
            data = data['test']
        elif 'data' in data:
            data = data['data']
        elif 'problems' in data:
            data = data['problems']
    
    return data


def format_question_for_model(item: Dict[str, Any], dataset: str) -> str:
    """Format a question item into a prompt for the model."""
    if dataset == 'svamp':
        body = item.get('Body', item.get('Question', ''))
        question = item.get('Question', '')
        return f"{body}\n{question}" if body and question else (body or question)
    
    elif dataset == 'gsm8k':
        return item.get('question', item.get('problem', ''))
    
    elif dataset == 'numglue':
        return item.get('question', item.get('input', item.get('problem', '')))
    
    elif dataset == 'mathematica':
        return item.get('question', item.get('problem', item.get('input', '')))
    
    return ""


def get_ground_truth(item: Dict[str, Any], dataset: str) -> str:
    """Extract ground truth answer from dataset item."""
    if dataset == 'svamp':
        return str(item.get('Answer', item.get('answer', ''))).strip()
    
    elif dataset == 'gsm8k':
        answer = item.get('answer', '')
        # GSM8K format is usually "### Answer: X" or just "X"
        if isinstance(answer, str):
            if '###' in answer:
                answer = answer.split('###')[-1].strip()
            if 'Answer:' in answer or 'answer:' in answer:
                answer = answer.split(':')[-1].strip()
        return str(answer).strip()
    
    elif dataset == 'numglue':
        return str(item.get('answer', item.get('output', ''))).strip()
    
    elif dataset == 'mathematica':
        return str(item.get('answer', item.get('output', ''))).strip()
    
    return ""


def get_item_id(item: Dict[str, Any], dataset: str, index: int) -> str:
    """Get unique identifier for an item."""
    # Try various ID fields
    id_fields = ['id', 'ID', 'idx', 'index', 'question_id', 'problem_id']
    
    for field in id_fields:
        if field in item:
            return str(item[field])
    
    # Fallback to index
    return str(index)

