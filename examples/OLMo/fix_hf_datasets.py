#!/usr/bin/env python3
"""
Fix broken HuggingFace datasets by re-downloading and saving them properly.
This script fixes datasets that have arrow files but are missing dataset_info.json and state.json.
"""
import os
import sys
from pathlib import Path
from typing import Optional

import datasets

# Add parent directory to path to import olmo.util
sys.path.insert(0, str(Path(__file__).parent))
from olmo.util import get_data_path

# Mapping of dataset paths to their HuggingFace identifiers
DATASET_MAPPING = {
    "ai2_arc": "ai2_arc",
    "boolq": "boolq",
    "glue": "glue",
    "hellaswag": "hellaswag",
    "nq_open": "natural_questions",
    "openbookqa": "openbookqa",
    "piqa": "piqa",
    "sciq": "sciq",
    "social_i_qa": "social_i_qa",
    "super_glue": "super_glue",
    "trivia_qa": "trivia_qa",
    "winogrande": "winogrande",
}

# Name mappings for datasets with specific configurations
NAME_MAPPING = {
    "ai2_arc": {
        "ARC-Challenge": "ARC-Challenge",
        "ARC-Easy": "ARC-Easy",
    },
    "glue": {
        "mrpc": "mrpc",
        "rte": "rte",
        "sst2": "sst2",
    },
    "nq_open": {
        "none": None,  # natural_questions doesn't use a name
    },
    "openbookqa": {
        "main": "main",
    },
    "piqa": {
        "plain_text": "plain_text",
    },
    "super_glue": {
        "cb": "cb",
        "copa": "copa",
    },
    "trivia_qa": {
        "rc.wikipedia.nocontext": "rc.wikipedia.nocontext",
    },
    "winogrande": {
        "winogrande_xl": "winogrande_xl",
    },
}


def find_broken_datasets(datasets_dir: Path):
    """Find all datasets that have arrow files but are missing metadata files."""
    broken_datasets = []
    
    for dataset_path in sorted(datasets_dir.iterdir()):
        if not dataset_path.is_dir():
            continue
        
        dataset_name = dataset_path.name
        if dataset_name not in DATASET_MAPPING:
            print(f"Warning: Unknown dataset '{dataset_name}', skipping...")
            continue
        
        # 查找所有 split 目录 (train, validation, test)
        for name_dir in dataset_path.iterdir():
            if not name_dir.is_dir():
                continue
            
            name = name_dir.name if name_dir.name != "none" else None
            
            # Check if this name is valid for the dataset
            if dataset_name in NAME_MAPPING:
                if name not in NAME_MAPPING[dataset_name] and name_dir.name != "none":
                    continue
            
            for split_dir in name_dir.iterdir():
                if not split_dir.is_dir() or split_dir.name not in ['train', 'validation', 'test']:
                    continue
                
                has_arrow = any(f.suffix == '.arrow' for f in split_dir.iterdir())
                has_info = (split_dir / 'dataset_info.json').exists()
                has_state = (split_dir / 'state.json').exists()
                
                if has_arrow and (not has_info or not has_state):
                    broken_datasets.append({
                        'dataset': dataset_name,
                        'name': name,
                        'split': split_dir.name,
                        'path': split_dir
                    })
    
    return broken_datasets


def fix_dataset(dataset: str, name: Optional[str], split: str, output_path: Path):
    """Download and save a dataset from HuggingFace Hub."""
    hf_dataset_name = DATASET_MAPPING[dataset]
    
    # Get the correct name for HuggingFace
    if dataset in NAME_MAPPING:
        hf_name = NAME_MAPPING[dataset].get(name, name)
    else:
        hf_name = name
    
    print(f"Loading {hf_dataset_name} (name={hf_name}, split={split}) from HuggingFace Hub...")
    try:
        if hf_name is None:
            dataset_obj = datasets.load_dataset(hf_dataset_name, split=split)
        else:
            dataset_obj = datasets.load_dataset(hf_dataset_name, name=hf_name, split=split)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to {output_path}...")
        dataset_obj.save_to_disk(str(output_path))
        print(f"✓ Successfully fixed {dataset}/{name or 'none'}/{split}")
        return True
    except Exception as e:
        print(f"✗ Failed to fix {dataset}/{name or 'none'}/{split}: {e}")
        return False


def main():
    # Get the datasets directory
    script_dir = Path(__file__).parent
    datasets_dir = script_dir / "olmo_data" / "hf_datasets"
    
    if not datasets_dir.exists():
        print(f"Error: Datasets directory not found at {datasets_dir}")
        return 1
    
    print(f"Scanning for broken datasets in {datasets_dir}...")
    broken_datasets = find_broken_datasets(datasets_dir)
    
    if not broken_datasets:
        print("No broken datasets found!")
        return 0
    
    print(f"\nFound {len(broken_datasets)} broken datasets:")
    for d in broken_datasets:
        print(f"  - {d['dataset']}/{d['name'] or 'none'}/{d['split']}")
    
    print(f"\nFixing {len(broken_datasets)} datasets...\n")
    
    success_count = 0
    for d in broken_datasets:
        success = fix_dataset(
            dataset=d['dataset'],
            name=d['name'],
            split=d['split'],
            output_path=d['path']
        )
        if success:
            success_count += 1
        print()  # Empty line for readability
    
    print(f"\nSummary: Fixed {success_count}/{len(broken_datasets)} datasets")
    return 0 if success_count == len(broken_datasets) else 1


if __name__ == "__main__":
    sys.exit(main())

