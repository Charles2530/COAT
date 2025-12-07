#!/usr/bin/env python3
"""
Python script to download math reasoning datasets: SVAMP, GSM8K, NumGLUE, and MATH
Supports downloading from HuggingFace, GitHub, and other sources
"""

import os
import json
import argparse
import requests
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not found. Install with: pip install datasets")


def download_file(url: str, output_path: str, description: str = "") -> bool:
    """Download a file from URL."""
    try:
        print(f"Downloading {description or url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  ✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def download_svamp(data_dir: str) -> bool:
    """Download SVAMP dataset."""
    print("\n--- Downloading SVAMP ---")
    svamp_dir = os.path.join(data_dir, "svamp")
    os.makedirs(svamp_dir, exist_ok=True)
    
    # Try GitHub raw file
    url = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
    output_file = os.path.join(svamp_dir, "test.json")
    
    if download_file(url, output_file, "SVAMP dataset"):
        return True
    
    # Try HuggingFace as fallback
    if HAS_DATASETS:
        try:
            dataset = load_dataset("svamp", split="test")
            dataset.to_json(output_file)
            print(f"  ✓ Downloaded SVAMP from HuggingFace to {output_file}")
            return True
        except:
            pass
    
    print("  ✗ SVAMP download failed. Please download manually from:")
    print("    https://github.com/arkilpatel/SVAMP")
    return False


def download_gsm8k(data_dir: str) -> bool:
    """Download GSM8K dataset."""
    print("\n--- Downloading GSM8K ---")
    gsm8k_dir = os.path.join(data_dir, "gsm8k")
    os.makedirs(gsm8k_dir, exist_ok=True)
    
    if not HAS_DATASETS:
        print("  ✗ Please install datasets library: pip install datasets")
        print("  Then download from: https://huggingface.co/datasets/gsm8k")
        return False
    
    try:
        # Download test split - must specify 'main' config
        print("  Loading GSM8K test split (config: 'main')...")
        test_dataset = load_dataset("gsm8k", "main", split="test")
        test_file = os.path.join(gsm8k_dir, "test.jsonl")
        test_dataset.to_json(test_file)
        print(f"  ✓ Downloaded GSM8K test to {test_file}")
        
        # Also download train for reference
        print("  Loading GSM8K train split (config: 'main')...")
        train_dataset = load_dataset("gsm8k", "main", split="train")
        train_file = os.path.join(gsm8k_dir, "train.jsonl")
        train_dataset.to_json(train_file)
        print(f"  ✓ Downloaded GSM8K train to {train_file}")
        return True
    except ValueError as e:
        if "Config name is missing" in str(e):
            print(f"  ✗ GSM8K download failed: Config name is required")
            print("  Attempting with explicit 'main' config...")
            try:
                test_dataset = load_dataset("gsm8k", "main", split="test")
                test_file = os.path.join(gsm8k_dir, "test.jsonl")
                test_dataset.to_json(test_file)
                print(f"  ✓ Downloaded GSM8K test to {test_file}")
                return True
            except Exception as e2:
                print(f"  ✗ Retry failed: {e2}")
        else:
            print(f"  ✗ GSM8K download failed: {e}")
    except Exception as e:
        print(f"  ✗ GSM8K download failed: {e}")
        print("  Please download manually from: https://huggingface.co/datasets/gsm8k")
        print("  Note: Use 'main' config: load_dataset('gsm8k', 'main')")
    return False


def download_numglue(data_dir: str) -> bool:
    """Download NumGLUE dataset."""
    print("\n--- Downloading NumGLUE ---")
    numglue_dir = os.path.join(data_dir, "numglue")
    os.makedirs(numglue_dir, exist_ok=True)
    
    if not HAS_DATASETS:
        print("  ✗ Please install datasets library: pip install datasets")
        print("  Then download from: https://github.com/allenai/numglue")
        return False
    
    # Try various HuggingFace paths
    hf_paths = [
        "allenai/numglue",
        "numglue",
    ]
    
    for hf_path in hf_paths:
        try:
            dataset = load_dataset(hf_path, split="test")
            output_file = os.path.join(numglue_dir, "test.json")
            dataset.to_json(output_file)
            print(f"  ✓ Downloaded NumGLUE to {output_file}")
            return True
        except:
            continue
    
    # Try GitHub
    github_urls = [
        "https://raw.githubusercontent.com/allenai/numglue/main/data/test.json",
        "https://github.com/allenai/numglue/raw/main/data/test.jsonl",
    ]
    
    output_file = os.path.join(numglue_dir, "test.json")
    for url in github_urls:
        if download_file(url, output_file, "NumGLUE dataset"):
            return True
    
    print("  ✗ NumGLUE download failed. Please download manually from:")
    print("    https://github.com/allenai/numglue")
    return False


def download_math(data_dir: str) -> bool:
    """Download MATH (Mathematica) dataset."""
    print("\n--- Downloading MATH (Mathematica) ---")
    math_dir = os.path.join(data_dir, "mathematica")
    os.makedirs(math_dir, exist_ok=True)
    
    if not HAS_DATASETS:
        print("  ✗ Please install datasets library: pip install datasets")
        print("  Then download from: https://huggingface.co/datasets/hendrycks/competition_math")
        return False
    
    # Try various HuggingFace paths
    hf_paths = [
        "qwedsacf/competition_math",  # Primary source
        "hendrycks/competition_math",
        "lighteval/math",
        "math",
    ]
    
    for hf_path in hf_paths:
        try:
            print(f"  Trying {hf_path}...")
            dataset = load_dataset(hf_path, split="test")
            output_file = os.path.join(math_dir, "test.json")
            dataset.to_json(output_file)
            print(f"  ✓ Downloaded MATH to {output_file}")
            return True
        except Exception as e:
            print(f"  Failed with {hf_path}: {str(e)[:100]}")
            continue
    
    print("  ✗ MATH dataset download failed. Please download manually from:")
    print("    https://huggingface.co/datasets/qwedsacf/competition_math")
    print("    https://huggingface.co/datasets/hendrycks/competition_math")
    print("    https://github.com/hendrycks/math")
    return False


def main():
    parser = argparse.ArgumentParser(description='Download math reasoning datasets')
    parser.add_argument('--data_dir', type=str, default='data/math_datasets',
                        help='Directory to save datasets (default: data/math_datasets)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        choices=['svamp', 'gsm8k', 'numglue', 'math', 'all'],
                        default=['all'],
                        help='Which datasets to download (default: all)')
    
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['svamp', 'gsm8k', 'numglue', 'math']
    
    print("=" * 50)
    print("Downloading Math Reasoning Datasets")
    print("=" * 50)
    print(f"Output directory: {args.data_dir}")
    print(f"Datasets: {', '.join(datasets_to_download)}")
    print("")
    
    results = {}
    if 'svamp' in datasets_to_download:
        results['svamp'] = download_svamp(args.data_dir)
    
    if 'gsm8k' in datasets_to_download:
        results['gsm8k'] = download_gsm8k(args.data_dir)
    
    if 'numglue' in datasets_to_download:
        results['numglue'] = download_numglue(args.data_dir)
    
    if 'math' in datasets_to_download:
        results['math'] = download_math(args.data_dir)
    
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {dataset.upper()}")
    
    print(f"\nData saved to: {args.data_dir}")
    
    if not HAS_DATASETS:
        print("\nNote: Install datasets library for better support:")
        print("  pip install datasets")


if __name__ == "__main__":
    main()

