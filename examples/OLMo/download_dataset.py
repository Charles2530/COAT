#!/usr/bin/env python3
"""
Download a single HuggingFace dataset and save it to olmo_data/hf_datasets.

Usage:
    python download_dataset.py <dataset_name> [--name <name>] [--split <split>] [--output-dir <dir>]

Examples:
    # Download hellaswag validation split
    python download_dataset.py hellaswag --split validation

    # Download GLUE MRPC validation split
    python download_dataset.py glue --name mrpc --split validation

    # Download AI2 ARC Challenge validation split
    python download_dataset.py ai2_arc --name ARC-Challenge --split validation

    # Download with custom output directory
    python download_dataset.py hellaswag --split validation --output-dir /path/to/output
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import datasets

# Add parent directory to path to import olmo.util
sys.path.insert(0, str(Path(__file__).parent))
try:
    from olmo.util import save_hf_dataset_to_disk
    USE_OLMO_UTIL = True
except ImportError:
    USE_OLMO_UTIL = False
    print("Warning: Could not import olmo.util.save_hf_dataset_to_disk, using direct save instead.")


def download_dataset(
    dataset_name: str,
    name: Optional[str] = None,
    split: str = "validation",
    output_dir: Optional[str] = None,
):
    """
    Download a HuggingFace dataset and save it to disk.

    Args:
        dataset_name: The HuggingFace dataset identifier (e.g., "hellaswag", "glue")
        name: Optional dataset configuration name (e.g., "mrpc" for GLUE)
        split: Dataset split to download (default: "validation")
        output_dir: Custom output directory (default: olmo_data/hf_datasets)
    """
    print(f"Downloading dataset: {dataset_name}")
    if name:
        print(f"  Configuration: {name}")
    print(f"  Split: {split}")
    
    # Load dataset from HuggingFace Hub
    try:
        if name:
            print(f"\nLoading from HuggingFace Hub...")
            dataset = datasets.load_dataset(dataset_name, name=name, split=split)
        else:
            print(f"\nLoading from HuggingFace Hub...")
            dataset = datasets.load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    print(f"Dataset loaded successfully. Size: {len(dataset)} examples")
    
    # Determine output path
    if output_dir:
        output_path = Path(output_dir)
    else:
        script_dir = Path(__file__).parent
        output_path = script_dir / "olmo_data" / "hf_datasets" / dataset_name / (name or "none") / split
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    print(f"\nSaving dataset to: {output_path}")
    try:
        if USE_OLMO_UTIL:
            save_hf_dataset_to_disk(
                dataset=dataset,
                hf_path=dataset_name,
                name=name,
                split=split,
                datasets_dir=script_dir / "olmo_data" / "hf_datasets"
            )
        else:
            dataset.save_to_disk(str(output_path))
        print(f"✓ Successfully saved dataset to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download a single HuggingFace dataset and save it to olmo_data/hf_datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="HuggingFace dataset identifier (e.g., 'hellaswag', 'glue', 'ai2_arc')"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset configuration name (e.g., 'mrpc' for GLUE, 'ARC-Challenge' for AI2 ARC)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to download (default: 'validation')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: olmo_data/hf_datasets/{dataset_name}/{name or 'none'}/{split})"
    )
    
    args = parser.parse_args()
    
    success = download_dataset(
        dataset_name=args.dataset_name,
        name=args.name,
        split=args.split,
        output_dir=args.output_dir
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

