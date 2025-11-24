#!/usr/bin/env python3
"""
Finetune the causal distilled model on NEW steering/throttle data format.
This is a convenience wrapper that uses process_new_data.py for the dataset.

Usage:
    python finetune_new_data.py

Or with custom parameters:
    python finetune_new_data.py --data_dir /path/to/data --num_epochs 20
"""

import sys
import os

# Replace the dataset import before importing the main module
import convert_unreal_data
from process_new_data import NewDataset

# Monkey-patch the UnrealDataset to use NewDataset
convert_unreal_data.UnrealDataset = NewDataset

# Now import and run the main finetuning script
from finetune_causal_distilled_lora import main, parse_args

if __name__ == "__main__":
    # Override default data_dir if not specified
    args = sys.argv[1:]
    if not any('--data_dir' in arg for arg in args):
        # Set default to the new data location
        sys.argv.extend(['--data_dir', '/media/kristofe/eight/data/'])

    print("=" * 60)
    print("FINE-TUNING WITH NEW DATA FORMAT (steering/throttle)")
    print("=" * 60)
    print("Using process_new_data.py for dataset loading")
    print()

    main()
