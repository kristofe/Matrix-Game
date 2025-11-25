#!/usr/bin/env python3
"""
Finetune the causal distilled model on NEW steering/throttle data format.
This is a convenience wrapper that uses process_new_data.py for the dataset.

Usage:
    python finetune_new_data.py

Quick test (2 epochs, small subset):
    python finetune_new_data.py --quick_train

Or with custom parameters:
    python finetune_new_data.py --data_dir /path/to/data --num_epochs 20
"""

import sys
import os
import argparse

# Replace the dataset import before importing the main module
import convert_unreal_data
from process_new_data import NewDataset

# Monkey-patch the UnrealDataset to use NewDataset
convert_unreal_data.UnrealDataset = NewDataset

# Now import and run the main finetuning script
from finetune_causal_distilled_lora import main as original_main

def main():
    # Parse our custom arguments first
    parser = argparse.ArgumentParser(description="Finetune with new data format", add_help=False)
    parser.add_argument("--quick_train", action="store_true",
                       help="Quick training mode: 2 epochs, smaller batch, fewer sequences for testing")
    parser.add_argument("--quick_sequences", type=int, default=1000,
                       help="Number of sequences to use in quick_train mode (default: 100)")

    # Parse known args, leave the rest for the main script
    quick_args, remaining_args = parser.parse_known_args()

    # Override default data_dir if not specified
    if not any('--data_dir' in arg for arg in remaining_args):
        # Set default to the new data location
        remaining_args.extend(['--data_dir', '/media/kristofe/eight/data/'])

    # Apply quick_train settings
    if quick_args.quick_train:
        print("=" * 60)
        print("QUICK TRAINING MODE ENABLED")
        print("=" * 60)
        print(f"Settings:")
        print(f"  - Epochs: 2")
        print(f"  - Sequences: Limited to first {quick_args.quick_sequences}")
        print(f"  - Batch size: 4")
        print(f"  - Gradient accumulation: 1")
        print(f"  - Save every: 1 epoch")
        print()

        # Set quick training parameters if not already specified
        if not any('--num_epochs' in arg for arg in remaining_args):
            remaining_args.extend(['--num_epochs', '2'])
        if not any('--batch_size' in arg for arg in remaining_args):
            remaining_args.extend(['--batch_size', '4'])
        if not any('--gradient_accumulation_steps' in arg for arg in remaining_args):
            remaining_args.extend(['--gradient_accumulation_steps', '4'])
        if not any('--save_every' in arg for arg in remaining_args):
            remaining_args.extend(['--save_every', '1'])
        if not any('--checkpoint_dir' in arg for arg in remaining_args):
            remaining_args.extend(['--checkpoint_dir', 'checkpoints_quick'])

        # Monkey-patch the dataset to limit sequences
        original_init = NewDataset.__init__
        def limited_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if len(self.sequences) > quick_args.quick_sequences:
                print(f"\n*** QUICK MODE: Limiting dataset from {len(self.sequences)} to {quick_args.quick_sequences} sequences ***\n")
                self.sequences = self.sequences[:quick_args.quick_sequences]

        NewDataset.__init__ = limited_init

    # Update sys.argv for the main script
    sys.argv = [sys.argv[0]] + remaining_args

    print("=" * 60)
    print("FINE-TUNING WITH NEW DATA FORMAT (steering/throttle)")
    print("=" * 60)
    print("Using process_new_data.py for dataset loading")
    print()

    original_main()

if __name__ == "__main__":
    main()
