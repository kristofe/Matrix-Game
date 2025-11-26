#!/usr/bin/env python3
"""
Convert Unreal Engine demo data to Matrix-Game format
Simple script to convert frames + CSV input to training format
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob
import hashlib

class UnrealDataset(Dataset):
    """Dataset for Unreal Engine demo data with optional latent caching."""

    def __init__(self, data_dir="data", sequence_length=30, fps=25, cache_latents=True):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.fps = fps
        self.cache_latents = cache_latents

        # Cache directory for latents
        self.cache_dir = os.path.join(data_dir, "cached_latents")
        if cache_latents:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load CSV data
        self.csv_data = self._load_csv_data()

        # Get all frame files
        self.frame_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.png")))
        print(f"Found {len(self.frame_files)} frames")

        # Create sequences
        self.sequences = self._create_sequences()
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")

        # Check cache status
        if cache_latents:
            cached_count = self._count_cached_sequences()
            print(f"Latent cache: {cached_count}/{len(self.sequences)} sequences cached")
    
    def _load_csv_data(self):
        """Load and parse the CSV input data."""
        csv_path = os.path.join(self.data_dir, "input.csv")
        csv_data = {}
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    key, time, frame_num = row[0], float(row[1]), int(row[2])
                    csv_data[frame_num] = {
                        'key': key,
                        'time': time
                    }
        
        print(f"Loaded {len(csv_data)} input records")
        return csv_data
    
    def _create_sequences(self):
        """Create sequences of frames for training."""
        sequences = []
        max_frames = len(self.frame_files)

        # Create overlapping sequences
        for start_idx in range(0, max_frames - self.sequence_length + 1, self.sequence_length // 2):
            end_idx = start_idx + self.sequence_length
            if end_idx <= max_frames:
                sequences.append((start_idx, end_idx))

        return sequences

    def _get_cache_path(self, idx):
        """Get cache file path for a sequence index."""
        start_idx, end_idx = self.sequences[idx]
        return os.path.join(self.cache_dir, f"seq_{start_idx:06d}_{end_idx:06d}.pt")

    def _count_cached_sequences(self):
        """Count how many sequences have cached latents."""
        count = 0
        for idx in range(len(self.sequences)):
            if os.path.exists(self._get_cache_path(idx)):
                count += 1
        return count

    def has_cached_latents(self, idx):
        """Check if latents are cached for a sequence."""
        return os.path.exists(self._get_cache_path(idx))

    def load_cached_latents(self, idx):
        """Load cached latents for a sequence."""
        cache_path = self._get_cache_path(idx)
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)
        return None

    def save_cached_latents(self, idx, latents, visual_context):
        """Save latents and CLIP embeddings to cache."""
        cache_path = self._get_cache_path(idx)
        cache_data = {
            'latents': latents.cpu(),
            'visual_context': visual_context.cpu(),
        }
        torch.save(cache_data, cache_path)
    
    def _key_to_universal_format(self, key):
        """
        Convert key to 4-dim universal keyboard format.
        Universal format: [W, A, S, D] for movement
        """
        # Initialize all keys as 0 (not pressed)
        keyboard = [0.0, 0.0, 0.0, 0.0]  # [W, A, S, D]
        
        if key == "=" or key == "":
            # No input
            pass
        elif key.upper() == "W":
            keyboard[0] = 1.0  # Forward
        elif key.upper() == "A":
            keyboard[1] = 1.0  # Left
        elif key.upper() == "S":
            keyboard[2] = 1.0  # Backward
        elif key.upper() == "D":
            keyboard[3] = 1.0  # Right
        elif key.upper() == "WA" or key.upper() == "AW":
            keyboard[0] = 1.0  # Forward + Left
            keyboard[1] = 1.0
        elif key.upper() == "WD" or key.upper() == "DW":
            keyboard[0] = 1.0  # Forward + Right
            keyboard[3] = 1.0
        elif key.upper() == "SA" or key.upper() == "AS":
            keyboard[2] = 1.0  # Backward + Left
            keyboard[1] = 1.0
        elif key.upper() == "SD" or key.upper() == "DS":
            keyboard[2] = 1.0  # Backward + Right
            keyboard[3] = 1.0
        # Add more key combinations as needed
        
        return keyboard
    
    def _get_mouse_actions(self, sequence_length):
        """
        Generate dummy mouse actions for now.
        In a real implementation, you'd extract mouse data from your input.
        """
        # Return small random movements (normalized to [-1, 1])
        # mouse_actions = np.random.normal(0, 0.1, (sequence_length, 2)).astype(np.float32)

        #actuall lets zero them out
        mouse_actions = np.zeros((sequence_length, 2)).astype(np.float32)
        return mouse_actions
    
    def _resizecrop(self, image, th, tw):
        """
        Crop image to target aspect ratio (th x tw) before resizing.
        This matches the preprocessing in inference.py.
        """
        w, h = image.size
        if h / w > th / tw:
            # Image is too tall - crop height
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            # Image is too wide - crop width
            new_h = int(h)
            new_w = int(new_h * tw / th)
        # Center crop
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    def _load_frame(self, frame_path):
        """Load and preprocess a single frame."""
        try:
            image = Image.open(frame_path).convert('RGB')
            # Crop to target aspect ratio (matching inference.py preprocessing)
            image = self._resizecrop(image, 352, 640)
            # Resize to expected dimensions (352x640 for the model)
            image = image.resize((640, 352))
            # Convert to tensor and normalize to [0, 1]
            frame = torch.from_numpy(np.array(image)).float() / 255.0
            return frame
        except Exception as e:
            print(f"Error loading frame {frame_path}: {e}")
            # Return black frame as fallback
            return torch.zeros(352, 640, 3)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.sequences[idx]

        # Load video frames
        video_frames = []
        keyboard_actions = []

        for i in range(start_idx, end_idx):
            # Load frame
            frame_path = self.frame_files[i]
            frame = self._load_frame(frame_path)
            video_frames.append(frame)

            # Get keyboard action for this frame
            frame_num = i + 1  # Frame numbers start from 1
            if frame_num in self.csv_data:
                key = self.csv_data[frame_num]['key']
            else:
                key = "="  # No input if not found

            keyboard = self._key_to_universal_format(key)
            keyboard_actions.append(keyboard)

        # Convert to tensors
        video_frames = torch.stack(video_frames)  # (T, H, W, C)
        keyboard_actions = torch.tensor(keyboard_actions, dtype=torch.float32)  # (T, 4)

        # Generate dummy mouse actions
        mouse_actions = torch.tensor(
            self._get_mouse_actions(len(video_frames)),
            dtype=torch.float32
        )  # (T, 2)

        result = {
            'video_frames': video_frames,
            'keyboard_actions': keyboard_actions,
            'mouse_actions': mouse_actions,
            'sequence_idx': idx,  # For caching
        }

        # Load cached latents if available
        if self.cache_latents:
            cached = self.load_cached_latents(idx)
            if cached is not None:
                result['latents'] = cached['latents']
                result['visual_context'] = cached['visual_context']
                result['cached'] = True
            else:
                # Use empty tensors as placeholders - training will compute actual values
                result['latents'] = torch.empty(0)
                result['visual_context'] = torch.empty(0)
                result['cached'] = False

        return result

def test_dataset():
    """Test the dataset to make sure it works."""
    print("Testing UnrealDataset...")
    
    dataset = UnrealDataset(data_dir="data", sequence_length=30)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample video frames shape: {sample['video_frames'].shape}")
        print(f"Sample keyboard actions shape: {sample['keyboard_actions'].shape}")
        print(f"Sample mouse actions shape: {sample['mouse_actions'].shape}")
        print(f"First few keyboard actions: {sample['keyboard_actions'][:5]}")
        print("Dataset test successful!")
    else:
        print("No sequences found in dataset!")

if __name__ == "__main__":
    test_dataset()
