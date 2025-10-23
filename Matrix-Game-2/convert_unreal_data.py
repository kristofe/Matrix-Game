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

class UnrealDataset(Dataset):
    """Dataset for Unreal Engine demo data."""
    
    def __init__(self, data_dir="data", sequence_length=30, fps=25):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.fps = fps
        
        # Load CSV data
        self.csv_data = self._load_csv_data()
        
        # Get all frame files
        self.frame_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.png")))
        print(f"Found {len(self.frame_files)} frames")
        
        # Create sequences
        self.sequences = self._create_sequences()
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
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
        mouse_actions = np.random.normal(0, 0.1, (sequence_length, 2)).astype(np.float32)
        return mouse_actions
    
    def _load_frame(self, frame_path):
        """Load and preprocess a single frame."""
        try:
            image = Image.open(frame_path).convert('RGB')
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
        
        return {
            'video_frames': video_frames,
            'keyboard_actions': keyboard_actions,
            'mouse_actions': mouse_actions
        }

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
