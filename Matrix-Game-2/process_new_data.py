#!/usr/bin/env python3
"""
Process new data format from /media/kristofe/eight/data/ to Matrix-Game format.

New format:
- steering: -1 to 1 (where -1 = A/left, 1 = D/right)
- throttle: 0 to 1 (where 1 = W/forward)
- brake: 0 to 1 (where 1 = S/backward) - not present in current data

Converts to WASD format: [W, A, S, D]
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob
from pathlib import Path


class NewDataset(Dataset):
    """Dataset for new steering/throttle format data with optional latent caching."""

    def __init__(self, data_dir, sequence_length=30, fps=25, cache_latents=True):
        self.data_root = data_dir
        self.sequence_length = sequence_length
        self.fps = fps
        self.cache_latents = cache_latents

        # Cache directory for latents
        self.cache_dir = os.path.join(self.data_root, "cached_latents")
        if cache_latents:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Find all runs across all session folders
        self.runs = self._find_all_runs()
        print(f"Found {len(self.runs)} runs across all sessions")

        # Create sequences from all runs
        self.sequences = []
        for run_path in self.runs:
            run_sequences = self._create_sequences_for_run(run_path)
            self.sequences.extend(run_sequences)

        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")

        # Check cache status
        if cache_latents:
            cached_count = self._count_cached_sequences()
            print(f"Latent cache: {cached_count}/{len(self.sequences)} sequences cached")

    def _find_all_runs(self):
        """Find all Run_XXXXXX folders across all session folders."""
        runs = []
        data_path = Path(self.data_root)

        # Iterate through session folders (timestamp folders)
        for session_folder in data_path.iterdir():
            if session_folder.is_dir() and not session_folder.name.startswith('.'):
                # Look for Run_XXXXXX folders inside each session
                run_folders = list(session_folder.glob("Run_*"))
                for run_folder in run_folders:
                    if run_folder.is_dir():
                        # Check if it has required files
                        if (run_folder / "input.csv").exists():
                            runs.append(run_folder)
                            print(f"Found run: {run_folder}")

        return sorted(runs)

    def _load_run_data(self, run_path):
        """Load CSV data and info for a specific run."""
        csv_path = run_path / "input.csv"
        info_path = run_path / "info.txt"

        # Load info
        fps = 25
        num_frames = 0
        if info_path.exists():
            with open(info_path, 'r') as f:
                for line in f:
                    if line.startswith("FPS"):
                        fps = int(line.split(":")[1].strip())
                    elif line.startswith("NumFrame"):
                        num_frames = int(line.split(":")[1].strip())

        # Load CSV data - new format has action_type, timestamp, frame, value
        actions_by_frame = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4:
                    action_type = row[0]  # 'steering' or 'throttle'
                    timestamp = float(row[1])
                    frame_num = int(row[2])
                    value = float(row[3])

                    # Validate ranges
                    if action_type == 'steering':
                        # steering should be -1 to 1
                        value = max(-1.0, min(1.0, value))
                    elif action_type == 'throttle':
                        # throttle should be 0 to 1
                        value = max(0.0, min(1.0, value))
                    elif action_type == 'brake':
                        # brake should be 0 to 1
                        value = max(0.0, min(1.0, value))

                    if frame_num not in actions_by_frame:
                        actions_by_frame[frame_num] = {}

                    actions_by_frame[frame_num][action_type] = value

        # Get frame files
        frame_files = sorted(list(run_path.glob("frame_*.png")))

        return {
            'actions_by_frame': actions_by_frame,
            'frame_files': frame_files,
            'fps': fps,
            'num_frames': num_frames
        }

    def _create_sequences_for_run(self, run_path):
        """Create sequences for a single run."""
        run_data = self._load_run_data(run_path)
        frame_files = run_data['frame_files']

        sequences = []
        max_frames = len(frame_files)

        # Create overlapping sequences
        for start_idx in range(0, max_frames - self.sequence_length + 1, self.sequence_length // 2):
            end_idx = start_idx + self.sequence_length
            if end_idx <= max_frames:
                sequences.append({
                    'run_path': run_path,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'run_data': run_data
                })

        return sequences

    def _get_cache_path(self, idx):
        """Get cache file path for a sequence index."""
        seq_info = self.sequences[idx]
        run_name = seq_info['run_path'].name
        start_idx = seq_info['start_idx']
        end_idx = seq_info['end_idx']
        return os.path.join(self.cache_dir, f"{run_name}_seq_{start_idx:06d}_{end_idx:06d}.pt")

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

    def _actions_to_wasd(self, steering, throttle, brake=0.0):
        """
        Convert steering/throttle/brake to WASD format [W, A, S, D].

        steering: -1 (full left/A) to 1 (full right/D)
        throttle: 0 to 1 (W)
        brake: 0 to 1 (S)

        Returns: [W, A, S, D] where each value is 0 to 1
        """
        wasd = [0.0, 0.0, 0.0, 0.0]

        # W (forward) = throttle
        wasd[0] = throttle

        # A (left) = negative steering (map -1 to 1, 0 stays 0)
        if steering < 0:
            wasd[1] = abs(steering)

        # S (backward) = brake
        wasd[2] = brake

        # D (right) = positive steering (map 0 to 1)
        if steering > 0:
            wasd[3] = steering

        return wasd

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

    def _get_mouse_actions(self, sequence_length):
        """
        Generate dummy mouse actions for now.
        In a real implementation, you'd extract mouse data from your input.
        """
        # Zero out mouse movements
        mouse_actions = np.zeros((sequence_length, 2)).astype(np.float32)
        return mouse_actions

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        run_data = seq_info['run_data']
        start_idx = seq_info['start_idx']
        end_idx = seq_info['end_idx']
        frame_files = run_data['frame_files']
        actions_by_frame = run_data['actions_by_frame']

        # Load video frames
        video_frames = []
        keyboard_actions = []

        for i in range(start_idx, end_idx):
            # Load frame
            frame_path = frame_files[i]
            frame = self._load_frame(frame_path)
            video_frames.append(frame)

            # Get actions for this frame
            frame_num = i  # Frame numbers start from 0 in new format
            steering = 0.0
            throttle = 0.0
            brake = 0.0

            if frame_num in actions_by_frame:
                steering = actions_by_frame[frame_num].get('steering', 0.0)
                throttle = actions_by_frame[frame_num].get('throttle', 0.0)
                brake = actions_by_frame[frame_num].get('brake', 0.0)

            # Convert to WASD format
            wasd = self._actions_to_wasd(steering, throttle, brake)
            keyboard_actions.append(wasd)

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
            'run_path': str(seq_info['run_path'])  # For debugging
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


def test_dataset(data_dir="/media/kristofe/eight/data/"):
    """Test the dataset to make sure it works."""
    print("Testing NewDataset...")
    print("=" * 60)

    dataset = NewDataset(
        data_dir=data_dir,
        sequence_length=30
    )

    if len(dataset) > 0:
        print(f"\nDataset successfully created with {len(dataset)} sequences")
        print("=" * 60)

        # Test first sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Video frames shape: {sample['video_frames'].shape}")
        print(f"  Keyboard actions shape: {sample['keyboard_actions'].shape}")
        print(f"  Mouse actions shape: {sample['mouse_actions'].shape}")
        print(f"  Run path: {sample['run_path']}")

        # Show first few actions
        print(f"\n  First 10 keyboard actions (WASD format):")
        print(f"  {'Frame':<6} {'W':<8} {'A':<8} {'S':<8} {'D':<8}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for i in range(min(10, len(sample['keyboard_actions']))):
            w, a, s, d = sample['keyboard_actions'][i]
            print(f"  {i:<6} {w:<8.3f} {a:<8.3f} {s:<8.3f} {d:<8.3f}")

        # Show statistics
        print(f"\n  Action statistics:")
        keyboard_np = sample['keyboard_actions'].numpy()
        print(f"  W (throttle) - min: {keyboard_np[:, 0].min():.3f}, max: {keyboard_np[:, 0].max():.3f}, mean: {keyboard_np[:, 0].mean():.3f}")
        print(f"  A (left)     - min: {keyboard_np[:, 1].min():.3f}, max: {keyboard_np[:, 1].max():.3f}, mean: {keyboard_np[:, 1].mean():.3f}")
        print(f"  S (brake)    - min: {keyboard_np[:, 2].min():.3f}, max: {keyboard_np[:, 2].max():.3f}, mean: {keyboard_np[:, 2].mean():.3f}")
        print(f"  D (right)    - min: {keyboard_np[:, 3].min():.3f}, max: {keyboard_np[:, 3].max():.3f}, mean: {keyboard_np[:, 3].mean():.3f}")

        print("\n" + "=" * 60)
        print("Dataset test successful!")
    else:
        print("ERROR: No sequences found in dataset!")


if __name__ == "__main__":
    test_dataset()
