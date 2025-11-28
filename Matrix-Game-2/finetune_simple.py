"""
Simple finetuning script - built from scratch to understand every piece.
"""

import torch
import os
import glob
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
  def __init__(self, data_dir="/media/kristofe/eight/data", sequence_length=9):
      self.sequence_length = sequence_length

      # Find all runs
      self.runs = []
      for run in sorted(glob.glob(os.path.join(data_dir, "*/Run_*"))):
          # Get frames and sort numerically by frame number
          frame_pattern = os.path.join(run, "frame_*.png")
          frames = glob.glob(frame_pattern)
          # Sort by extracting frame number from filename (e.g., frame_0001.png -> 1)
          frames = sorted(frames, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

          if len(frames) >= sequence_length:
              self.runs.append({
                  'path': run,
                  'frames': frames,
                  'inputs': self._load_csv(os.path.join(run, "input.csv"))
              })
      print(f"Found {len(self.runs)} runs")
      # Build list of all valid sequences
      self.sequences = []
      for run_idx, run in enumerate(self.runs):
          num_seq = len(run['frames']) - sequence_length + 1
          for start in range(num_seq):
              self.sequences.append((run_idx, start))
      print(f"Total sequences: {len(self.sequences)}")

  def _load_csv(self, csv_path):
      """Load steering/throttle per frame"""
      data = {}
      debug_counter = 0
      with open(csv_path, 'r') as f:
          for row in csv.reader(f):
              action, time, frame_num, value = row[0], float(row[1]), int(row[2]), float(row[3])
              if frame_num not in data:
                  data[frame_num] = {'steering': 0.0, 'throttle': 0.0}
              debug_counter += 1
              if debug_counter < 3:
                  print(f"action {action} frame_num {frame_num} value {value}")
              data[frame_num][action] = value
      return data

  def _load_frame(self, path):
      """Load image, resize to 352x640, normalize to [0,1]"""
      img = Image.open(path).convert('RGB')
      img = img.resize((640, 352))
      return torch.from_numpy(np.array(img)).float() / 255.0

  def _to_gta_drive_format(self, steering, throttle, brake=0.0):
      """
      Convert steering/throttle/brake to gta_drive format.

      For gta_drive mode (see utils/conditions.py):
        - keyboard_condition: [forward, back] where forward=throttle, back=brake
        - mouse_condition: [vertical, horizontal] where horizontal=steering

      Args:
          steering: -1 (full left) to 1 (full right) - maps to mouse horizontal
          throttle: 0 to 1 - maps to keyboard forward
          brake: 0 to 1 - maps to keyboard back

      Returns:
          keyboard: [forward, back]
          mouse: [vertical, horizontal]
      """
      keyboard = [
          max(0.0, min(1.0, throttle)),  # forward
          max(0.0, min(1.0, brake))       # back
      ]
      mouse = [
          0.0,      # vertical (not used for driving)
          steering  # horizontal (steering = camera rotation)
      ]
      return keyboard, mouse

  def __len__(self):
      return len(self.sequences)

  def __getitem__(self, idx):
      run_idx, start = self.sequences[idx]
      run = self.runs[run_idx]

      frames = []
      keyboard_actions = []
      mouse_actions = []

      for i in range(self.sequence_length):
          frame_idx = start + i
          frames.append(self._load_frame(run['frames'][frame_idx]))

          # Get raw inputs for this frame
          inp = run['inputs'].get(frame_idx, {'steering': 0.0, 'throttle': 0.0})
          steering = inp.get('steering', 0.0)
          throttle = inp.get('throttle', 0.0)
          brake = inp.get('brake', 0.0)

          # Convert to gta_drive format
          keyboard, mouse = self._to_gta_drive_format(steering, throttle, brake)
          keyboard_actions.append(keyboard)
          mouse_actions.append(mouse)

      return {
          'video_frames': torch.stack(frames),                                      # [T, H, W, C] = [9, 352, 640, 3]
          'keyboard_actions': torch.tensor(keyboard_actions, dtype=torch.float32),  # [T, 2] = [9, 2] (forward, back)
          'mouse_actions': torch.tensor(mouse_actions, dtype=torch.float32),        # [T, 2] = [9, 2] (vertical, horizontal/steering)
      }



def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  dataset = SimpleDataset(data_dir="/media/kristofe/eight/data", sequence_length=9)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

if __name__ == "__main__":
  main()

