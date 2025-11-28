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
          frames = sorted(glob.glob(os.path.join(run, "frame_*.png")))
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

  def __len__(self):
      return len(self.sequences)

  def __getitem__(self, idx):
      run_idx, start = self.sequences[idx]
      run = self.runs[run_idx]

      frames = []
      actions = []

      for i in range(self.sequence_length):
          frame_idx = start + i
          frames.append(self._load_frame(run['frames'][frame_idx]))
          inp = run['inputs'].get(frame_idx, {'steering': 0.0, 'throttle': 0.0})
          actions.append([inp['steering'], inp['throttle']])

      return {
          'frames': torch.stack(frames),                          # [9, 352, 640, 3]
          'actions': torch.tensor(actions, dtype=torch.float32)   # [9, 2]
      }



def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  dataset = SimpleDataset(data_dir="/media/kristofe/eight/data", sequence_length=9)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

if __name__ == "__main__":
  main()

