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
from omegaconf import OmegaConf
from safetensors.torch import load_file
from utils.wan_wrapper import WanDiffusionWrapper


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
      arr = torch.from_numpy(np.array(img)).float() / 255.0
      return arr * 2.0 - 1.0  # Normalize to [-1, 1]

  def _to_gta_drive_format(self, steering, throttle, brake=0.0):
      """
      Convert steering/throttle/brake to gta_drive format.

      For gta_drive mode (see utils/conditions.py):
        - keyboard_condition: [forward, back] where forward=throttle, back=brake
        - mouse_condition: [vertical, horizontal] where horizontal=steering

      Args:
          steering: 0 to 0.1
          throttle: 0 to 1 - maps to keyboard forward
          brake: 0 to 1 - maps to keyboard back

      Returns:
          keyboard: [forward, back]
          mouse: [vertical, horizontal]
      """
      keyboard = [
        1.0 if throttle > 0.1 else 0.0,  # forward binary
        1.0 if brake > 0.1 else 0.0      # back binary
      ]
      mouse = [
          0.0,      # vertical (not used for driving)
          steering * 0.1 # horizontal (steering = camera rotation)
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

def verify_batch(batch):
    frames = batch['video_frames']
    keyboard = batch['keyboard_actions']
    mouse = batch['mouse_actions']

    print("\n=== Verification ===")

    # 1. Frame range should be [-1, 1]
    fmin, fmax = frames.min().item(), frames.max().item()
    print(f"Frame range: min {fmin:.3f}, max {fmax:.3f}")

    # 2 keyboard should be binary 0 or 1
    unique_keyboard = torch.unique(keyboard).tolist()
    print(f"Keyboard unique values: {unique_keyboard} (expect [0.0, 1.0])")

    # 3. Mouse horizontal (steering) should be in range [-1, 1]
    steering = mouse[:, :, 1]
    smin, smax = steering.min().item(), steering.max().item()
    print(f"Steering range: min {smin:.3f}, max {smax:.3f}")

    # 4 Print shapes
    print(f"Video frames shape: {frames.shape}")        # expect [B, T, H, W, C]
    print(f"Keyboard actions shape: {keyboard.shape}")  # expect [B, T, 2]
    print(f"Mouse actions shape: {mouse.shape}")        # expect [B, T, 2]

def load_model(device):
    #load config
    config = OmegaConf.load("configs/inference_yaml/inference_gta_drive.yaml")
    config.model_kwargs.model_config = "configs/distilled_model/gta_drive"

    #create model
    model = WanDiffusionWrapper(**config.model_kwargs, is_causal=True)

    #load weights
    checkpoint = "models/gta_distilled_model/gta_keyboard2dim.safetensors"
    state_dict = load_file(checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device, dtype=torch.bfloat16)

    #load vae
    from wan.vae.wanx_vae import get_wanx_vae_wrapper
    vae = get_wanx_vae_wrapper("models/", torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device, dtype=torch.float16)
    return model, vae


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  dataset = SimpleDataset(data_dir="/media/kristofe/eight/data", sequence_length=9)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

  # 1. Grab one batch
  batch = next(iter(dataloader))
  # 2. Print shapes
  print(f"video_frames: {batch['video_frames'].shape}")      # expect [2, 9, 352, 640, 3]
  print(f"keyboard_actions: {batch['keyboard_actions'].shape}")  # expect [2, 9, 2]
  print(f"mouse_actions: {batch['mouse_actions'].shape}")        # expect [2, 9, 2]
  # 3. Check value ranges
  print(f"frames min/max: {batch['video_frames'].min():.3f} / {batch['video_frames'].max():.3f}")  # expect 0-1
  print(f"keyboard min/max: {batch['keyboard_actions'].min():.3f} / {batch['keyboard_actions'].max():.3f}")  # expect 0-1
  print(f"mouse (steering) range: {batch['mouse_actions'][:,:,1].min():.3f} / {batch['mouse_actions'][:,:,1].max():.3f}")  # expect -1 to 1

  # 4. Save first frame to verify visually
  first_frame = ((batch['video_frames'][0, 0] + 1)* 127.5).byte().numpy()
  Image.fromarray(first_frame).save("test_frame.png")
  print("Saved test_frame.png - check it looks correct")

  verify_batch(batch)


  # Load model
  print("\nLoading model...")
  model, vae = load_model(device)
  print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
  print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
  print("Vae loaded.")
  '''
      # 1. Grab one batch
      batch = next(iter(dataloader))

      # 2. Print shapes
      print(f"video_frames: {batch['video_frames'].shape}")      # expect [2, 9, 352, 640, 3]
      print(f"keyboard_actions: {batch['keyboard_actions'].shape}")  # expect [2, 9, 2]
      print(f"mouse_actions: {batch['mouse_actions'].shape}")        # expect [2, 9, 2]

      # 3. Check value ranges
      print(f"frames min/max: {batch['video_frames'].min():.3f} / {batch['video_frames'].max():.3f}")  # expect 0-1
      print(f"keyboard min/max: {batch['keyboard_actions'].min():.3f} / {batch['keyboard_actions'].max():.3f}")  # expect 0-1
      print(f"mouse (steering) range: {batch['mouse_actions'][:,:,1].min():.3f} / {batch['mouse_actions'][:,:,1].max():.3f}")  # expect -1 to 1

      # 4. Save first frame to verify visually
      first_frame = (batch['video_frames'][0, 0] * 255).byte().numpy()
      Image.fromarray(first_frame).save("test_frame.png")
      print("Saved test_frame.png - check it looks correct")

  This will confirm:
  - Batching works
  - Shapes are correct
  - Values are in expected ranges
  - Frames load correctly (visual check)

  '''

if __name__ == "__main__":
  main()

