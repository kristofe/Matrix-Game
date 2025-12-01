"""
Simple finetuning script - built from scratch to understand every piece.
"""

from datetime import datetime
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
import tqdm
from torch.utils.tensorboard import SummaryWriter

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

def train_step(model, vae, batch, scheduler, optimizer, device):
    frames = batch['video_frames'].to(device, dtype=torch.float16)
    frames = frames.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]

    with torch.no_grad():
        latents = vae.encode(frames, device=device).to(dtype=torch.bfloat16)
        visual_context = vae.clip.encode_video(frames).to(device=device, dtype=torch.bfloat16)
        model.model.num_frame_per_block = latents.shape[2]

    #prepare conditions
    num_latent_frames = latents.shape[2]
    num_action_steps = 1 + 4 * (num_latent_frames - 1)
    keyboard = batch['keyboard_actions'].to(device, dtype=torch.bfloat16)[:, :num_action_steps]
    mouse = batch['mouse_actions'].to(device, dtype=torch.bfloat16)[:, :num_action_steps]

    #mask cond
    mask_cond = torch.ones_like(latents[:,:4])
    mask_cond[:,:, 0] = 0 # first frame known/conditional
    cond_concat = torch.cat([mask_cond, latents], dim=1)

    #sample random timestep
    batch_size = latents.shape[0]
    t_scalar = torch.rand(1,device=device) *0.9 + 0.05  #avoid 0 and 1
    t = t_scalar.expand(batch_size)
    timestep = t.unsqueeze(1).expand(batch_size, num_latent_frames).to(dtype=torch.bfloat16)

    #add noise
    #per sample timesteps (different noise levels per sample)
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t_scalar)

    #forward pass
    conditional_dict = {
        "cond_concat": cond_concat,
        "visual_context": visual_context,
        "keyboard_cond": keyboard,
        "mouse_cond": mouse,
    }
    flow_pred, pred_x0 = model(
        noisy_image_or_video=noisy_latents,
        timestep = timestep,
        conditional_dict = conditional_dict,
    )   

    # flow matching loss: predict velocity (latents - noise)
    target = scheduler.training_target(latents, noise, t_scalar)
    loss = torch.nn.functional.mse_loss(flow_pred, target)

    #backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def test_logic(dataloader, model, vae, scheduler, device):
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


  # Test the VAE
  print("\nTesting VAE encoding/decoding...")
  with torch.no_grad():
    #get frames [B, T, H, W, C] -> [B, C, T, H, W]
    frames = batch['video_frames'].to(device, dtype=torch.float16)
    frames = frames.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]

    # Encode
    latents = vae.encode(frames, device=device)
    model.model.num_frame_per_block = latents.shape[2]

    print(f"Input frames shape: {frames.shape}")
    print(f"Latents shape: {latents.shape}") 

    # Decode
    recon_frames = vae.decode(latents, device=device)
    print(f"Reconstructed frames shape: {recon_frames.shape}")
    # Save first reconstructed frame
    recon_first_frame = ((recon_frames[0, :, 0] + 1) * 127.5).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(recon_first_frame).save("recon_test_frame.png")
    print("Saved recon_test_frame.png - check it looks correct")

    # Test a forward pass through the model
    print("\nTesting model forward pass...")

    # Prepare Conditions
    keyboard = batch['keyboard_actions'].to(device, dtype=torch.bfloat16) # [B, T, 2]
    mouse = batch['mouse_actions'].to(device, dtype=torch.bfloat16)       # [B, T, 2]
    latents = latents.to(dtype=torch.bfloat16)

    # expand actions to match latents spatial dims
    num_latent_frames = latents.shape[2]
    num_action_steps = 1 + 4 * (num_latent_frames - 1)
    keyboard = keyboard[:, :num_action_steps]
    mouse = mouse[:, :num_action_steps]

    # create cond_concat (mask + latents)
    mask_cond = torch.zeros_like(latents[:,:4]) # [B, 4, T, H, W]
    mask_cond[:,:, 0] = 0 # first frame known/conditional
    cond_concat = torch.cat([mask_cond, latents], dim=1) # [B, 20, T, H, W]

    # get visual context from CLIP
    visual_context = vae.clip.encode_video(frames).to(device=device, dtype=torch.bfloat16)  # [B, C, T, H, W]

    # add noise
    noise = torch.randn_like(latents)
    timestep_scalar = torch.tensor([0.5], device=device, dtype=torch.bfloat16)  # midpoint
    timestep = torch.full((1, num_latent_frames),0.5, device=device, dtype=torch.bfloat16) # midpoint
    noisy_latents = scheduler.add_noise(latents, noise, timestep_scalar)

    conditional_dict = {
        "cond_concat": cond_concat,
        "visual_context": visual_context,
        "keyboard_cond": keyboard,
        "mouse_cond": mouse,
    }
    # Forward pass
    pred = model(
        noisy_image_or_video=noisy_latents,
        timestep = timestep,
        conditional_dict = conditional_dict,
    )
    print(f'model output: {len(pred)} tensors, shapes {[p.shape for p in pred]}')
    print("Forward pass successful.")

    # Decode the pred_x0 back to video frames.
    flow_pred, pred_x0 = pred
    print(f'flow_pred shape: {flow_pred.shape}, pred_x0 shape: {pred_x0.shape}')

    # Decode the latents to frames
    decoded = vae.decode(pred_x0.to(torch.float16), device=device)

    # Save first decoded frame
    pred_frame = ((decoded[0, :, 0] + 1) * 127.5).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(pred_frame).save("pred_test_frame.png")
    print("Saved pred_test_frame.png - check it looks correct")



def overfit_test(model, vae, dataloader, scheduler, device):
    """Train on single batch - loss should go to near 0"""
    print("\n=== Overfit Test ===")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # higher LR
    batch = next(iter(dataloader))
    
    for step in range(100):
        loss = train_step(model, vae, batch, scheduler, optimizer, device)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
    
    print(f"Final loss: {loss:.6f}")
    if loss < 0.1:
        print("PASS: Loss decreased significantly - training is working")
    else:
        print("FAIL: Loss did not decrease enough - check training loop")
    return loss

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  dataset = SimpleDataset(data_dir="/media/kristofe/eight/data", sequence_length=9)
  dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
  print("\nLoading model...")
  model, vae = load_model(device)
  print("Model loaded.")

  print("Creating scheduler...")
  from utils.scheduler import FlowMatchScheduler
  scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)

  '''
  # Run overfit test first
  overfit_test(model, vae, dataloader, scheduler, device)
  # Reload fresh model for actual training
  model, vae = load_model(device)
  '''

  print("\n=== Starting training loop ===")
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  writer = SummaryWriter(log_dir=f"logs/finetune_simple_{timestamp}")
  # initialize tqdm progress bar
  total_steps = 50
  pbar = tqdm.tqdm(enumerate(dataloader), total=total_steps)  # total steps for demo
  # run training loop with tqdm progress bar
  for step, batch in pbar:
      loss = train_step(model, vae, batch, scheduler, optimizer, device)
      #update progress bar
      pbar.set_description(f"Step {step}, Loss: {loss:.6f}")
      writer.add_scalar("Loss/train", loss, step)
      if step >= total_steps:  # Just run a few steps for demo
          break
  writer.close()

  print("training loop complete.")

if __name__ == "__main__":
  main()

