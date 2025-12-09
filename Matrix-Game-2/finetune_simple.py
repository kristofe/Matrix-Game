"""
Simple finetuning script - built from scratch to understand every piece.
"""

from datetime import datetime
import torch
import os

# Memory optimization for CUDA - reduces fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import glob
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from safetensors.torch import load_file
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import FlowMatchScheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.visualize import process_video
import lpips
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# ============================================================================
# DDP (Distributed Data Parallel) IMPORTS
# These are needed for multi-GPU training
# ============================================================================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ============================================================================
# DDP HELPER FUNCTIONS
# ============================================================================
def setup_distributed():
    """
    Initialize the distributed process group for multi-GPU training.

    This function:
    1. Checks if we're running in distributed mode (via torchrun)
    2. Initializes the NCCL backend (optimized for NVIDIA GPUs)
    3. Returns the local rank (which GPU this process should use)

    Returns:
        rank: The global rank of this process (0, 1, 2, ...)
        local_rank: Which GPU on this machine to use
        world_size: Total number of GPUs across all machines
    """
    # Check if running with torchrun (sets these env vars automatically)
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Initialize process group with NCCL backend (fastest for NVIDIA GPUs)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        # Set the GPU for this process
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        # Single GPU mode - no distributed training
        return 0, 0, 1


def cleanup_distributed():
    """
    Clean up the distributed process group.
    Call this at the end of training to properly shut down.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """
    Check if this is the main process (rank 0).

    Only the main process should:
    - Print progress
    - Save checkpoints
    - Log to TensorBoard
    - Generate videos

    This prevents duplicate outputs and file conflicts.
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True

class SimpleDataset(Dataset):
  def __init__(self, data_dir, sequence_length=9, max_sequences=-1):
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
      #only take max_sequences if specified
      if max_sequences > 0:
          self.sequences = self.sequences[:max_sequences]
      print(f"subsampled sequences: {len(self.sequences)}")

  def _load_csv(self, csv_path):
      """Load steering/throttle per frame"""
      data = {}
      debug_counter = 0
      with open(csv_path, 'r') as f:
          for row in csv.reader(f):
              action, time, frame_num, value = row[0], float(row[1]), int(row[2]), float(row[3])
              if frame_num not in data:
                  data[frame_num] = {'steering': 0.0, 'throttle': 0.0}
              #debug_counter += 1
              #if debug_counter < 3:
              #    print(f"action {action} frame_num {frame_num} value {value}")
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
    model.eval()

    #load vae
    from wan.vae.wanx_vae import get_wanx_vae_wrapper
    vae = get_wanx_vae_wrapper("models/", torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device, dtype=torch.float16)


    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.requires_grad_(False)
    lpips_fn.eval()
    return model, vae, lpips_fn

def get_sequence_config(latent_frames, a100=False):
    """
    Calculate video frames and recommended batch size for a given number of latent frames.
    
    Formula: video_frames = 1 + 4 * (latent_frames - 1)
    
    Common configs:
      latent_frames=3  → video_frames=9   
      latent_frames=5  → video_frames=17
      latent_frames=9  → video_frames=33
      latent_frames=13 → video_frames=49
      latent_frames=21 → video_frames=81  (3.2s at 25fps)
    """
    video_frames = 1 + 4 * (latent_frames - 1)
    
    # Memory scales roughly linearly with latent frames
    # Base: latent_frames=3 uses ~27GB per sample at inference
    # Training uses more due to gradients
    memory_estimates = {
        3: {"batch_size": 3, "grad_accum": 4},   # ~80GB total
        5: {"batch_size": 2, "grad_accum": 6},   # ~90GB total
        9: {"batch_size": 1, "grad_accum": 12},  # ~80GB total
        13: {"batch_size": 1, "grad_accum": 12}, # ~95GB total (might be tight)
        21: {"batch_size": 1, "grad_accum": 12}, # Won't fit in 96GB
    }
    
    config = memory_estimates.get(latent_frames, {"batch_size": 1, "grad_accum": 8})

    if a100:
        # A100 40GB is more constrained - use smaller batch and more accumulation
        config["batch_size"] = 1
        config["grad_accum"] = 8  # Increased to compensate for batch_size=1
    
    return {
        "latent_frames": latent_frames,
        "video_frames": video_frames,
        "batch_size": config["batch_size"],
        "grad_accum_steps": config["grad_accum"],
        "effective_batch_size": config["batch_size"] * config["grad_accum"],
    }

def train_step(model, vae, lpips_fn, batch, scheduler, accumulation_steps, device, lpips_weight=0.6):
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

    #TODO: CONFIRM THIS IS CORRECT
    #mask cond: 1 = known, 0 = to generate (matching inference.py)
    mask_cond = torch.zeros_like(latents[:,:4])
    mask_cond[:,:, 0] = 1  # first frame known/conditional
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
    flow_loss = torch.nn.functional.mse_loss(flow_pred, target)

    # LPIPS loss on reconstructed frames (detached - no gradients through VAE/LPIPS)
    # Detach pred_x0 to prevent gradient flow through LPIPS - saves significant memory
    with torch.no_grad():
        pred_x0_detached = pred_x0.detach()
        pred_frames = vae.decode(pred_x0_detached.to(torch.float16), device=device)
        
        # LPIPS expects [B, C, H, W] so we flatten time dimension
        B, C, T, H, W = frames.shape
        pred_flat = pred_frames.reshape(B * T, C, H, W)
        gt_flat = frames.reshape(B * T, C, H, W)
        lpips_loss = lpips_fn(pred_flat, gt_flat).mean()

    # Combined loss - LPIPS provides guidance signal but gradients only flow through flow_loss
    loss = flow_loss + lpips_weight * lpips_loss
    loss = loss / accumulation_steps  # normalize for gradient accumulation

    # Free memory before backward pass
    del frames, latents, noise, noisy_latents, flow_pred, cond_concat, conditional_dict
    del pred_frames, pred_flat, gt_flat
    torch.cuda.empty_cache()

    loss.backward()

    #clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    return loss.item(), flow_loss.item(), lpips_loss.item(), pred_x0_detached 

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

def generate_video(model, vae, initial_frame, keyboard_actions, mouse_actions, device):
    '''
    Generate a 3-second video (90 frames = 23 latent frames) given an initial frame and actions.

    Args:
        model: the trained diffusion model
        vae: the VAE for encoding/decoding
        scheduler: the diffusion scheduler
        initial_frame: [1, C, H, W] tensor, initial frame in [-1, 1]
        keyboard_actions: [num_action_steps, 2] tensor (forward, back)
        mouse_actions: [num_action_steps,  2] tensor, (vertical, horizontal/steering) scaled by 0.1
        device: torch device
        num_inference_steps: number of diffusion steps
    '''
    from einops import rearrange
    from pipeline import CausalInferencePipeline
    from demo_utils.vae_block3 import VAEDecoderWrapper

    num_output_frames = 21  # latent frames = 81 video frames (~3.2s at 25fps)
    num_action_steps = 1 + 4 * (num_output_frames - 1)  # 81

    #process initial frame if needed
    if not isinstance(initial_frame, torch.Tensor):
        # PIL image
        from torchvision.transforms import v2
        transform = v2.Compose([
            v2.Resize((352, 640),antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        initial_frame = transform(initial_frame).unsqueeze(0)  # [1, C, H, W]
    elif initial_frame.ndim == 3:
        # [H, W, C] tensor from dataset - already in [-1, 1]
        initial_frame = initial_frame.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

    weight_dtype = torch.bfloat16

    # Build VAE decoder for pipeline
    vae_decoder = VAEDecoderWrapper()
    vae_state_dict = torch.load("models/Wan2.1_VAE.pth", map_location="cpu")
    decoder_state_dict = {}
    for key, value in vae_state_dict.items():
        if 'decoder.' in key or 'conv2' in key:
            decoder_state_dict[key] = value
    vae_decoder.load_state_dict(decoder_state_dict)
    vae_decoder.to(device, torch.float16)
    vae_decoder.requires_grad_(False)
    vae_decoder.eval()

    # Load config and build pipeline
    config = OmegaConf.load("configs/inference_yaml/inference_gta_drive.yaml")
    pipeline = CausalInferencePipeline(config, generator=model, vae_decoder=vae_decoder)
    pipeline = pipeline.to(device=device, dtype=weight_dtype)
    pipeline.vae_decoder.to(torch.float16)

    # Match inference.py exactly: image is [1, C, 1, H, W] in weight_dtype
    initial_frame = initial_frame.to(device=device, dtype=weight_dtype)
    image = initial_frame.unsqueeze(2)  # [1, C, 1, H, W]

    # Padding matches image dtype (like zeros_like in inference.py)
    padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (num_output_frames - 1), 1, 1)
    img_cond_input = torch.cat([image, padding_video], dim=2)

    with torch.no_grad():
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = vae.encode(img_cond_input.to(torch.float16), device=device, **tiler_kwargs).to(device)

    # Build conditioning like inference.py
    mask_cond = torch.ones_like(img_cond)
    mask_cond[:, :, 1:] = 0
    cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
    visual_context = vae.clip.encode_video(image.to(torch.float16))

    # Prepare noise and actions
    sampled_noise = torch.randn(1, 16, num_output_frames, 44, 80, device=device, dtype=weight_dtype)

    keyboard = keyboard_actions[:num_action_steps].unsqueeze(0).to(device, dtype=weight_dtype)
    mouse = mouse_actions[:num_action_steps].unsqueeze(0).to(device, dtype=weight_dtype)

    conditional_dict = {
        "cond_concat": cond_concat.to(device=device, dtype=weight_dtype),
        "visual_context": visual_context.to(device=device, dtype=weight_dtype),
        "keyboard_cond": keyboard,
        "mouse_cond": mouse,
    }

    # Run inference using the proper pipeline
    with torch.no_grad():
        videos = pipeline.inference(
            noise=sampled_noise,
            conditional_dict=conditional_dict,
            return_latents=False,
            mode='gta_drive',
            profile=False
        )

    # Convert output
    videos_tensor = torch.cat(videos, dim=1)
    videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
    video = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
    video = np.ascontiguousarray(video)

    return video

def generate_video_file(model, vae, initial_frame, device, path="output.mp4"):
    from PIL import Image
    import numpy as np

    if initial_frame is None:
        initial_frame = Image.open("demo_images/gta/0000.png")

    # Create constant actions (driving forward)
    num_action_steps = 89
    keyboard = torch.zeros(num_action_steps, 2)
    keyboard[:, 0] = 1  # forward

    # steer in a sloted sine wave
    steer_amplitude = 0.05
    steer_frequency = 2 * torch.pi / num_action_steps * 2  # 2 full waves over the video
    steer_values = steer_amplitude * torch.sin(torch.linspace(0, steer_frequency * num_action_steps, num_action_steps))
    mouse = torch.zeros(num_action_steps, 2)
    mouse[:, 1] = steer_values  # horizontal steering

    # Generate
    video = generate_video(model, vae, initial_frame, keyboard, mouse, device)

    # Save
    import torchvision.io
    video_tensor = torch.from_numpy(video)  # [T, H, W, C]
    torchvision.io.write_video(path, video_tensor, fps=25)
    no_icons_video = video.copy()

    # Build config tuple (keyboard_actions, mouse_actions)
    config = (
        keyboard.cpu().numpy(),  # [num_action_steps, 2]
        mouse.cpu().numpy()      # [num_action_steps, 2]
    )

    #modify path to include "_with_icons"
    base, ext = os.path.splitext(path)
    icons_path = f"{base}_with_icons{ext}"

    process_video(
        video,                    # [T, H, W, C] uint8
        icons_path,                     
        config,                   # tuple of (keyboard, mouse) arrays
        'assets/images/mouse.png',
        mouse_scale=0.1,
        process_icon=True,        # or False to skip icons
        mode='gta_drive'
    )

    return no_icons_video, video

def test_generate_video(img, device):
    from PIL import Image
    import numpy as np

    # Load finetuned model
    model, vae, lpips_fn = load_model(device)

    # Create constant actions (driving forward)
    num_action_steps = 89
    keyboard = torch.zeros(num_action_steps, 2)
    keyboard[:, 0] = 1  # forward

    #mouse = torch.zeros(num_action_steps, 2)  # straight
    # steer in a sloted sine wave
    steer_amplitude = 0.05
    steer_frequency = 2 * torch.pi / num_action_steps * 2  # 2 full waves over the video
    steer_values = steer_amplitude * torch.sin(torch.linspace(0, steer_frequency * num_action_steps, num_action_steps))
    mouse = torch.zeros(num_action_steps, 2)
    mouse[:, 1] = steer_values  # horizontal steering


    # Generate
    video = generate_video(model, vae, img, keyboard, mouse, device)

    # Save - video is [T, H, W, C] uint8
    import torchvision.io
    video_tensor = torch.from_numpy(video)  # [T, H, W, C]
    torchvision.io.write_video("output.mp4", video_tensor, fps=25)

    # Build config tuple (keyboard_actions, mouse_actions)
    config = (
        keyboard.cpu().numpy(),  # [num_action_steps, 2]
        mouse.cpu().numpy()      # [num_action_steps, 2]
    )

    process_video(
        video,                    # [T, H, W, C] uint8
        "output_with_icons.mp4",                     
        config,                   # tuple of (keyboard, mouse) arrays
        'assets/images/mouse.png',
        mouse_scale=0.1,
        process_icon=True,        # or False to skip icons
        mode='gta_drive'
    )
    return video_tensor


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  num_epochs = 5
  max_sequences = 1000  # limit dataset size for quick testing
  # === SEQUENCE LENGTH CONFIG ===
  # Choose latent frames: 3, 5, 9, 13 (higher = better temporal consistency but more memory)
  # A100 40GB: use latent_frames=3 (video_frames=9). latent_frames=5 causes OOM during backward.
  latent_frames = 3  # Reduced for A100 40GB memory constraints
  seq_config = get_sequence_config(latent_frames, a100=False)
  
  print(f"\n=== Sequence Config ===")
  print(f"Latent frames: {seq_config['latent_frames']}")
  print(f"Video frames: {seq_config['video_frames']}")
  print(f"Batch size: {seq_config['batch_size']}")
  print(f"Grad accum: {seq_config['grad_accum_steps']}")
  print(f"Effective batch: {seq_config['effective_batch_size']}")
  
  batch_size = seq_config['batch_size']
  sequence_length = seq_config['video_frames']
  grad_accum_steps = seq_config['grad_accum_steps']

  dataset = SimpleDataset(data_dir="/media/kristofe/eight/data/", sequence_length=sequence_length, max_sequences=max_sequences)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  print("\nLoading model...")
  model, vae, lpips_fn = load_model(device)
  print("Model loaded.")


  print("Creating scheduler...")
  scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)

  '''
  # Run overfit test first
  overfit_test(model, vae, dataloader, scheduler, device)
  # Reload fresh model for actual training
  model, vae = load_model(device)
  '''

  # creating output folder
  timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
  output_dir = f"outputs/finetune_simple_{timestamp}"
  os.makedirs(output_dir, exist_ok=True)    
  print(f"Outputs will be saved to: {output_dir}")

  #freeze action modules - preserve learned action-video mapping
  frozen_count = 0
  frozen_modules = ['action_model']
  for name, param in model.named_parameters():
      if any(fm in name for fm in frozen_modules):
          param.requires_grad = False
          frozen_count += param.numel()

  trainable_params = [p for p in model.parameters() if p.requires_grad]
  trainable_count = sum(p.numel() for p in trainable_params)
  total_count = sum(p.numel() for p in model.parameters())
  print(f"Froze {frozen_count} parameters in action modules.")
  print(f"Trainable {{trainable_count}} parameters.")
  print(f"Trainable parameters: {trainable_count} / {total_count} ({100.0 * trainable_count / total_count:.2f}%)")

  print("\n=== Starting training loop ===")
  model.train()
  lr = 5e-6
  optimizer = torch.optim.AdamW(trainable_params, lr=lr)

  # Learning rate scheduler with warmup
  warmup_steps = 300 # 1000
  grad_accum_steps = 4
  total_steps = num_epochs * len(dataloader)
  warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
  cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
  lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

  initial_frame = dataloader.dataset[0]['video_frames'][0]  # first frame of first sequence [H, W, C]
  #test_generate_video(initial_frame, device)
  generate_video_file(model, vae, initial_frame, device, path=f"{output_dir}/initial_output.mp4")

  # initialize tqdm progress bar
  reduced_steps = -1
  curr_step = 0
  hyperparams = {
    "lr": lr, 
    "batch_size": batch_size, 
    "sequence_length": sequence_length,
    "latent_frames": latent_frames,
    "epochs": num_epochs, 
    "warmup_steps": warmup_steps, 
    "grad_accum_steps": grad_accum_steps,
    "effective_batch_size": batch_size * grad_accum_steps,
    "frozen_modules": '_'.join(frozen_modules) if frozen_modules else 'none',
  }
  run_name = f"lr={hyperparams['lr']}_bs={hyperparams['batch_size']}_ep={hyperparams['epochs']}_sl={hyperparams['sequence_length']}_ga={hyperparams['grad_accum_steps']}_ts={timestamp}_ws={hyperparams['warmup_steps']}_fm={'_'.join(hyperparams['frozen_modules'])}"
  writer = SummaryWriter(log_dir=f"logs/{run_name}")
  for epoch in range(num_epochs):
    # min of len(dataloader) and total_steps   
    prog_bar_steps = len(dataloader) if total_steps < 0 else min(len(dataloader), total_steps)
    pbar = tqdm(enumerate(dataloader), total=prog_bar_steps)  # total steps for demo    
    # run training loop with tqdm progress bar
    for step, batch in pbar:
        curr_step += 1
        loss, flow_loss, lpips_loss, pred_x0 = train_step(model, vae, lpips_fn, batch, scheduler, grad_accum_steps, device)

        if step % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        #update progress bar
        pbar.set_description(f"Ep {epoch}-{step}, flow_loss: {flow_loss:.6f}, lpips_loss: {lpips_loss:.6f}")
        writer.add_scalar("Loss/train", loss, curr_step)
        writer.add_scalar("Flow Loss/train", flow_loss, curr_step)
        writer.add_scalar("LPIPS Loss/train", lpips_loss, curr_step)
        writer.add_scalar("StdDev Batch", pred_x0.std().item(), curr_step)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], curr_step)
        if reduced_steps > 0 and step >= reduced_steps:  # Just run a few steps for demo
            break
    
    #generate a video at the end of each epoch
    # get a random initial squence from the dataloader
    initial_frame = dataloader.dataset[ np.random.randint(0, len(dataloader.dataset))]['video_frames'][0]  # first frame of 9 frame random sequence [H, W, C]
    vid, icons_vid = generate_video_file(model, vae, initial_frame, device, path=f"{output_dir}/output_e{epoch}.mp4")
    # extract first middle and last frames for tensorboard
    mid_frame = vid[vid.shape[0] // 2]
    first_frame = vid[0]
    last_frame = vid[-1]
    # add to tensorboard as a image grid.  Use torchvision.utils.make_grid
    import torchvision.utils as vutils  
    grid = vutils.make_grid(torch.from_numpy(np.stack([first_frame, mid_frame, last_frame])).permute(0, 3, 1, 2).float() / 255.0, nrow=3)
    writer.add_image(f"Generated Video Frames", grid, epoch)
    #save grid as png
    vutils.save_image(grid, f"{output_dir}/generated_frames_epoch{epoch}.png")

    # save checkpoint
    checkpoint_frequency = 1  # save every epoch
    if (epoch + 1) % checkpoint_frequency == 0:
        checkpoint_path = f"{output_dir}/finetuned_model_epoch{epoch}.safetensors"
        os.makedirs("checkpoints", exist_ok=True)
        from safetensors.torch import save_file
        save_file(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

  # final metrics for comparison against other runs.
  writer.add_hparams(hparam_dict=hyperparams,metric_dict={
    "final_loss": loss,
    "final_flow_loss": flow_loss,
    "final_lpips_loss": lpips_loss,
  })
  writer.close()

  print("training loop complete.")
  initial_frame = dataloader.dataset[0]['video_frames'][0]  # first frame of first sequence [H, W, C]
  generate_video_file(model, vae, initial_frame, device, path=f"{output_dir}/final_output.mp4")
if __name__ == "__main__":
  main()

