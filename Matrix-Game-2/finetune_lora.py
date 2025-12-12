"""
LoRA finetuning script with DDP support for multi-GPU training.

This script uses Low-Rank Adaptation (LoRA) to efficiently finetune the model
on new domains while preserving the base model's capabilities.

To run on multiple GPUs:
    torchrun --nproc_per_node=NUM_GPUS finetune_lora.py

Example with 2 GPUs:
    torchrun --nproc_per_node=2 finetune_lora.py

For single GPU:
    python finetune_lora.py


######################
# OVERFITTING MODE
######################

# Train on 7 consecutive 3-frame blocks (= 21 latent frames worth of video)
python finetune_lora.py --latent_frames 3 --overfit --overfit_blocks 7

# Train on 23 consecutive blocks (= 69 latent frames worth of data)
python finetune_lora.py --latent_frames 3 --overfit --overfit_blocks 23

LoRA-specific arguments:
    --lora_rank: Rank of LoRA matrices (default: 64)
    --lora_alpha: Scaling factor (default: 128)
    --lora_dropout: Dropout for LoRA layers (default: 0.0)
    --lora_checkpoint: Path to LoRA weights for resuming training
"""

from datetime import datetime
import argparse
import math
import torch
import torch.nn as nn
import os

#bugfix on wsl linux
if torch.cuda.is_available():
    torch.cuda.init()
# Memory optimization for CUDA - reduces fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import glob
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import FlowMatchScheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.visualize import process_video
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def parse_args():
    parser = argparse.ArgumentParser(description="DDP-enabled finetuning for Matrix-Game-2")

    # Data settings
    parser.add_argument("--data_dir", type=str, 
                        #default="/mnt/d/data_640_360_300_sessions",
                        default="/media/kristofe/eight/data",
                        #default="/mnt/s3/uedata",
                        help="Path to training data directory")
    parser.add_argument("--max_sequences", type=int, default=-1,
                        help="Max sequences to use from dataset (-1 for all)")
    parser.add_argument("--overfit", action="store_true",
                        help="Overfit on single sequence (debug mode - repeats first sequence 100 times)")
    parser.add_argument("--overfit_blocks", type=int, default=0,
                        help="Overfit mode: create N consecutive 3-frame blocks from first run (0=disabled, requires --overfit)")

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1,
                        help="Number of warmup steps for LR scheduler")

    # LoRA settings
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=512,
                        help="Scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout for LoRA layers")
    parser.add_argument("--lora_checkpoint", type=str, default="",
                        help="Path to LoRA weights for resuming training")
    parser.add_argument("--lora_targets", type=str, default="all",
                        help="Which layers to inject LoRA into: 'all', 'cross_attn', 'self_attn', 'ffn', or comma-separated combo (e.g., 'cross_attn,ffn')")

    # Sequence/memory settings
    parser.add_argument("--latent_frames", type=int, default=3,
                        help="Number of latent frames. Must be divisible by 3 (num_frame_per_block). Valid: 3,6,9,12,15,18,21. video_frames = 1 + 4*(latent_frames-1)")
    parser.add_argument("--gpu", type=str, default="rtx6000", choices=["a100", "rtx6000"],
                        help="GPU type for memory config")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves ~30-50%% memory, ~20-30%% slower)")

    # Batch size overrides (optional - auto-configured if not specified)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (default: auto from latent_frames)")
    parser.add_argument("--grad_accum_steps", type=int, default=None,
                        help="Override gradient accumulation steps (default: auto)")

    # Logging/output settings
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/finetune_lora_<timestamp>)")
    parser.add_argument("--write_video_interval", type=int, default=100,
                        help="Generate video every N steps")
    parser.add_argument("--checkpoint_frequency", type=int, default=1,
                        help="Save checkpoint every N epochs")

    # Model settings
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


# ============================================================================
# LoRA Implementation
# ============================================================================
class LoRALayer(nn.Module):
    """
    Low_Rank Adaptation layer that wraps a linear layer.
    
    Instead of updating W directly, we learn W + BA where:
    - B is a (out_features x rank) matrix
    - A is a (rank x in_features) matrix
    """
    def __init__(self, original_layer, rank=64, alpha=128, dropout=0.0):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA Matrices
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A and B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        #freeze original layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return original_out + lora_out
    
def inject_lora_layers(model, rank: int = 64, alpha: int = 128, dropout: float = 0.0, targets: str = "all"):
    """
    Inject LoRA layers into the model's linear layers.

    Returns:
        lora_params: List of LoRA parameters for optimization.
        num_lora_params: Total number of LoRA parameters added.
        lora_layers: list of LoRALayer instances added to the model.
    Args:
        model: The model to modify
        rank: Rank of LoRA matrices
        alpha: Scaling factor
        dropout: Dropout probability for LoRA layers
        targets: Which layers to inject - 'all', 'cross_attn', 'self_attn', 'ffn', or comma-separated combo
    """
    lora_layers = []

    # Parse targets
    if targets == "all":
        inject_self_attn = True
        inject_cross_attn = True
        inject_ffn = True
    else:
        target_list = [t.strip() for t in targets.split(",")]
        inject_self_attn = "self_attn" in target_list
        inject_cross_attn = "cross_attn" in target_list
        inject_ffn = "ffn" in target_list

    print(f"LoRA targets: self_attn={inject_self_attn}, cross_attn={inject_cross_attn}, ffn={inject_ffn}")

    # Get the inner WanModel
    wan_model = model.model if hasattr(model, 'model') else model

    for block_idx, block in enumerate(wan_model.blocks):
        # Self-attention Q, K, V, O
        if inject_self_attn:
            for layer_name in ['q', 'k', 'v', 'o']:
                original = getattr(block.self_attn, layer_name)
                lora_layer = LoRALayer(original, rank=rank, alpha=alpha, dropout=dropout)
                setattr(block.self_attn, layer_name, lora_layer)
                lora_layers.append(lora_layer)

        # cross-attention Q, K, V, O
        if inject_cross_attn:
            for layer_name in ['q', 'k', 'v', 'o']:
                original = getattr(block.cross_attn, layer_name)
                lora_layer = LoRALayer(original, rank=rank, alpha=alpha, dropout=dropout)
                setattr(block.cross_attn, layer_name, lora_layer)
                lora_layers.append(lora_layer)

        # FFN layers (index 0 and 2 in the sequential)
        if inject_ffn:
            for ffn_idx in [0, 2]:
                original = block.ffn[ffn_idx]
                lora_layer = LoRALayer(original, rank=rank, alpha=alpha, dropout=dropout)
                block.ffn[ffn_idx] = lora_layer
                lora_layers.append(lora_layer)

    # Collect LoRA parameters
    lora_params = []
    num_lora_params = 0
    for layer in lora_layers:
        lora_params.extend([layer.lora_A, layer.lora_B])
        num_lora_params += layer.lora_A.numel() + layer.lora_B.numel()
    print(f"Injected LoRA layers into {len(lora_layers)} layers")
    print(f"Total LoRA parameters: {num_lora_params / 1e6:.2f} million")

    return lora_params, num_lora_params, lora_layers

def get_lora_state_dict(model):
    """
    Extract only the LoRA parameters from the model for saving.
    """
    lora_state_dict  = {}

    # Handle DDP wrapper
    if hasattr(model, 'module'):
        wan_model = model.module.model # DDP -> WanDiffusionWrapper -> WanModel
    elif hasattr(model, 'model'):
        wan_model = model.model # WanDiffusionWrapper -> WanModel
    else:
        wan_model = model
    
    for block_idx, block in enumerate(wan_model.blocks):
        # self-attention
        for layer_name in ['q', 'k', 'v', 'o']:
            layer = getattr(block.self_attn, layer_name)
            if isinstance(layer, LoRALayer):
                lora_state_dict[f'blocks.{block_idx}.self_attn.{layer_name}.lora_A'] = layer.lora_A.data
                lora_state_dict[f'blocks.{block_idx}.self_attn.{layer_name}.lora_B'] = layer.lora_B.data
        
        # cross-attention
        for layer_name in ['q', 'k', 'v', 'o']:
            layer = getattr(block.cross_attn, layer_name)
            if isinstance(layer, LoRALayer):
                lora_state_dict[f'blocks.{block_idx}.cross_attn.{layer_name}.lora_A'] = layer.lora_A.data
                lora_state_dict[f'blocks.{block_idx}.cross_attn.{layer_name}.lora_B'] = layer.lora_B.data

        # FFN layers
        for ffn_idx in [0, 2]:
            layer = block.ffn[ffn_idx]
            if isinstance(layer, LoRALayer):
                lora_state_dict[f'blocks.{block_idx}.ffn.{ffn_idx}.lora_A'] = layer.lora_A.data
                lora_state_dict[f'blocks.{block_idx}.ffn.{ffn_idx}.lora_B'] = layer.lora_B.data
        
    return lora_state_dict

def load_lora_state_dict(model, lora_path):
    """
    Load LoRA parameters from a checkpoint into the model.
    """
    lora_state_dict = load_file(lora_path)

    # Handle DDP wrapper
    if hasattr(model, 'module'):
        wan_model = model.module.model # DDP -> WanDiffusionWrapper -> WanModel
    elif hasattr(model, 'model'):
        wan_model = model.model # WanDiffusionWrapper -> WanModel
    else:
        wan_model = model
    
    for key, value in lora_state_dict.items():
        parts = key.split('.')
        block_idx = int(parts[1])
        block = wan_model.blocks[block_idx]

        if parts[2] == 'self_attn':
            layer = getattr(block.self_attn, parts[3])
        elif parts[2] == 'cross_attn':
            layer = getattr(block.cross_attn, parts[3])
        elif parts[2] == 'ffn':
            layer = block.ffn[int(parts[3])]
        
        if isinstance(layer, LoRALayer):
            param_name = parts[-1] # lora_A or lora_B
            if param_name == 'lora_A':
                layer.lora_A.data = value.to(layer.lora_A.device, dtype=layer.lora_A.dtype)
            elif param_name == 'lora_B':
                layer.lora_B.data = value.to(layer.lora_B.device, dtype=layer.lora_B.dtype)

    print(f"Loaded LoRA weights from {lora_path}")



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
  def __init__(self, data_dir, sequence_length=9, max_sequences=-1, overfit=False, overfit_blocks=0):
      self.sequence_length = sequence_length
      self.overfit = overfit
      self.overfit_blocks = overfit_blocks

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

      # Overfit mode: create consecutive blocks from first run
      if self.overfit and len(self.sequences) > 0:
          if self.overfit_blocks > 0:
              # Get first run info
              first_run_idx = self.sequences[0][0]
              first_run = self.runs[first_run_idx]
              num_frames_available = len(first_run['frames'])

              # Create consecutive non-overlapping blocks
              # Each block is sequence_length video frames (9 frames = 3 latent frames)
              # Blocks start at frame 0, 8, 16, 24... (stride = sequence_length - 1 = 8)
              stride = self.sequence_length - 1  # 8 for 9-frame sequences
              consecutive_sequences = []
              for block_idx in range(self.overfit_blocks):
                  start_frame = block_idx * stride
                  if start_frame + self.sequence_length <= num_frames_available:
                      consecutive_sequences.append((first_run_idx, start_frame))

              if len(consecutive_sequences) < self.overfit_blocks:
                  print(f"WARNING: Only {len(consecutive_sequences)} blocks available (requested {self.overfit_blocks})")
                  print(f"  Need {self.overfit_blocks * stride + 1} frames, have {num_frames_available}")

              # Repeat these consecutive blocks 100 times for overfitting
              self.sequences = consecutive_sequences * 100
              print(f"OVERFIT MODE: {len(consecutive_sequences)} consecutive blocks, repeated 100x = {len(self.sequences)} sequences")
          else:
              # Original overfit: just repeat first sequence
              self.sequences = [self.sequences[0]] * 100
              print(f"OVERFIT MODE: repeating first sequence 100 times")

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
        throttle * 0.5,  # forward - use actual throttle value (0-1) instead of binary
        brake      # back - use actual brake value (0-1) instead of binary
      ]
      mouse = [
          0.0,      # vertical (not used for driving)
          steering * 0.05 # horizontal (steering = camera rotation), reduced from 0.1
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

def visualize_training_sequence(dataset, output_path="sequence_grid.png", thumb_size=64, overfit_blocks=0):
    """
    Visualize training sequence(s) as a grid image and video.

    In regular mode: visualizes first sample from dataset.
    In overfit_blocks mode: chains consecutive blocks together to show full sequence.

    Args:
        dataset: SimpleDataset instance
        output_path: Path to save the grid image (video saved with same base name + .mp4)
        thumb_size: Size of each thumbnail (will be thumb_size x thumb_size * aspect_ratio)
        overfit_blocks: Number of consecutive blocks to chain (0 = use single sample)

    Returns:
        PIL Image of the grid
    """
    from PIL import Image
    import math
    import torchvision.io

    # Get frames - either single sample or chained blocks
    if overfit_blocks > 0:
        # Chain consecutive blocks together
        num_blocks_available = len(dataset.sequences) // 100  # sequences repeat 100x
        num_blocks_to_use = min(overfit_blocks, num_blocks_available)

        all_frames = []
        for i in range(num_blocks_to_use):
            sample = dataset[i]
            block_frames = sample['video_frames']  # [T, H, W, C]
            # Remove overlap frame (last frame) except for final block
            if i < num_blocks_to_use - 1:
                block_frames = block_frames[:-1]
            all_frames.append(block_frames)

        frames = torch.cat(all_frames, dim=0)  # [total_T, H, W, C]
        print(f"Chained {num_blocks_to_use} blocks: {len(all_frames)} segments -> {frames.shape[0]} total frames")
    else:
        # Regular mode: single sample
        sample = dataset[0]
        frames = sample['video_frames']  # [T, H, W, C] in [-1, 1]

    T, H, W, C = frames.shape

    # Convert from [-1, 1] to [0, 255]
    frames_uint8 = ((frames + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()

    # === Save as video ===
    base_path = output_path.rsplit('.', 1)[0]  # Remove extension
    video_path = f"{base_path}.mp4"
    video_tensor = torch.from_numpy(frames_uint8)  # [T, H, W, C]
    torchvision.io.write_video(video_path, video_tensor, fps=25)
    print(f"Saved sequence video ({T} frames) to {video_path}")

    # === Create grid image ===
    # Calculate thumbnail dimensions preserving aspect ratio
    aspect = W / H
    thumb_h = thumb_size
    thumb_w = int(thumb_size * aspect)

    # Calculate grid dimensions (roughly square)
    cols = math.ceil(math.sqrt(T * aspect))
    rows = math.ceil(T / cols)

    # Create output image
    grid_w = cols * thumb_w
    grid_h = rows * thumb_h
    grid_img = Image.new('RGB', (grid_w, grid_h), (0, 0, 0))

    for i in range(T):
        row = i // cols
        col = i % cols

        # Convert frame to PIL and resize
        frame_pil = Image.fromarray(frames_uint8[i])
        frame_thumb = frame_pil.resize((thumb_w, thumb_h), Image.LANCZOS)

        # Paste into grid
        grid_img.paste(frame_thumb, (col * thumb_w, row * thumb_h))

    grid_img.save(output_path)
    print(f"Saved sequence grid ({T} frames, {cols}x{rows}) to {output_path}")
    return grid_img


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

def load_model(device, gradient_checkpointing=False):
    #load config
    config = OmegaConf.load("configs/inference_yaml/inference_gta_drive.yaml")
    config.model_kwargs.model_config = "configs/distilled_model/gta_drive"

    #create model
    model = WanDiffusionWrapper(**config.model_kwargs, is_causal=True)

    #load weights
    checkpoint = "models/gta_distilled_model/gta_keyboard2dim.safetensors"
    state_dict = load_file(checkpoint)
    model.load_state_dict(state_dict, strict=False)

    # Enable gradient checkpointing to save ~30-50% activation memory (slower training)
    if gradient_checkpointing:
        model.model.gradient_checkpointing = True
        print("Gradient checkpointing ENABLED - saves memory but ~20-30% slower")

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    #load vae
    from wan.vae.wanx_vae import get_wanx_vae_wrapper
    vae = get_wanx_vae_wrapper("models/", torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device, dtype=torch.float16)

    return model, vae

def get_sequence_config(latent_frames, gpu="rtx6000", gradient_checkpointing=False):
    """
    Config for LoRA Training.

    Args:
        latent_frames: Number of latent frames (video_frames = 1 + 4 * (latent_frames - 1))
        gpu: "a100" (40GB) or "rtx6000" (96GB)
        gradient_checkpointing: If True, uses configs optimized for ~30-50% less memory
    """
    video_frames = 1 + 4 * (latent_frames - 1)

    # IMPORTANT: latent_frames must be divisible by num_frame_per_block (hardcoded to 3)
    # Valid values: 3, 6, 9, 12, 15, 18, 21, 24, ...
    # video_frames = 1 + 4*(latent-1): 3->9, 6->21, 9->33, 12->45, 15->57, 18->69, 21->81
    if latent_frames % 3 != 0:
        print(f"WARNING: latent_frames={latent_frames} is not divisible by 3 (num_frame_per_block).")
        print(f"         This may cause issues during inference. Consider using 3, 6, 9, 12, 15, 18, or 21.")

    if gradient_checkpointing:
        # With gradient checkpointing: ~30-50% memory savings
        # latent_frames must be divisible by 3 (num_frame_per_block)
        configs = {
            "a100": {  # 40GB - checkpointing enables longer sequences
                3:  {"batch_size": 3, "grad_accum": 4},   # ~25GB, video=9 frames
                6:  {"batch_size": 2, "grad_accum": 6},   # ~32GB, video=21 frames
                9:  {"batch_size": 1, "grad_accum": 12},  # ~38GB, video=33 frames
            },
            "rtx6000": {  # 96GB
                3:  {"batch_size": 6, "grad_accum": 2},   # ~25GB, video=9 frames
                6:  {"batch_size": 4, "grad_accum": 3},   # ~42GB, video=21 frames
                9:  {"batch_size": 3, "grad_accum": 4},   # ~55GB, video=33 frames
                12: {"batch_size": 2, "grad_accum": 6},   # ~68GB, video=45 frames
                15: {"batch_size": 1, "grad_accum": 12},  # ~75GB, video=57 frames
                18: {"batch_size": 1, "grad_accum": 12},  # ~88GB, video=69 frames - MAX for 96GB
                # 21+ causes OOM on 96GB even with checkpointing
            },
        }
    else:
        # Without gradient checkpointing (faster but more memory)
        # latent_frames must be divisible by 3 (num_frame_per_block)
        configs = {
            "a100": {  # 40GB
                3:  {"batch_size": 2, "grad_accum": 6},   # ~30GB, video=9 frames
                6:  {"batch_size": 1, "grad_accum": 12},  # ~38GB, video=21 frames
            },
            "rtx6000": {  # 96GB
                3:  {"batch_size": 4, "grad_accum": 3},   # ~35GB, video=9 frames
                6:  {"batch_size": 3, "grad_accum": 4},   # ~55GB, video=21 frames
                9:  {"batch_size": 2, "grad_accum": 6},   # ~75GB, video=33 frames
                12: {"batch_size": 1, "grad_accum": 12},  # ~92GB, video=45 frames (tight fit)
            },
        }

    gpu_configs = configs.get(gpu, configs["a100"])
    config = gpu_configs.get(latent_frames, {"batch_size": 1, "grad_accum": 12})

    return {
        "latent_frames": latent_frames,
        "video_frames": video_frames,
        "batch_size": config["batch_size"],
        "grad_accum_steps": config["grad_accum"],
        "effective_batch_size": config["batch_size"] * config["grad_accum"],
    }

def train_step(model, vae, batch, scheduler, accumulation_steps, device ):
    frames = batch['video_frames'].to(device, dtype=torch.float16)
    frames = frames.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]

    # For DDP, access underlying model with model.module if wrapped
    raw_model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        latents = vae.encode(frames, device=device).to(dtype=torch.bfloat16)
        first_frame = frames[:, :, :1, :, :]  # [B, C, 1, H, W]
        visual_context = vae.clip.encode_video(first_frame).to(device=device, dtype=torch.bfloat16)
        raw_model.model.num_frame_per_block = 3 # its always 3 latents.shape[2]

    # Free frames early - no longer needed after encoding
    del frames

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
    # t_scalar is in [0.05, 0.95] representing sigma interpolation
    # scheduler.timesteps are in [0, num_train_timesteps] range
    # We need to scale t_scalar to match the timesteps range for add_noise
    batch_size = latents.shape[0]
    t_scalar = torch.rand(1, device=device) * 0.9 + 0.05  # [0.05, 0.95] avoid extremes

    # Scale to timestep range for scheduler.add_noise (expects values matching self.timesteps)
    # self.timesteps = self.sigmas * num_train_timesteps, so we scale accordingly
    t_scaled = t_scalar * scheduler.num_train_timesteps  # [50, 950] for 1000 timesteps

    # For model forward pass, use scaled timestep
    t = t_scaled.expand(batch_size)
    timestep = t.unsqueeze(1).expand(batch_size, num_latent_frames).to(dtype=torch.bfloat16)

    #add noise using scaled timestep
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t_scaled)

    #forward pass
    conditional_dict = {
        "cond_concat": cond_concat,
        "visual_context": visual_context,
        "keyboard_cond": keyboard,
        "mouse_cond": mouse,
    }

    # Use autocast for bfloat16 on the forward to save some memory
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        flow_pred, pred_x0 = model(
            noisy_image_or_video=noisy_latents,
            timestep = timestep,
            conditional_dict = conditional_dict,
        )   

        # flow matching loss: predict velocity (latents - noise)
        target = scheduler.training_target(latents, noise, t_scalar)
        flow_loss = torch.nn.functional.mse_loss(flow_pred, target)

    loss = flow_loss
    loss = loss / accumulation_steps  # normalize for gradient accumulation

    # Free memory before backward pass (del allows Python/CUDA to reclaim)
    del latents, noise, noisy_latents, flow_pred, cond_concat, conditional_dict
    del target, visual_context, keyboard, mouse, mask_cond

    loss.backward()

    # Clip gradients only on LoRA parameters (the only ones with gradients)
    lora_params_for_clip = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(lora_params_for_clip, max_norm=1.0)

    return loss.item(), flow_loss.item(),pred_x0 

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
    model.model.num_frame_per_block = 3 # its always 3.  latents.shape[2]

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

def generate_video(model, vae, initial_frame, keyboard_actions, mouse_actions, device, num_output_frames=21):
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
        # VAE is in float16, so input must be float16 for encoding
        img_cond = vae.encode(img_cond_input.to(torch.float16), device=device, **tiler_kwargs).to(device)

    # Build conditioning like inference.py
    mask_cond = torch.ones_like(img_cond)
    mask_cond[:, :, 1:] = 0
    cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
    # CLIP encoder also expects float16
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

def get_chained_block_data(dataset, num_blocks):
    """
    Chain consecutive blocks together for longer video generation.

    When using overfit_blocks mode, the dataset contains consecutive 3-latent-frame
    sequences from the same video. This function chains them together by:
    1. Getting each block's actions
    2. Removing the overlap frame between blocks (stride = sequence_length - 1)
    3. Returning concatenated actions and the initial frame

    Args:
        dataset: SimpleDataset with consecutive block sequences
        num_blocks: Number of blocks to chain together

    Returns:
        dict with:
            - initial_frame: First frame from first block
            - keyboard_actions: Concatenated keyboard actions
            - mouse_actions: Concatenated mouse actions
            - num_output_frames: Total latent frames (3 per block)
    """
    num_blocks_available = len(dataset.sequences) // 100  # sequences repeat 100x
    num_blocks_to_use = min(num_blocks, num_blocks_available)

    all_keyboard = []
    all_mouse = []
    for i in range(num_blocks_to_use):
        sample = dataset[i]  # Gets block i (sequences repeat 100x)
        all_keyboard.append(sample['keyboard_actions'])
        all_mouse.append(sample['mouse_actions'])

    # Concatenate actions (removing overlap frame between blocks)
    # Each block has sequence_length video frames of actions
    # Blocks overlap by 1 frame, so remove last frame except for final block
    keyboard_actions = torch.cat([k[:-1] if i < len(all_keyboard)-1 else k for i, k in enumerate(all_keyboard)])
    mouse_actions = torch.cat([m[:-1] if i < len(all_mouse)-1 else m for i, m in enumerate(all_mouse)])

    # Use first frame from first block
    initial_frame = dataset[0]['video_frames'][0]

    # 3 latent frames per block
    num_output_frames = 3 * num_blocks_to_use

    return {
        'initial_frame': initial_frame,
        'keyboard_actions': keyboard_actions,
        'mouse_actions': mouse_actions,
        'num_output_frames': num_output_frames
    }

def generate_training_video(model, vae, dataset, device, output_path, overfit_blocks=0):
    """
    Generate a video from training data for evaluation.

    Handles both regular mode (single sample) and overfit_blocks mode
    (chained consecutive blocks for longer video).

    Args:
        model: The diffusion model
        vae: VAE encoder/decoder
        dataset: SimpleDataset instance
        device: torch device
        output_path: Path to save the video
        overfit_blocks: Number of consecutive blocks to chain (0 = use single sample)

    Returns:
        vid: Generated video array [T, H, W, C]
        icons_vid: Video with icons overlay
    """
    if overfit_blocks > 0:
        # Chain consecutive blocks for longer video generation
        block_data = get_chained_block_data(dataset, overfit_blocks)
        vid, icons_vid = generate_video_file(
            model, vae, block_data['initial_frame'], device,
            path=output_path,
            keyboard_actions=block_data['keyboard_actions'],
            mouse_actions=block_data['mouse_actions'],
            num_output_frames=block_data['num_output_frames'])
    else:
        # Regular mode: use first sample
        sample = dataset[0]
        initial_frame = sample['video_frames'][0]
        keyboard_actions = sample['keyboard_actions']
        mouse_actions = sample['mouse_actions']
        vid, icons_vid = generate_video_file(
            model, vae, initial_frame, device,
            path=output_path,
            keyboard_actions=keyboard_actions,
            mouse_actions=mouse_actions)

    return vid, icons_vid

def generate_video_file(model, vae, initial_frame, device, path="output.mp4", keyboard_actions=None, mouse_actions=None, num_output_frames=21):
    from PIL import Image
    import numpy as np

    num_action_steps = 1 + 4 * (num_output_frames - 1)  # 81 when num_output_frames=21

    # Use provided actions or create defaults
    if keyboard_actions is not None:
        keyboard = keyboard_actions[:num_action_steps]
        # Pad if needed
        if len(keyboard) < num_action_steps:
            padding = torch.zeros(num_action_steps - len(keyboard), keyboard.shape[-1])
            keyboard = torch.cat([keyboard, padding], dim=0)
    else:
        # Default: driving forward
        keyboard = torch.zeros(num_action_steps, 2)
        keyboard[:, 0] = 1  # forward

    if mouse_actions is not None:
        mouse = mouse_actions[:num_action_steps]
        # Pad if needed
        if len(mouse) < num_action_steps:
            padding = torch.zeros(num_action_steps - len(mouse), mouse.shape[-1])
            mouse = torch.cat([mouse, padding], dim=0)
    else:
        # Default: sine wave steering
        steer_amplitude = 0.05
        steer_frequency = 2 * torch.pi / num_action_steps * 2
        steer_values = steer_amplitude * torch.sin(torch.linspace(0, steer_frequency * num_action_steps, num_action_steps))
        mouse = torch.zeros(num_action_steps, 2)
        mouse[:, 1] = steer_values

    # Generate
    video = generate_video(model, vae, initial_frame, keyboard, mouse, device, num_output_frames=num_output_frames)

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
    model, vae = load_model(device)

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

def pprint(str):
    ''' 
    Print only from main process 
    '''
    if is_main_process():
        print(str)

def main():
  # Parse command line arguments
  args = parse_args()

  # ==========================================================================
  # DDP SETUP - Initialize distributed training
  # ==========================================================================
  # This returns (0, 0, 1) for single GPU, or actual values for multi-GPU
  rank, local_rank, world_size = setup_distributed()

  # Each process uses its assigned GPU
  # local_rank maps to the GPU index on this machine (0, 1, 2, ...)
  device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

  # is_main controls logging/saving - only rank 0 should do these
  is_main = is_main_process()

  pprint(f"Using device: {device}")
  pprint(f"World size (total GPUs): {world_size}")

  # Get sequence config based on latent_frames and GPU type
  seq_config = get_sequence_config(
      args.latent_frames,
      gpu=args.gpu,
      gradient_checkpointing=args.gradient_checkpointing
  )

  # Allow command-line overrides for batch_size and grad_accum_steps
  batch_size = args.batch_size if args.batch_size is not None else seq_config['batch_size']
  grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else seq_config['grad_accum_steps']
  sequence_length = seq_config['video_frames']

  pprint(f"\n=== Training Config ===")
  pprint(f"Data dir: {args.data_dir}")
  pprint(f"Max sequences: {args.max_sequences}")
  pprint(f"Epochs: {args.num_epochs}")
  pprint(f"Learning rate: {args.lr}")
  pprint(f"Warmup steps: {args.warmup_steps}")
  pprint(f"\n=== Sequence Config ===")
  pprint(f"Gradient checkpointing: {args.gradient_checkpointing}")
  pprint(f"Latent frames: {args.latent_frames}")
  pprint(f"Video frames: {sequence_length}")
  pprint(f"Batch size per GPU: {batch_size}")
  pprint(f"Grad accum: {grad_accum_steps}")
  pprint(f"Effective batch (with {world_size} GPUs): {batch_size * grad_accum_steps * world_size}")

  dataset = SimpleDataset(data_dir=args.data_dir, sequence_length=sequence_length, max_sequences=args.max_sequences, overfit=args.overfit, overfit_blocks=args.overfit_blocks)

  # ==========================================================================
  # DDP DATALOADER - Use DistributedSampler for multi-GPU
  # ==========================================================================
  # DistributedSampler automatically splits data across GPUs
  # - GPU 0 gets samples 0, 2, 4, 6, ...
  # - GPU 1 gets samples 1, 3, 5, 7, ...
  # This ensures each GPU sees different data each batch
  if world_size > 1:
      sampler = DistributedSampler(
          dataset,
          num_replicas=world_size,  # Total number of GPUs
          rank=rank,                 # This GPU's rank
          shuffle=True,              # Shuffle within each epoch
      )
      # When using sampler, set shuffle=False in DataLoader (sampler handles it)
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
  else:
      # Single GPU - use regular shuffle
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  if is_main:
    # Visualize training sequence (handles both regular and overfit_blocks modes)
    visualize_training_sequence(dataset, f"{output_dir}/training_sequence.png",
                                 thumb_size=128, overfit_blocks=args.overfit_blocks)

  pprint("\nLoading model...")
  model, vae = load_model(device, gradient_checkpointing=args.gradient_checkpointing)
  pprint("Model loaded.")

  # ==========================================================================
  # LORA INJECTION - Inject LoRA layers into the model, before DDP wrapping
  # ==========================================================================
  # Freeze all parameters first
  for param in model.parameters():
    param.requires_grad = False

  # Inject LoRA layers into the model
  lora_params, num_lora_params, lora_layers = inject_lora_layers(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=args.lora_targets
  )
  if args.lora_checkpoint:
      load_lora_state_dict(model, args.lora_checkpoint)

  total_params = sum(p.numel() for p in model.parameters())
  pprint(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
  pprint(f"LoRA parameters: {num_lora_params:,} / {total_params:,} ({100.0 * num_lora_params / total_params:.2f}%)")

  # ==========================================================================
  # DDP MODEL WRAPPING - Wrap model with DistributedDataParallel
  # ==========================================================================
  # DDP wraps your model and handles:
  # 1. Broadcasting initial weights from rank 0 to all GPUs
  # 2. Averaging gradients across GPUs during backward pass
  # 3. Keeping model weights synchronized
  if world_size > 1:
      # find_unused_parameters=True is needed if some parameters don't receive gradients
      # (like our frozen action_model parameters)
      model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
      pprint(f"Model wrapped with DDP across {world_size} GPUs")

  pprint("Creating scheduler...")
  scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
  # Set timesteps for training (1000 steps for full resolution)
  scheduler.set_timesteps(1000, training=True)

  # creating output folder (only on main process)
  from zoneinfo import ZoneInfo
  timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d-%H_%M_%S")
  output_dir = args.output_dir if args.output_dir else f"outputs/finetune_lora_{timestamp}"
  if is_main:
      os.makedirs(output_dir, exist_ok=True)
  pprint(f"Outputs will be saved to: {output_dir}")

  # Synchronize all processes before continuing
  # This ensures the output directory is created before other processes try to use it
  if world_size > 1:
      dist.barrier()



  pprint("\n=== Starting training loop ===")
  model.train()
  optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

  # Learning rate scheduler with warmup
  total_steps = args.num_epochs * len(dataloader)
  warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
  cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps)
  lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps])

  # Generate initial video only on main process
  if is_main:
      # For DDP, access underlying model with model.module
      raw_model = model.module if world_size > 1 else model
      generate_training_video(raw_model, vae, dataloader.dataset, device,
                              output_path=f"{output_dir}/initial_output.mp4",
                              overfit_blocks=args.overfit_blocks)

  # initialize tqdm progress bar
  reduced_steps = -1
  curr_step = 0
  hyperparams = {
    "lr": args.lr,
    "batch_size": batch_size,
    "sequence_length": sequence_length,
    "latent_frames": args.latent_frames,
    "epochs": args.num_epochs,
    "warmup_steps": args.warmup_steps,
    "grad_accum_steps": grad_accum_steps,
    "effective_batch_size": batch_size * grad_accum_steps * world_size,
    "gradient_checkpointing": args.gradient_checkpointing,
    "lora_rank": args.lora_rank,
    "lora_alpha": args.lora_alpha,
    "lora_dropout": args.lora_dropout,
    "lora_targets": args.lora_targets,
    "world_size": world_size,
  }

  # TensorBoard writer only on main process
  writer = None
  if is_main:
      run_name = f"{timestamp}_lr={args.lr}_bs={batch_size}_ep={args.num_epochs}_lf={args.latent_frames}_ga={grad_accum_steps}_ws={args.warmup_steps}_gpus={world_size}"
      writer = SummaryWriter(log_dir=f"logs/{run_name}")

  for epoch in range(args.num_epochs):
    # ==========================================================================
    # DDP EPOCH SYNC - Set epoch for DistributedSampler
    # ==========================================================================
    # This ensures different shuffling each epoch across all GPUs
    if world_size > 1:
        dataloader.sampler.set_epoch(epoch)

    # min of len(dataloader) and total_steps
    prog_bar_steps = len(dataloader) if total_steps < 0 else min(len(dataloader), total_steps)

    # Only show progress bar on main process
    if is_main:
        pbar = tqdm(enumerate(dataloader), total=prog_bar_steps)
    else:
        pbar = enumerate(dataloader)

    # run training loop with tqdm progress bar
    for step, batch in pbar:
        curr_step += 1
        loss, flow_loss, pred_x0 = train_step(model, vae, batch, scheduler, grad_accum_steps, device)

        if step % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # Logging only on main process
            if is_main:
                writer.add_scalar("Loss/train", loss, curr_step)
                writer.add_scalar("Flow Loss/train", flow_loss, curr_step)
                writer.add_scalar("StdDev Batch", pred_x0.std().item(), curr_step)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], curr_step)

                # Monitor LoRA weight norms to detect collapse
                lora_A_norm = sum(layer.lora_A.norm().item() for layer in lora_layers) / len(lora_layers)
                lora_B_norm = sum(layer.lora_B.norm().item() for layer in lora_layers) / len(lora_layers)
                writer.add_scalar("LoRA/avg_A_norm", lora_A_norm, curr_step)
                writer.add_scalar("LoRA/avg_B_norm", lora_B_norm, curr_step)

        if is_main:
                pbar.set_description(f"Ep {epoch}-{step}, flow_loss: {flow_loss:.6f}")
                
        if reduced_steps > 0 and step >= reduced_steps:  # Just run a few steps for demo
            break
        if is_main and curr_step % args.write_video_interval == 0:
            # For DDP, access underlying model with model.module
            raw_model = model.module if world_size > 1 else model

            # Switch to eval mode for generation
            raw_model.eval()

            # Generate a video using same initial frame and actions from training data
            vid, icons_vid = generate_training_video(raw_model, vae, dataloader.dataset, device,
                                                      output_path=f"{output_dir}/output_step_{curr_step}.mp4",
                                                      overfit_blocks=args.overfit_blocks)

            # Switch back to train mode
            raw_model.train()

            # Extract first middle and last frames for tensorboard
            mid_frame = vid[vid.shape[0] // 2]
            first_frame = vid[0]
            last_frame = vid[-1]
            import torchvision.utils as vutils
            grid = vutils.make_grid(torch.from_numpy(np.stack([first_frame, mid_frame, last_frame])).permute(0, 3, 1, 2).float() / 255.0, nrow=3)
            writer.add_image(f"Generated Video Frames", grid, curr_step)
            vutils.save_image(grid, f"{output_dir}/generated_frames_step{curr_step}.png")

    # Video generation and checkpointing only on main process
    if is_main:
        # For DDP, access underlying model with model.module
        raw_model = model.module if world_size > 1 else model

        # Switch to eval mode for generation
        raw_model.eval()

        # Generate a video using same initial frame and actions from training data
        vid, icons_video = generate_training_video(raw_model, vae, dataloader.dataset, device,
                                                    output_path=f"{output_dir}/output_epoch{epoch}.mp4",
                                                    overfit_blocks=args.overfit_blocks)

        # Switch back to train mode
        raw_model.train()

        # Extract first middle and last frames for tensorboard
        mid_frame = vid[vid.shape[0] // 2]
        first_frame = vid[0]
        last_frame = vid[-1]
        import torchvision.utils as vutils
        grid = vutils.make_grid(torch.from_numpy(np.stack([first_frame, mid_frame, last_frame])).permute(0, 3, 1, 2).float() / 255.0, nrow=3)
        writer.add_image(f"Generated Video Frames", grid, epoch)
        vutils.save_image(grid, f"{output_dir}/generated_frames_epoch{epoch}.png")

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_frequency == 0:
            checkpoint_path = f"{output_dir}/lora_epoch{epoch}.safetensors"
            lora_state_dict = get_lora_state_dict(model)
            os.makedirs("checkpoints", exist_ok=True)
            from safetensors.torch import save_file
            # Save the unwrapped model state dict (not DDP wrapper)
            save_file(lora_state_dict, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # ==========================================================================
    # DDP SYNC - Synchronize all processes at end of epoch
    # ==========================================================================
    # This ensures all GPUs finish the epoch before starting the next
    if world_size > 1:
        dist.barrier()

  # Final logging only on main process
  if is_main:
      writer.add_hparams(hparam_dict=hyperparams, metric_dict={
        "final_loss": loss,
        "final_flow_loss": flow_loss,
      })
      writer.close()

      print("Training loop complete.")
      raw_model = model.module if world_size > 1 else model
      generate_training_video(raw_model, vae, dataloader.dataset, device,
                              output_path=f"{output_dir}/final_output.mp4",
                              overfit_blocks=args.overfit_blocks)

  # ==========================================================================
  # DDP CLEANUP - Clean up distributed processes
  # ==========================================================================
  cleanup_distributed()


if __name__ == "__main__":
  main()

