"""
DDP-enabled finetuning script for the universal model.

Adapted from finetune_ddp.py to work with the universal model configuration.

To run on multiple GPUs:
    torchrun --nproc_per_node=NUM_GPUS finetune_universal.py

Example with 2 GPUs:
    torchrun --nproc_per_node=2 finetune_universal.py

For single GPU:
    python finetune_universal.py
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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# ============================================================================
# DDP (Distributed Data Parallel) IMPORTS
# ============================================================================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ============================================================================
# DDP HELPER FUNCTIONS
# ============================================================================
def setup_distributed():
    """Initialize the distributed process group for multi-GPU training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


# ============================================================================
# DATASET - TODO: Adapt this for universal model data format
# ============================================================================
class SimpleDataset(Dataset):
    def __init__(self, data_dir, sequence_length=9, max_sequences=-1):
        self.sequence_length = sequence_length
        # Reference: configs/distilled_model/universal/config.json shows keyboard_dim_in: 4
        self.runs = []

        # Find all runs across all sessions
        self.runs = []
        for session_dir in sorted(glob.glob(os.path.join(data_dir, "*/"))):
            if "__MACOSX" in session_dir:
                continue
            for run in sorted(glob.glob(os.path.join(session_dir, "Run_*"))):
                # Get frames and sort numerically
                frame_pattern = os.path.join(run, "frame_*.png")
                frames = glob.glob(frame_pattern)
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
        if max_sequences > 0:
            self.sequences = self.sequences[:max_sequences]
        print(f"Subsampled sequences: {len(self.sequences)}")

    def _load_csv(self, csv_path):
        """Load steering/throttle per frame"""
        data = {}
        with open(csv_path, 'r') as f:
            for row in csv.reader(f):
                action, time, frame_num, value = row[0], float(row[1]), int(row[2]), float(row[3])
                if frame_num not in data:
                    data[frame_num] = {'steering': 0.0, 'throttle': 0.0}
                data[frame_num][action] = value
        return data

    def _load_frame(self, path):
        """Load image, resize to 352x640, normalize to [-1, 1]"""
        img = Image.open(path).convert('RGB')
        img = img.resize((640, 352))  # Universal model expects 352 height
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        return arr * 2.0 - 1.0
    
    def __len__(self):
        return len(self.sequences)


    def _to_universal_format(self, steering, throttle, brake=0.0):
        """
        Convert steering/throttle/brake to universal format.
        
        Universal mode expects:
        - keyboard_condition: [forward, back, left, right] - 4D binary
        - mouse_condition: [vertical, horizontal] - 2D continuous
        
        Args:
            steering: -1 to 1 (negative = left, positive = right)
            throttle: 0 to 1
            brake: 0 to 1
        
        Returns:
            keyboard: [forward, back, left, right]
            mouse: [vertical, horizontal]
        """
        # Keyboard - binary actions
        keyboard = [
            1.0 if throttle > 0.1 else 0.0,  # forward
            1.0 if brake > 0.1 else 0.0,      # back
            1.0 if steering < -0.1 else 0.0,  # left (negative steering)
            1.0 if steering > 0.1 else 0.0,   # right (positive steering)
        ]
        
        # Mouse - camera control (scaled by 0.1 like in conditions.py)
        mouse = [
            0.0,             # vertical (not used for driving)
            0.0,  #no camera control steering * 0.1   # horizontal (camera rotation)
        ]
        
        return keyboard, mouse
    
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

            # Convert to universal format
            keyboard, mouse = self._to_universal_format(steering, throttle, brake)
            keyboard_actions.append(keyboard)
            mouse_actions.append(mouse)

        return {
            'video_frames': torch.stack(frames),                                      # [T, H, W, C]
            'keyboard_actions': torch.tensor(keyboard_actions, dtype=torch.float32),  # [T, 4]
            'mouse_actions': torch.tensor(mouse_actions, dtype=torch.float32),        # [T, 2]
        }
# ============================================================================
# MODEL LOADING - TODO: Adapt for universal model
# ============================================================================
def load_model(device, gradient_checkpointing=False):
    """Load the universal model with its configuration."""
    config = OmegaConf.load("configs/inference_yaml/inference_universal.yaml")
    config.model_kwargs.model_config = "configs/distilled_model/universal"

    model = WanDiffusionWrapper(**config.model_kwargs, is_causal=True)

    checkpoint = "models/base_distilled_model/base_distill.safetensors"
    state_dict = load_file(checkpoint)
    model.load_state_dict(state_dict, strict=False)

    if gradient_checkpointing:
        model.model.gradient_checkpointing = True
        print("Gradient checkpointing ENABLED")

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    # Load VAE (shared across models)
    from wan.vae.wanx_vae import get_wanx_vae_wrapper
    vae = get_wanx_vae_wrapper("models/", torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device, dtype=torch.float16)

    return model, vae


def generate_video_file(model, vae, initial_frame, device, path="output.mp4"):
    from PIL import Image
    import numpy as np

    if initial_frame is None:
        initial_frame = Image.open("demo_images/gta/0000.png")

    # Create constant actions (driving forward)
    num_action_steps = 89
    keyboard = torch.zeros(num_action_steps, 4)
    keyboard[:, 0] = 1  # forward
    #TODO: make the steering in sine wave
    keyboard[:, 2] = (steer_values < -0.1).float()  # left
    keyboard[:, 3] = (steer_values > 0.1).float()   # right

    mouse = torch.zeros(num_action_steps, 2)  # no camera movement

    '''
    # steer in a sloted sine wave
    steer_amplitude = 0.05
    steer_frequency = 2 * torch.pi / num_action_steps * 2  # 2 full waves over the video
    steer_values = steer_amplitude * torch.sin(torch.linspace(0, steer_frequency * num_action_steps, num_action_steps))
    mouse = torch.zeros(num_action_steps, 2)
    mouse[:, 1] = steer_values  # horizontal steering
    '''

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
        mode='universal'
    )

    return no_icons_video, video


# ============================================================================
# TRAINING - TODO: Adapt conditional_dict for universal model
# ============================================================================
def train_step(model, vae, batch, scheduler, accumulation_steps, device):
    """Training step adapted for universal model."""
    frames = batch['video_frames'].to(device, dtype=torch.float16)
    frames = frames.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]

    raw_model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        latents = vae.encode(frames, device=device).to(dtype=torch.bfloat16)
        visual_context = vae.clip.encode_video(frames).to(device=device, dtype=torch.bfloat16)
        raw_model.model.num_frame_per_block = latents.shape[2]

    del frames

    # Prepare conditions
    num_latent_frames = latents.shape[2]
    num_action_steps = 1 + 4 * (num_latent_frames - 1)

    # TODO: keyboard should be [B, T, 4] for universal model (not [B, T, 2])
    keyboard = batch['keyboard_actions'].to(device, dtype=torch.bfloat16)[:, :num_action_steps]
    mouse = batch['mouse_actions'].to(device, dtype=torch.bfloat16)[:, :num_action_steps]

    # Mask cond: 1 = known, 0 = to generate
    mask_cond = torch.zeros_like(latents[:,:4])
    mask_cond[:,:, 0] = 1
    cond_concat = torch.cat([mask_cond, latents], dim=1)

    # Sample random timestep
    batch_size = latents.shape[0]
    t_scalar = torch.rand(1, device=device) * 0.9 + 0.05
    t = t_scalar.expand(batch_size)
    timestep = t.unsqueeze(1).expand(batch_size, num_latent_frames).to(dtype=torch.bfloat16)

    # Add noise
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t_scalar)

    # Forward pass
    conditional_dict = {
        "cond_concat": cond_concat,
        "visual_context": visual_context,
        "keyboard_cond": keyboard,
        "mouse_cond": mouse,
    }

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        flow_pred, pred_x0 = model(
            noisy_image_or_video=noisy_latents,
            timestep=timestep,
            conditional_dict=conditional_dict,
        )

        target = scheduler.training_target(latents, noise, t_scalar)
        flow_loss = torch.nn.functional.mse_loss(flow_pred, target)

    loss = flow_loss / accumulation_steps

    del latents, noise, noisy_latents, flow_pred, cond_concat, conditional_dict
    del target, visual_context, keyboard, mouse, mask_cond

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    return loss.item(), flow_loss.item(), pred_x0


# ============================================================================
# VIDEO GENERATION - TODO: Adapt for universal model
# ============================================================================
def generate_video_file(model, vae, initial_frame, device, path="output.mp4"):
    """Generate video using universal model."""
    # TODO: Adapt this to use universal model actions and mode
    # Reference inference.py for correct universal mode usage
    raise NotImplementedError("Video generation needs adaptation for universal model")


def pprint(str):
    """Print only from main process."""
    if is_main_process():
        print(str)


def get_sequence_config(latent_frames, gpu="rtx6000", gradient_checkpointing=False):
    """Config for LoRA Training - same as GTA model."""
    video_frames = 1 + 4 * (latent_frames - 1)

    if gradient_checkpointing:
        configs = {
            "a100": {
                3:  {"batch_size": 3, "grad_accum": 4},
                5:  {"batch_size": 2, "grad_accum": 6},
                9:  {"batch_size": 1, "grad_accum": 12},
            },
            "rtx6000": {
                3:  {"batch_size": 6, "grad_accum": 2},
                5:  {"batch_size": 4, "grad_accum": 3},
                9:  {"batch_size": 3, "grad_accum": 5},
                11: {"batch_size": 2, "grad_accum": 4},
                13: {"batch_size": 2, "grad_accum": 6},
                15: {"batch_size": 1, "grad_accum": 12},
                17: {"batch_size": 1, "grad_accum": 12},
            },
        }
    else:
        configs = {
            "a100": {
                3:  {"batch_size": 2, "grad_accum": 6},
                5:  {"batch_size": 1, "grad_accum": 12},
            },
            "rtx6000": {
                3:  {"batch_size": 4, "grad_accum": 3},
                5:  {"batch_size": 3, "grad_accum": 4},
                9:  {"batch_size": 2, "grad_accum": 6},
                11: {"batch_size": 1, "grad_accum": 12},
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


def main():
    # DDP SETUP
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = is_main_process()

    pprint(f"Using device: {device}")
    pprint(f"World size (total GPUs): {world_size}")

    # Training config
    latent_frames = 9
    num_epochs = 5
    max_sequences = 400
    lr = 1e-6
    warmup_steps = 100
    write_video_interval = 100
    enable_gradient_checkpointing = True

    seq_config = get_sequence_config(latent_frames, gpu="rtx6000", gradient_checkpointing=enable_gradient_checkpointing)

    pprint(f"\n=== Sequence Config ===")
    pprint(f"Gradient checkpointing: {enable_gradient_checkpointing}")
    pprint(f"Latent frames: {seq_config['latent_frames']}")
    pprint(f"Video frames: {seq_config['video_frames']}")
    pprint(f"Batch size per GPU: {seq_config['batch_size']}")
    pprint(f"Grad accum: {seq_config['grad_accum_steps']}")
    pprint(f"Effective batch (with {world_size} GPUs): {seq_config['batch_size'] * seq_config['grad_accum_steps'] * world_size}")

    batch_size = seq_config['batch_size']
    sequence_length = seq_config['video_frames']
    grad_accum_steps = seq_config['grad_accum_steps']

    # dataset = SimpleDataset(data_dir="/media/kristofe/eight/data", sequence_length=sequence_length, max_sequences=max_sequences)
     dataset = SimpleDataset(data_dir="/mnt/d/data_640_360_300_sessions", sequence_length=sequence_length, max_sequences=max_sequences)

    cleanup_distributed()


if __name__ == "__main__":
    main()
