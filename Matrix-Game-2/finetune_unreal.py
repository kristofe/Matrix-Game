"""
DDP-enabled finetuning script for the Unreal model.

Uses keyboard for forward/back and mouse for steering (like GTA model)
but based on universal model weights to avoid GTA HUD artifacts.

To run on multiple GPUs:
    torchrun --nproc_per_node=NUM_GPUS finetune_unreal.py

Example with 2 GPUs:
    torchrun --nproc_per_node=2 finetune_unreal.py

For single GPU:
    python finetune_unreal.py
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
# DATASET - Unreal driving data format
# ============================================================================
class SimpleDataset(Dataset):
    def __init__(self, data_dir, sequence_length=9, max_sequences=-1):
        self.sequence_length = sequence_length
        # Unreal config: keyboard_dim_in: 2 (forward, back), mouse_dim_in: 2 (vertical, horizontal/steering)
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
        img = img.resize((640, 352))  # Model expects 352 height
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        return arr * 2.0 - 1.0

    def __len__(self):
        return len(self.sequences)


    def _to_unreal_format(self, steering, throttle, brake=0.0):
        """
        Convert steering/throttle/brake to Unreal format.

        Unreal mode expects (like GTA):
        - keyboard_condition: [forward, back] - 2D binary
        - mouse_condition: [vertical, horizontal] - 2D continuous (horizontal = steering)

        Args:
            steering: -1 to 1 (negative = left, positive = right)
            throttle: 0 to 1
            brake: 0 to 1

        Returns:
            keyboard: [forward, back]
            mouse: [vertical, horizontal]
        """
        # Keyboard - binary actions for gas/brake
        keyboard = [
            1.0 if throttle > 0.1 else 0.0,  # forward (gas)
            1.0 if brake > 0.1 else 0.0,      # back (brake)
        ]

        # Mouse - steering control (scaled by 0.1 like in conditions.py)
        mouse = [
            0.0,                # vertical (not used for driving)
            steering * 0.1,    # horizontal = steering
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

            # Convert to unreal format (keyboard for gas/brake, mouse for steering)
            keyboard, mouse = self._to_unreal_format(steering, throttle, brake)
            keyboard_actions.append(keyboard)
            mouse_actions.append(mouse)

        return {
            'video_frames': torch.stack(frames),                                      # [T, H, W, C]
            'keyboard_actions': torch.tensor(keyboard_actions, dtype=torch.float32),  # [T, 2]
            'mouse_actions': torch.tensor(mouse_actions, dtype=torch.float32),        # [T, 2]
        }
# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(device, gradient_checkpointing=False):
    """Load the unreal model with its configuration."""
    config = OmegaConf.load("configs/inference_yaml/inference_unreal.yaml")
    config.model_kwargs.model_config = "configs/distilled_model/unreal"

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


def generate_video(model, vae, initial_frame, keyboard_actions, mouse_actions, device):
    '''
    Generate a 3-second video (90 frames = 23 latent frames) given an initial frame and actions.

    Args:
        model: the trained diffusion model
        vae: the VAE for encoding/decoding
        scheduler: the diffusion scheduler
        initial_frame: [1, C, H, W] tensor, initial frame in [-1, 1]
        keyboard_actions: [num_action_steps, 2] tensor (forward, back)
        mouse_actions: [num_action_steps, 2] tensor, (vertical, horizontal/steering) scaled by 0.1
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
    config = OmegaConf.load("configs/inference_yaml/inference_unreal.yaml")
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
            mode='unreal',
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

    # Create constant actions (driving forward with slight steering)
    num_action_steps = 89
    keyboard = torch.zeros(num_action_steps, 2)
    keyboard[:, 0] = 1  # forward (gas)

    # Steer in a sinusoidal pattern
    steer_amplitude = 0.5  # max steering angle
    steer_frequency = 2 * torch.pi / num_action_steps * 2  # 2 full waves over the video
    steer_values = steer_amplitude * torch.sin(torch.linspace(0, steer_frequency * num_action_steps, num_action_steps))

    mouse = torch.zeros(num_action_steps, 2)
    mouse[:, 1] = steer_values * 0.1  # horizontal steering (scaled by 0.1)

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
        mode='unreal'
    )

    return no_icons_video, video


# ============================================================================
# TRAINING
# ============================================================================
def train_step(model, vae, batch, scheduler, accumulation_steps, device):
    """Training step adapted for unreal model."""
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

    # keyboard: [B, T, 2] for unreal model (forward, back)
    # mouse: [B, T, 2] for unreal model (vertical, horizontal/steering)
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

    dataset = SimpleDataset(data_dir="/media/kristofe/eight/data", sequence_length=sequence_length, max_sequences=max_sequences)
    # dataset = SimpleDataset(data_dir="/mnt/d/data_640_360_300_sessions", sequence_length=sequence_length, max_sequences=max_sequences)

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

    pprint("\nLoading model...")
    # gradient_checkpointing=True saves ~30-50% memory but ~20-30% slower training
    model, vae = load_model(device, gradient_checkpointing=enable_gradient_checkpointing)
    pprint("Model loaded.")
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

    # creating output folder (only on main process)
    timestamp = datetime.now().strftime("%Y%m%d-%H_%M_%S")
    output_dir = f"outputs/finetune_unreal_{timestamp}"
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    pprint(f"Outputs will be saved to: {output_dir}")

    # Synchronize all processes before continuing
    # This ensures the output directory is created before other processes try to use it
    if world_size > 1:
        dist.barrier()

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
    pprint(f"Froze {frozen_count} parameters in action modules.")
    pprint(f"Trainable parameters: {trainable_count} / {total_count} ({100.0 * trainable_count / total_count:.2f}%)")



    pprint("\n=== Starting training loop ===")
    model.train()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # Learning rate scheduler with warmup
    total_steps = num_epochs * len(dataloader)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # Generate initial video only on main process
    if is_main:
        # For DDP, access underlying model with model.module
        raw_model = model.module if world_size > 1 else model
        initial_frame = dataloader.dataset[0]['video_frames'][0]
        generate_video_file(raw_model, vae, initial_frame, device, path=f"{output_dir}/initial_output.mp4")

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
        "effective_batch_size": batch_size * grad_accum_steps * world_size,  # Include world_size
        "frozen_modules": '_'.join(frozen_modules) if frozen_modules else 'none',
        "world_size": world_size,  # Track how many GPUs used
    }

    # TensorBoard writer only on main process
    writer = None
    if is_main:
        run_name = f"ft_unreal_lr={hyperparams['lr']}_bs={hyperparams['batch_size']}_ep={hyperparams['epochs']}_sl={hyperparams['sequence_length']}_ga={hyperparams['grad_accum_steps']}_ts={timestamp}_ws={hyperparams['warmup_steps']}_fm={'_'.join(hyperparams['frozen_modules'])}_gpus={world_size}"
        writer = SummaryWriter(log_dir=f"logs/{run_name}")

    for epoch in range(num_epochs):
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

            if is_main:
                    pbar.set_description(f"Ep {epoch}-{step}, flow_loss: {flow_loss:.6f}")

            if reduced_steps > 0 and step >= reduced_steps:  # Just run a few steps for demo
                break
            if is_main and curr_step % write_video_interval == 0:
                # For DDP, access underlying model with model.module
                raw_model = model.module if world_size > 1 else model

                # Generate a video at the end of each epoch
                initial_frame = dataloader.dataset[np.random.randint(0, len(dataloader.dataset))]['video_frames'][0]
                vid, icons_vid = generate_video_file(raw_model, vae, initial_frame, device, path=f"{output_dir}/output_s{curr_step}.mp4")

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

            # Generate a video at the end of each epoch
            initial_frame = dataloader.dataset[np.random.randint(0, len(dataloader.dataset))]['video_frames'][0]
            vid, icons_vid = generate_video_file(raw_model, vae, initial_frame, device, path=f"{output_dir}/output_e{epoch}.mp4")

            # Extract first middle and last frames for tensorboard
            mid_frame = vid[vid.shape[0] // 2]
            first_frame = vid[0]
            last_frame = vid[-1]
            import torchvision.utils as vutils
            grid = vutils.make_grid(torch.from_numpy(np.stack([first_frame, mid_frame, last_frame])).permute(0, 3, 1, 2).float() / 255.0, nrow=3)
            writer.add_image(f"Generated Video Frames", grid, epoch)
            vutils.save_image(grid, f"{output_dir}/generated_frames_epoch{epoch}.png")

            # Save checkpoint
            checkpoint_frequency = 1  # save every epoch
            if (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_path = f"{output_dir}/finetuned_model_epoch{epoch}.safetensors"
                os.makedirs("checkpoints", exist_ok=True)
                from safetensors.torch import save_file
                # Save the unwrapped model state dict (not DDP wrapper)
                save_file(raw_model.state_dict(), checkpoint_path)
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
        initial_frame = dataloader.dataset[0]['video_frames'][0]
        generate_video_file(raw_model, vae, initial_frame, device, path=f"{output_dir}/final_output.mp4")

    # ==========================================================================
    # DDP CLEANUP - Clean up distributed processes
    # ==========================================================================
    cleanup_distributed()

if __name__ == "__main__":
    main()
