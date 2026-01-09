# Training the DiT (Diffusion Transformer) for Matrix-Game-2

A comprehensive guide to training the Diffusion Transformer using a pre-trained VAE, written for ML researchers implementing their own version.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Data Preparation](#3-data-preparation)
4. [Flow Matching Diffusion](#4-flow-matching-diffusion)
5. [Simplified Training Code](#5-simplified-training-code)
6. [Conditioning Details](#6-conditioning-details)
7. [Hyperparameters & Tips](#7-hyperparameters--tips)
8. [Inference After Training](#8-inference-after-training)
9. [Quick Reference](#9-quick-reference)

**Appendices**
- [Appendix A: Flow Matching vs DDPM](#appendix-a-flow-matching-vs-ddpm)
- [Appendix B: Memory Optimization](#appendix-b-memory-optimization)

---

## 1. Overview

### What We're Training

The DiT (Diffusion Transformer) is trained to **generate video in VAE latent space**, not pixel space. This is critical for efficiency:

```
TRAINING PIPELINE
─────────────────

Video Frames [B, 3, T, H, W]
        │
        ▼ (frozen VAE encoder)
VAE Latents [B, 16, T', H/8, W/8]
        │
        ▼ (add noise)
Noisy Latents
        │
        ▼ (DiT forward pass)
Predicted Flow ────► Loss ────► Backprop ────► Update DiT weights
        │
        ▼ (compare to)
Target Flow = noise - latents
```

### Why Train in Latent Space?

| Space | Resolution | Tokens (17 frames, 720p) | Memory |
|-------|------------|--------------------------|--------|
| Pixel | [3, 17, 720, 1280] | 15,667,200 | ~60 GB |
| VAE Latent | [16, 5, 90, 160] | 72,000 | ~2 GB |
| After Patching | [1536, 5, 45, 80] | 18,000 | ~0.5 GB |

**256× fewer tokens** makes attention tractable!

### The Complete System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING (this document)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│   │  Video   │ ──► │   VAE    │ ──► │  DiT     │ ──► │  Loss    │          │
│   │  Frames  │     │ (frozen) │     │ (train)  │     │          │          │
│   └──────────┘     └──────────┘     └──────────┘     └──────────┘          │
│        │                                  ▲                                  │
│        │                                  │                                  │
│        └──────────────► CLIP ─────────────┘                                 │
│                      (frozen)    visual conditioning                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE (after training)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│   │  Noise   │ ──► │   DiT    │ ──► │  Clean   │ ──► │  Video   │          │
│   │          │     │ denoise  │     │  Latent  │     │  Frames  │          │
│   └──────────┘     └──────────┘     └──────────┘     └──────────┘          │
│                          ▲                  │                                │
│                          │                  ▼                                │
│                    Conditions          VAE Decoder                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites

### 2.1 Trained VAE

You should have a trained `SimpleVideoVAE` from VAE_DOC.md Section 7:

```python
# Load your trained VAE
from vae import SimpleVideoVAE

vae = SimpleVideoVAE(z_dim=16, base_dim=96)
vae.load_state_dict(torch.load("checkpoints/vae_epoch_100.pth"))
vae.eval()
vae.requires_grad_(False)  # CRITICAL: VAE is always frozen during DiT training
```

**Key VAE Properties (SimpleVideoVAE):**
- Input: `[B, 3, T, H, W]` video in `[-1, 1]`
- Output: `[B, 16, T/4, H/8, W/8]` latents
- Spatial compression: 8×
- Temporal compression: 4× (via two strided temporal convolutions)

**Important: SimpleVideoVAE vs StreamableVideoVAE vs Full WAN VAE**

| Property | SimpleVideoVAE | StreamableVideoVAE | Full WAN VAE |
|----------|----------------|-------------------|--------------|
| Temporal formula | `T_latent = T_video // 4` | `T_latent = 1 + (T_video - 1) // 4` | Same as Streamable |
| Valid T_video | Multiples of 4 (4, 8, 12...) | 1 + 4k (1, 5, 9, 13, 17...) | Same as Streamable |
| Processing | Single forward pass | Chunked (1, then groups of 4) | Same as Streamable |
| Streaming support | No | Yes | Yes |
| Feature caching | No | Yes | Yes |

**See VAE_DOC.md Appendix D for the complete StreamableVideoVAE implementation.**

If you trained your **SimpleVideoVAE on 4 frames**, each training sample produces **1 latent frame**.
If you trained your **StreamableVideoVAE on 5 frames**, each training sample produces **2 latent frames**.

```
Example with T_video=4:
  Video:  [B, 3, 4, H, W]
  Latent: [B, 16, 1, H/8, W/8]  ← Single latent frame per 4 video frames
```

### 2.2 Training Data Format

Your training data should be organized as:

```
data/
├── Run_001/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   ├── ...
│   └── input.csv          # Optional: action data
├── Run_002/
│   └── ...
```

**Video requirements for SimpleVideoVAE:**
- Resolution divisible by 8 (e.g., 352×640, 720×1280)
- Frame count: **multiples of 4** (4, 8, 12, 16, 20...) for SimpleVideoVAE
- RGB images normalized to `[-1, 1]`

**Recommended training configurations:**

| Video Frames | Latent Frames | Use Case |
|--------------|---------------|----------|
| 4 | 1 | Minimal (matches your VAE training) |
| 8 | 2 | Short sequences |
| 12 | 3 | Medium sequences |
| 16 | 4 | Longer sequences |

### 2.3 Hardware Requirements

| Setup | VRAM | Batch Size | Latent Frames | Video Frames |
|-------|------|------------|---------------|--------------|
| RTX 4090 (24GB) | 24 GB | 1 | 1 | 4 |
| RTX 4090 (24GB) | 24 GB | 1 | 2 | 8 |
| A100 (40GB) | 40 GB | 2 | 2 | 8 |
| A100 (80GB) | 80 GB | 4 | 4 | 16 |

With gradient checkpointing, you can roughly double these numbers.

**Note:** For SimpleVideoVAE, `video_frames = latent_frames × 4`.

### 2.4 Training with Single Latent Frames

If you trained your VAE on **4 video frames** (producing **1 latent frame**), you have options for DiT training:

**Option A: Train with T_latent=1 (matches VAE training)**
- Pros: Consistent with how VAE was trained
- Cons: DiT sees no temporal context, learns single-frame prediction only
- Good for: Initial experiments, testing the pipeline

**Option B: Train with T_latent=2+ (longer sequences)**
- Pros: DiT learns temporal dynamics across latent frames
- Cons: VAE sees longer sequences than it was trained on
- Good for: Production models, learning motion

```
Option A (T_latent=1):
  Video: [B, 3, 4, H, W] → VAE → Latent: [B, 16, 1, H/8, W/8]
  DiT predicts a single latent frame. No temporal attention across latents.

Option B (T_latent=2):
  Video: [B, 3, 8, H, W] → VAE → Latent: [B, 16, 2, H/8, W/8]
  DiT can learn temporal relationships between 2 latent frames.
  VAE generalizes to longer sequences (usually works well).
```

**Recommendation:** Start with T_latent=1 to verify training works, then increase to T_latent=2-4 for learning motion.

---

## 3. Data Preparation

### 3.1 Dataset Class

```python
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd

class VideoDataset(Dataset):
    """
    Dataset that loads video sequences and optional action data.

    Directory structure:
        data_root/
        ├── Run_001/
        │   ├── frame_0000.png
        │   ├── frame_0001.png
        │   └── input.csv (optional)
        └── Run_002/
            └── ...
    """
    def __init__(self, data_root, num_latent_frames=1, image_size=(352, 640)):
        self.data_root = Path(data_root)
        self.image_size = image_size  # (H, W)

        # Calculate video frames from latent frames
        # SimpleVideoVAE has 4× temporal compression: T_latent = T_video // 4
        # So T_video = T_latent * 4
        self.num_video_frames = num_latent_frames * 4

        # Find all runs
        self.runs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])

        # Build index of valid sequences
        self.sequences = []
        for run_dir in self.runs:
            frames = sorted(run_dir.glob("frame_*.png"))
            # Each run can have multiple sequences
            for start in range(0, len(frames) - self.num_video_frames + 1, self.num_video_frames):
                self.sequences.append((run_dir, start))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        run_dir, start = self.sequences[idx]

        # Load video frames
        frames = []
        for i in range(start, start + self.num_video_frames):
            img_path = run_dir / f"frame_{i:04d}.png"
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size[1], self.image_size[0]))  # PIL uses (W, H)
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1  # [-1, 1]
            frames.append(img)

        video = torch.stack(frames, dim=1)  # [C, T, H, W]

        # Load actions if available
        action_file = run_dir / "input.csv"
        if action_file.exists():
            actions = pd.read_csv(action_file)
            mouse = torch.tensor(actions[['mouse_x', 'mouse_y']].values[start:start+self.num_video_frames], dtype=torch.float32)
            keyboard = torch.tensor(actions[['key_w', 'key_s']].values[start:start+self.num_video_frames], dtype=torch.float32)
        else:
            # Default: no actions
            mouse = torch.zeros(self.num_video_frames, 2)
            keyboard = torch.zeros(self.num_video_frames, 2)

        return {
            'video': video,           # [3, T, H, W]
            'mouse': mouse,           # [T, 2]
            'keyboard': keyboard,     # [T, 2]
        }
```

### 3.2 VAE Encoding (Always Frozen)

```python
def encode_batch(vae, video_batch, device):
    """
    Encode video to VAE latent space.

    Args:
        vae: Trained SimpleVideoVAE (frozen)
        video_batch: [B, 3, T, H, W] in [-1, 1]
        device: torch device

    Returns:
        latents: [B, 16, T', H/8, W/8] normalized latents
    """
    vae.eval()
    with torch.no_grad():
        video_batch = video_batch.to(device)
        mu, _ = vae.encode(video_batch)  # We only use mu, not sampling
        return mu  # Already normalized by VAE
```

### 3.3 Building Conditioning Tensors

The DiT receives conditioning through multiple channels:

```python
def prepare_conditioning(latents, first_frame_latent):
    """
    Build the conditioning concatenation tensor.

    For image-to-video generation:
    - First frame is known (mask = 1)
    - Rest are generated (mask = 0)

    Args:
        latents: [B, 16, T', H', W'] - target latents
        first_frame_latent: [B, 16, 1, H', W'] - encoded first frame

    Returns:
        cond_concat: [B, 20, T', H', W'] - mask(4) + context(16)
    """
    B, C, T, H, W = latents.shape
    device = latents.device

    # Mask: 4 channels indicating which frames are known
    # 1 = known (conditioned), 0 = generate
    mask = torch.zeros(B, 4, T, H, W, device=device)
    mask[:, :, 0, :, :] = 1.0  # First frame is known

    # Context: first frame latent repeated/padded
    context = torch.zeros(B, 16, T, H, W, device=device)
    context[:, :, 0, :, :] = first_frame_latent[:, :, 0, :, :]

    # Concatenate: [B, 4+16, T, H, W] = [B, 20, T, H, W]
    cond_concat = torch.cat([mask, context], dim=1)

    return cond_concat
```

### 3.4 Visual Context (CLIP Features)

For the simplified implementation, we use a placeholder. In the full system, CLIP provides visual features:

```python
def get_visual_context(first_frame, clip_model=None):
    """
    Get visual context for cross-attention.

    In full system: CLIP encodes first frame to [B, 257, 1280]
    Simplified: Use learned embedding or random

    Args:
        first_frame: [B, 3, 1, H, W]
        clip_model: Optional CLIP encoder

    Returns:
        visual_context: [B, 257, 1280] (or [B, 257, dim] for simplified)
    """
    B = first_frame.shape[0]
    device = first_frame.device

    if clip_model is not None:
        # Full system
        with torch.no_grad():
            visual_context = clip_model.encode_video(first_frame)
    else:
        # Simplified: random context (replace with learned embedding in practice)
        visual_context = torch.randn(B, 257, 1280, device=device)

    return visual_context
```

---

## 4. Flow Matching Diffusion

### 4.1 What is Flow Matching?

Flow matching is an alternative to DDPM (Denoising Diffusion Probabilistic Models). Instead of predicting noise, we predict the **velocity** (or flow) that moves from noise to data:

```
DDPM (Noise Prediction):
────────────────────────
Forward:  x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
Target:   noise
Loss:     MSE(predicted_noise, noise)

Flow Matching (Velocity Prediction):
────────────────────────────────────
Forward:  x_t = (1 - sigma_t) * x_0 + sigma_t * noise
Target:   flow = noise - x_0
Loss:     MSE(predicted_flow, flow)
```

**Why Flow Matching?**
- Simpler math (linear interpolation)
- Faster convergence
- Works well with few inference steps (3-10)
- Used by WAN, Stable Diffusion 3, and other modern models

### 4.2 Forward Process (Adding Noise)

```python
def add_noise(latents, noise, sigma):
    """
    Add noise to latents using flow matching interpolation.

    Formula: x_t = (1 - sigma) * x_0 + sigma * noise

    Args:
        latents: [B, C, T, H, W] - clean latents (x_0)
        noise: [B, C, T, H, W] - random noise
        sigma: [B] or scalar - noise level in [0, 1]

    Returns:
        noisy_latents: [B, C, T, H, W]
    """
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor([sigma], device=latents.device)

    # Expand sigma to match latent dimensions
    sigma = sigma.view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]

    noisy_latents = (1 - sigma) * latents + sigma * noise
    return noisy_latents
```

### 4.3 Training Target

```python
def compute_flow_target(latents, noise):
    """
    Compute the flow matching target.

    Flow = noise - x_0 (velocity from data to noise)

    The model predicts this flow, and we can recover x_0 as:
        x_0 = x_t - sigma * predicted_flow

    Args:
        latents: [B, C, T, H, W] - clean latents
        noise: [B, C, T, H, W] - noise

    Returns:
        flow: [B, C, T, H, W] - target velocity
    """
    return noise - latents
```

### 4.4 Timestep Scheduling

Matrix-Game-2 uses specific timesteps aligned with inference:

```python
class FlowMatchScheduler:
    """
    Simplified flow matching scheduler.

    Key insight: Training should sample from the same timesteps used in inference.
    WAN uses 3 inference steps: t = [1000, 666, 333] → sigma = [1.0, 0.667, 0.333]
    """
    def __init__(self, num_train_timesteps=1000, shift=5.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        # Inference timesteps (what we'll use at test time)
        # Scaled from [1000, 666, 333] to indices
        self.inference_timesteps = torch.tensor([0, 334, 667])

    def get_sigma(self, timestep):
        """
        Convert timestep to sigma (noise level).

        Uses shifted schedule for better training dynamics.
        """
        t = timestep / self.num_train_timesteps  # Normalize to [0, 1]
        sigma = t / (t + (1 - t) * self.shift)   # Shift schedule
        return sigma

    def sample_timestep(self, batch_size, device):
        """
        Sample timesteps for training.

        Samples from inference timesteps to align train/test distributions.
        """
        indices = torch.randint(0, len(self.inference_timesteps), (batch_size,))
        timesteps = self.inference_timesteps[indices].to(device)
        return timesteps

    def training_weight(self, sigma):
        """
        Weight loss by timestep.

        Gaussian weighting centered at sigma=0.5 to emphasize mid-diffusion.
        """
        # Gaussian centered at 0.5
        weight = torch.exp(-((sigma - 0.5) ** 2) / (2 * 0.25 ** 2))
        return weight + 0.05  # Add epsilon to avoid zero weights
```

### 4.5 Loss Calculation

```python
def flow_matching_loss(predicted_flow, target_flow, sigma, scheduler):
    """
    Compute weighted flow matching loss.

    Args:
        predicted_flow: [B, C, T, H, W] - model output
        target_flow: [B, C, T, H, W] - ground truth flow
        sigma: [B] - noise levels
        scheduler: FlowMatchScheduler for weights

    Returns:
        loss: scalar
    """
    # Per-element squared error
    sq_error = (predicted_flow - target_flow) ** 2

    # Get timestep weights
    weights = scheduler.training_weight(sigma)  # [B]
    weights = weights.view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]

    # Weighted mean
    loss = (weights * sq_error).mean()

    return loss
```

---

## 5. Simplified Training Code

### 5.1 Complete Training Script

```python
"""
Simplified DiT Training Script for Matrix-Game-2

This script trains a Diffusion Transformer (DiT) in VAE latent space
using flow matching, consistent with WAN and Matrix-Game-2.

Usage:
    python train_dit.py --data_root data/ --vae_checkpoint vae.pth --epochs 100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from pathlib import Path
from tqdm import tqdm

# Import your implementations
from vae import SimpleVideoVAE
from dit import SimpleDiT  # From DIT_DOC.md Section 7


class FlowMatchScheduler:
    """Flow matching noise scheduler."""

    def __init__(self, num_train_timesteps=1000, shift=5.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.inference_timesteps = torch.tensor([0, 334, 667])

    def get_sigma(self, timestep, device):
        t = timestep.float() / self.num_train_timesteps
        sigma = t / (t + (1 - t) * self.shift)
        return sigma.to(device)

    def sample_timestep(self, batch_size, device):
        indices = torch.randint(0, len(self.inference_timesteps), (batch_size,), device=device)
        return self.inference_timesteps[indices].to(device)

    def training_weight(self, sigma):
        weight = torch.exp(-((sigma - 0.5) ** 2) / (2 * 0.25 ** 2))
        return weight + 0.05


def train_step(model, vae, batch, scheduler, device):
    """
    Single training step.

    Returns:
        loss: scalar loss value
        pred_x0: [B, C, T, H, W] predicted clean latents (for visualization)
    """
    # Move data to device
    video = batch['video'].to(device)  # [B, 3, T, H, W]
    mouse = batch['mouse'].to(device)  # [B, T, 2]
    keyboard = batch['keyboard'].to(device)  # [B, T, 2]

    B = video.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Encode video to latent space (VAE is frozen)
    # ─────────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        latents, _ = vae.encode(video)  # [B, 16, T', H/8, W/8]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Prepare conditioning
    # ─────────────────────────────────────────────────────────────────────────
    first_frame_latent = latents[:, :, :1, :, :]  # [B, 16, 1, H', W']

    # Build cond_concat: mask (4 ch) + context (16 ch) = 20 ch
    T_latent = latents.shape[2]
    H_latent, W_latent = latents.shape[3], latents.shape[4]

    mask = torch.zeros(B, 4, T_latent, H_latent, W_latent, device=device)
    mask[:, :, 0, :, :] = 1.0  # First frame is known

    context = torch.zeros(B, 16, T_latent, H_latent, W_latent, device=device)
    context[:, :, 0, :, :] = first_frame_latent[:, :, 0, :, :]

    cond_concat = torch.cat([mask, context], dim=1)  # [B, 20, T', H', W']

    # Visual context (simplified: random, replace with CLIP in production)
    visual_context = torch.randn(B, 257, 1280, device=device)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Sample timestep and add noise
    # ─────────────────────────────────────────────────────────────────────────
    timesteps = scheduler.sample_timestep(B, device)  # [B]
    sigma = scheduler.get_sigma(timesteps, device)    # [B]

    noise = torch.randn_like(latents)

    # Flow matching forward: x_t = (1 - sigma) * x_0 + sigma * noise
    sigma_expanded = sigma.view(-1, 1, 1, 1, 1)
    noisy_latents = (1 - sigma_expanded) * latents + sigma_expanded * noise

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Forward pass through DiT
    # ─────────────────────────────────────────────────────────────────────────
    # Scale timesteps to expected range
    t_scaled = timesteps.float()

    predicted_flow = model(
        noisy_latents,
        t_scaled,
        cond_concat,
        visual_context
    )  # [B, 16, T', H', W']

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Compute loss
    # ─────────────────────────────────────────────────────────────────────────
    # Target: flow = noise - x_0
    target_flow = noise - latents

    # Weighted MSE loss
    weights = scheduler.training_weight(sigma).view(-1, 1, 1, 1, 1)
    loss = (weights * (predicted_flow - target_flow) ** 2).mean()

    # Recover predicted x_0 for visualization
    # x_0 = x_t - sigma * flow
    with torch.no_grad():
        pred_x0 = noisy_latents - sigma_expanded * predicted_flow

    return loss, pred_x0


def train(
    data_root: str,
    vae_checkpoint: str,
    output_dir: str = "outputs",
    epochs: int = 100,
    batch_size: int = 1,
    learning_rate: float = 5e-6,
    grad_accum_steps: int = 4,
    num_latent_frames: int = 1,  # 1 latent frame = 4 video frames (SimpleVideoVAE)
    image_size: tuple = (352, 640),
    checkpoint_every: int = 10,
    device: str = "cuda",
):
    """
    Main training function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Setup models
    # ─────────────────────────────────────────────────────────────────────────
    print("Loading VAE...")
    vae = SimpleVideoVAE(z_dim=16, base_dim=96)
    vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to(device)

    print("Creating DiT...")
    dit = SimpleDiT(
        in_channels=36,       # 16 latent + 20 conditioning
        out_channels=16,
        dim=1536,
        ffn_dim=8960,
        num_heads=12,
        num_layers=30,
        patch_size=(1, 2, 2),
    ).to(device)

    print(f"DiT parameters: {sum(p.numel() for p in dit.parameters()):,}")

    # ─────────────────────────────────────────────────────────────────────────
    # Setup training
    # ─────────────────────────────────────────────────────────────────────────
    scheduler = FlowMatchScheduler()

    optimizer = AdamW(dit.parameters(), lr=learning_rate, weight_decay=0.01)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = VideoDataset(data_root, num_latent_frames=num_latent_frames, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(epochs):
        dit.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Forward + backward
            loss, pred_x0 = train_step(dit, vae, batch, scheduler, device)
            loss = loss / grad_accum_steps  # Scale for accumulation
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix(loss=loss.item() * grad_accum_steps)

        lr_scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.6f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = output_dir / f"dit_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': dit.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    torch.save(dit.state_dict(), output_dir / "dit_final.pth")
    print("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DiT for video generation")
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to trained VAE")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_latent_frames", type=int, default=1, help="1 latent = 4 video frames")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    train(
        data_root=args.data_root,
        vae_checkpoint=args.vae_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accum_steps=args.grad_accum_steps,
        num_latent_frames=args.num_latent_frames,
        device=args.device,
    )
```

### 5.2 Training Loop Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING LOOP                                   │
└─────────────────────────────────────────────────────────────────────────────┘

For each epoch:
│
├─► For each batch:
│   │
│   ├─► Step 1: Load video [B, 3, T, H, W]
│   │
│   ├─► Step 2: VAE Encode (no_grad)
│   │       video ──► latents [B, 16, T', H/8, W/8]
│   │
│   ├─► Step 3: Build conditioning
│   │       ├─ mask [B, 4, T', H', W']     (1 for known frames)
│   │       ├─ context [B, 16, T', H', W'] (first frame latent)
│   │       └─ cond_concat [B, 20, T', H', W']
│   │
│   ├─► Step 4: Sample timestep t ~ {0, 334, 667}
│   │
│   ├─► Step 5: Add noise
│   │       sigma = schedule(t)
│   │       noisy = (1-sigma) * latents + sigma * noise
│   │
│   ├─► Step 6: DiT forward
│   │       predicted_flow = DiT(noisy, t, cond_concat, visual_context)
│   │
│   ├─► Step 7: Compute loss
│   │       target = noise - latents
│   │       loss = weighted_mse(predicted_flow, target)
│   │
│   ├─► Step 8: Backward
│   │       loss.backward()
│   │
│   └─► Step 9: Optimizer step (every grad_accum_steps)
│           clip_grad_norm_(1.0)
│           optimizer.step()
│           optimizer.zero_grad()
│
└─► End epoch: lr_scheduler.step(), save checkpoint
```

---

## 6. Conditioning Details

### 6.1 Image Conditioning (Cross-Attention)

The first frame provides visual context through cross-attention:

```
Image Conditioning Pipeline:
────────────────────────────

First Frame [B, 3, 1, H, W]
        │
        ▼ CLIP Encoder (frozen)
CLIP Features [B, 257, 1280]
        │
        ▼ MLPProj (Linear → GELU → Linear)
Visual Context [B, 257, 1536]
        │
        ▼ Cross-Attention in each DiT block
        │
        Q = proj(hidden_states)     [B, L, 1536]
        K = proj(visual_context)    [B, 257, 1536]  ← from CLIP
        V = proj(visual_context)    [B, 257, 1536]  ← from CLIP
        │
        ▼ Attention(Q, K, V)
Output: hidden states informed by first frame appearance
```

**Why 257 tokens?**
- 1 CLS token (global image summary)
- 256 patch tokens (16×16 grid of image patches)

### 6.2 Action Conditioning

For interactive video generation, actions (mouse/keyboard) condition the model:

```python
class SimpleActionModule(nn.Module):
    """
    Simplified action conditioning module.

    In full model: Separate attention for mouse and keyboard
    Simplified: Concatenate and project
    """
    def __init__(self, mouse_dim=2, keyboard_dim=2, hidden_dim=128, model_dim=1536):
        super().__init__()
        self.action_proj = nn.Sequential(
            nn.Linear(mouse_dim + keyboard_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, hidden_states, mouse, keyboard, grid_sizes):
        """
        Args:
            hidden_states: [B, L, model_dim] - from transformer
            mouse: [B, T_action, 2] - mouse x,y per frame
            keyboard: [B, T_action, 2] - keyboard states per frame
            grid_sizes: (T, H, W) - spatial dimensions

        Returns:
            hidden_states: [B, L, model_dim] - conditioned
        """
        B, L, D = hidden_states.shape
        T, H, W = grid_sizes

        # Combine actions
        actions = torch.cat([mouse, keyboard], dim=-1)  # [B, T_action, 4]

        # Expand to match spatial dimensions
        # Each latent frame corresponds to 4 video frames (VAE compression)
        # So we average/subsample actions to match
        if actions.shape[1] > T:
            # Subsample: take every 4th action starting from 0
            indices = torch.arange(0, T * 4, 4, device=actions.device)[:T]
            actions = actions[:, indices, :]  # [B, T, 4]

        # Project to model dimension
        action_emb = self.action_proj(actions)  # [B, T, model_dim]

        # Expand to all spatial positions
        action_emb = action_emb.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, model_dim]
        action_emb = action_emb.expand(-1, -1, H, W, -1)   # [B, T, H, W, model_dim]
        action_emb = action_emb.reshape(B, L, D)           # [B, L, model_dim]

        # Add to hidden states
        return hidden_states + action_emb
```

### 6.3 Mask Conditioning

The mask tells the model which frames are known vs. to-be-generated:

```
Mask Channels: [B, 4, T', H', W']
─────────────────────────────────

Channel values:
  1.0 = Known frame (don't modify, use as context)
  0.0 = Generate this frame

Example for image-to-video (first frame known):
  Frame 0: mask = [1, 1, 1, 1]  ← all 4 channels = 1
  Frame 1: mask = [0, 0, 0, 0]  ← all 4 channels = 0
  Frame 2: mask = [0, 0, 0, 0]
  ...

Why 4 channels?
  Historical: matches VAE's 4x temporal compression
  Practical: allows for more complex masking patterns (e.g., interpolation)
```

---

## 7. Hyperparameters & Tips

### 7.1 Recommended Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 5e-6 | For finetuning; 1e-4 for from-scratch |
| Weight decay | 0.01 | Standard for AdamW |
| Batch size | 1-4 | Depends on VRAM |
| Grad accum | 4-12 | Effective batch = batch × accum |
| Grad clip | 1.0 | Prevents exploding gradients |
| Epochs | 50-200 | Until convergence |
| Warmup steps | 100-500 | Linear warmup |
| Scheduler | Cosine | Annealing to 0 |

### 7.2 Batch Size vs. Gradient Accumulation

```
Memory-limited? Use gradient accumulation:
──────────────────────────────────────────

batch_size=1, grad_accum=4   →  effective_batch=4,  memory: ★☆☆☆☆
batch_size=2, grad_accum=2   →  effective_batch=4,  memory: ★★★☆☆
batch_size=4, grad_accum=1   →  effective_batch=4,  memory: ★★★★★

All produce same gradients, but:
- Higher batch_size = faster (fewer optimizer steps)
- Higher grad_accum = lower memory
```

### 7.3 Memory Optimization

```python
# 1. Gradient checkpointing (30-50% memory savings)
from torch.utils.checkpoint import checkpoint

class DiTBlockWithCheckpoint(nn.Module):
    def forward(self, x, *args):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward, x, *args, use_reentrant=False)
        return self._forward(x, *args)

# 2. Mixed precision (40-50% memory savings)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(dtype=torch.bfloat16):
    loss, _ = train_step(model, vae, batch, scheduler, device)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 3. Delete tensors immediately
del noisy_latents, predicted_flow
torch.cuda.empty_cache()
```

### 7.4 Training Timestep Alignment

**Critical insight:** Train on the same timesteps used for inference!

```python
# BAD: Random timesteps across full range
timesteps = torch.randint(0, 1000, (batch_size,))

# GOOD: Sample from inference timesteps
inference_timesteps = [0, 334, 667]  # WAN uses 3 steps
timesteps = inference_timesteps[torch.randint(0, 3, (batch_size,))]
```

**Why?**
- Model only sees timesteps it will use at inference
- Better sample quality with fewer steps
- Matches training distribution to test distribution

### 7.5 Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Loss NaN | Training crashes | Lower learning rate, add grad clipping |
| Blurry output | Low-frequency artifacts | Train longer, check VAE quality |
| Color shift | Hue/saturation wrong | Normalize data properly [-1, 1] |
| Temporal flicker | Frame-to-frame inconsistency | Increase temporal context, check mask |
| Action ignored | No response to input | Verify action module is training |

---

## 8. Inference After Training

### 8.1 Denoising Loop

```python
@torch.no_grad()
def generate_video(
    dit,
    vae,
    first_frame,
    num_latent_frames=4,
    num_steps=3,
    device="cuda",
):
    """
    Generate video from first frame using trained DiT.

    Args:
        dit: Trained SimpleDiT
        vae: Trained SimpleVideoVAE (frozen)
        first_frame: [1, 3, H, W] in [-1, 1]
        num_latent_frames: Number of latent frames to generate (output = 4× this)
        num_steps: Denoising steps (default 3 for flow matching)

    Returns:
        video: [1, 3, T, H, W] generated video where T = num_latent_frames * 4
    """
    dit.eval()
    vae.eval()

    # For SimpleVideoVAE, we need 4 frames to produce 1 latent frame
    # To encode the first frame, we repeat it 4 times
    first_frame_4 = first_frame.unsqueeze(2).expand(-1, -1, 4, -1, -1)  # [1, 3, 4, H, W]
    first_latent, _ = vae.encode(first_frame_4)  # [1, 16, 1, H', W']

    # Get latent dimensions
    _, C, _, H_lat, W_lat = first_latent.shape

    # Initialize with pure noise
    latents = torch.randn(1, C, num_latent_frames, H_lat, W_lat, device=device)

    # Set first frame from encoding
    latents[:, :, 0, :, :] = first_latent[:, :, 0, :, :]

    # Build conditioning
    mask = torch.zeros(1, 4, num_latent_frames, H_lat, W_lat, device=device)
    mask[:, :, 0, :, :] = 1.0

    context = torch.zeros(1, 16, num_latent_frames, H_lat, W_lat, device=device)
    context[:, :, 0, :, :] = first_latent[:, :, 0, :, :]

    cond_concat = torch.cat([mask, context], dim=1)
    visual_context = torch.randn(1, 257, 1280, device=device)  # Placeholder

    # Denoising schedule (reverse: from noise to clean)
    scheduler = FlowMatchScheduler()
    timesteps = [667, 334, 0][:num_steps]  # High to low noise

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device)
        sigma = scheduler.get_sigma(t_tensor, device)

        # Predict flow
        predicted_flow = dit(latents, t_tensor.float(), cond_concat, visual_context)

        # Update latents: move toward clean
        # x_0 = x_t - sigma * flow
        sigma_expanded = sigma.view(-1, 1, 1, 1, 1)
        latents = latents - sigma_expanded * predicted_flow

        # Preserve first frame
        latents[:, :, 0, :, :] = first_latent[:, :, 0, :, :]

        # If not last step, re-add noise for next timestep
        if i < len(timesteps) - 1:
            next_t = timesteps[i + 1]
            next_sigma = scheduler.get_sigma(torch.tensor([next_t], device=device), device)
            noise = torch.randn_like(latents)
            next_sigma_expanded = next_sigma.view(-1, 1, 1, 1, 1)
            latents = (1 - next_sigma_expanded) * latents + next_sigma_expanded * noise
            latents[:, :, 0, :, :] = first_latent[:, :, 0, :, :]

    # Decode to video
    video = vae.decode(latents)  # [1, 3, T, H, W]

    return video.clamp(-1, 1)
```

### 8.2 Usage Example

```python
# Load models
vae = SimpleVideoVAE(z_dim=16, base_dim=96)
vae.load_state_dict(torch.load("vae_final.pth"))

dit = SimpleDiT(in_channels=36, out_channels=16, dim=1536, ...)
dit.load_state_dict(torch.load("dit_final.pth"))

# Load first frame
from PIL import Image
first_frame = Image.open("frame_0000.png").convert('RGB')
first_frame = torch.tensor(np.array(first_frame)).permute(2, 0, 1).float() / 127.5 - 1
first_frame = first_frame.unsqueeze(0).cuda()  # [1, 3, H, W]

# Generate video
# num_latent_frames=4 → output 16 video frames (4 latent × 4 = 16 video frames)
video = generate_video(dit, vae, first_frame, num_latent_frames=4, num_steps=3)

# Save frames
print(f"Generated {video.shape[2]} frames")  # 16 frames
for i in range(video.shape[2]):
    frame = video[0, :, i, :, :]  # [3, H, W]
    frame = ((frame + 1) * 127.5).clamp(0, 255).byte()
    frame = frame.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(frame).save(f"output_frame_{i:04d}.png")
```

---

## 9. Quick Reference

### Tensor Shapes

```
TRAINING (SimpleVideoVAE with 4-frame input):
─────────────────────────────────────────────
Input video:        [B, 3, T, H, W]           e.g., [2, 3, 4, 352, 640]
VAE latent:         [B, 16, T/4, H/8, W/8]    e.g., [2, 16, 1, 44, 80]
Conditioning:       [B, 20, T/4, H/8, W/8]    e.g., [2, 20, 1, 44, 80]
DiT input:          [B, 36, T/4, H/8, W/8]    e.g., [2, 36, 1, 44, 80]
After patch embed:  [B, L, 1536]              e.g., [2, 1760, 1536]  (1×44×40)
Visual context:     [B, 257, 1536]            e.g., [2, 257, 1536]
Timestep:           [B]                        e.g., [2]
DiT output:         [B, 16, T/4, H/8, W/8]    e.g., [2, 16, 1, 44, 80]

TRAINING (SimpleVideoVAE with 8-frame input):
─────────────────────────────────────────────
Input video:        [B, 3, 8, H, W]           e.g., [2, 3, 8, 352, 640]
VAE latent:         [B, 16, 2, H/8, W/8]      e.g., [2, 16, 2, 44, 80]
After patch embed:  [B, L, 1536]              e.g., [2, 3520, 1536]  (2×44×40)

INFERENCE:
──────────
First frame:        [1, 3, H, W]              e.g., [1, 3, 352, 640]
Generated video:    [1, 3, T, H, W]           e.g., [1, 3, 16, 352, 640]
```

### Key Formulas

```
VAE Temporal Compression (SimpleVideoVAE):
  T_latent = T_video // 4
  T_video = T_latent * 4

  NOTE: Different from full WAN VAE which uses:
    T_latent = 1 + (T_video - 1) // 4
    T_video = 1 + 4 * (T_latent - 1)

Flow Matching Forward:
  x_t = (1 - σ_t) * x_0 + σ_t * ε      where ε ~ N(0,1)

Flow Matching Target:
  flow = ε - x_0

Flow Matching Update (inference):
  x_0 = x_t - σ_t * flow_pred

Loss:
  L = E[w(t) * ||flow_pred - flow_target||²]
  where w(t) = exp(-(σ-0.5)²/0.125) + 0.05
```

### Hyperparameters Table

| Category | Parameter | Default | Range |
|----------|-----------|---------|-------|
| **Model** | dim | 1536 | 768-2048 |
| | num_heads | 12 | 8-16 |
| | num_layers | 30 | 12-40 |
| | ffn_dim | 8960 | 4×dim to 6×dim |
| **Training** | lr | 5e-6 | 1e-6 to 1e-4 |
| | batch_size | 1 | 1-8 |
| | grad_accum | 4 | 1-16 |
| | epochs | 100 | 50-500 |
| **Diffusion** | num_steps | 3 | 3-10 |
| | shift | 5.0 | 1.0-10.0 |
| **Data** | image_size | (352, 640) | divisible by 8 |
| | num_latent_frames | 1 | 1-4 (SimpleVideoVAE) |
| | num_video_frames | 4 | 4, 8, 12, 16... |

---

## Appendix A: Flow Matching vs DDPM

### Side-by-Side Comparison

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Forward process | `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε` | `x_t = (1-σ_t) x_0 + σ_t ε` |
| Prediction target | Noise ε | Flow v = ε - x_0 |
| Update rule | Complex (multiple terms) | Simple: `x_0 = x_t - σ_t v` |
| Typical steps | 20-1000 | 3-10 |
| Schedule | Discrete βs | Continuous σ(t) |

### Why Flow Matching for Video?

1. **Fewer steps**: 3 steps vs 20+ for DDPM means 7× faster generation
2. **Simpler math**: Linear interpolation is easier to implement and debug
3. **Better for long sequences**: Each frame must be consistent; fewer steps = less drift
4. **Industry adoption**: Used by Stable Video Diffusion, WAN, etc.

---

## Appendix B: Memory Optimization

### Memory Breakdown (30-layer DiT, batch=1, 17 frames @ 720p)

```
Component                    Memory (GB)
─────────────────────────────────────────
VAE encoding (peak)          2.0
DiT parameters               5.2
DiT activations (fwd)        8.0
DiT gradients (bwd)          5.2
Optimizer states (AdamW)     10.4
──────────────────────────────────────
Total (no optimization)      ~31 GB

With gradient checkpointing:
DiT activations (fwd)        2.0  (vs 8.0)
──────────────────────────────────────
Total (checkpointing)        ~25 GB

With bfloat16:
All components               ~12-15 GB
```

### Gradient Checkpointing Implementation

```python
class DiTBlockCheckpointed(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.use_checkpoint = True
        # ... same as regular DiT block

    def forward(self, x, t_emb, context):
        if self.training and self.use_checkpoint:
            # Checkpoint: don't store activations, recompute in backward
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x, t_emb, context,
                use_reentrant=False,
            )
        return self._forward_impl(x, t_emb, context)

    def _forward_impl(self, x, t_emb, context):
        # Actual forward pass
        x = x + self.self_attn(self.norm1(x), t_emb)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.ffn(self.norm3(x), t_emb)
        return x
```

### Multi-GPU Training (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model
dit = dit.to(local_rank)
dit = DDP(dit, device_ids=[local_rank])

# Use DistributedSampler
sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# In training loop
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Important for shuffling
    for batch in dataloader:
        ...
```

---

## Summary

Training the DiT for Matrix-Game-2 follows this pipeline:

1. **Encode** training videos to VAE latent space (frozen VAE)
2. **Add noise** using flow matching: `x_t = (1-σ) x_0 + σ ε`
3. **Predict flow** with DiT: `v = DiT(x_t, t, conditions)`
4. **Compute loss**: `L = ||v - (ε - x_0)||²` with timestep weighting
5. **Update** DiT parameters via backprop

Key points:
- VAE is **always frozen** - only DiT trains
- Use **flow matching** (velocity prediction), not DDPM (noise prediction)
- Train on **inference timesteps** [0, 334, 667] for best results
- **Gradient accumulation** and **checkpointing** are essential for memory

After training, generate videos by iteratively denoising from pure noise, conditioned on the first frame.
