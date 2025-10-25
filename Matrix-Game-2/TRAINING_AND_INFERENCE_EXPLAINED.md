# Training and Inference Explained

## Overview

The Matrix-Game-2.0 model uses **Flow Matching** for video generation. This document explains how training and inference work with the correct implementation.

---

## What is Flow Matching?

Flow Matching is a diffusion approach that learns to transform noise into data through a continuous flow.

### Key Concepts

**Traditional Diffusion (DDPM/DDIM):**
- Model predicts **noise** (ε)
- Formula: `x₀ = (xₜ - √(1-αₜ) · ε) / √αₜ`
- Uses alpha/beta schedules

**Flow Matching (What we use):**
- Model predicts **velocity** (v = ε - x₀)
- Formula: `xₜ₊₁ = xₜ + v · (σₜ₊₁ - σₜ)`
- Uses sigma schedules with shifting

### Why Flow Matching?

- ✅ More stable training
- ✅ Better sample quality
- ✅ Flexible timestep scheduling
- ✅ Natural interpolation properties

---

## Training Process

### Common Components (Both Scripts)

1. **Initialize FlowMatchScheduler**
   ```python
   scheduler = FlowMatchScheduler(
       shift=5.0,           # Shifts the noise schedule
       sigma_min=0.0,       # Minimum noise level
       extra_one_step=True  # Additional timestep for stability
   )
   scheduler.set_timesteps(1000, training=True)
   ```

2. **Sample Random Timesteps**
   ```python
   # Sample random timesteps for each batch item
   timestep_indices = torch.randint(0, 1000, (batch_size,), device='cpu')
   timesteps = scheduler.timesteps[timestep_indices].to(device, dtype=bfloat16)
   ```

3. **Add Noise to Clean Data**
   ```python
   # Start with clean latents
   latents = torch.randn(...)  # Placeholder for actual encoded frames
   noise = torch.randn_like(latents)
   
   # Add noise using Flow Matching schedule
   noisy_latents = scheduler.add_noise(latents, noise, timestep_indices)
   # Formula: noisy = (1 - σₜ) · latents + σₜ · noise
   ```

4. **Model Forward Pass**
   ```python
   # Model predicts velocity (not noise!)
   predicted_velocity = model(
       noisy_latents,
       timesteps,
       visual_context,      # CLIP embeddings
       cond_concat,         # Conditional info
       mouse_cond,          # Mouse actions
       keyboard_cond        # Keyboard actions
   )
   ```

5. **Compute Target and Loss**
   ```python
   # Compute velocity target: v = noise - x₀
   target_velocity = scheduler.training_target(latents, noise, timestep_indices)
   
   # Train to predict velocity
   loss = MSE(predicted_velocity, target_velocity)
   ```

### Base Model Training (`finetune_base_model.py`)

**Purpose:** Fine-tune the model on your game data to learn game-specific patterns.

**Process:**
```
Input: 9 video frames → [B, 9, 352, 640, 3]
   ↓
Encode to latents → [B, 16, 3, 44, 80]  (VAE compression: 8x spatial, 4x temporal)
   ↓
Add noise at random timestep t → noisy_latents
   ↓
Model predicts velocity → predicted_v
   ↓
Loss = MSE(predicted_v, target_v)
   ↓
Backprop & update weights
```

**What it learns:**
- How video frames evolve over time
- Game-specific visual patterns
- Action-conditioned motion (keyboard/mouse → video changes)

---

### Autoregressive Training (`finetune_autoregressive.py`)

**Purpose:** Teach the model to generate long videos by chaining predictions.

**Process:**
```
Input: 18 frames split into:
  - Context: first 9 frames (what we've generated so far)
  - Target: next 9 frames (what we want to generate)
  
Training:
  ↓
Encode both to latents
  ↓
For TARGET frames:
  1. Add noise at random timestep
  2. Model predicts velocity (conditioned on context)
  3. Loss = MSE(predicted_v, target_v)
  ↓
(Optional) Curriculum Learning - first 1/3 of epochs:
  For CONTEXT frames:
  1. Also train to reconstruct context
  2. Helps model understand video continuity
  3. Combined loss = 0.3 · context + 0.7 · target
  ↓
Backprop & update
```

**What it learns:**
- How to continue a video from previous frames
- Temporal consistency across long sequences
- Autoregressive conditioning (using past to predict future)

**Key Difference from Base:**
- Base model: predicts frames independently
- Autoregressive: predicts frames **given previous frames**

---

## Inference Process

### Autoregressive Inference (`inference_autoregressive.py`)

**Purpose:** Generate long videos by iteratively generating chunks.

**Setup:**
```python
# Initialize scheduler for inference
scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
scheduler.set_timesteps(100)  # 100 denoising steps
```

**Process:**

```
Start with initial image
   ↓
Encode to latent → first_latent [1, 16, 1, 44, 80]
   ↓
┌─────────────────────────────────────────────┐
│ LOOP: Generate 9 frames at a time          │
│                                             │
│  1. Sample pure noise → [1, 16, 3, 44, 80] │
│     (3 latent frames = 9 video frames)      │
│                                             │
│  2. Denoising Loop (100 steps):             │
│     for timestep in [999, 989, 979, ... 0]: │
│        ↓                                     │
│        Model predicts velocity              │
│        ↓                                     │
│        Update: latents += v · Δσ            │
│        (Flow Matching update rule)          │
│                                             │
│  3. Decode latents → 9 video frames         │
│                                             │
│  4. Use these as context for next chunk     │
│                                             │
└─────────────────────────────────────────────┘
   ↓
Repeat until desired length
   ↓
Concatenate all chunks → full video
   ↓
Add keyboard/mouse overlays
   ↓
Save video file
```

**Detailed Denoising:**

```python
# Start with pure noise
latents = torch.randn([1, 16, 3, 44, 80])

# Denoise step by step
for i, timestep in enumerate(scheduler.timesteps):
    # 1. Model predicts velocity at this noise level
    predicted_v = model(
        latents, 
        timestep,
        visual_context,   # From first frame CLIP encoding
        cond_concat,      # Previous frames + mask
        mouse_cond,       # Desired mouse movements
        keyboard_cond     # Desired key presses
    )
    
    # 2. Update latents using Flow Matching
    # Formula: x_next = x_current + v · (σ_next - σ_current)
    latents = scheduler.step(predicted_v, timestep, latents)

# After 100 steps: pure noise → clean latents
```

---

## Key Parameters

### Scheduler Parameters

- **`shift=5.0`**: Shifts the noise schedule to focus more steps on important noise levels
- **`sigma_min=0.0`**: Minimum noise (fully clean)
- **`sigma_max=1.0`**: Maximum noise (default, pure noise)
- **`num_timesteps=1000`**: Total timesteps during training
- **`num_inference_steps=100`**: Steps used during inference (fewer = faster, more = better quality)

### Model Inputs

- **`latents`**: [B, 16, T, 44, 80] - The noisy video latents
- **`timestep`**: [B] - Current noise level (0-1000)
- **`visual_context`**: [B, 1280] - CLIP embedding of first frame
- **`cond_concat`**: [B, 20, T, 44, 80] - Mask + encoded previous frames
- **`mouse_cond`**: [B, num_frames, 2] - Mouse movements (dx, dy)
- **`keyboard_cond`**: [B, num_frames, 4] - Keyboard state (W, A, S, D)

### Output

- **`predicted_velocity`**: [B, 16, T, 44, 80] - The velocity field v = ε - x₀

---

## Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Input** | Clean latents + noise | Pure noise |
| **Timesteps** | Random single timestep per batch | All timesteps (100 steps) |
| **Goal** | Learn to predict velocity | Generate clean video |
| **Loss** | MSE(predicted_v, target_v) | No loss (sampling) |
| **Duration** | 1 forward pass per batch | 100 forward passes per chunk |
| **Output** | Loss value | Video frames |

---

## Flow Diagram

### Training Flow
```
Video Frames (ground truth)
        ↓
    VAE Encode
        ↓
Clean Latents (x₀)
        ↓
Sample Random Noise (ε) & Timestep (t)
        ↓
Add Noise: xₜ = (1-σₜ)·x₀ + σₜ·ε
        ↓
Model Forward: predict velocity v
        ↓
Compute Target: v_target = ε - x₀
        ↓
Loss = MSE(v, v_target)
        ↓
Backprop & Update Weights
```

### Inference Flow
```
Start: Pure Noise (x₁₀₀₀)
        ↓
    ┌─────────┐
    │ Loop:   │
    │ t=999→0 │
    └─────────┘
        ↓
Predict Velocity: v = model(xₜ, t, conditions)
        ↓
Update: xₜ₋₁ = xₜ + v·(σₜ₋₁ - σₜ)
        ↓
    (repeat)
        ↓
End: Clean Latents (x₀)
        ↓
    VAE Decode
        ↓
Video Frames (generated)
```

---

## Why This Matters

### Before the Fix ❌

**Training:**
- Used custom linear schedule
- Predicted noise (ε)
- Target: `ε`

**Inference:**
- Used custom DDIM schedule
- Interpreted output as noise
- Formula: `x₀ = (xₜ - √(1-α)·ε) / √α`

**Problem:** Training and inference used different math! Like training in metric but measuring in imperial.

### After the Fix ✅

**Training:**
- Uses FlowMatchScheduler
- Predicts velocity (v)
- Target: `v = ε - x₀`

**Inference:**
- Uses FlowMatchScheduler
- Interprets output as velocity
- Formula: `xₜ₊₁ = xₜ + v·Δσ`

**Result:** Perfect alignment! Training and inference use the same mathematical framework.

---

## Quick Reference

### Start Training (Base Model)
```bash
python finetune_base_model.py
```
- Trains on 9-frame sequences
- Learns basic video generation
- ~10 epochs recommended

### Start Training (Autoregressive)
```bash
python finetune_autoregressive.py
```
- Trains on 18-frame sequences (9 context + 9 target)
- Learns long-form generation
- ~100 epochs with curriculum learning

### Run Inference
```bash
python inference_autoregressive.py \
    --checkpoint_path checkpoints/autoregressive_best.safetensors \
    --num_output_frames 81 \
    --action_mode random
```
- Generates 81-frame video (9 steps × 9 frames)
- Uses trained autoregressive model
- Outputs to `outputs/autoregressive_demo.mp4`

---

## Troubleshooting

### Common Issues

**Training loss not decreasing:**
- Check learning rate (default: 1e-5 base, 5e-6 autoregressive)
- Ensure dataset has enough variation
- Verify actions match your game (WASD layout)

**Generated videos are blurry:**
- Train for more epochs
- Use smaller learning rate
- Check if VAE decoder is properly loaded

**Videos have temporal inconsistency:**
- Train autoregressive model longer
- Increase num_inference_steps (try 200)
- Ensure context frames are correctly passed

**CUDA out of memory:**
- Reduce batch_size (default: 8 base, 1 autoregressive)
- Use gradient checkpointing (if available)
- Reduce num_output_frames during inference

---

## Advanced Topics

### Curriculum Learning

In autoregressive training, the first 33 epochs also train on context reconstruction:
- Helps model learn temporal continuity
- Weight: 30% context loss + 70% target loss
- After epoch 33: only target loss

### Timestep Shifting

The `shift=5.0` parameter modifies the noise schedule:
```python
# Without shift: uniform timesteps
σ = linspace(σ_min, σ_max, 1000)

# With shift: more focus on mid-range noise
σ_shifted = shift · σ / (1 + (shift - 1) · σ)
```
This allocates more denoising steps to the noise levels that matter most.

### Action Conditioning

Mouse and keyboard actions are embedded and added to the latent features:
- **Mouse**: 2D continuous (dx, dy) → embedded per frame spatially
- **Keyboard**: 4D binary (W, A, S, D) → embedded per frame globally
- Each transformer block has dedicated action attention

### Autoregressive Context

During generation, previous latents are encoded in `cond_concat`:
- First 4 channels: **mask** (1 for past frames, 0 for future)
- Next 16 channels: **encoded latents** from previous generations
- Total: 20 channels concatenated with current 16 latent channels → 36 input channels

---

## Summary

**Flow Matching** provides a stable, high-quality diffusion framework for video generation.

**Training** teaches the model to predict velocity fields that transform noise into video frames.

**Inference** uses those learned velocity predictions to iteratively denoise random noise into coherent videos.

**Autoregressive generation** chains multiple denoising processes together, using previous outputs as context for future predictions, enabling long-form video generation.

The key to success is **alignment**: training and inference must use the same mathematical framework (Flow Matching) for optimal results.

