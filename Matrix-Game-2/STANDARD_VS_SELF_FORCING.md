# Standard Training vs Self-Forcing: Side-by-Side Comparison

## Quick Answer

**Q: Should I use self-forcing?**

A: If you're generating videos longer than ~50 frames, **YES**. Self-forcing prevents quality degradation over time.

## Visual Comparison

### Standard Training Flow
```
┌─────────────┐
│ Load Frame  │
│ from Data   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Encode to   │
│ Latents     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Add Noise   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Train Model │
└─────────────┘
```

### Self-Forcing Training Flow
```
┌─────────────┐
│ Load Frame  │
│ from Data   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Encode to   │
│ Latents     │
└──────┬──────┘
       │
       ├──────────────────────────┐
       │                          │
       ▼                          ▼
  [50% chance]              [50% chance]
┌─────────────┐          ┌─────────────┐
│ Use GT      │          │ Generate    │
│ Latents     │          │ from Model  │
└──────┬──────┘          └──────┬──────┘
       │                        │
       └────────────┬───────────┘
                    ▼
              ┌─────────────┐
              │ Add Noise   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Train Model │
              └─────────────┘
```

## Code Comparison

### Standard Training
```python
# finetune_causal_distilled.py (line ~190-236)

# Encode frames to latents
latents = vae.encode(frames)

# Prepare conditioning
img_cond = latents.clone()
cond_concat = torch.cat([mask_cond, img_cond], dim=1)

# Add noise
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

# Train
predicted = model(noisy_latents, conditioning)
loss = mse_loss(predicted, target)
loss.backward()
```

### Self-Forcing Training
```python
# finetune_causal_distilled_self_forcing.py (line ~330-420)

# Encode frames to latents
latents_gt = vae.encode(frames)

# SELF-FORCING DECISION
use_self_forcing = (np.random.rand() < self_forcing_prob)

if use_self_forcing:
    # Generate frames from model
    model.eval()
    with torch.no_grad():
        generated = generate_frame_chunk(
            model, latent_cond, actions, ...
        )
        latents = torch.cat([latent_cond, generated], dim=2)
    model.train()
else:
    # Use ground truth
    latents = latents_gt

# Prepare conditioning (same as standard)
img_cond = latents.clone()
cond_concat = torch.cat([mask_cond, img_cond], dim=1)

# Add noise (same as standard)
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

# Train (same as standard, but supervise against GT)
predicted = model(noisy_latents, conditioning)
target = compute_target(latents_gt, noise, timesteps)  # Note: GT target!
loss = mse_loss(predicted, target)
loss.backward()
```

## Feature Comparison Table

| Feature | Standard | Self-Forcing |
|---------|----------|--------------|
| **Conditioning Source** | Always GT | Mix of GT + Model |
| **Train/Test Match** | ❌ Poor | ✅ Excellent |
| **Error Accumulation** | ❌ High | ✅ Low |
| **Short Videos (<30f)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Long Videos (>100f)** | ⭐⭐ | ⭐⭐⭐⭐ |
| **Training Time** | 1x | ~2x |
| **Training Stability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Implementation** | Simple | Moderate |
| **Memory Usage** | Lower | Higher |

## Parameter Differences

### Standard Training Command
```bash
python finetune_causal_distilled.py \
    --data_dir data \
    --model_variant gta_drive \
    --batch_size 1 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4
```

### Self-Forcing Training Command
```bash
python finetune_causal_distilled_self_forcing.py \
    --data_dir data \
    --model_variant gta_drive \
    --batch_size 1 \
    --num_epochs 10 \
    --learning_rate 5e-6 \              # Lower!
    --gradient_accumulation_steps 4 \
    --self_forcing_mode curriculum \    # New!
    --self_forcing_prob_start 0.0 \     # New!
    --self_forcing_prob_end 0.9 \       # New!
    --num_conditioning_frames 1         # New!
```

## When to Use Each Approach

### Use Standard Training When:
- ✅ Generating short videos (<30 frames)
- ✅ You need fastest training time
- ✅ You have limited compute budget
- ✅ You're doing quick prototyping
- ✅ Model quality is already excellent

### Use Self-Forcing Training When:
- ✅ Generating long videos (>50 frames)
- ✅ You observe quality degradation over time
- ✅ You can afford ~2x training time
- ✅ You need production-quality long generation
- ✅ Error accumulation is a problem

## Real-World Impact

### Example: 150-frame Video Generation

**Standard Training Results:**
```
Frame   1-30:  ⭐⭐⭐⭐⭐ Perfect quality
Frame  31-60:  ⭐⭐⭐⭐   Slight blur
Frame  61-90:  ⭐⭐⭐     Noticeable artifacts
Frame 91-120:  ⭐⭐       Significant degradation
Frame 121-150: ⭐         Unusable quality
```

**Self-Forcing Training Results:**
```
Frame   1-30:  ⭐⭐⭐⭐⭐ Perfect quality
Frame  31-60:  ⭐⭐⭐⭐⭐ Still excellent
Frame  61-90:  ⭐⭐⭐⭐   Slight blur
Frame 91-120:  ⭐⭐⭐⭐   Maintained quality
Frame 121-150: ⭐⭐⭐     Acceptable quality
```

## Training Progress Comparison

### Standard Training Output
```
Epoch 5/10: 100%|████████| loss: 0.0234, lr: 1.00e-05
```

### Self-Forcing Training Output
```
Epoch 5/10: 100%|████████| loss: 0.0234, sf_prob: 0.45, sf_count: 127, gt_count: 138, lr: 5.00e-06
                                         ^^^^^^^^^^^  ^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
                                         Self-forcing Self-forcing   Ground truth
                                         probability  samples        samples
```

## Loss Curves

### Expected Training Loss

```
Standard Training:
Loss │
     │ ╲
     │  ╲___________
     │              ────────
     │
     └────────────────────────> Epoch

Self-Forcing Training:
Loss │
     │ ╲
     │  ╲___
     │      ╲______________
     │                     ─────   (Slightly higher, but better generalization)
     │
     └────────────────────────> Epoch
```

**Note**: Self-forcing loss may be 10-20% higher, but this is expected and acceptable. The model is learning a harder task.

## Memory Usage Comparison

### Standard Training
```
GPU Memory Usage:
├── Model: ~8 GB
├── Optimizer: ~8 GB
├── Activations: ~4 GB
├── VAE (temp): ~2 GB
└── Total: ~22 GB
```

### Self-Forcing Training
```
GPU Memory Usage:
├── Model: ~8 GB
├── Optimizer: ~8 GB
├── Activations: ~4 GB
├── VAE (temp): ~2 GB
├── Generation Buffer: ~2 GB  ← Extra
└── Total: ~24 GB
```

## Inference Quality Metrics

Assuming you test on 100-frame generation:

| Metric | Standard | Self-Forcing | Improvement |
|--------|----------|--------------|-------------|
| **PSNR (avg)** | 24.5 dB | 27.2 dB | +11% |
| **SSIM (avg)** | 0.75 | 0.86 | +15% |
| **FVD (lower better)** | 450 | 280 | -38% |
| **Perceptual Quality** | 3.2/5 | 4.1/5 | +28% |
| **Temporal Consistency** | 0.68 | 0.84 | +24% |

*Note: These are illustrative numbers. Actual results depend on your specific data and model.*

## Migration Path

### Step 1: Baseline (Current State)
```bash
# Your current training
python finetune_causal_distilled.py --data_dir data --num_epochs 10
```

### Step 2: Test Self-Forcing (Conservative)
```bash
# Try self-forcing with low probability
python finetune_causal_distilled_self_forcing.py \
    --data_dir data \
    --self_forcing_mode scheduled \
    --self_forcing_prob_end 0.3 \
    --num_epochs 5
```

### Step 3: Full Self-Forcing (Recommended)
```bash
# Use curriculum learning for best results
python finetune_causal_distilled_self_forcing.py \
    --data_dir data \
    --self_forcing_mode curriculum \
    --self_forcing_prob_end 0.9 \
    --num_epochs 10
```

### Step 4: Comparison
```bash
# Generate long sequences with both models
python inference.py --checkpoint_path standard_model.safetensors --num_output_frames 150
python inference.py --checkpoint_path sf_model.safetensors --num_output_frames 150

# Compare quality visually and with metrics
```

## Common Questions

### Q: Can I use both approaches?
**A:** Yes! Train with standard first (fast convergence), then finetune with self-forcing (better long-term quality).

```bash
# Stage 1: Standard (5 epochs)
python finetune_causal_distilled.py --num_epochs 5

# Stage 2: Self-forcing (10 epochs)
python finetune_causal_distilled_self_forcing.py \
    --pretrained_checkpoint checkpoints/causal_distilled_best.safetensors \
    --num_epochs 10
```

### Q: Is self-forcing always better?
**A:** For long sequences (>50 frames), yes. For short sequences (<30 frames), the difference is minimal.

### Q: Why is training slower?
**A:** Self-forcing generates frames during training, which requires forward passes through the model. This approximately doubles training time.

### Q: Can I reduce the computational cost?
**A:** Yes, use lower `self_forcing_prob_end` (e.g., 0.5 instead of 0.9). You'll still get benefits with lower cost.

### Q: Does self-forcing work with non-distilled models?
**A:** Yes! Just adjust `--inference_steps` to match your model (e.g., 50 for non-distilled).

## Decision Tree

```
Do you need to generate videos > 50 frames?
│
├─ NO → Use Standard Training
│        ✓ Faster
│        ✓ Simpler
│        ✓ Good quality for short videos
│
└─ YES → Do you observe error accumulation?
         │
         ├─ NO → Start with Standard, monitor quality
         │
         └─ YES → Use Self-Forcing Training
                  ✓ Reduces error accumulation
                  ✓ Better long-term quality
                  ✓ More stable generation
```

## Summary

| Aspect | Standard | Self-Forcing | Winner |
|--------|----------|--------------|--------|
| **Short videos** | Excellent | Excellent | 🤝 Tie |
| **Long videos** | Poor | Excellent | 🏆 Self-Forcing |
| **Training speed** | Fast | Slower | 🏆 Standard |
| **Memory usage** | Lower | Higher | 🏆 Standard |
| **Implementation** | Simple | Complex | 🏆 Standard |
| **Error accumulation** | High | Low | 🏆 Self-Forcing |
| **Train/test match** | Poor | Excellent | 🏆 Self-Forcing |
| **Production quality** | Good | Excellent | 🏆 Self-Forcing |

## Conclusion

- For **quick prototyping** and **short videos**: Use standard training
- For **production deployment** and **long videos**: Use self-forcing training
- For **best of both worlds**: Train standard first, then finetune with self-forcing

**Bottom Line**: If your videos are longer than 50 frames and quality matters, the 2x training cost of self-forcing is worth it for the significant quality improvement.

