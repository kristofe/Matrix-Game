# Self-Forcing Training for Causal Video Models

## Overview

This document explains the **self-forcing training** implementation in `finetune_causal_distilled_self_forcing.py` and why it's critical for addressing error accumulation in autoregressive video generation.

## The Problem: Error Accumulation

### Standard Training (Teacher Forcing)
In traditional training:
1. Model conditions on **ground truth** previous frames
2. Predicts next frame
3. During inference, model conditions on **its own predictions**

This creates a **train/test mismatch**:
- **Training**: Model always sees perfect, ground truth frames
- **Inference**: Model sees its own imperfect predictions, which accumulate errors

### Example of Error Accumulation
```
Frame 1 (GT) → Model predicts Frame 2 (small error)
Frame 2 (predicted, slightly off) → Model predicts Frame 3 (larger error)
Frame 3 (predicted, more off) → Model predicts Frame 4 (even larger error)
...
Frame N (predicted, completely degraded)
```

## The Solution: Self-Forcing

### What is Self-Forcing?

Self-forcing trains the model to condition on **its own predictions** rather than ground truth, matching what happens during inference.

### How it Works

1. **Ground Truth Path** (probability `1 - p`):
   - Encode frames from dataset to latents
   - Use these GT latents as conditioning
   - Train with flow matching on GT latents

2. **Self-Forcing Path** (probability `p`):
   - Use first frame(s) from dataset as initial conditioning
   - **Generate** next frames using the model itself
   - Use these generated latents as conditioning
   - Train with flow matching, but supervise against GT

### Key Implementation Details

```python
# Decide whether to use self-forcing
use_self_forcing = (np.random.rand() < self_forcing_prob)

if use_self_forcing:
    # Generate frames from model's distribution
    model.eval()  # Use eval mode for generation
    with torch.no_grad():
        generated_latents = generate_frame_chunk(
            model, latent_cond, actions, ...
        )
        latents = torch.cat([latent_cond, generated_latents], dim=2)
    model.train()  # Back to training mode
else:
    # Use ground truth latents
    latents = latents_gt

# Train on the latents (GT or generated)
# But supervise against GT!
predicted = model(noisy_latents, conditioning, timesteps)
target = compute_target(latents_gt, noise, timesteps)
loss = mse_loss(predicted, target)
```

## Self-Forcing Modes

The script supports three training strategies:

### 1. Scheduled Sampling (`--self_forcing_mode scheduled`)
- Fixed probability throughout training
- E.g., 50% GT, 50% self-forcing for entire training
```bash
--self_forcing_mode scheduled \
--self_forcing_prob_end 0.5
```

### 2. Curriculum Learning (`--self_forcing_mode curriculum`)
- Gradually increase self-forcing probability
- Start with mostly GT, end with mostly self-forcing
- Recommended for stable training
```bash
--self_forcing_mode curriculum \
--self_forcing_prob_start 0.0 \
--self_forcing_prob_end 0.9
```

### 3. Full Self-Forcing (`--self_forcing_mode full`)
- Always use model predictions (after epoch 0)
- Most aggressive, may be unstable
```bash
--self_forcing_mode full
```

## Compatibility with Flow Matching

**Yes, self-forcing is fully compatible with flow matching!**

They address different aspects:
- **Flow Matching**: How to train the denoising process (velocity prediction)
- **Self-Forcing**: What to condition on (GT vs. model predictions)

The implementation:
1. Uses self-forcing to generate conditioning latents
2. Applies flow matching training on those latents
3. Supervises against ground truth targets

This is similar to how self-forcing works with other generative models (GANs, VAEs, autoregressive transformers).

## Usage Examples

### Basic Self-Forcing Training
```bash
python finetune_causal_distilled_self_forcing.py \
    --data_dir data/my_gameplay \
    --checkpoint_dir checkpoints_sf \
    --model_variant gta_drive \
    --self_forcing_mode curriculum \
    --self_forcing_prob_start 0.0 \
    --self_forcing_prob_end 0.8 \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4
```

### Conservative Self-Forcing (Safer)
```bash
python finetune_causal_distilled_self_forcing.py \
    --data_dir data/my_gameplay \
    --self_forcing_mode scheduled \
    --self_forcing_prob_end 0.3 \
    --num_conditioning_frames 2 \
    --learning_rate 5e-6
```

### Aggressive Self-Forcing (More Realistic)
```bash
python finetune_causal_distilled_self_forcing.py \
    --data_dir data/my_gameplay \
    --self_forcing_mode curriculum \
    --self_forcing_prob_start 0.2 \
    --self_forcing_prob_end 0.95 \
    --num_conditioning_frames 1 \
    --learning_rate 3e-6
```

## Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--self_forcing_mode` | Training strategy | `curriculum` |
| `--self_forcing_prob_start` | Initial SF probability | `0.0` |
| `--self_forcing_prob_end` | Final SF probability | `0.8-0.9` |
| `--num_conditioning_frames` | GT frames to start with | `1` |
| `--inference_steps` | Denoising steps for generation | `1` (distilled) |
| `--learning_rate` | Learning rate | `5e-6` (lower than standard) |

## Expected Benefits

1. **Reduced Error Accumulation**
   - Model learns to handle its own imperfect predictions
   - More stable long-horizon generation

2. **Better Train/Test Alignment**
   - Training distribution matches inference distribution
   - Model sees realistic conditioning during training

3. **Improved Robustness**
   - Model becomes resilient to its own mistakes
   - Less catastrophic failures in long rollouts

## Computational Considerations

Self-forcing has **~2x computational cost**:
- Standard training: Encode frames → Train
- Self-forcing: Encode frames → Generate frames → Train

Mitigations:
- Use `--inference_steps 1` for distilled models (already fast)
- Use scheduled sampling with lower probability (e.g., 0.3-0.5)
- Reduce batch size if needed

## Monitoring Training

Watch for these metrics:
- **Loss**: May be slightly higher than standard training (expected)
- **SF Count vs GT Count**: Check the ratio matches your probability
- **Generated Quality**: Monitor if model generates coherent frames

## Comparison with Standard Training

| Aspect | Standard Training | Self-Forcing Training |
|--------|------------------|----------------------|
| Conditioning | Always GT | Mix of GT + model predictions |
| Train/Test Match | Poor | Excellent |
| Training Stability | High | Medium (requires tuning) |
| Long-horizon Quality | Degrades | Stable |
| Training Time | 1x | ~2x |
| Error Accumulation | High | Low |

## Research Background

Self-forcing is based on:
- **Scheduled Sampling** (Bengio et al., 2015)
- **Professor Forcing** (Lamb et al., 2016)
- **Data Augmentation with Model Predictions**

It's widely used in:
- Autoregressive language models
- Video prediction models
- Sequence-to-sequence models
- Reinforcement learning (on-policy training)

## Troubleshooting

### Training is unstable
- Start with lower `self_forcing_prob_end` (e.g., 0.5)
- Use `curriculum` mode with gradual increase
- Increase `num_conditioning_frames` to 2-3
- Lower learning rate (e.g., 3e-6)

### Generated frames are blurry
- Check that model is in eval mode during generation
- Ensure `inference_steps` matches your model (1 for distilled)
- Verify VAE is working correctly

### No improvement over baseline
- Make sure self-forcing is actually being used (check SF count)
- Try increasing `self_forcing_prob_end` to 0.8-0.9
- Train for more epochs (self-forcing needs time to learn)

## Future Enhancements

Possible improvements:
1. **Dynamic self-forcing**: Adjust probability based on model quality
2. **Multi-step self-forcing**: Generate multiple frames at once
3. **Adversarial self-forcing**: Add discriminator for generated frames
4. **Mixture of distributions**: Blend GT and generated latents

## References

- Bengio, S., et al. (2015). "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"
- Lamb, A., et al. (2016). "Professor Forcing: A New Algorithm for Training Recurrent Networks"
- Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling"

