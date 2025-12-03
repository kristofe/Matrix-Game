# Plan: Proper Finetuning of Matrix-Game-2 Distilled Model

## User Context
- **Goal**: Visual domain adaptation (new scenery/environment)
- **Data**: 90,000+ sequences (can generate more)
- **Hardware**: RTX 6000 Pro (96GB VRAM), can scale to multi-GPU
- **Priority**: Quick iteration to verify training is working before scaling
- **Known Issue**: Previous methods caused "blur to gray" - need to prevent this

## Overview

This plan outlines the proper approach to finetuning the Matrix-Game-2 distilled model, which is an autoregressive world model distilled from Wan2.1 architecture.

## Model Architecture Summary

### Base Architecture
- **Type**: Causal Diffusion Transformer (1.62B parameters)
- **Base**: Distilled from Wan2.1 / SkyReels-V2 (88% parameter reduction from 14B)
- **Key Innovation**: Block-wise causal attention with KV caching for autoregressive generation

### Core Components
1. **Patch Embedding**: Conv3d (36 input channels → 1536 dim)
2. **30 Transformer Blocks**: Each with self-attention, cross-attention, FFN
3. **Action Modules**: Integrated into first 15 blocks (205M params, 12.7% of model)
   - Mouse branch: 86.9M params (camera/steering)
   - Keyboard branch: 71.0M params (gas/brake)
4. **Causal Masking**: Block-wise attention preventing future frame leakage

### Distillation Characteristics
- Uses **Flow Matching** instead of DDPM (fewer denoising steps)
- Discrete timesteps at inference: `[1000, 666, 333]` (3 steps)
- Continuous timesteps during training: uniform sampling in `[0.05, 0.95]`

## Current Training Implementation Analysis

### What's Currently in finetune_simple.py

1. **Loss Function**: Flow Matching MSE + LPIPS
   - `target = noise - latents` (velocity prediction)
   - `flow_loss = MSE(flow_pred, target)`
   - `lpips_loss` on decoded frames (optional)

2. **Timestep Sampling**: Uniform `[0.05, 0.95]`
   - Same timestep for all frames in sequence
   - Avoids extreme values

3. **Conditioning**:
   - `cond_concat`: mask + latents (20 channels)
   - `visual_context`: CLIP features from first frame
   - `keyboard_cond`: [forward, back] binary actions
   - `mouse_cond`: [vertical, horizontal] steering

## Current Implementation Status

### Already Implemented ✓
- [x] Flow matching loss computation
- [x] Proper timestep sampling (continuous `[0.05, 0.95]`)
- [x] Correct masking (first frame = 1, rest = 0)
- [x] Action conditioning (keyboard + mouse)
- [x] LPIPS perceptual loss
- [x] TensorBoard logging
- [x] Video generation per epoch
- [x] Checkpoint saving
- [x] Timestamped output directories

### To Be Added
- [ ] EMA (Exponential Moving Average)
- [ ] Learning rate scheduler with warmup
- [ ] Gradient clipping
- [ ] BSMNTW timestep weighting
- [ ] Collapse detection metrics
- [ ] More frequent video generation

## Key Files to Reference

- `wan/modules/causal_model.py` - Model architecture with causal attention
- `wan/modules/action_module.py` - Action conditioning modules
- `utils/wan_wrapper.py` - Wrapper with Flow Matching scheduler
- `utils/scheduler.py` - FlowMatchScheduler with BSMNTW weighting
- `pipeline/causal_inference.py` - Inference with KV caching
- `configs/distilled_model/gta_drive/` - Model configuration

## Root Cause Analysis: "Blur to Gray" Problem

This is a common failure mode in diffusion finetuning. Likely causes:

1. **Mode Collapse / Posterior Collapse**: Model learns to predict the mean (gray) because it minimizes MSE
2. **Learning Rate Too High**: Catastrophic forgetting of pretrained features
3. **Missing Perceptual Loss**: Pure MSE loss rewards blurry outputs
4. **Timestep Distribution Mismatch**: Training on different timestep distribution than inference

### Prevention Strategies (Implemented in Plan)
- **LPIPS loss** (already added) - prevents blur by penalizing perceptual differences
- **Lower learning rate** with warmup - prevent catastrophic forgetting
- **EMA (Exponential Moving Average)** - stabilizes training
- **Gradient clipping** - prevents sudden parameter jumps
- **Monitor intermediate timesteps** - catch collapse early

---

## Recommended Implementation Changes

### Phase 1: Quick Iteration Setup (Verify Training Works)

**Goal**: Confirm loss decreases and generated videos improve before scaling.

#### 1.1 Add Gradient Clipping
```python
# After loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 1.2 Add EMA for Stability
```python
# In load_model() or main()
from copy import deepcopy
ema_model = deepcopy(model)
ema_decay = 0.9999

# After each optimizer.step():
with torch.no_grad():
    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(ema_decay).add_(model_p, alpha=1 - ema_decay)
```

#### 1.3 Increase Batch Size (You Have 96GB!)
```python
# Current: batch_size=3, sequence_length=9
# Recommended: batch_size=8-16, sequence_length=9-17
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
```

#### 1.4 Add Learning Rate Scheduler with Warmup
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

# Warmup for first 1000 steps, then cosine decay
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=1000)
cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)
lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[1000])

# Call after each optimizer.step():
lr_scheduler.step()
```

#### 1.5 Add Timestep Weighting (BSMNTW)
```python
# First, enable training mode in scheduler setup:
scheduler.set_timesteps(1000, training=True)

# In train_step(), after computing flow_loss:
weight = scheduler.training_weight(t_scalar * 1000)  # Convert to [0, 1000] range
weighted_flow_loss = flow_loss * weight.mean()
```

### Phase 2: Enhanced Monitoring

#### 2.1 Track Multiple Metrics
```python
# Log these to TensorBoard:
writer.add_scalar("Loss/flow_weighted", weighted_flow_loss, step)
writer.add_scalar("Loss/lpips", lpips_loss, step)
writer.add_scalar("Metrics/pred_x0_std", pred_x0.std().item(), step)  # Should NOT go to 0
writer.add_scalar("Metrics/flow_pred_std", flow_pred.std().item(), step)
writer.add_scalar("LR", optimizer.param_groups[0]['lr'], step)
```

#### 2.2 Early Warning: Detect Collapse
```python
# If pred_x0 std drops significantly, training is collapsing
if pred_x0.std().item() < 0.1:
    print("WARNING: Possible mode collapse detected!")
```

#### 2.3 Generate Videos at Multiple Checkpoints
```python
# Generate every N steps (not just epochs) for quick feedback
if step % 500 == 0:
    generate_video_file(ema_model, vae, ...)  # Use EMA model for eval
```

### Phase 3: Scale Up (After Verifying It Works)

#### 3.1 Increase Data Usage
```python
# Use full dataset
dataset = SimpleDataset(data_dir="...", sequence_length=17, max_sequences=-1)  # All 90K sequences
```

#### 3.2 Longer Sequences
```python
# sequence_length=17 → 5 latent frames → more temporal context
# sequence_length=33 → 9 latent frames → even better but more memory
```

#### 3.3 Multi-GPU (When Ready)
```python
# Using PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])
```

---

## File Modifications Summary

**File**: `finetune_simple.py`

| Location | Change |
|----------|--------|
| Imports | Add `from copy import deepcopy` and LR scheduler imports |
| After `load_model()` | Create EMA model with `deepcopy(model)` |
| `train_step()` | Return additional metrics (pred_x0_std, flow_pred_std) |
| After `loss.backward()` | Add gradient clipping |
| After `optimizer.step()` | Add EMA update and LR scheduler step |
| Training loop | Add enhanced TensorBoard logging |
| Training loop | Add video generation every N steps |

---

## Recommended Hyperparameters

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| Learning Rate | 1e-5 | 5e-6 | Lower to prevent forgetting |
| Batch Size | 3 | 8-12 | Use your VRAM |
| Sequence Length | 9 | 17 | More temporal context |
| LPIPS Weight | 0.1 | 0.1-0.3 | Increase if blur appears |
| EMA Decay | N/A | 0.9999 | Standard for diffusion |
| Gradient Clip | N/A | 1.0 | Stability |
| Warmup Steps | N/A | 1000 | Prevent early instability |

---

## Quick Validation Checklist

Before long training runs, verify:
- [ ] Loss decreases over first 100 steps
- [ ] `pred_x0.std()` stays above 0.3 (no collapse)
- [ ] Generated video at step 500 shows domain characteristics
- [ ] No NaN/Inf in gradients
- [ ] Learning rate follows expected schedule

---

## Implementation Order

1. **Add gradient clipping** (1 line) - immediate stability
2. **Add EMA** (~10 lines) - use for evaluation
3. **Increase batch size** (1 line) - use your hardware
4. **Add LR scheduler with warmup** (~5 lines) - prevent forgetting
5. **Add collapse detection logging** (~5 lines) - early warning
6. **Add BSMNTW weighting** (~3 lines) - better loss landscape
7. **Generate videos more frequently** - faster iteration

---

---

## Selective Weight Training Strategies

Since your goal is **visual domain adaptation** (not learning new action patterns), you have several options for which weights to train:

### Option 1: Full Finetuning (Current Approach)
**Train**: All 1.62B parameters
**Pros**: Maximum flexibility, can adapt everything
**Cons**: Risk of catastrophic forgetting, slower, more memory
```python
# Current default - all parameters trainable
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
```

### Option 2: Freeze Action Modules (RECOMMENDED for Visual Adaptation)
**Train**: ~1.4B params (diffusion backbone only)
**Freeze**: 205M params (action modules)
**Pros**: Preserves learned action-video mapping, faster training, less memory
**Cons**: Can't adapt to new action patterns
```python
# Freeze action modules - they already know how to translate actions to video changes
for name, param in model.named_parameters():
    if 'action_module' in name or 'mouse' in name or 'keyboard' in name:
        param.requires_grad = False

# Only optimize trainable params
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=5e-6)
print(f"Training {sum(p.numel() for p in trainable_params)/1e6:.1f}M parameters")
```

### Option 3: Only Train Action Modules
**Train**: 205M params (action modules only)
**Freeze**: 1.4B params (diffusion backbone)
**Pros**: Very fast, minimal risk of visual degradation
**Cons**: Won't adapt visual style at all
```python
# Freeze everything except action modules
for name, param in model.named_parameters():
    if 'action_module' not in name and 'mouse' not in name and 'keyboard' not in name:
        param.requires_grad = False
```

### Option 4: LoRA (Low-Rank Adaptation)
**Train**: ~10-50M new parameters (injected adapters)
**Freeze**: All 1.62B original parameters
**Pros**: Very efficient, can merge back, easy to experiment
**Cons**: Requires additional setup, may limit adaptation capacity
```python
# Using PEFT library
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # attention layers
    lora_dropout=0.05,
)
model = get_peft_model(model.model, lora_config)
model.print_trainable_parameters()  # Will show ~1-3% trainable
```

### Option 5: Layer-wise Learning Rate Decay
**Train**: All parameters but with different learning rates
**Pros**: Fine-grained control, early layers (low-level features) change less
**Cons**: More hyperparameters to tune
```python
# Higher LR for later layers (high-level features), lower for early layers
param_groups = []
num_layers = 30

for name, param in model.named_parameters():
    layer_num = 0
    # Extract layer number from parameter name
    for i in range(num_layers):
        if f'.{i}.' in name or f'blocks.{i}' in name:
            layer_num = i
            break

    # Decay factor: earlier layers get smaller LR
    lr_scale = 0.1 + 0.9 * (layer_num / num_layers)  # 0.1 to 1.0
    param_groups.append({
        'params': [param],
        'lr': 5e-6 * lr_scale
    })

optimizer = torch.optim.AdamW(param_groups)
```

### Recommendation for Your Use Case

Since you want **visual domain adaptation**:

1. **Start with Option 2** (Freeze Action Modules):
   - The action modules already learned the mapping from controls → video changes in GTA
   - Your new domain likely has similar driving dynamics
   - This preserves the "physics" while adapting the visuals

2. **If that doesn't work well**, try **Option 1** (Full Finetuning) with:
   - Very low learning rate (1e-6)
   - Strong gradient clipping
   - EMA for stability

3. **For quick experiments**, try **Option 4** (LoRA):
   - Fastest to iterate
   - Easy to compare different adaptations
   - Can always go back to original model

### How to Identify Action Module Parameters

To see what parameters are in the action modules:
```python
for name, param in model.named_parameters():
    if any(x in name.lower() for x in ['action', 'mouse', 'keyboard']):
        print(f"{name}: {param.numel()/1e6:.2f}M params")
```

### Memory Savings from Freezing

| Strategy | Trainable Params | Optimizer States | Approx VRAM Saved |
|----------|-----------------|------------------|-------------------|
| Full | 1.62B | 6.5GB | 0 |
| Freeze Actions | 1.4B | 5.6GB | ~1GB |
| Only Actions | 205M | 0.8GB | ~5.7GB |
| LoRA (r=16) | ~50M | 0.2GB | ~6.3GB |

---

## Notes

- The scheduler's `training_weight()` method uses BSMNTW (Biased Signal-to-Noise Timestep Weighting) which applies a Gaussian weighting centered at the middle of the timestep range
- EMA should be used for **evaluation only** - train with the main model, generate videos with EMA model
- When freezing parameters, make sure to only pass trainable parameters to the optimizer (otherwise you'll get warnings and waste memory)
