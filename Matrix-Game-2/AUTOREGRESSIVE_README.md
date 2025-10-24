# Autoregressive Training for Long Videos

## The Problem
`finetune_base_model.py` only generates 9-frame videos. Can't make longer videos.

## The Solution
Two new scripts for autoregressive generation (unlimited length):

### 1. Train: `finetune_autoregressive.py`
Trains model to continue videos (context → next frames)

```bash
python finetune_autoregressive.py
```

**What it does:**
- Splits sequences into pairs: [9 context frames] → [9 target frames]
- Teaches model to predict next 9 frames given previous 9
- Uses curriculum learning (easier tasks first)
- Saves best checkpoint to `checkpoints/autoregressive_best.safetensors`

**Requirements:**
- Data sequences ≥18 frames (9 context + 9 target)
- ~10 min/epoch training time
- ~18 GB VRAM

### 2. Inference: `inference_autoregressive.py`
Generates long videos by chaining predictions

```bash
# Generate 81 frames (9 generations)
python inference_autoregressive.py --num_output_frames 81 --action_mode forward

# Generate 243 frames
python inference_autoregressive.py --num_output_frames 243
```

**How it works:**
```
Step 1: [Image] + Actions → Frames₁ (0-8)
Step 2: [Frames₁[-1]] + Actions → Frames₂ (9-17)
Step 3: [Frames₂[-1]] + Actions → Frames₃ (18-26)
...
Result: Long coherent video
```

## Key Differences

| Feature | Basic | Autoregressive |
|---------|-------|----------------|
| Max length | 9 frames | Unlimited |
| Training | Independent clips | Context → Target pairs |
| Output | Single clip | Chained continuations |

## Configuration

Edit in `finetune_autoregressive.py`:
```python
CONTEXT_FRAMES = 9      # Input length
TARGET_FRAMES = 9       # Prediction length
BATCH_SIZE = 4          # Reduce if OOM
NUM_EPOCHS = 15         # More = better quality
USE_CURRICULUM = True   # Recommended
```

## Troubleshooting

**"No sequences long enough"** → Need 18+ frame sequences  
**Out of memory** → Reduce `BATCH_SIZE = 2`  
**Video has jumps** → Train longer or check you're using autoregressive checkpoint

That's it!

