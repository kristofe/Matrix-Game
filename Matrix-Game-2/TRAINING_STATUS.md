# Matrix-Game Base Model Fine-tuning Status

## ✅ TRAINING IS RUNNING!

**Process ID:** 253631  
**Status:** Active and training  
**Model:** WanModel (base_model)  
**Strategy:** action_only (22.5% of parameters)

---

## Critical Discovery: Sequence Length Requirements

The base model's action module has **strict requirements** for sequence lengths:

### The Formula
With VAE temporal compression (4x) and `num_frame_per_block=3`:
- Original frames must satisfy: `(N_frames - 1) % 4 == 0`
- Compressed frames must equal: `(N_frames - 1) // 4 + 1`
- **For training: compressed frames MUST = 3** (the default block size)

### Valid Sequence Lengths
- ✅ **9 frames** → 3 latent frames (WORKING!)
- ❌ 13 frames → 4 latent frames (fails)
- ❌ 25 frames → 7 latent frames (fails)
- ❌ 57 frames → 15 latent frames (fails)

### Why This Matters
The WanModel's action module hard-codes `num_frame_per_block=3` and doesn't handle other sizes correctly in non-causal training mode. This is a limitation of the current implementation.

---

## Current Training Configuration

```python
SEQUENCE_LENGTH = 9          # 9 frames per sequence
BATCH_SIZE = 1               # Single batch
NUM_EPOCHS = 10              # 10 full epochs
LEARNING_RATE = 1e-5         # Fine-tuning rate
TRAINING_STRATEGY = 'action_only'  # Train action modules only
```

### Dataset Statistics
- **Total Frames:** 1,403
- **Sequences Created:** ~310 sequences (9 frames each with overlap)
- **Batches per Epoch:** ~310
- **Total Training Steps:** ~3,100 steps

### Model Architecture
- **Total Parameters:** 1,824,405,824 (1.8B)
- **Trainable Parameters:** 410,327,040 (410M)
- **Training:** 22.49% of model
- **Frozen:** 77.51% of model

---

## What Was Fixed

### Issue 1: Keyboard Dimension Mismatch
**Problem:** Base model trained with 6-dim keyboard, we use 4-dim WASD  
**Solution:** Filter incompatible weights, reinitialize keyboard embeddings  
**Result:** ✅ 30 keyboard embedding layers reinitialized

### Issue 2: Incorrect Latent Frame Count
**Problem:** Created latents with original frame count (57) instead of compressed count (15)  
**Solution:** Calculate compressed frames: `(N_frames - 1) // 4 + 1`  
**Result:** ✅ Latents now have correct temporal dimension

### Issue 3: Action Module Block Size Mismatch
**Problem:** Model expects exactly 3 latent frames, we had 15  
**Solution:** Use sequence length of 9 frames → 3 latent frames  
**Result:** ✅ Training now runs without errors

---

## Files Modified

1. **`finetune_base_model.py`**
   - Fixed keyboard weight loading with filtering
   - Corrected VAE compression calculation
   - Set sequence length to 9 frames
   - Added comprehensive debug output

2. **`convert_unreal_data.py`**
   - No changes needed
   - Works perfectly with any sequence length

---

## Monitoring Training

### Check if Running
```bash
ps aux | grep finetune_base_model
```

### Monitor GPU
```bash
watch -n 1 nvidia-smi
```

### Check Checkpoints
```bash
ls -lh checkpoints/
cat checkpoints/training_info.txt  # After training completes
```

---

## Expected Outputs

Training will save:
- `checkpoints/base_finetuned_epoch2.safetensors`
- `checkpoints/base_finetuned_epoch4.safetensors`
- `checkpoints/base_finetuned_epoch6.safetensors`
- `checkpoints/base_finetuned_epoch8.safetensors`
- `checkpoints/base_finetuned_epoch10.safetensors`
- `checkpoints/base_finetuned_final.safetensors`
- `checkpoints/training_info.txt`

---

## Performance Notes

### Training Speed
- **~1-2 batches/second** (depending on GPU)
- **~5 minutes per epoch** (310 batches)
- **~50 minutes total** (10 epochs)

### Memory Usage
- **Model:** ~7GB VRAM
- **Training:** ~3-4GB VRAM
- **Total:** ~10-12GB VRAM

---

## Limitations Discovered

### 1. Sequence Length Constraint
The base model can only train with 9-frame sequences due to hard-coded `num_frame_per_block=3` in the action module. This is a significant limitation for learning long-term dependencies.

### 2. Non-Causal Training Mode
The WanModel's action module has bugs when used in non-causal training mode with sequences that don't match the expected block size.

### 3. Keyboard Dimension Incompatibility
The base model was trained with 6-dim keyboard input (unknown format), but we're using 4-dim WASD. The keyboard embeddings are trained from scratch.

---

## Recommendations

### For Better Results
1. **Collect More Data:** 9-frame sequences are short; you'll need many examples
2. **Data Augmentation:** Add brightness/contrast variations
3. **Longer Training:** If loss is decreasing, train for more epochs
4. **Consider Distilled Models:** The GTA or Temple Run distilled models might be more flexible

### Alternative Approaches
1. **Use Distilled Models:** Check `models/gta_distilled_model/` or `models/templerun_distilled_model/`
2. **Modify Action Module:** Fix the hard-coded block size (advanced)
3. **Use CausalWanModel:** The universal model (if the time embedding issue can be resolved)

---

## Success Criteria

Training is successful if:
- ✅ Process runs without errors
- ✅ Loss decreases over epochs
- ✅ Checkpoints are saved
- ✅ Model generates coherent video with actions

---

**Last Updated:** Training started at 22:10  
**Status:** RUNNING ✅  
**ETA:** ~50 minutes from start

