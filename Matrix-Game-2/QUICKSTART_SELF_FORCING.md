# Quick Start: Self-Forcing Training

## TL;DR

Self-forcing trains your model on its own predictions, not just ground truth. This prevents error accumulation in long video generation.

```bash
# Standard training (OLD way - has error accumulation)
python finetune_causal_distilled.py --data_dir data --num_epochs 10

# Self-forcing training (NEW way - reduces error accumulation)
python finetune_causal_distilled_self_forcing.py \
    --data_dir data \
    --self_forcing_mode curriculum \
    --num_epochs 10
```

## 30-Second Explanation

**Problem**: During training, your model sees perfect ground truth frames. During inference, it sees its own imperfect predictions. This mismatch causes quality to degrade over time.

**Solution**: Train the model on its own predictions so it learns to handle realistic (imperfect) inputs.

## Getting Started

### Step 1: Verify Your Setup

Make sure you have:
- ✓ Training data in `data/` directory
- ✓ Pretrained model weights
- ✓ Standard training script works (optional but recommended)

### Step 2: Run Self-Forcing Training

#### Option A: Recommended (Curriculum Learning)
Gradually increases self-forcing from 0% to 90% over training:

```bash
python finetune_causal_distilled_self_forcing.py \
    --data_dir data/my_gameplay \
    --checkpoint_dir checkpoints_sf \
    --model_variant gta_drive \
    --self_forcing_mode curriculum \
    --self_forcing_prob_start 0.0 \
    --self_forcing_prob_end 0.9 \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6
```

#### Option B: Conservative (Safer)
Fixed 30% self-forcing throughout training:

```bash
python finetune_causal_distilled_self_forcing.py \
    --data_dir data/my_gameplay \
    --self_forcing_mode scheduled \
    --self_forcing_prob_end 0.3 \
    --num_epochs 10
```

#### Option C: Aggressive (Best Results)
High self-forcing (95%) for maximum train/test alignment:

```bash
python finetune_causal_distilled_self_forcing.py \
    --data_dir data/my_gameplay \
    --self_forcing_mode curriculum \
    --self_forcing_prob_start 0.2 \
    --self_forcing_prob_end 0.95 \
    --num_epochs 15 \
    --learning_rate 3e-6
```

### Step 3: Monitor Training

Watch the progress bar for these indicators:

```
Epoch 5/10:  50%|████▌    | loss: 0.0234, sf_prob: 0.45, sf_count: 127, gt_count: 138
                                           ^^^^^^^^  ^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^
                                           Current    Self-     Ground truth vs
                                           prob.      forcing   self-forcing samples
```

### Step 4: Test Your Model

After training, test on long-horizon generation:

```bash
python inference.py \
    --config_path configs/inference_yaml/inference_gta_drive.yaml \
    --checkpoint_path checkpoints_sf/causal_distilled_sf_best.safetensors \
    --img_path demo_images/universal/0000.png \
    --num_output_frames 300  # Test long generation!
```

Compare with standard model on the **same** long sequence.

## Key Parameters Explained

| Parameter | What It Does | Recommended Value |
|-----------|--------------|-------------------|
| `--self_forcing_mode` | How to schedule self-forcing | `curriculum` |
| `--self_forcing_prob_start` | Starting % of self-forcing | `0.0` |
| `--self_forcing_prob_end` | Final % of self-forcing | `0.8-0.9` |
| `--num_conditioning_frames` | How many GT frames to start with | `1` |
| `--learning_rate` | Learning rate (lower than standard!) | `3e-6` to `5e-6` |

## Expected Results

### Short Videos (< 30 frames)
- Standard and self-forcing should perform similarly
- Both produce high-quality output

### Medium Videos (30-100 frames)
- Self-forcing starts to show benefits
- Less quality degradation over time
- More stable motion

### Long Videos (100+ frames)
- **Big difference!**
- Standard: Quality degrades, artifacts accumulate
- Self-forcing: Maintains quality, stable generation

## Troubleshooting

### "Training is too slow"
- Self-forcing is ~2x slower (it generates frames during training)
- Reduce `self_forcing_prob_end` to 0.5 for faster training
- Use fewer `inference_steps` (should be 1 for distilled models)

### "Loss is higher than standard training"
- **This is normal!** Self-forcing is a harder task
- Focus on inference quality, not training loss
- If loss is much higher (>2x), reduce self-forcing probability

### "Generated frames are blurry"
- Check that model is using `eval()` mode during generation
- Verify `inference_steps` is correct for your model
- May need more training epochs

### "No improvement over baseline"
- Make sure `self_forcing_prob_end` is high enough (>0.7)
- Test on longer sequences (100+ frames) to see benefits
- Check that self-forcing is actually being used (sf_count > 0)

## Visualize the Difference

Generate comparison plots:

```bash
python compare_training_approaches.py
```

This creates visualizations in `visualizations/` showing:
- How standard vs self-forcing training works
- Error accumulation comparison
- Training schedule options

## FAQ

**Q: Will this work with my custom model?**
A: Yes! As long as you have an autoregressive/causal model that generates video frames sequentially.

**Q: Can I use this with non-distilled models?**
A: Yes! Just adjust `--inference_steps` to match your model (e.g., 50 for non-distilled).

**Q: Does this work with flow matching?**
A: Yes! Self-forcing is about what you condition on, flow matching is about how you train. They're compatible.

**Q: Should I always use self-forcing?**
A: For long-horizon video generation, yes! For short clips, standard training is fine.

**Q: How much better is self-forcing?**
A: Depends on sequence length. For 100+ frame generation, you can see 2-5x reduction in quality degradation.

## Next Steps

1. ✅ Train with self-forcing
2. ✅ Compare with standard model on long sequences
3. ✅ Tune hyperparameters for your specific use case
4. ✅ Deploy and enjoy stable long-horizon generation!

## Advanced: Hybrid Approach

For best results, consider a two-stage approach:

```bash
# Stage 1: Warmup with standard training (fast, stable)
python finetune_causal_distilled.py \
    --data_dir data \
    --num_epochs 5

# Stage 2: Finetune with self-forcing (better long-term)
python finetune_causal_distilled_self_forcing.py \
    --data_dir data \
    --pretrained_checkpoint checkpoints/causal_distilled_best.safetensors \
    --self_forcing_mode curriculum \
    --num_epochs 10
```

This gives you:
- Fast initial convergence (stage 1)
- Robust long-horizon performance (stage 2)

## Support

For detailed explanation, see: `SELF_FORCING_README.md`

For issues or questions, check the troubleshooting section above.

---

**Remember**: Self-forcing is about training your model to be resilient to its own mistakes, just like training in the real world!

