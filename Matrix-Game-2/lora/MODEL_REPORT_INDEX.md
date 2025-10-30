# GTA Distilled Model - Complete Documentation Package

This package contains comprehensive documentation for implementing LoRA on the GTA distilled model.

## Generated Files

### 1. `GTA_MODEL_ANALYSIS_REPORT.txt`
**Comprehensive technical analysis covering:**
- Basic model information (1.62B parameters, 30 layers, 12 heads)
- Action module architecture (mouse + keyboard control)
- Parameter breakdown by module type
- Layer-by-layer structure details
- Attention mechanism specifications
- LoRA implementation recommendations
- Code samples for PEFT integration
- Training considerations and hyperparameters

**Key Statistics:**
- Total Parameters: 1,619,238,464 (~1.62B)
- Self-Attention: 283.4M params (17.5%)
- Cross-Attention: 283.4M params (17.5%)  
- Feed-Forward: 826.1M params (51.0%)
- Action Modules: 205.2M params (12.7%)
- Model Size: 6.03 GB

### 2. `LORA_IMPLEMENTATION_GUIDE.md`
**Practical implementation guide with:**
- Why LoRA is ideal for this model (4-6x speedup, 80% memory reduction)
- Four LoRA strategies (attention-only, attention-ffn, full, action-focused)
- Complete training scripts with PEFT
- Manual LoRA implementation examples
- Hyperparameter recommendations
- Troubleshooting common issues
- Advanced topics (QLoRA, multi-adapter, etc.)

**Recommended Starting Point:**
- Strategy: `attention_ffn`
- Rank: 16, Alpha: 32
- Learning Rate: 1e-4
- Expected trainable params: ~20-40M (~1.5% of model)

### 3. `analyze_gta_model.py`
**Automated analysis script that:**
- Loads model weights and configuration
- Analyzes layer structure and parameters
- Counts parameters by module type
- Examines attention mechanisms
- Deep dives into action modules
- Generates LoRA recommendations
- Produces the detailed text report

**Usage:**
```bash
python analyze_gta_model.py | tee GTA_MODEL_ANALYSIS_REPORT.txt
```

## Model Architecture Summary

### Core Components
```
GTA Distilled Model (CausalWanModel)
├── 30 Transformer Blocks
│   ├── Self-Attention (RoPE, 12 heads, dim 128)
│   ├── Cross-Attention (image conditioning)
│   ├── Feed-Forward Network (1536 → 8960 → 1536)
│   └── Action Module (in 15 blocks)
│       ├── Mouse: 2D input → 1024 hidden → attention
│       └── Keyboard: 2D input → 128 hidden → attention
├── Patch Embedding (3D conv)
├── Time Embedding (sinusoidal, 256-dim)
├── Image Embedding (CLIP, 1280 → 1536)
└── Output Head (deconv)
```

### Action Module Details
- **Enabled Blocks:** 0-14 (first 15 of 30 blocks)
- **Mouse Input:** 2D (x, y camera movement)
- **Keyboard Input:** 2D (gas/brake for GTA, vs 4D WASD for base)
- **Temporal Window:** 3 frames with VAE 4x compression
- **Architecture:** Dual-branch (mouse + keyboard) with cross-attention

## LoRA Strategy Comparison

| Strategy | Target Modules | Rank | Params | Use Case |
|----------|---------------|------|--------|----------|
| Attention Only | Q,K,V,O (self+cross) | 8 | ~5-10M | Small data, fast |
| Attention + FFN | Above + FFN layers | 16 | ~20-40M | **Recommended** |
| Full LoRA | Above + action modules | 32 | ~80-120M | Large data, max quality |
| Action Focused | Action + self-attn | 16 | ~15-25M | Control fine-tuning |

## Implementation Checklist

### Prerequisites
- [x] Model files in `models/gta_distilled_model/`
- [x] Python 3.10+ with PyTorch 2.0+
- [x] CUDA GPU with 20GB+ VRAM (24GB recommended)
- [x] Install PEFT: `pip install peft transformers`

### Steps to Implement LoRA

1. **Choose Strategy**
   ```python
   # For most use cases, start here:
   strategy = "attention_ffn"
   rank = 16
   alpha = 32
   ```

2. **Load Model**
   ```python
   from wan.modules.causal_model import CausalWanModel
   model = CausalWanModel.from_pretrained("models/gta_distilled_model")
   ```

3. **Configure LoRA**
   ```python
   from peft import LoraConfig, get_peft_model
   
   config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=[
           "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
           "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
           "ffn.0", "ffn.2",
       ],
       lora_dropout=0.05,
       bias="none",
   )
   
   model = get_peft_model(model, config)
   ```

4. **Train**
   ```bash
   python finetune_gta_lora.py \
       --data_dir data/gta_gameplay \
       --lora_strategy attention_ffn \
       --lora_rank 16 \
       --num_epochs 10
   ```

5. **Inference**
   ```python
   from peft import PeftModel
   
   base = CausalWanModel.from_pretrained("models/gta_distilled_model")
   model = PeftModel.from_pretrained(base, "checkpoints/lora_adapter_best")
   model.eval()
   ```

## Expected Performance

### Training Metrics
| Metric | Full Fine-tuning | LoRA (r=16) | Improvement |
|--------|------------------|-------------|-------------|
| VRAM | 24-32 GB | 18-22 GB | 25-30% less |
| Speed | 0.5 it/s | 2-3 it/s | 4-6x faster |
| Epoch Time | 8-12 min | 2-3 min | 4-6x faster |
| Checkpoint Size | 6.0 GB | 50-200 MB | 30-120x smaller |

### Quality
- LoRA achieves 95-99% of full fine-tuning quality
- Better generalization due to implicit regularization
- Faster convergence with higher learning rates

## Key Insights for LoRA Implementation

### 1. Why This Model Benefits from LoRA
- **Large size (1.6B params):** Full fine-tuning is expensive
- **Repetitive structure (30 identical blocks):** Low-rank updates are efficient  
- **Attention-heavy (35% of params):** LoRA excels on attention layers
- **Modular actions (12.7% of params):** Easy to target specific components

### 2. Module Selection Priority
1. **High Priority:** Self-attention Q,K,V,O (most important for generation)
2. **Medium Priority:** Cross-attention Q,K,V,O (image conditioning)
3. **Medium Priority:** FFN layers (complex transformations)
4. **Low Priority:** Action modules (already specialized, may not need adaptation)

### 3. Rank Selection Guidelines
- **Rank 4-8:** Minimal changes, good for small datasets
- **Rank 16:** Sweet spot for most use cases
- **Rank 32-64:** Large datasets, maximum quality
- **Rule of thumb:** Start low, increase if underfitting

### 4. Common Pitfalls
- ❌ Applying LoRA to normalization layers (not needed, adds overhead)
- ❌ Using rank > 64 (diminishing returns, risk of overfitting)
- ❌ Training with full model learning rate (use 2-10x higher for LoRA)
- ❌ Forgetting to load adapter during inference
- ✅ Start with attention-only, add FFN if needed
- ✅ Use gradient checkpointing to save memory
- ✅ Monitor both training and validation loss

## Additional Resources

### Related Documentation in This Repository
- `CAUSAL_FINETUNING_GUIDE.md` - Guide for full fine-tuning (compare with LoRA)
- `FINETUNING_GUIDE.md` - General fine-tuning overview
- `finetune_causal_distilled.py` - Reference implementation (adapt for LoRA)
- `wan/modules/causal_model.py` - Model architecture source code
- `wan/modules/action_module.py` - Action module implementation

### External References
1. LoRA Paper: https://arxiv.org/abs/2106.09685
2. QLoRA Paper: https://arxiv.org/abs/2305.14314
3. PEFT Documentation: https://huggingface.co/docs/peft
4. Matrix-Game 2.0 Paper: https://arxiv.org/abs/2508.13009

## Version Information

- **Model Version:** GTA Distilled Model (keyboard 2D)
- **Model Class:** CausalWanModel (WanModel with causal attention)
- **Diffusers Version:** 0.33.1
- **Base Model:** Skywork/SkyReels-V2-I2V-1.3B-540P
- **Documentation Generated:** October 30, 2025

## Quick Reference

### File Sizes
- Model weights: 6.03 GB
- Config file: <1 KB  
- LoRA adapter (r=16): ~50-200 MB
- Full training checkpoint: 6.03 GB
- LoRA training checkpoint: 50-200 MB

### Key Dimensions
- Hidden dim: 1536
- FFN dim: 8960
- Attention heads: 12
- Head dim: 128
- Layers: 30
- Latent channels: 36 (in), 16 (out)

### Action Dimensions
- Mouse: 2D (x, y)
- Keyboard: 2D (gas, brake) for GTA
- Keyboard: 4D (W,A,S,D) for base model
- Temporal window: 3 frames
- VAE compression: 4x

---

## Getting Started

**Recommended reading order:**

1. Start with this index for overview
2. Read `LORA_IMPLEMENTATION_GUIDE.md` for practical steps
3. Refer to `GTA_MODEL_ANALYSIS_REPORT.txt` for technical details
4. Use `analyze_gta_model.py` to analyze your own checkpoints

**For immediate implementation:**

```bash
# 1. Analyze the model
python analyze_gta_model.py

# 2. Start training with recommended settings  
python finetune_gta_lora.py \
    --data_dir data/gta_gameplay \
    --lora_strategy attention_ffn \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_epochs 10 \
    --learning_rate 1e-4

# 3. Test inference
python inference.py \
    --config_path configs/inference_yaml/inference_gta.yaml \
    --checkpoint_path models/gta_distilled_model/gta_keyboard2dim.safetensors \
    --lora_adapter checkpoints/lora_adapter_best \
    --img_path demo_images/gta_drive/0000.png \
    --output_folder outputs
```

---

## Contact & Support

For questions or issues:
- Review the troubleshooting sections in `LORA_IMPLEMENTATION_GUIDE.md`
- Check existing documentation in this repository
- Refer to the PEFT library documentation
- Review the Matrix-Game 2.0 paper for architectural details

**Happy fine-tuning! 🚗💨**
