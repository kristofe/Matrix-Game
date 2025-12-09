# Matrix-Game-2 Model Architecture

> **Model:** Causal Diffusion Transformer (distilled from Wan2.1)
> **Parameters:** ~1.42 billion
> **Purpose:** Action-conditioned video generation for driving games

---

## Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                     WanDiffusionWrapper                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    CausalWanModel                         │  │
│  │                                                           │  │
│  │   [Patch Embed] ──► [30 Transformer Blocks] ──► [Head]    │  │
│  │        │                     │                     │      │  │
│  │        │              ┌──────┴──────┐              │      │  │
│  │        │              │   Per Block │              │      │  │
│  │        │              ├─────────────┤              │      │  │
│  │        │              │ Self-Attn   │              │      │  │
│  │        │              │ Cross-Attn  │              │      │  │
│  │        │              │ Action*     │◄── blocks 0-14      │  │
│  │        │              │ FFN         │              │      │  │
│  │        │              └─────────────┘              │      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Model Configuration

| Property | Value |
|----------|-------|
| **Hidden Dimension** | 1536 |
| **FFN Dimension** | 8960 |
| **Transformer Layers** | 30 |
| **Attention Heads** | 12 |
| **Head Dimension** | 128 |
| **Input Channels** | 36 (VAE latent + mask) |
| **Output Channels** | 16 (VAE latent) |
| **Patch Size** | (1, 2, 2) temporal, height, width |

---

## 2. Architecture Diagram

```
INPUT
  │
  │  video_latents: [B, 36, T, H/8, W/8]
  │  timestep: [B]
  │  visual_context: [B, 1280]  (CLIP)
  │  keyboard_cond: [B, T, 2]
  │  mouse_cond: [B, T, 2]
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PATCH EMBEDDING                            │
│  Conv3d(36 → 1536, kernel=(1,2,2), stride=(1,2,2))              │
│  Output: [B, 1536, T, H/16, W/16]                               │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TIME EMBEDDING                              │
│  Sinusoidal(256) → Linear(256→1536) → SiLU → Linear(1536→1536)  │
│  → Linear(1536→9216) → reshape to 6 modulation vectors          │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    IMAGE EMBEDDING                              │
│  LayerNorm(1280) → Linear(1280→1536) → GELU → Linear → LN       │
│  (Projects CLIP visual features to model dimension)             │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
╔═════════════════════════════════════════════════════════════════╗
║            TRANSFORMER BLOCKS (×30)                             ║
╠═════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │ SELF-ATTENTION (Causal with RoPE)                       │    ║
║  │   norm1(x) → Q, K, V projections → FlexAttention        │    ║
║  │   Q/K: RMSNorm applied before attention                 │    ║
║  │   Supports KV-caching for autoregressive generation     │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                            │                                    ║
║                            ▼                                    ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │ CROSS-ATTENTION (Image Conditioning)                    │    ║
║  │   Q from hidden, K/V from image embeddings              │    ║
║  │   Injects visual context from input frame               │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                            │                                    ║
║                            ▼                                    ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │ ACTION MODULE (blocks 0-14 only)                        │    ║
║  │   ┌─────────────┐     ┌──────────────┐                  │    ║
║  │   │ Mouse Ctrl  │     │ Keyboard Ctrl│                  │    ║
║  │   │ MLP → Attn  │     │ Embed → Attn │                  │    ║
║  │   │ → proj      │     │ → proj       │                  │    ║
║  │   └─────────────┘     └──────────────┘                  │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                            │                                    ║
║                            ▼                                    ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │ FEED-FORWARD NETWORK                                    │    ║
║  │   norm2(x) → Linear(1536→8960) → GELU → Linear(→1536)   │    ║
║  │   Uses adaptive modulation from timestep                │    ║
║  └─────────────────────────────────────────────────────────┘    ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT HEAD                               │
│  LayerNorm → Linear(1536 → 64) → Unpatchify                     │
│  Output: [B, 16, T, H/8, W/8] (VAE latent space)                │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
OUTPUT (velocity prediction for flow matching)
```

---

## 3. Parameter Breakdown

### By Component

```
┌────────────────────────┬────────────────┬─────────┐
│ Component              │ Parameters     │ % Total │
├────────────────────────┼────────────────┼─────────┤
│ Patch Embedding        │ 221K           │ 0.02%   │
│ Time Embeddings        │ 16.9M          │ 1.2%    │
│ Image Embedding        │ 6.7M           │ 0.5%    │
│ Self-Attention (×30)   │ 283M           │ 20%     │
│ Cross-Attention (×30)  │ 283M           │ 20%     │
│ FFN Networks (×30)     │ 823M           │ 58%     │
│ Action Modules (×15)   │ 95M            │ 6.7%    │
│ Output Head            │ 98K            │ 0.01%   │
├────────────────────────┼────────────────┼─────────┤
│ TOTAL                  │ ~1.42B         │ 100%    │
└────────────────────────┴────────────────┴─────────┘
```

### Visual Breakdown

```
Parameter Distribution:
═══════════════════════════════════════════════════════════

FFN (58%)         ████████████████████████████████████████████
Self-Attn (20%)   ███████████████
Cross-Attn (20%)  ███████████████
Action (6.7%)     █████
Other (1.7%)      █

═══════════════════════════════════════════════════════════
```

### Per Transformer Block

```
Single Block (~47M params):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Self-Attention    ████████░░░░░░░░░░  9.4M  (20%)      │
│  Cross-Attention   ████████░░░░░░░░░░  9.4M  (20%)      │
│  FFN               ████████████████████ 27.5M (58%)     │
│  Action Module*    ███░░░░░░░░░░░░░░░░  6.3M  (13%)*    │
│  Norms/Modulation  ░░░░░░░░░░░░░░░░░░░  0.4M  (1%)      │
│                                                         │
│  * Only in blocks 0-14                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Attention Layer Details

### Self-Attention Structure

```
                    Input: x [B, seq, 1536]
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Q proj  │     │ K proj  │     │ V proj  │
        │1536→1536│     │1536→1536│     │1536→1536│
        └────┬────┘     └────┬────┘     └────┬────┘
             │               │               │
             ▼               ▼               │
        ┌─────────┐     ┌─────────┐          │
        │ RMSNorm │     │ RMSNorm │          │
        └────┬────┘     └────┬────┘          │
             │               │               │
             ▼               ▼               ▼
        ┌─────────────────────────────────────────┐
        │          FlexAttention                  │
        │   • 12 heads × 128 dim                  │
        │   • RoPE positional encoding            │
        │   • Causal mask (autoregressive)        │
        │   • KV-cache for inference              │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
                    ┌─────────┐
                    │ O proj  │
                    │1536→1536│
                    └────┬────┘
                         │
                         ▼
                 Output: [B, seq, 1536]
```

### Parameter Names (for LoRA)

```python
# Self-Attention (per block i)
blocks.{i}.self_attn.q.weight      # [1536, 1536] - 2.36M params
blocks.{i}.self_attn.k.weight      # [1536, 1536] - 2.36M params
blocks.{i}.self_attn.v.weight      # [1536, 1536] - 2.36M params
blocks.{i}.self_attn.o.weight      # [1536, 1536] - 2.36M params
blocks.{i}.self_attn.norm_q.weight # [128]
blocks.{i}.self_attn.norm_k.weight # [128]

# Cross-Attention (per block i)
blocks.{i}.cross_attn.q.weight     # [1536, 1536]
blocks.{i}.cross_attn.k.weight     # [1536, 1536]
blocks.{i}.cross_attn.v.weight     # [1536, 1536]
blocks.{i}.cross_attn.o.weight     # [1536, 1536]
```

---

## 5. Action Module Details

### Enabled in Blocks 0-14 (First 15 of 30)

```
┌─────────────────────────────────────────────────────────────────┐
│                       ACTION MODULE                             │
│                                                                 │
│  ┌─────────────────────────┐    ┌─────────────────────────┐     │
│  │     MOUSE CONTROL       │    │    KEYBOARD CONTROL     │     │
│  │                         │    │                         │     │
│  │  mouse_cond [B,T,2]     │    │  keyboard_cond [B,T,2]  │     │
│  │         │               │    │         │               │     │
│  │         ▼               │    │         ▼               │     │
│  │  ┌─────────────┐        │    │  ┌─────────────┐        │     │
│  │  │  mouse_mlp  │        │    │  │keyboard_emb │        │     │
│  │  │  (4 layers) │        │    │  │ 2→128→128   │        │     │
│  │  └──────┬──────┘        │    │  └──────┬──────┘        │     │
│  │         │               │    │         │               │     │
│  │         ▼               │    │         ▼               │     │
│  │  ┌─────────────┐        │    │  ┌─────────────┐        │     │
│  │  │   t_qkv     │        │    │  │  attn_q     │←─ x    │     │
│  │  │  QKV proj   │        │    │  │  attn_kv    │        │     │
│  │  └──────┬──────┘        │    │  └──────┬──────┘        │     │
│  │         │               │    │         │               │     │
│  │         ▼               │    │         ▼               │     │
│  │  ┌─────────────┐        │    │  ┌─────────────┐        │     │
│  │  │  Attention  │        │    │  │  Attention  │        │     │
│  │  │  (RoPE)     │        │    │  │  (Cross)    │        │     │
│  │  └──────┬──────┘        │    │  └──────┬──────┘        │     │
│  │         │               │    │         │               │     │
│  │         ▼               │    │         ▼               │     │
│  │  ┌─────────────┐        │    │  ┌─────────────┐        │     │
│  │  │ proj_mouse  │        │    │  │proj_keyboard│        │     │
│  │  │ 1024→1536   │        │    │  │ 1024→1536   │        │     │
│  │  └──────┬──────┘        │    │  └──────┬──────┘        │     │
│  │         │               │    │         │               │     │
│  └─────────┼───────────────┘    └─────────┼───────────────┘     │
│            │                              │                     │
│            └──────────────┬───────────────┘                     │
│                           ▼                                     │
│                    x = x + mouse_out + keyboard_out             │
└─────────────────────────────────────────────────────────────────┘
```

### Action Module Parameters

```python
# Mouse control (per block i, blocks 0-14)
blocks.{i}.action_model.mouse_mlp.0.weight    # Expansion
blocks.{i}.action_model.mouse_mlp.2.weight    # Hidden
blocks.{i}.action_model.t_qkv.weight          # QKV projection
blocks.{i}.action_model.proj_mouse.weight     # 1024→1536

# Keyboard control
blocks.{i}.action_model.keyboard_embed.0.weight  # 2→128
blocks.{i}.action_model.keyboard_embed.2.weight  # 128→128
blocks.{i}.action_model.mouse_attn_q.weight      # Query from x
blocks.{i}.action_model.keyboard_attn_kv.weight  # KV from keyboard
blocks.{i}.action_model.proj_keyboard.weight     # 1024→1536
```

### Action Module Config

```json
{
  "blocks": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
  "mouse_dim_in": 2,
  "keyboard_dim_in": 2,
  "hidden_size": 128,
  "mouse_hidden_dim": 1024,
  "keyboard_hidden_dim": 1024,
  "heads_num": 16,
  "window_size": 3,
  "vae_time_compression_ratio": 4,
  "rope_theta": 256
}
```

---

## 6. FFN (Feed-Forward Network)

```
         Input: x [B, seq, 1536]
                    │
                    ▼
           ┌────────────────┐
           │   LayerNorm    │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │    Linear      │
           │  1536 → 8960   │  ← 13.7M params (expansion)
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │     GELU       │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │    Linear      │
           │  8960 → 1536   │  ← 13.7M params (projection)
           └───────┬────────┘
                   │
                   ▼
         Output: [B, seq, 1536]

FFN Parameters per block: ~27.5M
FFN Parameters total (30 blocks): ~823M
```

---

## 7. LoRA Targeting Strategies

### Strategy 1: Attention Only (Conservative)

**Best for:** Small datasets, preserving original behavior

```python
target_modules = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
]
lora_config = LoraConfig(r=8, lora_alpha=16, ...)
```

```
Trainable params: ~5-10M (0.5% of model)
Memory savings:   ~60-70%
```

### Strategy 2: Attention + FFN (Balanced)

**Best for:** Medium datasets, good quality/efficiency balance

```python
target_modules = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
]
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
```

```
Trainable params: ~20-40M (2-3% of model)
Memory savings:   ~50-60%
```

### Strategy 3: Visual Only (Freeze Actions)

**Best for:** Domain adaptation (your use case!)

```python
target_modules = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
]
# Freeze all action_model parameters separately
modules_to_freeze = ["action_model"]
```

```
Trainable params: ~30-50M
Action behavior:  Preserved from original
```

### Strategy 4: Full LoRA (Maximum Adaptation)

**Best for:** Large datasets, significant domain shift

```python
target_modules = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
    "action_model.proj_mouse", "action_model.proj_keyboard",
]
lora_config = LoraConfig(r=32, lora_alpha=64, ...)
```

```
Trainable params: ~80-120M (6-8% of model)
Memory savings:   ~40-50%
```

---

## 8. Memory Estimates

### Training Memory (A100 40GB)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY USAGE COMPARISON                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full Finetuning (current):                                     │
│  ╔══════════════════════════════════════════╗                   │
│  ║ Model weights    ████████  3.2 GB        ║                   │
│  ║ Gradients        ████████  3.2 GB        ║                   │
│  ║ Optimizer (AdamW)████████████████ 6.4 GB ║                   │
│  ║ Activations      ████████████████████ 15-25 GB               │
│  ║ ─────────────────────────────────────────║                   │
│  ║ TOTAL            ~28-38 GB (OOM on A100) ║                   │
│  ╚══════════════════════════════════════════╝                   │
│                                                                 │
│  LoRA Finetuning (r=16):                                        │
│  ╔══════════════════════════════════════════╗                   │
│  ║ Model weights    ████████  3.2 GB        ║  (frozen)         │
│  ║ LoRA adapters    ░  ~50 MB               ║                   │
│  ║ Gradients        ░  ~50 MB               ║  (LoRA only)      │
│  ║ Optimizer        ░  ~100 MB              ║  (LoRA only)      │
│  ║ Activations      ████████████  8-12 GB   ║  (reduced)        │
│  ║ ─────────────────────────────────────────║                   │
│  ║ TOTAL            ~12-16 GB ✓             ║                   │
│  ╚══════════════════════════════════════════╝                   │
│                                                                 │
│  LoRA + LPIPS (with gradients):                                 │
│  ╔══════════════════════════════════════════╗                   │
│  ║ LoRA training    ████████████  12-16 GB  ║                   │
│  ║ LPIPS gradients  ████████  5-8 GB        ║                   │
│  ║ ─────────────────────────────────────────║                   │
│  ║ TOTAL            ~17-24 GB ✓             ║                   │
│  ╚══════════════════════════════════════════╝                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `utils/wan_wrapper.py` | Main wrapper for diffusion |
| `wan/modules/causal_model.py` | CausalWanModel architecture |
| `wan/modules/model.py` | Base WanModel |
| `wan/modules/action_module.py` | Action conditioning |
| `wan/modules/attention.py` | Attention implementations |
| `wan/modules/posemb_layers.py` | RoPE embeddings |
| `configs/distilled_model/gta_drive/config.json` | Model config |

---

## 10. Quick Code Reference

### Load Model

```python
from utils.wan_wrapper import WanDiffusionWrapper
from omegaconf import OmegaConf

config = OmegaConf.load("configs/inference_yaml/inference_gta_drive.yaml")
model = WanDiffusionWrapper(**config.model_kwargs, is_causal=True)
```

### Print Layer Names

```python
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### Apply LoRA

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Freeze Action Modules

```python
for name, param in model.named_parameters():
    if "action_model" in name:
        param.requires_grad = False
```

---

## 11. Key Insights for Finetuning

1. **Action modules are in first 15 blocks only** - Freeze them to preserve action behavior

2. **FFN is 58% of parameters** - Targeting FFN gives biggest impact per LoRA rank

3. **Attention Q/K/V/O are good LoRA targets** - Standard and effective

4. **RoPE is non-trainable** - Position encoding is pre-computed

5. **Cross-attention conditions on CLIP** - Image understanding happens here

6. **Causal generation uses KV-cache** - Important for inference, not training

7. **Time modulation affects all layers** - Timestep is critical for diffusion

---

*Generated for Matrix-Game-2 finetuning project*
