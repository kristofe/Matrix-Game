# Matrix-Game-2 DiT (Diffusion Transformer) Documentation

A comprehensive guide to the WAN Diffusion Transformer used in Matrix-Game-2, written for ML researchers who want to implement their own version.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Data Flow](#3-data-flow)
4. [Core Components](#4-core-components)
5. [Key Functions](#5-key-functions)
6. [Action Module](#6-action-module)
7. [Simplified Implementation](#7-simplified-implementation)
8. [File Reference](#8-file-reference)

**Appendices**
- [Appendix A: Causal vs Bidirectional](#appendix-a-causal-vs-bidirectional)
- [Appendix B: RoPE 3D Positional Embeddings](#appendix-b-rope-3d-positional-embeddings)

---

## 1. Overview

The DiT (Diffusion Transformer) is the core denoising model that generates video from noise, conditioned on image context and action inputs.

### What is a DiT?

A **Diffusion Transformer** replaces the traditional U-Net architecture in diffusion models with a Transformer. Instead of convolutions, it uses:
- **Patch embedding** to convert video into tokens
- **Self-attention** for global context
- **Cross-attention** for conditioning (image, actions)
- **Time modulation** to condition on diffusion timestep

### Key Specifications

| Property | Value | Why This Value? |
|----------|-------|-----------------|
| Model Type | Image-to-Video (i2v) | Generates video conditioned on first frame |
| Hidden Dimension | 1536 | Must be divisible by heads (1536/12=128 per head) |
| FFN Dimension | 8960 | ~5.8× hidden dim (typical is 4×, larger = more capacity) |
| Attention Heads | 12 | Each head has 128 dims, attends to different patterns |
| Head Dimension | 128 | 1536 ÷ 12 = 128 (standard size, same as LLaMA) |
| Transformer Layers | 30 | Deep network for complex video dynamics |
| Patch Size | (1, 2, 2) | No temporal patching, 2×2 spatial (token = 4 latent pixels) |
| Input Channels | 36 | 16 VAE latent + 20 conditioning (mask, first frame, etc.) |
| Output Channels | 16 | VAE latent channels (predicts noise in latent space) |
| Parameters | ~1.4 billion | Large enough for video, fits on consumer GPUs |

**Why 12 heads with 128 dimensions each?**
- Multi-head attention lets different heads focus on different patterns
- One head might track motion, another color consistency, another object identity
- 128 dims per head is standard (GPT-3, LLaMA use same)
- Total: 12 × 128 = 1536 hidden dimension

**Why 1536 hidden dimension?**
- Larger than typical vision transformers (ViT uses 768)
- Video is harder than images - needs more capacity
- Sweet spot for ~1B parameter models
- Divisible by many head counts (12, 16, 24, 32)

### Model Variants

| Model | File | Attention | Use Case |
|-------|------|-----------|----------|
| **WanModel** | `wan/modules/model.py` | Bidirectional | Training, offline generation |
| **CausalWanModel** | `wan/modules/causal_model.py` | Causal (block-wise) | Streaming inference |

This document focuses on **WanModel** (bidirectional), with CausalWanModel covered in the appendix.

---

## 2. Architecture Diagram

### High-Level Overview

```
Input: VAE Latent [B, 16, T, H/8, W/8] + Conditioning [B, 20, T, H/8, W/8]
                              ↓
                    ┌─────────────────────┐
                    │   Patch Embedding   │  Conv3d(36→1536, kernel=(1,2,2), stride=(1,2,2))
                    │   [B, L, 1536]      │  where L = T × (H/16) × (W/16)
                    └─────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │   Time Embedding    │  sinusoidal → MLP → 6 modulation params
                    │   [B, 6, 1536]      │
                    └─────────────────────┘
                              ↓
              ┌───────────────────────────────────┐
              │                                   │
              │   ×30 WanAttentionBlock           │
              │   ┌─────────────────────────┐     │
              │   │ LayerNorm + Modulation  │     │
              │   │         ↓               │     │
              │   │ Self-Attention (+ RoPE) │←────┼──── 3D Rotary Position Embedding
              │   │         ↓               │     │
              │   │ Cross-Attention         │←────┼──── Image Context (CLIP features)
              │   │         ↓               │     │
              │   │ Action Module           │←────┼──── Mouse + Keyboard conditioning
              │   │         ↓               │     │
              │   │ FFN + Modulation        │     │
              │   └─────────────────────────┘     │
              │                                   │
              └───────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │   Head (unpatchify)  │  Linear(1536 → 16×1×2×2) + reshape
                    │   [B, 16, T, H/8, W/8]│
                    └─────────────────────┘
                              ↓
Output: Predicted Noise [B, 16, T, H/8, W/8]
```

### Single WanAttentionBlock

```
Input x [B, L, 1536]
    │
    ├──────────────────────────────────────────────────┐
    │                                                  │ (residual)
    ↓                                                  │
┌─────────────────────────────────────────┐            │
│ norm1(x) * (1 + e[1]) + e[0]            │  ← Time modulation (shift + scale)
└─────────────────────────────────────────┘            │
    ↓                                                  │
┌─────────────────────────────────────────┐            │
│ WanSelfAttention                        │            │
│   Q = norm_q(W_q(x))                    │            │
│   K = norm_k(W_k(x))                    │            │
│   V = W_v(x)                            │            │
│   Q, K = rope_apply(Q, K, freqs)        │  ← 3D RoPE │
│   out = flash_attention(Q, K, V)        │            │
│   out = W_o(out)                        │            │
└─────────────────────────────────────────┘            │
    ↓                                                  │
    + e[2] * y  ←──────────────────────────────────────┘
    │                                                  │
    ├──────────────────────────────────────────────────┐
    ↓                                                  │ (residual)
┌─────────────────────────────────────────┐            │
│ WanI2VCrossAttention                    │            │
│   Q = norm_q(W_q(x))                    │            │
│   K = norm_k(W_k(context))   ← cached   │            │
│   V = W_v(context)           ← cached   │            │
│   out = flash_attention(Q, K, V)        │            │
└─────────────────────────────────────────┘            │
    ↓                                                  │
    + cross_attn_output ←──────────────────────────────┘
    │
    ↓
┌─────────────────────────────────────────┐
│ ActionModule (if enabled)               │
│   Mouse attention + Keyboard attention  │
└─────────────────────────────────────────┘
    │
    ├──────────────────────────────────────────────────┐
    ↓                                                  │ (residual)
┌─────────────────────────────────────────┐            │
│ norm2(x) * (1 + e[4]) + e[3]            │  ← Time modulation
└─────────────────────────────────────────┘            │
    ↓                                                  │
┌─────────────────────────────────────────┐            │
│ FFN: Linear(1536→8960) → GELU → Linear  │            │
└─────────────────────────────────────────┘            │
    ↓                                                  │
    + e[5] * y  ←──────────────────────────────────────┘
    │
    ↓
Output x [B, L, 1536]
```

---

## 3. Data Flow

### Input Preparation

```
VAE Latent:     [B, 16, T, H/8, W/8]     ← From VAE encoder (or noise during generation)
Conditioning:   [B, 20, T, H/8, W/8]     ← Mask, first frame latent, etc.
                        ↓
Concatenate:    [B, 36, T, H/8, W/8]     ← in_dim = 36
```

### Through the Model

```
Step 1: Patch Embedding
────────────────────────
Input:  [B, 36, T, H/8, W/8]
        ↓ Conv3d(36, 1536, kernel=(1,2,2), stride=(1,2,2))
Output: [B, 1536, T, H/16, W/16]
        ↓ flatten + transpose
Tokens: [B, L, 1536]  where L = T × (H/16) × (W/16)

Example: For 720p video (1280×720) with 17 frames:
  - VAE latent: [B, 16, 17, 90, 160]  (after 8x spatial compression)
  - After patch: [B, 1536, 17, 45, 80]
  - Tokens: [B, 61200, 1536]  (17 × 45 × 80 = 61,200 tokens!)

Step 2: Time Embedding
──────────────────────
Timestep t: scalar (0 to 1000)
        ↓ sinusoidal_embedding_1d(256, t)
        [B, 256]
        ↓ MLP: Linear(256→1536) → SiLU → Linear(1536→1536)
        [B, 1536]
        ↓ time_projection: SiLU → Linear(1536→9216)
        ↓ reshape
        [B, 6, 1536]  ← 6 modulation vectors

Step 3: Image Context
─────────────────────
CLIP features: [B, 257, 1280]  ← 257 = 1 CLS + 256 patches
        ↓ MLPProj: LayerNorm → Linear → GELU → Linear → LayerNorm
        [B, 257, 1536]  ← context for cross-attention

Step 4: Transformer Blocks (×30)
────────────────────────────────
For each block:
  x = self_attention(x, grid_sizes, freqs)  ← with RoPE
  x = cross_attention(x, context)            ← image conditioning
  x = action_module(x, mouse, keyboard)      ← action conditioning
  x = ffn(x)

Step 5: Head (Unpatchify)
─────────────────────────
Tokens: [B, L, 1536]
        ↓ LayerNorm + modulation
        ↓ Linear(1536 → 16×1×2×2 = 64)
        [B, L, 64]
        ↓ reshape to [B, T, H/16, W/16, 1, 2, 2, 16]
        ↓ einsum rearrange
Output: [B, 16, T, H/8, W/8]  ← predicted noise in VAE latent space
```

### Token Count vs Resolution

| Resolution | VAE Latent | After Patch | Tokens |
|------------|------------|-------------|--------|
| 256×256, 17 frames | [17, 32, 32] | [17, 16, 16] | 4,352 |
| 512×512, 17 frames | [17, 64, 64] | [17, 32, 32] | 17,408 |
| 720p, 17 frames | [17, 90, 160] | [17, 45, 80] | 61,200 |
| 1080p, 17 frames | [17, 135, 240] | [17, 68, 120] | 138,720 |

The token count grows quadratically with resolution - this is why attention is expensive!

---

## 4. Core Components

### 4.1 Patch Embedding

**File:** `wan/modules/model.py:526-527`

```python
self.patch_embedding = nn.Conv3d(
    in_dim,           # 36 (16 latent + 20 conditioning)
    dim,              # 1536
    kernel_size=patch_size,   # (1, 2, 2)
    stride=patch_size         # (1, 2, 2)
)
```

Converts video latents into patch tokens:
- **Temporal:** No compression (stride 1)
- **Spatial:** 2× compression each dimension (stride 2)
- Each patch covers 1×2×2 = 4 latent pixels → 1 token

```
Input:  [B, 36, T, H, W]
        ↓ Conv3d with kernel (1,2,2), stride (1,2,2)
Output: [B, 1536, T, H/2, W/2]
        ↓ flatten(2).transpose(1,2)
Tokens: [B, T×H/2×W/2, 1536]
```

### 4.2 Time Embedding

**File:** `wan/modules/model.py:17-27, 532-535`

Two-stage embedding with 6 modulation outputs:

```python
# Stage 1: Sinusoidal embedding
def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).div(half))
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

# Stage 2: MLP projection
self.time_embedding = nn.Sequential(
    nn.Linear(freq_dim, dim),   # 256 → 1536
    nn.SiLU(),
    nn.Linear(dim, dim)         # 1536 → 1536
)
self.time_projection = nn.Sequential(
    nn.SiLU(),
    nn.Linear(dim, dim * 6)     # 1536 → 9216, then reshape to [B, 6, 1536]
)
```

**The 6 modulation vectors:**
```
e[0]: Self-attention shift
e[1]: Self-attention scale
e[2]: Self-attention output gate
e[3]: FFN shift
e[4]: FFN scale
e[5]: FFN output gate
```

Used as: `output = norm(x) * (1 + scale) + shift` and `x = x + gate * layer_output`

### 4.3 WanRMSNorm

**File:** `wan/modules/model.py:74-90`

RMS (Root Mean Square) normalization - simpler than LayerNorm:

```python
class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # RMS norm: x / sqrt(mean(x^2) + eps) * weight
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
```

**Difference from LayerNorm:**
- LayerNorm: `(x - mean) / std * gamma + beta`
- RMSNorm: `x / rms * gamma` (no mean subtraction, no bias)

RMSNorm is ~10-15% faster and works well for transformers.

### 4.4 WanSelfAttention

**File:** `wan/modules/model.py:106-160`

```python
class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1,-1), qk_norm=True, eps=1e-6):
        self.num_heads = num_heads      # 12
        self.head_dim = dim // num_heads  # 1536/12 = 128

        # Linear projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK normalization (stabilizes training)
        self.norm_q = WanRMSNorm(dim, eps)
        self.norm_k = WanRMSNorm(dim, eps)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Compute Q, K, V with QK normalization
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Apply 3D RoPE to Q and K
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # Flash attention (bidirectional - no causal mask!)
        x = flash_attention(q, k, v, k_lens=seq_lens, window_size=self.window_size)

        return self.o(x.flatten(2))
```

**Key features:**
- **QK normalization:** Normalizes Q and K before attention (stabilizes large models)
- **3D RoPE:** Positional info encoded in Q, K via rotary embeddings
- **Flash Attention:** Memory-efficient attention (O(N) memory vs O(N²))
- **Bidirectional:** No causal mask - each token attends to all tokens

### 4.5 WanI2VCrossAttention

**File:** `wan/modules/model.py:228-260`

Image-to-video cross-attention with K/V caching:

```python
class WanI2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Query from video tokens
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        # Key/Value from image context (cached after first use)
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]  # Reuse cached K
                v = crossattn_cache["v"]  # Reuse cached V
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v, k_lens=None)
        return self.o(x.flatten(2))
```

**Why cache K/V?**
- Image context is the same for all diffusion steps
- Computing K, V once and reusing saves ~10% compute

### 4.6 WanAttentionBlock

**File:** `wan/modules/model.py:273-369`

The full transformer block combining all components:

```python
class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, ...):
        # Norms
        self.norm1 = WanLayerNorm(dim)
        self.norm2 = WanLayerNorm(dim)
        self.norm3 = WanLayerNorm(dim) if cross_attn_norm else nn.Identity()

        # Attention
        self.self_attn = WanSelfAttention(dim, num_heads, ...)
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, ...)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),       # 1536 → 8960
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)        # 8960 → 1536
        )

        # Action module (optional)
        self.action_model = ActionModule(**action_config) if action_config else None

        # Time modulation (learned, added to time embedding)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, mouse_cond, keyboard_cond):
        # Combine learned modulation with time embedding
        e = (self.modulation + e).chunk(6, dim=1)

        # Self-attention with modulation
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
        x = x + y * e[2]

        # Cross-attention (image conditioning)
        x = x + self.cross_attn(self.norm3(x), context)

        # Action conditioning
        if self.action_model is not None:
            x = self.action_model(x, grid_sizes, mouse_cond, keyboard_cond)

        # FFN with modulation
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]

        return x
```

### 4.7 Head (Unpatchify)

**File:** `wan/modules/model.py:373-406, 700-721`

Converts tokens back to video latent format:

```python
class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        out_channels = math.prod(patch_size) * out_dim  # 1×2×2 × 16 = 64
        self.norm = WanLayerNorm(dim)
        self.head = nn.Linear(dim, out_channels)  # 1536 → 64
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x

# Unpatchify: reshape tokens back to video
def unpatchify(self, x, grid_sizes):
    # x: [B, L, 64] where 64 = out_dim × prod(patch_size)
    c = self.out_dim  # 16
    x = x.view(bs, *grid_sizes, *self.patch_size, c)
    # [B, T, H/2, W/2, 1, 2, 2, 16]
    x = torch.einsum("bfhwpqrc->bcfphqwr", x)
    # [B, 16, T, 1, H/2, 2, W/2, 2]
    x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
    # [B, 16, T, H, W]
    return x
```

### 4.8 MLPProj (Image Embedding)

**File:** `wan/modules/model.py:409-421`

Projects CLIP image features to model dimension:

```python
class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim):  # 1280 → 1536
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)
```

---

## 5. Key Functions

### 5.1 WanModel.__init__

**File:** `wan/modules/model.py:448-564`

```python
def __init__(self,
    model_type='i2v',
    patch_size=(1, 2, 2),
    in_dim=36,              # 16 VAE latent + 20 conditioning
    dim=1536,               # Hidden dimension
    ffn_dim=8960,           # FFN intermediate dimension
    freq_dim=256,           # Time embedding dimension
    out_dim=16,             # Output channels (VAE latent)
    num_heads=12,           # Attention heads
    num_layers=30,          # Transformer blocks
    qk_norm=True,           # QK normalization
    action_config={},       # Action module config
    ...
):
    # Patch embedding
    self.patch_embedding = nn.Conv3d(in_dim, dim, patch_size, stride=patch_size)

    # Time embedding
    self.time_embedding = nn.Sequential(Linear(freq_dim→dim), SiLU, Linear(dim→dim))
    self.time_projection = nn.Sequential(SiLU, Linear(dim→dim*6))

    # Transformer blocks
    self.blocks = nn.ModuleList([
        WanAttentionBlock(dim, ffn_dim, num_heads, action_config=action_config)
        for _ in range(num_layers)
    ])

    # Output head
    self.head = Head(dim, out_dim, patch_size)

    # RoPE frequencies (precomputed)
    d = dim // num_heads  # 128
    self.freqs = torch.cat([
        rope_params(1024, d - 4*(d//6)),   # Temporal: 128 - 84 = 44 dims
        rope_params(1024, 2*(d//6)),       # Height: 42 dims
        rope_params(1024, 2*(d//6))        # Width: 42 dims
    ], dim=1)

    # Image projection (for i2v)
    self.img_emb = MLPProj(1280, dim)
```

### 5.2 WanModel._forward

**File:** `wan/modules/model.py:580-699`

```python
def _forward(self, x, t, visual_context, cond_concat, mouse_cond=None, keyboard_cond=None):
    # Concatenate input with conditioning
    x = torch.cat([x, cond_concat], dim=1)  # [B, 36, T, H, W]

    # Patch embedding
    x = self.patch_embedding(x)              # [B, 1536, T, H/2, W/2]
    grid_sizes = torch.tensor(x.shape[2:])   # [T, H/2, W/2]
    x = x.flatten(2).transpose(1, 2)         # [B, L, 1536]
    seq_lens = torch.tensor([x.size(1)] * x.size(0))

    # Time embedding
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # [B, 6, 1536]

    # Image context
    context = self.img_emb(visual_context)   # [B, 257, 1536]

    # Transformer blocks
    kwargs = dict(e=e0, grid_sizes=grid_sizes, seq_lens=seq_lens,
                  freqs=self.freqs, context=context,
                  mouse_cond=mouse_cond, keyboard_cond=keyboard_cond)

    for block in self.blocks:
        x = block(x, **kwargs)

    # Output head
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)

    return x.float()
```

### 5.3 flash_attention

**File:** `wan/modules/attention.py:31-149`

```python
def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, causal=False, window_size=(-1,-1), ...):
    """
    q: [B, Lq, Nq, C1]  - Query
    k: [B, Lk, Nk, C1]  - Key
    v: [B, Lk, Nk, C2]  - Value
    k_lens: [B]         - Valid length per sequence (for masking padding)
    causal: bool        - Whether to apply causal mask
    window_size: tuple  - Local attention window (-1 = global)
    """
    # Flatten batch and sequence for variable-length flash attention
    q = q.flatten(0, 1)  # [B*Lq, Nq, C1]
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    # Compute cumulative sequence lengths for flash attention
    cu_seqlens_q = torch.cat([zeros, q_lens]).cumsum(0)
    cu_seqlens_k = torch.cat([zeros, k_lens]).cumsum(0)

    # Call flash attention (FA3 for Hopper, FA2 otherwise)
    if FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            softmax_scale=softmax_scale, causal=causal
        )
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size
        )
    else:
        # Fallback to PyTorch SDPA
        x = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return x.unflatten(0, (b, lq))
```

**Note:** WanModel uses `causal=False` - bidirectional attention!

---

## 6. Action Module

**File:** `wan/modules/action_module.py`

The ActionModule conditions video generation on user inputs (mouse, keyboard).

### 6.1 Overview

```
Mouse Input:    [B, N_frames, 2]     ← (x, y) coordinates per frame
Keyboard Input: [B, N_frames, 6]     ← 6 button states per frame (or 4 in some configs)
                        ↓
              ┌─────────────────────┐
              │   ActionModule      │
              │   • Mouse MLP       │
              │   • Mouse Attention │
              │   • Keyboard Embed  │
              │   • Keyboard Attn   │
              └─────────────────────┘
                        ↓
              Added to hidden states
```

### 6.2 Configuration

```python
ActionModule(
    mouse_dim_in=2,              # x, y coordinates
    keyboard_dim_in=6,           # 6 button states (4 in some configs)
    hidden_size=128,             # Keyboard embedding size
    img_hidden_size=1536,        # Model hidden dimension
    keyboard_hidden_dim=1024,    # Keyboard attention dimension
    mouse_hidden_dim=1024,       # Mouse attention dimension
    vae_time_compression_ratio=4,# VAE temporal compression
    windows_size=3,              # Temporal window for actions
    heads_num=16,                # Attention heads
    patch_size=[1, 2, 2],
    local_attn_size=6,           # Local attention window
    enable_mouse=True,
    enable_keyboard=True,
)
```

### 6.3 Mouse Conditioning

```python
# Mouse input processing
# 1. Window past actions (windows_size=3 means look at 12 raw frames = 3 latent frames)
pad_t = vae_time_compression_ratio * windows_size  # 4 × 3 = 12
mouse_condition = torch.cat([pad, mouse_condition], dim=1)
group_mouse = [mouse_condition[:, i-window:i, :] for i in range(N_feats)]

# 2. Concatenate with current hidden state
group_mouse = torch.cat([hidden_states, group_mouse], dim=-1)

# 3. Project through MLP
group_mouse = self.mouse_mlp(group_mouse)  # → [B*HW, T, 1024]

# 4. Self-attention over temporal dimension
q, k, v = self.t_qkv(group_mouse).chunk(3)
q, k = apply_rotary_emb(q, k, freqs_cis)
attn = flash_attn_func(q, k, v)

# 5. Project back and add residual
hidden_states = hidden_states + self.proj_mouse(attn)
```

### 6.4 Keyboard Conditioning

```python
# Keyboard input processing
# 1. Embed keyboard states
keyboard_condition = self.keyboard_embed(keyboard_condition)  # [B, T, 6] → [B, T, 128]

# 2. Window and flatten
group_keyboard = [keyboard_condition[:, i-window:i, :] for i in range(N_feats)]
group_keyboard = group_keyboard.reshape(B, T, window*128)  # Flatten window

# 3. Cross-attention: video queries, keyboard keys/values
q = self.mouse_attn_q(hidden_states)  # [B, L, 1024]
k, v = self.keyboard_attn_kv(group_keyboard).chunk(2)  # [B, T, 1024] each

# 4. Apply attention with RoPE
q, k = apply_rotary_emb(q, k, freqs_cis)
attn = flash_attn_func(q, k, v)

# 5. Project and add residual
hidden_states = hidden_states + self.proj_keyboard(attn)
```

### 6.5 Which Blocks Have Actions?

From config:
```json
"action_config": {
    "blocks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
}
```

In distilled models, only **blocks 0-14** (first 15 of 30) have action modules. This is a key distillation decision - action conditioning in early layers is sufficient.

---

## 7. Simplified Implementation

A minimal implementation to understand the core concepts:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SimpleSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.norm_q = SimpleRMSNorm(dim)
        self.norm_k = SimpleRMSNorm(dim)

    def forward(self, x):
        B, L, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # QK normalization
        q = self.norm_q(q.flatten(-2)).view(B, L, self.num_heads, self.head_dim)
        k = self.norm_k(k.flatten(-2)).view(B, L, self.num_heads, self.head_dim)

        # Attention
        q, k, v = [rearrange(t, 'b l h d -> b h l d') for t in [q, k, v]]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, 'b h l d -> b l (h d)')

        return self.out(attn)

class SimpleCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, L, C = x.shape

        q = self.q(x).view(B, L, self.num_heads, self.head_dim)
        kv = self.kv(context).view(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        q, k, v = [rearrange(t, 'b l h d -> b h l d') for t in [q, k, v]]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, 'b h l d -> b l (h d)')

        return self.out(attn)

class SimpleBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SimpleSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = SimpleCrossAttention(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim)
        )
        # Time modulation
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(self, x, e, context):
        # Add modulation to time embedding
        mods = (self.modulation + e).chunk(6, dim=1)

        # Self-attention with modulation
        y = self.self_attn(self.norm1(x) * (1 + mods[1]) + mods[0])
        x = x + y * mods[2]

        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), context)

        # FFN with modulation
        y = self.ffn(self.norm3(x) * (1 + mods[4]) + mods[3])
        x = x + y * mods[5]

        return x

class SimpleDiT(nn.Module):
    def __init__(self,
                 in_channels=36,
                 out_channels=16,
                 dim=1536,
                 ffn_dim=8960,
                 num_heads=12,
                 num_layers=30,
                 patch_size=(1, 2, 2)):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Patch embedding
        self.patch_embed = nn.Conv3d(in_channels, dim, patch_size, stride=patch_size)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Image embedding
        self.img_embed = nn.Sequential(
            nn.LayerNorm(1280), nn.Linear(1280, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)
        ])

        # Output head
        self.head_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels * patch_size[0] * patch_size[1] * patch_size[2])

    def sinusoidal_embed(self, t, dim=256):
        half = dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (math.log(10000) / half))
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, t, cond_concat, visual_context):
        # Concatenate input with conditioning
        x = torch.cat([x, cond_concat], dim=1)

        # Patch embed
        x = self.patch_embed(x)
        grid_sizes = x.shape[2:]
        x = rearrange(x, 'b c t h w -> b (t h w) c')

        # Time embedding
        t_emb = self.time_embed(self.sinusoidal_embed(t))
        t_emb = self.time_proj(t_emb).view(-1, 6, x.shape[-1])

        # Image context
        context = self.img_embed(visual_context)

        # Transformer
        for block in self.blocks:
            x = block(x, t_emb, context)

        # Head
        x = self.head(self.head_norm(x))

        # Unpatchify
        t, h, w = grid_sizes
        pt, ph, pw = self.patch_size
        x = x.view(-1, t, h, w, pt, ph, pw, self.out_channels)
        x = rearrange(x, 'b t h w pt ph pw c -> b c (t pt) (h ph) (w pw)')

        return x

# =============================================================================
# TODO: Steps to evolve this into the full WAN model
# =============================================================================
#
# LEVEL 1: Core Improvements
# --------------------------
# [ ] 1. Add QK Normalization to attention
#        - Replace: q = self.q(x)
#        - With:    q = self.norm_q(self.q(x))
#        - Same for k. Stabilizes training for large models.
#
# [ ] 2. Add 3D RoPE positional embeddings
#        - Precompute frequencies: rope_params(1024, head_dim) for T, H, W
#        - Split head_dim: [44, 42, 42] for temporal, height, width
#        - Apply to Q, K before attention: q, k = rope_apply(q, k, grid_sizes, freqs)
#        - See Appendix B for implementation details
#
# [ ] 3. Replace LayerNorm with RMSNorm
#        - RMSNorm: x * rsqrt(mean(x²) + eps) * weight
#        - No mean subtraction, no bias. ~10-15% faster.
#
# LEVEL 2: Efficiency
# -------------------
# [ ] 4. Add Flash Attention backend
#        - Replace F.scaled_dot_product_attention with flash_attn
#        - Use flash_attn_varlen_func for variable-length sequences
#        - Requires reshaping to [total_tokens, heads, head_dim]
#
# [ ] 5. Add K/V caching for cross-attention
#        - Image context doesn't change across diffusion steps
#        - Cache K, V after first computation: crossattn_cache["k"] = k
#        - Reuse on subsequent forward passes
#
# [ ] 6. Add gradient checkpointing
#        - Wrap block forward in torch.utils.checkpoint.checkpoint()
#        - Trades compute for memory during training
#
# LEVEL 3: Action Conditioning
# ----------------------------
# [ ] 7. Add ActionModule for mouse conditioning
#        - Input: [B, N_frames, 2] mouse (x, y) coordinates
#        - Window past 12 frames (windows_size=3 × vae_compression=4)
#        - Concat with hidden state, project through MLP
#        - Self-attention over temporal dimension with RoPE
#        - Add residual: x = x + proj_mouse(attn)
#
# [ ] 8. Add ActionModule for keyboard conditioning
#        - Input: [B, N_frames, 6] keyboard states (or 4)
#        - Embed through MLP: [B, T, 6] → [B, T, 128]
#        - Cross-attention: video queries, keyboard keys/values
#        - Add residual: x = x + proj_keyboard(attn)
#
# [ ] 9. Make action modules optional per block
#        - Only blocks 0-14 have action modules in distilled models
#        - Pass action_config["blocks"] = [0,1,2,...,14]
#
# LEVEL 4: Causal/Streaming Support
# ---------------------------------
# [ ] 10. Add block-wise causal attention mask
#         - Use flex_attention with BlockMask
#         - Mask function: kv_idx < ends[q_idx] (causal within frame blocks)
#         - Support local_attn_size for windowed attention
#
# [ ] 11. Add KV cache for streaming inference
#         - Maintain rolling buffer of K, V from past frames
#         - Evict oldest when buffer full (keep sink tokens)
#         - Update cache indices: global_end_index, local_end_index
#
# [ ] 12. Split forward into _forward_train and _forward_inference
#         - Training: full sequence, no cache
#         - Inference: incremental, use KV cache
#
# LEVEL 5: Production Ready
# -------------------------
# [ ] 13. Add proper weight initialization
#         - Xavier uniform for linear layers
#         - Normal(std=0.02) for time embedding
#         - Zero init for output head and action projections
#
# [ ] 14. Add diffusers compatibility
#         - Inherit from ModelMixin, ConfigMixin
#         - Use @register_to_config decorator
#         - Support gradient checkpointing flag
#
# [ ] 15. Add window attention option
#         - window_size parameter for local attention
#         - Useful for very high resolution
#
# =============================================================================
```

---

## 8. File Reference

| File | Description |
|------|-------------|
| `wan/modules/model.py` | WanModel (bidirectional DiT) |
| `wan/modules/causal_model.py` | CausalWanModel (causal DiT for streaming) |
| `wan/modules/attention.py` | Flash attention wrappers |
| `wan/modules/action_module.py` | Mouse/keyboard conditioning |
| `wan/modules/posemb_layers.py` | RoPE positional embeddings |
| `configs/foundation_model/config.json` | Full model config |
| `configs/distilled_model/*/config.json` | Distilled model configs |

---

## Appendix A: Causal vs Bidirectional

### Key Differences

| Aspect | WanModel (Bidirectional) | CausalWanModel (Causal) |
|--------|--------------------------|-------------------------|
| **Attention** | Full attention (each token sees all) | Block-wise causal (only past+current) |
| **KV Cache** | No | Yes (for streaming) |
| **Use Case** | Training, offline generation | Real-time streaming inference |
| **File** | `model.py` | `causal_model.py` |

### Causal Attention Mask

In CausalWanModel, attention is block-wise causal:

```python
def attention_mask(b, h, q_idx, kv_idx):
    if local_attn_size == -1:
        # Global causal: attend to all past + current
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
    else:
        # Local causal: attend to recent window only
        return ((kv_idx < ends[q_idx]) &
                (kv_idx >= ends[q_idx] - local_attn_size)) | (q_idx == kv_idx)
```

### Block-wise Processing

```
Frame blocks:  [Block 0] [Block 1] [Block 2] [Block 3] ...
                   ↓         ↓         ↓         ↓
Causal:       [  0  ]  [ 0,1 ]  [0,1,2] [1,2,3]  ← each sees only past
                                         ↑
                                    local window
```

With `local_attn_size=4`, block 3 only attends to blocks 0,1,2,3 (not all history).

### KV Cache

CausalWanModel maintains KV cache for efficient streaming:

```python
# During inference, cache K/V from previous frames
kv_cache["k"][:, start:end] = new_k
kv_cache["v"][:, start:end] = new_v

# When cache is full, evict oldest (rolling buffer)
if current_end > cache_size:
    # Roll cache: evict oldest, keep sink tokens
    kv_cache["k"][:, sink:sink+roll] = kv_cache["k"][:, sink+evict:...].clone()
```

---

## Appendix B: RoPE 3D Positional Embeddings

### Why Does Attention Need Positional Information?

**The Problem:** Attention is **permutation invariant**. The formula:
```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```
treats tokens as an unordered **set**. If you shuffle the input tokens, the output just gets shuffled the same way - attention doesn't know the difference!

**Why this matters for video:**
```
Frame sequence:  [1] [2] [3] [4] [5]
Shuffled:        [3] [1] [5] [2] [4]

Without position info, attention sees these as equivalent!
But video has temporal structure - frame 5 should "know" it comes after frame 4.
```

**The Solution:** Inject position information so tokens know where they are:
- Token at position 5 can reason "I'm near the end"
- Two tokens can compute "we're 3 positions apart"

### Positional Encoding Methods Compared

| Method | How It Works | Pros | Cons |
|--------|--------------|------|------|
| **Learned Absolute** | Add learned embedding per position | Simple, effective | Can't handle longer sequences than trained |
| **Sinusoidal** | Add fixed sin/cos patterns | Extrapolates to longer sequences | Doesn't capture relative distance well |
| **ALiBi** | Bias attention by distance | Simple, efficient | Linear bias may not fit all patterns |
| **RoPE** | Rotate Q, K by position angle | Relative + extrapolates + efficient | Slightly more complex |

### Why RoPE?

**RoPE (Rotary Position Embedding)** encodes position by **rotating** query and key vectors:

```
q_rotated = q × e^(i × m × θ)   where m = position, θ = frequency
k_rotated = k × e^(i × n × θ)   where n = position
```

When computing attention `q · k`, the rotation angles **subtract**:
```
q_rotated · k_rotated = |q||k| × cos((m - n) × θ)
```

This means attention depends on **(m - n)** = the **relative distance**, not absolute positions!

**Benefits of RoPE:**
1. **Relative position:** "3 frames apart" encoded, not "frame 47"
2. **Extrapolation:** Works for longer videos than training
3. **Efficient:** No extra parameters, applied on-the-fly
4. **3D extension:** Can encode temporal + spatial position separately

### 3D RoPE for Video

The model uses separate RoPE for temporal (T), height (H), and width (W):

```python
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2  # num_heads, head_dim/2

    # Split frequencies for T, H, W (different proportions)
    freqs = freqs.split([c - 2*(c//3), c//3, c//3], dim=1)
    # For head_dim=128: [44, 42, 42] dimensions

    f, h, w = grid_sizes.tolist()

    # Create 3D position grid
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),  # Temporal
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),  # Height
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),  # Width
    ], dim=-1).reshape(seq_len, 1, -1)

    # Apply as complex rotation
    x_complex = torch.view_as_complex(x.reshape(..., -1, 2))
    x_rotated = x_complex * freqs_i
    return torch.view_as_real(x_rotated).flatten(-2)
```

### Frequency Allocation

```
head_dim = 128 (per attention head)

Temporal (T): 128 - 2×(128÷6) = 128 - 84 = 44 dimensions
Height (H):   128÷6 × 2 = 42 dimensions
Width (W):    128÷6 × 2 = 42 dimensions
Total:        44 + 42 + 42 = 128 ✓
```

More dimensions for temporal because video has more variation in time than space.

### RoPE Parameters

```python
self.freqs = torch.cat([
    rope_params(1024, d - 4*(d//6)),  # Temporal: max 1024 positions
    rope_params(1024, 2*(d//6)),      # Height: max 1024 positions
    rope_params(1024, 2*(d//6))       # Width: max 1024 positions
], dim=1)
```

The `1024` max positions allow for very long videos/high resolutions before extrapolation is needed.
