# Matrix-Game-2 VAE Documentation

A comprehensive guide to the 3D Video VAE used in Matrix-Game-2, written for ML researchers who want to implement their own version.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Data Flow](#3-data-flow)
4. [Core Components](#4-core-components)
5. [Key Functions](#5-key-functions)
6. [Resolution Handling](#6-resolution-handling)
7. [Simplified Implementation](#7-simplified-implementation)
8. [File Reference](#8-file-reference)

---

## 1. Overview

The VAE is a **3D Variational Autoencoder** that compresses video into a latent space suitable for diffusion models.

### Key Specifications (from exploration)

| Property | Value |
|----------|-------|
| Input | `[B, 3, T, H, W]` RGB video, values in `[-1, 1]` |
| Output | `[B, 16, T', H/8, W/8]` latent tensor |
| Spatial compression | 8x |
| Temporal compression | ~4x (via chunked processing) |
| Latent channels | 16 |
| Base channels | 96 |

### How It's Used in inference.py

```python
# Line 74-77: Load the VAE
vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)

# Line 105: Encode input to latent space
# Input:  [1, 3, 597, 352, 640]  (597 frames = 1 + 4*149)
# Output: [1, 16, 150, 44, 80]   (150 latent frames)
img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs)

# Decoding happens in the pipeline - latent back to video
```

---

## 2. Architecture Diagram

```
                          ENCODER
═══════════════════════════════════════════════════════════════

Input Video [B, 3, T, H, W]
                │
                ▼
┌─────────────────────────────────────┐
│ conv1: CausalConv3d(3 → 96, k=3)    │
└─────────────────────────────────────┘
                │ [B, 96, T, H, W]
                ▼
┌─────────────────────────────────────┐
│ LEVEL 0                             │
│  ├─ ResBlock(96 → 96)               │
│  ├─ ResBlock(96 → 96)               │
│  └─ Downsample2D (H/2, W/2)         │
└─────────────────────────────────────┘
                │ [B, 192, T, H/2, W/2]
                ▼
┌─────────────────────────────────────┐
│ LEVEL 1                             │
│  ├─ ResBlock(96 → 192)              │
│  ├─ ResBlock(192 → 192)             │
│  └─ Downsample3D (H/4, W/4, T-ish)  │
└─────────────────────────────────────┘
                │ [B, 384, T, H/4, W/4]
                ▼
┌─────────────────────────────────────┐
│ LEVEL 2                             │
│  ├─ ResBlock(192 → 384)             │
│  ├─ ResBlock(384 → 384)             │
│  └─ Downsample3D (H/8, W/8, T-ish)  │
└─────────────────────────────────────┘
                │ [B, 384, T, H/8, W/8]
                ▼
┌─────────────────────────────────────┐
│ LEVEL 3 (no downsample)             │
│  ├─ ResBlock(384 → 384)             │
│  └─ ResBlock(384 → 384)             │
└─────────────────────────────────────┘
                │ [B, 384, T, H/8, W/8]
                ▼
┌─────────────────────────────────────┐
│ MIDDLE BLOCK                        │
│  ├─ ResBlock(384 → 384)             │
│  ├─ SpatialAttention(384)           │
│  └─ ResBlock(384 → 384)             │
└─────────────────────────────────────┘
                │ [B, 384, T, H/8, W/8]
                ▼
┌─────────────────────────────────────┐
│ HEAD                                │
│  ├─ RMSNorm(384)                    │
│  ├─ SiLU()                          │
│  └─ CausalConv3d(384 → 32, k=3)     │
└─────────────────────────────────────┘
                │ [B, 32, T, H/8, W/8]
                ▼
┌─────────────────────────────────────┐
│ conv1: CausalConv3d(32 → 32, k=1)   │
│ Split → mu[16], logvar[16]          │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ NORMALIZE                           │
│ z = (mu - mean) * (1/std)           │
└─────────────────────────────────────┘
                │
                ▼
        Latent z [B, 16, T', H/8, W/8]


                          DECODER
═══════════════════════════════════════════════════════════════

        Latent z [B, 16, T', H/8, W/8]
                │
                ▼
┌─────────────────────────────────────┐
│ DENORMALIZE                         │
│ z = z * std + mean                  │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ conv2: CausalConv3d(16 → 16, k=1)   │
└─────────────────────────────────────┘
                │ [B, 16, T', H/8, W/8]
                ▼
┌─────────────────────────────────────┐
│ conv1: CausalConv3d(16 → 384, k=3)  │
└─────────────────────────────────────┘
                │ [B, 384, T', H/8, W/8]
                ▼
┌─────────────────────────────────────┐
│ MIDDLE BLOCK                        │
│  ├─ ResBlock(384 → 384)             │
│  ├─ SpatialAttention(384)           │
│  └─ ResBlock(384 → 384)             │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ LEVEL 0 (upsample temporal+spatial) │
│  ├─ Upsample3D → [B, 192, T'*2, H/4, W/4]
│  ├─ ResBlock(192 → 384)             │
│  ├─ ResBlock(384 → 384)             │
│  └─ ResBlock(384 → 384)             │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ LEVEL 1 (upsample temporal+spatial) │
│  ├─ Upsample3D → [B, 192, T'*4, H/2, W/2]
│  ├─ ResBlock(192 → 192)             │
│  ├─ ResBlock(192 → 192)             │
│  └─ ResBlock(192 → 192)             │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ LEVEL 2 (upsample spatial only)     │
│  ├─ Upsample2D → [B, 96, T_out, H, W]
│  ├─ ResBlock(96 → 96)               │
│  ├─ ResBlock(96 → 96)               │
│  └─ ResBlock(96 → 96)               │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ HEAD                                │
│  ├─ RMSNorm(96)                     │
│  ├─ SiLU()                          │
│  └─ CausalConv3d(96 → 3, k=3)       │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ clamp(-1, 1)                        │
└─────────────────────────────────────┘
                │
                ▼
        Output Video [B, 3, T_out, H, W]
```

### Tested Tensor Shapes (from exploration script)

**Encoder (single forward pass - no temporal compression):**
```
Input Shape                    -> Output Shape              | Compression
---------------------------------------------------------------------------
[1, 3, 1, 352, 640]            -> [1, 32, 1, 44, 80]        | S:8x, T:1.0x
[1, 3, 17, 352, 640]           -> [1, 32, 17, 44, 80]       | S:8x, T:1.0x
[1, 3, 17, 720, 1280]          -> [1, 32, 17, 90, 160]      | S:8x, T:1.0x
```

**Full VAE Round-Trip (with chunked temporal processing):**
```
Input:   [1, 3, 17, 256, 256]
Latent:  [1, 16, 5, 32, 32]     <- Temporal compression happens here!
Recon:   [1, 3, 17, 256, 256]
```

---

## 3. Data Flow

### Important: Temporal Compression via Chunking

The temporal compression (4x) doesn't happen in a single forward pass. Instead, the `encode()` method processes video in chunks:

```python
def encode(self, x, scale):
    t = x.shape[2]  # Total frames
    iter_ = 1 + (t - 1) // 4  # Number of chunks

    for i in range(iter_):
        if i == 0:
            # First frame alone
            out = self.encoder(x[:, :, :1, :, :], ...)
        else:
            # Then groups of 4 frames
            out_ = self.encoder(x[:, :, 1+4*(i-1):1+4*i, :, :], ...)
            out = torch.cat([out, out_], dim=2)

    # Split to mu, logvar
    mu, logvar = self.conv1(out).chunk(2, dim=1)
    # Normalize
    mu = (mu - mean) * (1/std)
    return mu
```

This chunking + caching mechanism enables:
1. **Memory efficiency**: Don't need all frames in memory at once
2. **Streaming**: Can process video frame-by-frame
3. **Causal generation**: Each output only depends on past inputs

### Temporal Formulas

```
T_latent = 1 + (T_input - 1) // 4
T_output = T_latent * 4 - 3  (if decoding all at once)

Valid input sizes: 1, 5, 9, 13, 17, 21, 25, ... (T = 1 + 4k)
```

| T_input | T_latent | T_output | Match? |
|---------|----------|----------|--------|
| 1       | 1        | 1        | Yes    |
| 5       | 2        | 5        | Yes    |
| 17      | 5        | 17       | Yes    |
| 57      | 15       | 57       | Yes    |

### Latent Normalization

Per-channel statistics learned from training data:

```python
mean = [-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
         0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921]

std  = [ 2.8184,  1.4541,  2.3275,  2.6558,  1.2196,  1.7708,  2.6052,  2.0743,
         3.2687,  2.1526,  2.8652,  1.5579,  1.6382,  1.1253,  2.8251,  1.9160]

# Encoding: z_norm = (z_raw - mean) * (1/std)
# Decoding: z_raw = z_norm * std + mean
```

**Why?** Makes latent ~N(0,1) for stable diffusion training.

---

## 4. Core Components

### 4.1 CausalConv3d

**File:** `wan/modules/vae.py:17-36`

3D convolution that only looks at past frames (can't see the future).

```python
class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Asymmetric temporal padding: (left=2*p, right=0)
        self._padding = (
            self.padding[2], self.padding[2],  # W: symmetric
            self.padding[1], self.padding[1],  # H: symmetric
            2 * self.padding[0], 0             # T: causal (left only)
        )
        self.padding = (0, 0, 0)  # Disable built-in padding

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            # Use cached past frames instead of zero-padding
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)
```

**Tested behavior:**
```
CausalConv3d(3 -> 96, kernel=3, padding=1)
Internal padding (W_l, W_r, H_l, H_r, T_l, T_r): (1, 1, 1, 1, 2, 0)
```

### 4.2 RMS_norm

**File:** `wan/modules/vae.py:39-54`

Root Mean Square normalization (simpler than LayerNorm):

```python
class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma
```

### 4.3 ResidualBlock

**File:** `wan/modules/vae.py:186-220`

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        self.residual = nn.Sequential(
            RMS_norm(in_dim),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1)
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, ...):
        return self.residual(x) + self.shortcut(x)
```

### 4.4 AttentionBlock

**File:** `wan/modules/vae.py:223-262`

Self-attention per frame (2D spatial attention):

```python
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)  # Initialize to identity

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')  # Flatten time into batch
        x = self.norm(x)

        q, k, v = self.to_qkv(x).reshape(b*t, 1, c*3, h*w).permute(0,1,3,2).chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)

        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x + identity
```

### 4.5 Resample (Downsample/Upsample)

**File:** `wan/modules/vae.py:66-161`

```python
class Resample(nn.Module):
    def __init__(self, dim, mode):
        # mode: 'downsample2d', 'downsample3d', 'upsample2d', 'upsample3d'

        if mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=2)
            )
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(...)  # Spatial
            self.time_conv = CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1))

        elif mode == 'upsample2d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(dim, dim//2, 3, padding=1)  # Halves channels!
            )
        elif mode == 'upsample3d':
            self.time_conv = CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))
            # Doubles time by interleaving channels
```

---

## 5. Key Functions

### 5.1 VideoVAE_.__init__

**File:** `wan/vae/wanx_vae_src/vae.py:500-525`

```python
def __init__(self,
             dim=96,              # Base channels
             z_dim=16,            # Latent channels
             dim_mult=[1,2,4,4],  # Channel multipliers: 96→192→384→384
             num_res_blocks=2,
             attn_scales=[],      # Only middle block has attention
             temperal_downsample=[False, True, True],
             dropout=0.0):

    self.encoder = Encoder3d(dim, z_dim*2, ...)  # Output 32 ch (mu+logvar)
    self.conv1 = CausalConv3d(z_dim*2, z_dim*2, 1)
    self.conv2 = CausalConv3d(z_dim, z_dim, 1)
    self.decoder = Decoder3d(dim, z_dim, ...)
```

### 5.2 VideoVAE_.encode

**File:** `wan/vae/wanx_vae_src/vae.py:533-558`

```python
def encode(self, x, scale):
    self.clear_cache()
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4  # Number of chunks

    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :], feat_cache=..., feat_idx=...)
        else:
            out_ = self.encoder(x[:, :, 1+4*(i-1):1+4*i, :, :], ...)
            out = torch.cat([out, out_], dim=2)

    mu, log_var = self.conv1(out).chunk(2, dim=1)
    mu = (mu - scale[0].view(1, z_dim, 1, 1, 1)) * scale[1].view(1, z_dim, 1, 1, 1)
    return mu
```

### 5.3 VideoVAE_.decode

**File:** `wan/vae/wanx_vae_src/vae.py:560-583`

```python
def decode(self, z, scale):
    self.clear_cache()
    z = z / scale[1].view(...) + scale[0].view(...)  # Denormalize

    x = self.conv2(z)
    iter_ = z.shape[2]  # Process each latent frame

    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:, :, i:i+1, :, :], feat_cache=..., feat_idx=...)
        else:
            out_ = self.decoder(x[:, :, i:i+1, :, :], ...)
            out = torch.cat([out, out_], dim=2)
    return out
```

### 5.4 WanVAE.tiled_encode

**File:** `wan/vae/wanx_vae_src/vae.py:711-759`

For high-resolution videos that don't fit in memory:

```python
def tiled_encode(self, video, device, tile_size, tile_stride):
    # Split into overlapping tiles, encode each, blend results
    tasks = []
    for h in range(0, H, stride_h):
        for w in range(0, W, stride_w):
            tasks.append((h, h+size_h, w, w+size_w))

    for h, h_, w, w_ in tasks:
        tile = video[:, :, :, h:h_, w:w_]
        encoded = self.model.encode(tile, self.scale)
        mask = self.build_mask(...)  # Linear blend at edges
        values[..., target:target+size] += encoded * mask
        weight[..., target:target+size] += mask

    return values / weight
```

---

## 6. Resolution Handling

### Constraints

```python
# Spatial: must be divisible by 8
assert H % 8 == 0 and W % 8 == 0

# Temporal: T = 1 + 4k for perfect round-trip
# Valid: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57...
```

### Memory Estimates (bfloat16)

From exploration script:

| Resolution | Peak Memory |
|------------|-------------|
| 352x640 (default) | ~4 GB |
| 512x512 | ~4.6 GB |
| 720x1280 (720p) | ~16 GB |
| 1080x1920 | ~36 GB |

Use tiled encoding for larger resolutions.

### Changing Compression Ratio

To change spatial compression from 8x to 4x:
```python
dim_mult = [1, 2, 4]  # 3 levels instead of 4
temperal_downsample = [False, True]  # 2 levels
# Result: H/4, W/4, T/2
```

---

## 7. Simplified Implementation

Here's a minimal VAE for learning (without caching complexity):

```python
"""
Simplified 3D Video VAE - for learning purposes.
Removes caching, focuses on core architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalConv3d(nn.Module):
    """3D conv with causal temporal padding."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride)
        # Causal: pad 2*p on left, 0 on right for time
        self.pad = (padding[2], padding[2],   # W
                    padding[1], padding[1],   # H
                    2 * padding[0], 0)        # T (causal)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(in_ch), nn.SiLU(),
            CausalConv3d(in_ch, out_ch, 3, padding=1),
            RMSNorm(out_ch), nn.SiLU(),
            CausalConv3d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = CausalConv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class SpatialAttention(nn.Module):
    """Self-attention per frame."""
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x_flat = rearrange(x, 'b c t h w -> (b t) c h w')

        qkv = self.qkv(self.norm(x_flat))
        qkv = rearrange(qkv, 'bt (n c) h w -> bt n (h w) c', n=3)
        q, k, v = qkv.unbind(dim=1)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'bt (h w) c -> bt c h w', h=h)
        out = self.proj(out)

        return x + rearrange(out, '(b t) c h w -> b c t h w', t=t)


class Downsample(nn.Module):
    def __init__(self, dim, temporal=False):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=2)
        )
        self.temporal = temporal
        if temporal:
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.spatial(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        if self.temporal:
            x = self.time_conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, dim, temporal=False):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dim, dim // 2, 3, padding=1)
        )
        self.temporal = temporal
        if temporal:
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, x):
        b, c, t, h, w = x.shape
        if self.temporal:
            x = self.time_conv(x)
            x = rearrange(x, 'b (n c) t h w -> b c (t n) h w', n=2)
            t = t * 2
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.spatial(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, z_dim=16, base_dim=96):
        super().__init__()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*4]

        self.conv_in = CausalConv3d(3, dims[0], 3, padding=1)

        self.blocks = nn.ModuleList([
            nn.Sequential(ResBlock(dims[0], dims[0]), ResBlock(dims[0], dims[0])),
            nn.Sequential(ResBlock(dims[0], dims[1]), ResBlock(dims[1], dims[1])),
            nn.Sequential(ResBlock(dims[1], dims[2]), ResBlock(dims[2], dims[2])),
            nn.Sequential(ResBlock(dims[2], dims[3]), ResBlock(dims[3], dims[3])),
        ])

        self.downs = nn.ModuleList([
            Downsample(dims[0], temporal=False),
            Downsample(dims[1], temporal=True),
            Downsample(dims[2], temporal=True),
        ])

        self.mid = nn.Sequential(
            ResBlock(dims[3], dims[3]),
            SpatialAttention(dims[3]),
            ResBlock(dims[3], dims[3]),
        )

        self.conv_out = nn.Sequential(
            RMSNorm(dims[3]), nn.SiLU(),
            CausalConv3d(dims[3], z_dim * 2, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv_in(x)
        for i, (block, down) in enumerate(zip(self.blocks[:-1], self.downs)):
            x = block(x)
            x = down(x)
        x = self.blocks[-1](x)
        x = self.mid(x)
        x = self.conv_out(x)
        return x.chunk(2, dim=1)  # mu, logvar


class SimpleDecoder(nn.Module):
    def __init__(self, z_dim=16, base_dim=96):
        super().__init__()
        dims = [base_dim*4, base_dim*4, base_dim*2, base_dim]

        self.conv_in = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.mid = nn.Sequential(
            ResBlock(dims[0], dims[0]),
            SpatialAttention(dims[0]),
            ResBlock(dims[0], dims[0]),
        )

        self.ups = nn.ModuleList([
            Upsample(dims[0], temporal=True),
            Upsample(dims[1], temporal=True),
            Upsample(dims[2], temporal=False),
        ])

        self.blocks = nn.ModuleList([
            nn.Sequential(ResBlock(dims[0]//2, dims[1]), ResBlock(dims[1], dims[1]), ResBlock(dims[1], dims[1])),
            nn.Sequential(ResBlock(dims[1]//2, dims[2]), ResBlock(dims[2], dims[2]), ResBlock(dims[2], dims[2])),
            nn.Sequential(ResBlock(dims[2]//2, dims[3]), ResBlock(dims[3], dims[3]), ResBlock(dims[3], dims[3])),
        ])

        self.conv_out = nn.Sequential(
            RMSNorm(dims[3]), nn.SiLU(),
            CausalConv3d(dims[3], 3, 3, padding=1),
        )

    def forward(self, z):
        x = self.conv_in(z)
        x = self.mid(x)
        for up, block in zip(self.ups, self.blocks):
            x = up(x)
            x = block(x)
        return self.conv_out(x)


class SimpleVideoVAE(nn.Module):
    """
    Simplified 3D Video VAE.

    Usage:
        vae = SimpleVideoVAE()
        video = torch.randn(1, 3, 17, 256, 256)  # [B, C, T, H, W]

        # Encode
        mu, logvar = vae.encode(video)
        z = vae.reparameterize(mu, logvar)

        # Decode
        recon = vae.decode(z)

        # Training
        recon, mu, logvar = vae(video)
        loss = vae.loss(video, recon, mu, logvar)
    """
    def __init__(self, z_dim=16, base_dim=96):
        super().__init__()
        self.encoder = SimpleEncoder(z_dim, base_dim)
        self.decoder = SimpleDecoder(z_dim, base_dim)

        # Register normalization buffers (update these from training data)
        self.register_buffer('mean', torch.zeros(z_dim))
        self.register_buffer('std', torch.ones(z_dim))

    def encode(self, x):
        mu, logvar = self.encoder(x)
        # Normalize
        mu = (mu - self.mean.view(1, -1, 1, 1, 1)) / self.std.view(1, -1, 1, 1, 1)
        return mu, logvar

    def decode(self, z):
        # Denormalize
        z = z * self.std.view(1, -1, 1, 1, 1) + self.mean.view(1, -1, 1, 1, 1)
        return self.decoder(z).clamp(-1, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, recon, mu, logvar, kl_weight=1e-6):
        recon_loss = F.l1_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss


if __name__ == "__main__":
    vae = SimpleVideoVAE(z_dim=16, base_dim=64)
    x = torch.randn(1, 3, 17, 128, 128)

    print(f"Input: {list(x.shape)}")
    mu, logvar = vae.encode(x)
    print(f"Latent: {list(mu.shape)}")
    recon = vae.decode(mu)
    print(f"Output: {list(recon.shape)}")
    print(f"Params: {sum(p.numel() for p in vae.parameters()):,}")
```

---

## 8. File Reference

| File | Description |
|------|-------------|
| `inference.py` | Entry point - shows VAE usage in generation pipeline |
| `wan/modules/vae.py` | Core building blocks (CausalConv3d, ResBlock, Attention, Resample) |
| `wan/vae/wanx_vae_src/vae.py` | Main VAE classes (VideoVAE_, WanVAE with tiling) |
| `wan/vae/wanx_vae.py` | Wrapper combining VAE + CLIP encoder |
| `demo_utils/vae_block3.py` | Decoder wrapper for streaming inference |
| `temp_vae_exploration.py` | Exploration script (delete when done) |

---

## Quick Reference

```
ENCODING:
  Input:  [B, 3, T, H, W]           # RGB, [-1, 1]
  Output: [B, 16, T', H/8, W/8]     # T' = 1 + (T-1)//4

DECODING:
  Input:  [B, 16, T', H/8, W/8]
  Output: [B, 3, T_out, H, W]       # T_out = T'*4 - 3

ARCHITECTURE:
  Encoder: Conv3d → [ResBlock×2 → Down]×3 → ResBlock×2 → Mid → Head
  Decoder: Conv3d → Mid → [Up → ResBlock×3]×3 → Head

HYPERPARAMETERS:
  z_dim=16, base_dim=96, dim_mult=[1,2,4,4]
  temporal_downsample=[False, True, True]

NORMALIZATION (per-channel):
  Encode: z = (mu - mean) / std
  Decode: z = z * std + mean
```
