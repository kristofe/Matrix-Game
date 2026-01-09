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

**Appendices**
- [Appendix A: VAE in the Full Pipeline](#appendix-a-vae-in-the-full-pipeline)
- [Appendix B: CLIP Integration](#appendix-b-clip-integration)
- [Appendix C: Streaming Decoder](#appendix-c-streaming-decoder)

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
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)  # 1x1 conv = linear projection
        self.proj = nn.Conv2d(dim, dim, 1)        # 1x1 conv = linear projection
        nn.init.zeros_(self.proj.weight)  # Initialize to identity

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')  # Flatten time into batch
        x = self.norm(x)

        # Reshape for attention: each spatial position becomes a token
        q, k, v = self.to_qkv(x).reshape(b*t, 1, c*3, h*w).permute(0,1,3,2).chunk(3, dim=-1)
        # q, k, v shapes: [B*T, 1, H*W, C] - H*W tokens, each with C dimensions

        x = F.scaled_dot_product_attention(q, k, v)  # Standard matrix attention!

        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x + identity
```

**Important clarification:**
- The `nn.Conv2d(..., 1)` layers are **1x1 convolutions**, which are mathematically equivalent to `nn.Linear` applied independently at each spatial position. This is a common pattern in vision architectures.
- The actual attention mechanism uses **standard scaled dot-product attention**: `softmax(QK^T / sqrt(d)) * V`
- Each spatial position (pixel) becomes a "token", so attention is computed over H×W tokens per frame

### 4.5 Resample (Downsample/Upsample)

**File:** `wan/modules/vae.py:66-161`

Handles spatial (H, W) and temporal (T) resampling **separately**.

```python
class Resample(nn.Module):
    def __init__(self, dim, mode):
        # mode: 'downsample2d', 'downsample3d', 'upsample2d', 'upsample3d'

        if mode == 'downsample2d':
            # Spatial only: H,W → H/2, W/2
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),  # Asymmetric pad for odd dims
                nn.Conv2d(dim, dim, 3, stride=2)
            )

        elif mode == 'downsample3d':
            # Spatial: same as downsample2d
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=2)
            )
            # Temporal: T → T/2 (kernel covers 3 frames, stride 2)
            self.time_conv = CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1))

        elif mode == 'upsample2d':
            # Spatial only: H,W → 2H, 2W
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(dim, dim//2, 3, padding=1)  # Note: halves channels!
            )

        elif mode == 'upsample3d':
            # Spatial: same as upsample2d
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(dim, dim//2, 3, padding=1)
            )
            # Temporal: outputs 2x channels, then reshaped to 2x frames
            self.time_conv = CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))
```

**How temporal upsampling works (upsample3d):**
```python
# In forward(), after time_conv outputs [B, C*2, T, H, W]:
x = x.reshape(b, 2, c, t, h, w)              # Split channels into 2
x = torch.stack((x[:, 0], x[:, 1]), dim=3)   # Interleave along time axis
x = x.reshape(b, c, t * 2, h, w)             # Result: [B, C, T*2, H, W]
```
This is a **learned temporal upsample**: instead of interpolation, the conv predicts two frames per input frame.

**How spatial ops work (all modes):**
```python
# Flatten time into batch, apply 2D conv, restore
x = rearrange(x, 'b c t h w -> (b t) c h w')  # Each frame processed independently
x = self.resample(x)
x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
```

**Processing order:**
- `upsample3d`: temporal first → then spatial
- `downsample3d`: spatial first → then temporal

**Why asymmetric padding `ZeroPad2d((0, 1, 0, 1))`?**

The padding adds zeros only to the right and bottom edges:
```
nn.ZeroPad2d((left=0, right=1, top=0, bottom=1))
nn.Conv2d(dim, dim, kernel=3, stride=2, padding=0)
```

Tracing through an 8×8 input:
```
Input: 8×8

After ZeroPad2d(0,1,0,1) → 9×9:
    ┌─────────────────┬─┐
    │                 │0│
    │                 │0│
    │     pixels      │0│  ← zeros added to right
    │                 │0│
    │                 │0│
    ├─────────────────┼─┤
    │ 0 0 0 0 0 0 0 0 │0│  ← zeros added to bottom
    └─────────────────┴─┘

Then Conv2d(kernel=3, stride=2, padding=0):
    Output: (9 - 3) / 2 + 1 = 4×4
```

Why not symmetric padding on all sides?

1. **Alignment**: The first output pixel's 3×3 receptive field covers actual pixels `[0:3, 0:3]`, not padded zeros. The top-left corner stays anchored.

2. **Matches upsampling**: When you upsample (nearest neighbor) then downsample, asymmetric padding ensures correct spatial alignment.

3. **Handles odd dimensions**:
   ```
   7×7 → pad → 8×8 → stride 2 → 4×4
   8×8 → pad → 9×9 → stride 2 → 4×4
   ```
   Both map to `ceil(H/2)` output size.

Where the kernel lands with asymmetric padding (kernel=3, stride=2):
```
         col: 0   1   2   3   4   5   6   7   8
            ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
      row 0 │ A │ A │ * │ B │ * │ C │ * │ D │ 0 │
      row 1 │ A │ A │ * │ B │ * │ C │ * │ D │ 0 │
      row 2 │ * │ * │ * │ * │ * │ * │ * │ * │ 0 │  ← row 2 shared by kernels A,B,C,D and E,F,G,H
      row 3 │ E │ E │ * │ F │ * │ G │ * │ H │ 0 │
      row 4 │ E │ E │ * │ F │ * │ G │ * │ H │ 0 │
      row 5 │ * │ * │ * │ * │ * │ * │ * │ * │ 0 │
      row 6 │ I │ I │ * │ J │ * │ K │ * │ L │ 0 │
      row 7 │ I │ I │ * │ J │ * │ K │ * │ L │ 0 │
      row 8 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │  ← padded zeros
            └───┴───┴───┴───┴───┴───┴───┴───┴───┘

   * = overlap between adjacent kernels

Kernel A: rows 0-2, cols 0-2 → output[0,0]
Kernel B: rows 0-2, cols 2-4 → output[0,1]  (overlaps A at col 2)
Kernel C: rows 0-2, cols 4-6 → output[0,2]  (overlaps B at col 4)
Kernel D: rows 0-2, cols 6-8 → output[0,3]  (overlaps C at col 6)
...
Output: 4×4 from 9×9 padded input (originally 8×8)
```

With stride=2 and kernel=3, consecutive kernels **overlap by 1 pixel**.
The kernel starts at actual pixel [0,0], not at a padded zero.

**The `time_conv` - Temporal Convolution**

The `time_conv` handles temporal dimension changes. Its kernel shape is `(3, 1, 1)`:
```
Temporal: 3 frames  ← looks at 3 consecutive frames
Height:   1 pixel   ← no spatial mixing
Width:    1 pixel   ← no spatial mixing
```
This is a **purely temporal convolution** - it mixes information across time only, applied independently at each spatial location.

**For downsample3d** - halves temporal dimension:
```python
self.time_conv = CausalConv3d(dim, dim, (3,1,1), stride=(2,1,1), padding=(0,0,0))
```
```
Input frames:   [0] [1] [2] [3] [4] [5] [6] [7]
                 \_____/     \_____/     \_____/
Kernel covers 3,   ↓           ↓           ↓      stride=2
Output frames:    [0]         [1]         [2]     → T/2 frames
```

**For upsample3d** - doubles temporal dimension:
```python
self.time_conv = CausalConv3d(dim, dim*2, (3,1,1), padding=(1,0,0))
```

**Tracing the upsample padding:**
```
padding=(1, 0, 0) means pad_t=1, pad_h=0, pad_w=0

CausalConv3d converts this to:
  _padding = (0, 0,      # Width:  no padding
              0, 0,      # Height: no padding
              2*1, 0)    # Time:   2 past, 0 future  ← causal!

With kernel=3 and 2 frames of past padding:

  Input:    [frame0] [frame1] [frame2]
              ↓
  Padded:   [0] [0] [frame0] [frame1] [frame2]
             ↑   ↑
           zero padding (past only)

  Kernel positions (size 3, stride 1):
    Output 0: kernel sees [  0  ] [  0  ] [frame0] → outputs [0a, 0b]
    Output 1: kernel sees [  0  ] [frame0] [frame1] → outputs [1a, 1b]
    Output 2: kernel sees [frame0] [frame1] [frame2] → outputs [2a, 2b]
```
Each output only sees current + past frames, never future frames.

Outputs 2x channels, then reshaped to 2x frames:
```python
# After time_conv: [B, C*2, T, H, W]
x = x.reshape(b, 2, c, t, h, w)            # split channels into 2
x = torch.stack((x[:,0], x[:,1]), dim=3)   # interleave along time
x = x.reshape(b, c, t*2, h, w)             # now T*2 frames
```
```
Input frames:     [0]     [1]     [2]
                   ↓       ↓       ↓
Conv outputs:   [0a,0b] [1a,1b] [2a,2b]   ← 2 channels per frame
                   ↓  interleave  ↓
Output frames:  [0a][0b][1a][1b][2a][2b]  → 2x frames
```
This is a **learned temporal upsample** - the network predicts two frames from each input frame.

**Why "Causal" Conv?**

Looking at CausalConv3d padding (line 24-25 of vae.py):
```python
self._padding = (pad_w, pad_w,      # Width:  symmetric
                 pad_h, pad_h,      # Height: symmetric
                 2 * pad_t, 0)      # Time:   ALL on past, none on future!
```

Normal conv with kernel=3, padding=1 (symmetric):
```
past | current | future
 [1]     [2]      [3]    ← sees 1 past + 1 future frame
```

Causal conv with kernel=3, padding=1:
```
     past      | current
 [1]    [2]       [3]    ← sees 2 past + 0 future frames
```

**Why causal matters:** For video generation/streaming, future frames don't exist yet. Causal convolutions ensure each output only depends on past and current inputs, enabling frame-by-frame streaming decode.

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

**conv1 and conv2 - Latent Projections:**

```python
# Defined in __init__:
self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)  # kernel=1 (pointwise)
self.conv2 = CausalConv3d(z_dim, z_dim, 1)          # kernel=1 (pointwise)
```

These are **1x1x1 convolutions** - pointwise linear transforms with no spatial/temporal mixing:
- `conv1`: Refines encoder output before splitting into μ and log_var
- `conv2`: Refines latent before feeding to decoder

Why 1x1x1 conv instead of Linear? Same math, but keeps tensor in [B,C,T,H,W] format.

**Why different chunking: encode [1,4,4,4...] vs decode [1,1,1...]?**

The temporal compression is 4x (two `downsample3d` with stride=2 each):
```
temporal_downsample = [False, True, True]  → 1x * 2x * 2x = 4x compression
```

**Encode processes [1, 4, 4, 4, ...] input frames:**
```
Input frames:    [0] [1  2  3  4] [5  6  7  8] ...
                  ↓       ↓            ↓
Latent frames:   [0]     [1]         [2]       ...

- Frame 0 alone  → latent 0  (establishes causal cache, no past context)
- Frames 1-4     → latent 1  (4 frames compress to 1 via 4x downsample)
- Frames 5-8     → latent 2
```

**Decode processes [1, 1, 1, ...] latent frames:**
```
Latent frames:   [0]        [1]        [2]
                  ↓          ↓          ↓
Output frames:  [0 1 2 3]  [4 5 6 7]  [8 9 10 11]
                4 frames   4 frames    4 frames

temporal_upsample = [True, True, False]  → 2x * 2x * 1x = 4x expansion
```

**Key insight:** Each single latent frame "contains" ~4 video frames worth of information. The encoder compresses 4→1, the decoder expands 1→4.

Decoding one latent at a time enables:
1. **Streaming:** Output frames immediately, don't wait for all latents
2. **Memory efficiency:** Minimal data in GPU memory at once
3. **Causal coherence:** `feat_cache` maintains temporal context between chunks

```
ENCODE                              DECODE
[B,3,17,H,W]  →  [1,4,4,4] chunks   [B,16,5,h,w]  →  [1,1,1,1,1] chunks  →  [B,3,17,H,W]
                 ↓                                   ↓
              4x compress                         4x expand
              with caching                        with caching
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

---

## Appendix A: VAE in the Full Pipeline

### A.1 Overview: Where VAE Fits

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Input Image │ ──▶ │ VAE Encoder  │ ──▶ │   Diffusion  │ ──▶ │ VAE Decoder  │ ──▶ Output Video
│  [3,1,H,W]   │     │              │     │    Model     │     │  (Streaming) │     [3,T,H,W]
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │                    ▲
                            │                    │
                            ▼                    │
                     ┌──────────────┐     ┌──────────────┐
                     │    CLIP      │     │  Keyboard/   │
                     │  (Visual     │     │   Mouse      │
                     │   Context)   │     │   Actions    │
                     └──────────────┘     └──────────────┘
```

**From `inference.py`:**

```python
# Step 1: Load VAE + CLIP together
vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
# This loads BOTH:
#   - vae.vae: the VideoVAE model
#   - vae.clip: the CLIPModel for visual conditioning

# Step 2: Encode input image to latent space
img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs)
# Input:  [1, 3, 597, 352, 640]
# Output: [1, 16, 150, 44, 80]

# Step 3: Get visual context from CLIP (for conditioning diffusion)
visual_context = self.vae.clip.encode_video(image)
# Output: [B, 257, 1280] - CLIP features for cross-attention

# Step 4: Diffusion generates latents (block by block)
# Step 5: VAE Decoder (streaming) converts latents back to video
```

### A.2 Pipeline Flow in Detail

**From `pipeline/causal_inference.py`:**

```python
class CausalInferencePipeline:
    def inference(self, noise, conditional_dict, ...):
        # Initialize streaming VAE cache
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)  # 32 cache tensors
        videos = []

        # For each block of latent frames...
        for current_num_frames in all_num_frames:
            # 1. Denoise the latent (diffusion model)
            for timestep in denoising_steps:
                denoised_pred = self.generator(
                    noisy_input,
                    conditional_dict,  # Contains visual_context from CLIP
                    timestep,
                    kv_cache=...,
                )

            # 2. IMMEDIATELY decode to video (streaming)
            video, vae_cache = self.vae_decoder(denoised_pred, *vae_cache)
            videos.append(video)  # Accumulate decoded frames

        return videos  # List of video chunks
```

**Key insight**: The VAE decoder is called **after every diffusion block**, not at the end. This enables:
- **Low latency**: See frames as they're generated
- **Memory efficiency**: Don't store all latents before decoding
- **Streaming output**: Can display/save frames incrementally

---

## Appendix B: CLIP Integration

### B.1 Is CLIP Used in VAE Training?

**No.** CLIP and VAE are completely independent:

1. **VAE is trained separately** - standard reconstruction + KL loss
2. **CLIP is a pretrained model** - loaded from `open-clip-xlm-roberta-large-vit-huge-14.pth`
3. **They're only combined at inference time** for the diffusion model

**Evidence from `finetune_base.py`:**

```python
# Load VAE + CLIP
vae = get_wanx_vae_wrapper("models/", torch.float16)
vae.requires_grad_(False)  # VAE is FROZEN
vae.eval()

# During training:
def train_step(model, vae, batch, ...):
    # VAE encodes training data to latents
    latents = vae.encode(frames, device=device)

    # CLIP provides visual context (separate from VAE)
    visual_context = vae.clip.encode_video(first_frame)

    # Only the DIFFUSION MODEL is trained
    loss = diffusion_model(latents, visual_context, ...)
```

### B.2 How CLIP is Used

**File:** `wan/vae/wanx_vae_src/clip.py`

CLIP provides a 1280-dimensional visual embedding for conditioning:

```python
class CLIPModel:
    def encode_video(self, video):
        # video: [B, 3, T, H, W]
        b, c, t, h, w = video.shape

        # Reshape to process each frame
        video = video.transpose(1, 2).reshape(b * t, c, h, w)

        # Resize to CLIP's expected size (224x224)
        video = F.interpolate(video, size=(224, 224), mode='bicubic')

        # Normalize for CLIP
        video = self.transforms(video)  # CLIP normalization

        # Run through ViT (stopping before final pooling)
        out = self.model.visual(video, use_31_block=True)
        # Output: [B*T, 257, 1280]
        # 257 = 1 CLS token + 256 patch tokens (16x16 grid)

        return out
```

**The output is used for cross-attention in the diffusion model's transformer blocks.**

### B.3 VAE + CLIP Wrapper

**File:** `wan/vae/wanx_vae.py`

```python
class WanxVAEWrapper:
    def __init__(self, vae, clip):
        self.vae = vae   # VideoVAE for encode/decode
        self.clip = clip  # CLIPModel for visual conditioning

    def encode(self, x, device, tiled=False, ...):
        return self.vae.encode(x, device=device, tiled=tiled, ...)

    def decode(self, latents, device, tiled=False, ...):
        return self.vae.decode(latents, device=device, tiled=tiled, ...)

def get_wanx_vae_wrapper(model_path, weight_dtype):
    vae = WanVAE(pretrained_path=os.path.join(model_path, "Wan2.1_VAE.pth"))
    clip = CLIPModel(
        checkpoint_path=os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        tokenizer_path=os.path.join(model_path, 'xlm-roberta-large')
    )
    return WanxVAEWrapper(vae, clip)
```

---

## Appendix C: Streaming Decoder

### C.1 Why Streaming?

Normal VAE decoding:
```python
# Wait for ALL latents, then decode
all_latents = diffusion_model.generate()  # [B, 16, 150, 44, 80]
video = vae.decode(all_latents)           # Decode all at once
```

Streaming VAE decoding:
```python
# Decode each block immediately after it's generated
for block in diffusion_blocks:
    latent_block = diffusion_model.denoise(block)  # [B, 16, 3, 44, 80]
    video_block, cache = vae_decoder(latent_block, cache)
    yield video_block  # Output frames immediately!
```

### C.2 The VAE Cache Structure

**File:** `demo_utils/constant.py`

The streaming decoder maintains 32 cache tensors, one for each CausalConv3d:

```python
ZERO_VAE_CACHE = [
    # Layer 0: conv2 (latent input)
    torch.zeros(1, 16, 2, 44, 80),      # 44x80 (latent resolution)

    # Layers 1-11: Middle block + first upsample level (384 channels)
    torch.zeros(1, 384, 2, 44, 80),     # 11 tensors at 44x80

    # Layers 12-18: After first upsample (384→192 channels, 2x resolution)
    torch.zeros(1, 192, 2, 88, 160),    # 7 tensors at 88x160
    torch.zeros(1, 384, 2, 88, 160),

    # Layers 19-25: After second upsample (192→96 channels, 4x resolution)
    torch.zeros(1, 192, 2, 176, 320),   # 7 tensors at 176x320

    # Layers 26-31: After third upsample (96 channels, 8x resolution)
    torch.zeros(1, 96, 2, 352, 640),    # 6 tensors at 352x640 (output resolution)
]
```

**Each cache stores the last 2 frames** (CACHE_T = 2) of intermediate features.

### C.3 VAEDecoderWrapper

**File:** `demo_utils/vae_block3.py`

```python
class VAEDecoderWrapper(nn.Module):
    def __init__(self):
        self.decoder = VAEDecoder3d()
        self.conv2 = CausalConv3d(16, 16, 1)

        # Hardcoded normalization constants
        self.mean = torch.tensor([...])  # 16 values
        self.std = torch.tensor([...])   # 16 values
        self.z_dim = 16

    def forward(self, z, *feat_cache):
        # z: [B, T, C, H, W] -> transpose to [B, C, T, H, W]
        z = z.permute(0, 2, 1, 3, 4)
        feat_cache = list(feat_cache)

        # Denormalize
        scale = [self.mean, 1.0 / self.std]
        z = z / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)

        # Process through conv2
        x = self.conv2(z)

        # Decode each latent frame with caching
        for i in range(z.shape[2]):
            if i == 0:
                out, feat_cache = self.decoder(x[:, :, i:i+1, :, :], feat_cache)
            else:
                out_, feat_cache = self.decoder(x[:, :, i:i+1, :, :], feat_cache)
                out = torch.cat([out, out_], dim=2)

        # Clamp and transpose back
        out = out.float().clamp_(-1, 1)
        out = out.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        return out, feat_cache  # Return BOTH video AND updated cache
```

### C.4 How Caching Works

Each CausalConv3d layer:
1. **Receives** the cache from the previous call (last 2 frames of features)
2. **Concatenates** cache with current input
3. **Computes** the convolution (now has enough temporal context)
4. **Updates** the cache with the last 2 frames of current features
5. **Returns** output and updated cache

```python
def forward(self, x, cache_x=None):
    # cache_x: [B, C, 2, H, W] - last 2 frames from previous call

    if cache_x is not None:
        # Prepend cached frames to current input
        x = torch.cat([cache_x, x], dim=2)  # [B, C, 2+T, H, W]
        # Reduce the zero-padding since we have real data

    # Apply causal padding (only on left side of time)
    x = F.pad(x, self._padding)

    # Convolution
    return super().forward(x)
```

### C.5 Streaming Decode in the Pipeline

```python
# From CausalInferencePipeline.inference()

# Initialize cache with zeros (or None)
vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
for j in range(len(vae_cache)):
    vae_cache[j] = None  # Will be populated on first call

# For each block of latents...
for block_idx in range(num_blocks):
    # Diffusion generates 3 latent frames
    denoised_pred = diffusion_step(...)  # [B, 16, 3, 44, 80]

    # Immediately decode with cached state
    video_chunk, vae_cache = self.vae_decoder(
        denoised_pred.transpose(1, 2).half(),  # [B, 3, 16, 44, 80]
        *vae_cache  # Unpack 32 cache tensors
    )
    # video_chunk: [B, ~12, 3, 352, 640]  (3 latent frames → ~12 video frames)

    videos.append(video_chunk)

    # vae_cache is now updated for next iteration!
```

### C.6 Diagram: Streaming Decode

```
Block 0                          Block 1                          Block 2
────────                         ────────                         ────────

Latents: [16, 3, 44, 80]        Latents: [16, 3, 44, 80]        Latents: [16, 3, 44, 80]
         │                               │                               │
         ▼                               ▼                               ▼
┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
│  VAE Decoder    │             │  VAE Decoder    │             │  VAE Decoder    │
│  + Empty Cache  │ ──cache──▶  │  + Cache[0]     │ ──cache──▶  │  + Cache[1]     │
└─────────────────┘             └─────────────────┘             └─────────────────┘
         │                               │                               │
         ▼                               ▼                               ▼
Video: [3, 9, 352, 640]         Video: [3, 12, 352, 640]        Video: [3, 12, 352, 640]
(First frame + 2×4)             (3×4 frames)                    (3×4 frames)

Total: 9 + 12 + 12 + ... frames (no seams, smooth transitions)
```

The cache ensures temporal coherence between blocks - without it, each block would have discontinuities at the boundaries.

---

## Summary

| Component | Role | Training |
|-----------|------|----------|
| **VAE Encoder** | Compress video → latent | Trained separately (reconstruction + KL) |
| **VAE Decoder** | Decompress latent → video | Trained with encoder |
| **CLIP** | Visual conditioning for diffusion | Pretrained, frozen |
| **Diffusion Model** | Generate latents from noise | Trained with VAE+CLIP frozen |

**Key takeaways:**
1. VAE and CLIP are independent - CLIP is NOT used in VAE training
2. VAE decoder is called incrementally (streaming) for low latency
3. The 32-tensor cache enables temporal coherence in streaming mode
4. CLIP's 257×1280 features condition the diffusion model via cross-attention

---

## Appendix D: StreamableVideoVAE Implementation

This appendix provides a complete implementation of a **StreamableVideoVAE** that supports:
- Encoding 1 frame alone, then groups of 4 (matching the full WAN VAE)
- Streaming decode (one latent at a time)
- Feature caching for temporal coherence

### D.1 What's Needed for Streaming

To enable streaming, the VAE needs:

| Component | Requirement | Purpose |
|-----------|-------------|---------|
| **Causal Convolutions** | Pad only on the left (past) | No future frame dependency |
| **Feature Caching** | Store intermediate features | Temporal continuity across chunks |
| **Chunked Processing** | 1 frame, then 4s | Efficient incremental encoding |

### D.2 Complete StreamableVideoVAE Implementation

```python
"""
StreamableVideoVAE - Supports streaming encode/decode with caching.

Key differences from SimpleVideoVAE:
1. CausalConv3d with feature caching
2. Chunked encode: 1 frame, then groups of 4
3. Streaming decode: 1 latent at a time

Temporal formula: T_latent = 1 + (T_video - 1) // 4
Valid T_video: 1, 5, 9, 13, 17, 21... (1 + 4k)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, List, Tuple


class CausalConv3dCached(nn.Module):
    """
    3D Causal Convolution with feature caching for streaming.

    Key features:
    - Causal temporal padding (only looks at past)
    - Feature cache for streaming (maintains state between chunks)
    - Cache stores last (kernel_t - 1) frames of input
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        # Normalize to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

        # Cache for streaming: stores last (kernel_t - 1) frames
        self.cache: Optional[torch.Tensor] = None
        self.cache_size = kernel_size[0] - 1  # Temporal kernel minus 1

    def forward(self, x, use_cache=True):
        """
        Forward pass with optional caching.

        Args:
            x: [B, C, T, H, W] input tensor
            use_cache: If True, use/update cache for streaming

        Returns:
            output: [B, C_out, T_out, H_out, W_out]
        """
        # Prepend cache if available and using cache mode
        if use_cache and self.cache is not None:
            x = torch.cat([self.cache, x], dim=2)
            temporal_pad = max(0, self.cache_size - self.cache.shape[2])
        else:
            temporal_pad = self.cache_size + self.padding[0]

        # Causal padding: all temporal padding on left, symmetric spatial
        # Format: (W_left, W_right, H_left, H_right, T_left, T_right)
        pad = (
            self.padding[2], self.padding[2],  # Width: symmetric
            self.padding[1], self.padding[1],  # Height: symmetric
            temporal_pad, 0                     # Time: causal (left only)
        )
        x_padded = F.pad(x, pad)

        # Update cache with last cache_size frames of input (before padding)
        if use_cache and self.cache_size > 0:
            # Store the last cache_size frames for next chunk
            cache_start = max(0, x.shape[2] - self.cache_size)
            self.cache = x[:, :, cache_start:].detach().clone()

        return self.conv(x_padded)

    def clear_cache(self):
        """Clear the feature cache."""
        self.cache = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class CachedResBlock(nn.Module):
    """Residual block with cached causal convolutions."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = RMSNorm(in_ch)
        self.conv1 = CausalConv3dCached(in_ch, out_ch, 3, padding=1)
        self.norm2 = RMSNorm(out_ch)
        self.conv2 = CausalConv3dCached(out_ch, out_ch, 3, padding=1)
        self.skip = CausalConv3dCached(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, use_cache=True):
        h = self.act(self.norm1(x))
        h = self.conv1(h, use_cache=use_cache)
        h = self.act(self.norm2(h))
        h = self.conv2(h, use_cache=use_cache)

        if isinstance(self.skip, CausalConv3dCached):
            skip = self.skip(x, use_cache=use_cache)
        else:
            skip = self.skip(x)

        return h + skip

    def clear_cache(self):
        self.conv1.clear_cache()
        self.conv2.clear_cache()
        if isinstance(self.skip, CausalConv3dCached):
            self.skip.clear_cache()


class SpatialAttention(nn.Module):
    """Self-attention per frame (no caching needed - spatial only)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, use_cache=True):
        identity = x
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'bt (n c) h w -> bt n (h w) c', n=3)
        q, k, v = qkv.unbind(dim=1)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'bt (h w) c -> bt c h w', h=h)
        out = self.proj(out)

        out = rearrange(out, '(b t) c h w -> b c t h w', t=t)
        return identity + out

    def clear_cache(self):
        pass  # No cache for spatial attention


class CachedDownsample(nn.Module):
    """
    Downsampling with optional temporal stride and caching.

    KEY INSIGHT FROM WAN VAE:
    - First call: SKIP temporal downsample, save last frame to cache
    - Subsequent calls: concat cached frame + current, then strided conv

    This ensures:
    - 1 frame (first) → 1 latent (skip strided conv)
    - 4 frames (subsequent) → 1 latent (concat 1+4=5 → strided conv → ~2, then next layer)
    """
    def __init__(self, dim, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.is_first_call = True
        self.cached_frame = None  # Store last frame for next call

        # Spatial downsampling (2D, applied per-frame)
        self.spatial = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=2)
        )

        # Temporal downsampling (causal 3D conv with stride 2)
        # Note: kernel=3, stride=2, NO padding - the "causal" part is handled by concat
        if temporal:
            self.time_conv = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x, use_cache=True):
        b, c, t, h, w = x.shape

        # Spatial downsample (per-frame)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.spatial(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        # Temporal downsample
        if self.temporal:
            if use_cache and self.is_first_call:
                # FIRST CALL: Skip strided conv, just save state
                self.cached_frame = x[:, :, -1:, :, :].clone()
                self.is_first_call = False
                # x passes through unchanged (no temporal downsample)
            elif use_cache:
                # SUBSEQUENT CALLS: Concat cached frame + current, then strided conv
                x_with_context = torch.cat([self.cached_frame.to(x.device), x], dim=2)
                self.cached_frame = x[:, :, -1:, :, :].clone()  # Save last frame
                x = self.time_conv(x_with_context)
            else:
                # Non-cached mode: apply strided conv with zero padding
                x = F.pad(x, (0, 0, 0, 0, 2, 0))  # Pad 2 frames on left (temporal)
                x = self.time_conv(x)

        return x

    def clear_cache(self):
        self.is_first_call = True
        self.cached_frame = None


class CachedUpsample(nn.Module):
    """
    Upsampling with optional temporal expansion and caching.

    KEY INSIGHT FROM WAN VAE:
    - First latent: SKIP temporal upsample (only spatial) → 1 latent → 1 frame
    - Subsequent latents: DO temporal upsample → 1 latent → 2 frames (per layer)

    This is achieved by tracking whether we've seen the first latent via
    the `is_first_latent` flag, which is set on clear_cache() and cleared
    after the first forward pass.
    """
    def __init__(self, dim, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.is_first_latent = True  # Track if this is the first latent

        # Spatial upsampling (2D, applied per-frame)
        self.spatial = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dim, dim // 2, 3, padding=1)
        )

        # Temporal upsampling: conv outputs 2x channels, then reshape to 2x frames
        if temporal:
            self.time_conv = CausalConv3dCached(dim, dim * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, use_cache=True):
        b, c, t, h, w = x.shape

        # Temporal upsample (if enabled AND not the first latent)
        if self.temporal:
            if use_cache and self.is_first_latent:
                # FIRST LATENT: Skip temporal upsample entirely!
                # Don't call time_conv at all - leave its cache empty
                # On next call, time_conv will use zero padding (like fresh call)
                self.is_first_latent = False
                # t stays the same (no temporal expansion)
            else:
                # SUBSEQUENT LATENTS: Do temporal upsample
                # - Second latent: cache is None, uses zero padding, then saves cache
                # - Third+ latent: cache has data from previous call
                x = self.time_conv(x, use_cache=use_cache)
                # Reshape: [B, 2C, T, H, W] -> [B, C, 2T, H, W]
                x = rearrange(x, 'b (n c) t h w -> b c (t n) h w', n=2)
                t = t * 2

        # Spatial upsample (per-frame) - always applied
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.spatial(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        return x

    def clear_cache(self):
        self.is_first_latent = True  # Reset for next sequence
        if self.temporal:
            self.time_conv.clear_cache()


class StreamableEncoder(nn.Module):
    """Encoder with caching for streaming."""
    def __init__(self, z_dim=16, base_dim=96):
        super().__init__()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*4]

        self.conv_in = CausalConv3dCached(3, dims[0], 3, padding=1)

        # Encoder blocks
        self.blocks = nn.ModuleList([
            nn.ModuleList([CachedResBlock(dims[0], dims[0]), CachedResBlock(dims[0], dims[0])]),
            nn.ModuleList([CachedResBlock(dims[0], dims[1]), CachedResBlock(dims[1], dims[1])]),
            nn.ModuleList([CachedResBlock(dims[1], dims[2]), CachedResBlock(dims[2], dims[2])]),
            nn.ModuleList([CachedResBlock(dims[2], dims[3]), CachedResBlock(dims[3], dims[3])]),
        ])

        # Downsamplers: [no temporal, temporal, temporal]
        self.downs = nn.ModuleList([
            CachedDownsample(dims[0], temporal=False),
            CachedDownsample(dims[1], temporal=True),
            CachedDownsample(dims[2], temporal=True),
        ])

        # Middle block
        self.mid = nn.ModuleList([
            CachedResBlock(dims[3], dims[3]),
            SpatialAttention(dims[3]),
            CachedResBlock(dims[3], dims[3]),
        ])

        # Output head
        self.norm_out = RMSNorm(dims[3])
        self.conv_out = CausalConv3dCached(dims[3], z_dim * 2, 3, padding=1)

    def forward(self, x, use_cache=True):
        x = self.conv_in(x, use_cache=use_cache)

        # Encoder blocks with downsampling
        for i, (block_list, down) in enumerate(zip(self.blocks[:-1], self.downs)):
            for block in block_list:
                x = block(x, use_cache=use_cache)
            x = down(x, use_cache=use_cache)

        # Final block (no downsample)
        for block in self.blocks[-1]:
            x = block(x, use_cache=use_cache)

        # Middle
        x = self.mid[0](x, use_cache=use_cache)
        x = self.mid[1](x, use_cache=use_cache)
        x = self.mid[2](x, use_cache=use_cache)

        # Output
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x, use_cache=use_cache)

        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar

    def clear_cache(self):
        self.conv_in.clear_cache()
        for block_list in self.blocks:
            for block in block_list:
                block.clear_cache()
        for down in self.downs:
            down.clear_cache()
        for m in self.mid:
            m.clear_cache()
        self.conv_out.clear_cache()


class StreamableDecoder(nn.Module):
    """Decoder with caching for streaming."""
    def __init__(self, z_dim=16, base_dim=96):
        super().__init__()
        dims = [base_dim*4, base_dim*4, base_dim*2, base_dim]

        self.conv_in = CausalConv3dCached(z_dim, dims[0], 3, padding=1)

        # Middle block
        self.mid = nn.ModuleList([
            CachedResBlock(dims[0], dims[0]),
            SpatialAttention(dims[0]),
            CachedResBlock(dims[0], dims[0]),
        ])

        # Upsamplers: [temporal, temporal, no temporal]
        self.ups = nn.ModuleList([
            CachedUpsample(dims[0], temporal=True),
            CachedUpsample(dims[1], temporal=True),
            CachedUpsample(dims[2], temporal=False),
        ])

        # After upsample, channels are halved by the spatial conv
        self.blocks = nn.ModuleList([
            nn.ModuleList([CachedResBlock(dims[0]//2, dims[1]), CachedResBlock(dims[1], dims[1]), CachedResBlock(dims[1], dims[1])]),
            nn.ModuleList([CachedResBlock(dims[1]//2, dims[2]), CachedResBlock(dims[2], dims[2]), CachedResBlock(dims[2], dims[2])]),
            nn.ModuleList([CachedResBlock(dims[2]//2, dims[3]), CachedResBlock(dims[3], dims[3]), CachedResBlock(dims[3], dims[3])]),
        ])

        # Output head
        self.norm_out = RMSNorm(dims[3])
        self.conv_out = CausalConv3dCached(dims[3], 3, 3, padding=1)

    def forward(self, z, use_cache=True):
        x = self.conv_in(z, use_cache=use_cache)

        # Middle
        x = self.mid[0](x, use_cache=use_cache)
        x = self.mid[1](x, use_cache=use_cache)
        x = self.mid[2](x, use_cache=use_cache)

        # Decoder blocks with upsampling
        for up, block_list in zip(self.ups, self.blocks):
            x = up(x, use_cache=use_cache)
            for block in block_list:
                x = block(x, use_cache=use_cache)

        # Output
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x, use_cache=use_cache)

        return x

    def clear_cache(self):
        self.conv_in.clear_cache()
        for m in self.mid:
            m.clear_cache()
        for up in self.ups:
            up.clear_cache()
        for block_list in self.blocks:
            for block in block_list:
                block.clear_cache()
        self.conv_out.clear_cache()


class StreamableVideoVAE(nn.Module):
    """
    Streamable 3D Video VAE with caching support.

    Supports two modes:
    1. Batch mode: encode(video) - process entire video at once
    2. Streaming mode: encode_streaming(video) - process 1 frame, then 4s

    Temporal formula: T_latent = 1 + (T_video - 1) // 4
    Valid T_video: 1, 5, 9, 13, 17... (1 + 4k)

    Usage:
        vae = StreamableVideoVAE()

        # Batch mode (simple)
        video = torch.randn(1, 3, 17, 256, 256)
        latents = vae.encode(video)  # [1, 16, 5, 32, 32]
        recon = vae.decode(latents)  # [1, 3, 17, 256, 256]

        # Streaming mode (for inference)
        latents = vae.encode_streaming(video)  # Same result, chunked processing
        frames = []
        for i in range(latents.shape[2]):
            chunk = vae.decode_streaming(latents[:, :, i:i+1])
            frames.append(chunk)
        video = torch.cat(frames, dim=2)
    """
    def __init__(self, z_dim=16, base_dim=96):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = StreamableEncoder(z_dim, base_dim)
        self.decoder = StreamableDecoder(z_dim, base_dim)

        # Normalization buffers
        self.register_buffer('mean', torch.zeros(z_dim))
        self.register_buffer('std', torch.ones(z_dim))

    def encode(self, x, normalize=True):
        """
        Encode video to latents (batch mode).

        Args:
            x: [B, 3, T, H, W] video in [-1, 1]
            normalize: Whether to apply latent normalization

        Returns:
            mu: [B, 16, T', H/8, W/8] where T' = 1 + (T-1)//4
        """
        self.encoder.clear_cache()
        mu, logvar = self.encoder(x, use_cache=False)

        if normalize:
            mu = (mu - self.mean.view(1, -1, 1, 1, 1)) / self.std.view(1, -1, 1, 1, 1)

        return mu, logvar

    def encode_streaming(self, x, normalize=True):
        """
        Encode video to latents (streaming mode).

        Processes: 1 frame alone, then groups of 4.
        This matches the full WAN VAE behavior.

        Args:
            x: [B, 3, T, H, W] video in [-1, 1], T should be 1 + 4k
            normalize: Whether to apply latent normalization

        Returns:
            mu: [B, 16, T', H/8, W/8] where T' = 1 + (T-1)//4
        """
        self.encoder.clear_cache()

        t = x.shape[2]
        num_chunks = 1 + (t - 1) // 4

        outputs = []
        for i in range(num_chunks):
            if i == 0:
                # First frame alone
                chunk = x[:, :, :1]
            else:
                # Groups of 4
                start = 1 + 4 * (i - 1)
                end = min(1 + 4 * i, t)
                chunk = x[:, :, start:end]

            mu_chunk, _ = self.encoder(chunk, use_cache=True)
            outputs.append(mu_chunk)

        mu = torch.cat(outputs, dim=2)

        if normalize:
            mu = (mu - self.mean.view(1, -1, 1, 1, 1)) / self.std.view(1, -1, 1, 1, 1)

        return mu

    def decode(self, z, denormalize=True):
        """
        Decode latents to video (batch mode).

        Args:
            z: [B, 16, T', H/8, W/8] latents
            denormalize: Whether to apply latent denormalization

        Returns:
            video: [B, 3, T, H, W] where T = 1 + 4*(T'-1)
        """
        self.decoder.clear_cache()

        if denormalize:
            z = z * self.std.view(1, -1, 1, 1, 1) + self.mean.view(1, -1, 1, 1, 1)

        return self.decoder(z, use_cache=False).clamp(-1, 1)

    def decode_streaming(self, z, denormalize=True, is_first=None):
        """
        Decode latents to video (streaming mode).

        Call once per latent frame. First call should have is_first=True
        or will be auto-detected based on cache state.

        Args:
            z: [B, 16, 1, H/8, W/8] single latent frame
            denormalize: Whether to apply latent denormalization
            is_first: If True, clear cache first. Auto-detect if None.

        Returns:
            video: [B, 3, T_out, H, W] - 1 frame for first latent, 4 for subsequent
        """
        if is_first is None:
            # Auto-detect: if decoder has no cache, this is the first
            is_first = self.decoder.conv_in.cache is None

        if is_first:
            self.decoder.clear_cache()

        if denormalize:
            z = z * self.std.view(1, -1, 1, 1, 1) + self.mean.view(1, -1, 1, 1, 1)

        return self.decoder(z, use_cache=True).clamp(-1, 1)

    def clear_cache(self):
        """Clear all caches."""
        self.encoder.clear_cache()
        self.decoder.clear_cache()

    def forward(self, x):
        """Training forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def loss(self, x, recon, mu, logvar, kl_weight=1e-6):
        recon_loss = F.l1_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    print("Testing StreamableVideoVAE...")

    vae = StreamableVideoVAE(z_dim=16, base_dim=64)
    print(f"Parameters: {sum(p.numel() for p in vae.parameters()):,}")

    # Test batch mode
    print("\n--- Batch Mode ---")
    x = torch.randn(1, 3, 17, 128, 128)  # 17 = 1 + 4*4, valid for streaming
    print(f"Input: {list(x.shape)}")

    mu, _ = vae.encode(x)
    print(f"Latent (batch): {list(mu.shape)}")  # Should be [1, 16, 5, 16, 16]

    recon = vae.decode(mu)
    print(f"Recon (batch): {list(recon.shape)}")  # Should be [1, 3, 17, 128, 128]

    # Test streaming encode
    print("\n--- Streaming Encode ---")
    vae.clear_cache()
    mu_stream = vae.encode_streaming(x)
    print(f"Latent (streaming): {list(mu_stream.shape)}")
    print(f"Latents match: {torch.allclose(mu, mu_stream, atol=1e-5)}")

    # Test streaming decode
    print("\n--- Streaming Decode ---")
    vae.clear_cache()
    frames = []
    for i in range(mu.shape[2]):
        chunk = vae.decode_streaming(mu[:, :, i:i+1])
        frames.append(chunk)
        print(f"  Latent {i}: output shape {list(chunk.shape)}")

    recon_stream = torch.cat(frames, dim=2)
    print(f"Recon (streaming): {list(recon_stream.shape)}")

    # Note: streaming decode may not exactly match batch due to caching differences
    # but should be visually similar

    print("\n--- Training Example ---")
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    for step in range(3):
        recon, mu, logvar = vae(x)
        loss = vae.loss(x, recon, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}: loss = {loss.item():.4f}")

    print("\nDone!")
```

### D.3 Key Differences: SimpleVideoVAE vs StreamableVideoVAE

| Feature | SimpleVideoVAE | StreamableVideoVAE |
|---------|----------------|-------------------|
| Temporal formula | `T_latent = T_video // 4` | `T_latent = 1 + (T_video - 1) // 4` |
| Valid inputs | 4, 8, 12, 16... | 1, 5, 9, 13, 17... |
| Single frame encode | No (needs 4) | Yes |
| Streaming encode | No | Yes (`encode_streaming`) |
| Streaming decode | No | Yes (`decode_streaming`) |
| Feature caching | No | Yes |
| Matches WAN VAE | No | Yes |

### D.4 Training the StreamableVideoVAE

**Key insight:** Training uses **batch mode** (full video at once), while inference can use **streaming mode** (chunked). The "skip first latent" logic in CachedDownsample/CachedUpsample only activates during streaming.

#### Training vs Streaming

| Aspect | Training (Batch Mode) | Inference (Streaming Mode) |
|--------|----------------------|---------------------------|
| Method | `vae.forward(video)` or `vae.encode()`/`vae.decode()` | `vae.encode_streaming()` / `vae.decode_streaming()` |
| Processing | Full video at once | 1 frame, then groups of 4 |
| `use_cache` | `False` | `True` |
| Temporal skip logic | NOT active | Active (first latent skipped) |

#### Why Train on 5 Frames?

With the formula `T_latent = 1 + (T_video - 1) // 4`:
- **1 frame → 1 latent**: No temporal learning (single frame)
- **5 frames → 2 latents**: Minimum to learn temporal compression!

Training on 5 frames teaches the model:
1. How first frame contributes to first latent
2. How next 4 frames contribute to second latent
3. Temporal relationships between latent frames

#### Training Code

```python
# Training loop - uses batch mode (no caching)
vae = StreamableVideoVAE(z_dim=16, base_dim=96)
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in dataloader:
        # Valid T: 5, 9, 13, 17... (1 + 4k where k >= 1)
        video = batch['video']  # [B, 3, 5, H, W]

        # forward() uses encode/decode with use_cache=False
        recon, mu, logvar = vae(video)
        loss = vae.loss(video, recon, mu, logvar, kl_weight=1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### After Training: Streaming Inference

```python
# Inference - uses streaming mode (with caching)
vae.eval()
vae.clear_cache()

# First: encode first frame alone
first_latent = vae.encode_streaming(first_frame[:, :, :1])  # 1 frame → 1 latent

# Then: encode groups of 4
next_latent = vae.encode_streaming(frames[:, :, 1:5])  # 4 frames → 1 latent

# Decode one latent at a time
frame_0 = vae.decode_streaming(first_latent, is_first=True)   # 1 latent → 1 frame
frames_1_4 = vae.decode_streaming(next_latent)                # 1 latent → 4 frames
```

**Valid training sequence lengths:**
- 5 frames → 2 latents (recommended minimum)
- 9 frames → 3 latents
- 13 frames → 4 latents
- 17 frames → 5 latents

### D.5 What Else is Needed for Full Streaming?

The StreamableVideoVAE provides the **foundation** for streaming. For a complete streaming system:

| Component | Status | Notes |
|-----------|--------|-------|
| **VAE Encoder caching** | ✅ Done | `encode_streaming()` |
| **VAE Decoder caching** | ✅ Done | `decode_streaming()` |
| **DiT KV caching** | Separate | See `CausalWanModel` in DIT_DOC.md |
| **Block-wise diffusion** | Separate | Generate N latents at a time |
| **Action buffering** | Separate | Buffer past actions for conditioning |

For streaming inference:

```python
# Streaming inference loop
vae.clear_cache()
dit.clear_kv_cache()

# Encode first frame
first_frame = video[:, :, :1]  # [B, 3, 1, H, W]
first_latent = vae.encode_streaming(first_frame)

# Generate subsequent blocks
all_frames = [vae.decode_streaming(first_latent, is_first=True)]

for block_idx in range(num_blocks):
    # DiT generates next N latent frames
    noise = torch.randn(1, 16, 3, H_lat, W_lat)
    latent_block = dit.generate_block(noise, conditions, kv_cache)

    # Immediately decode to video
    for i in range(latent_block.shape[2]):
        video_chunk = vae.decode_streaming(latent_block[:, :, i:i+1])
        all_frames.append(video_chunk)
        yield video_chunk  # Stream to display/save
```

### D.6 Output Frame Counts

Understanding how many video frames each decode produces:

```
Decode (batch mode):
  T_latent=1 → T_video=1   (first frame only)
  T_latent=2 → T_video=5   (1 + 4)
  T_latent=3 → T_video=9   (1 + 4 + 4)
  T_latent=5 → T_video=17  (1 + 4 + 4 + 4 + 4)

Decode (streaming mode):
  Latent 0 → 1 frame   (first latent = first frame)
  Latent 1 → 4 frames  (second latent = 4 new frames)
  Latent 2 → 4 frames  (third latent = 4 new frames)
  ...
```

This matches the full WAN VAE behavior and enables true streaming video generation.
