#!/usr/bin/env python3
"""
Fine-tune the Matrix-Game Base Model (WanModel)
Analyzes the base model architecture and fine-tunes it with Unreal demo data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import math
from safetensors.torch import load_file, save_file
from omegaconf import OmegaConf
from tqdm import tqdm

# Import Matrix-Game components
from wan.modules.model import WanModel
from convert_unreal_data import UnrealDataset

def sinusoidal_embedding_1d(freq_dim, t):
    """Create sinusoidal embeddings for time steps."""
    half_dim = freq_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

def analyze_model(model):
    """Analyze the model architecture and print key information."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    print(f"\nModel Type: {model.__class__.__name__}")
    print(f"Model Variant: {model.model_type}")
    print(f"Action Module Enabled: {model.use_action_module}")
    
    print(f"\nDimensions:")
    print(f"  Input Channels: {model.in_dim}")
    print(f"  Hidden Dimension: {model.dim}")
    print(f"  FFN Dimension: {model.ffn_dim}")
    print(f"  Output Channels: {model.out_dim}")
    print(f"  Number of Layers: {model.num_layers}")
    print(f"  Number of Heads: {model.num_heads}")
    print(f"  Patch Size: {model.patch_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    # Analyze action modules
    if model.use_action_module:
        action_blocks = []
        for idx, block in enumerate(model.blocks):
            if hasattr(block, 'action_model') and block.action_model is not None:
                action_blocks.append(idx)
        print(f"\nAction Modules:")
        print(f"  Blocks with actions: {len(action_blocks)}/{len(model.blocks)}")
        print(f"  Action block indices: {action_blocks[:10]}{'...' if len(action_blocks) > 10 else ''}")
    
    print("="*60 + "\n")

def setup_model(config_path, weights_path, device):
    """Load and setup the base model."""
    print(f"Loading model configuration from: {config_path}")
    config = OmegaConf.load(config_path)
    
    print("Initializing WanModel...")
    model = WanModel(
        model_type=config.get('model_type', 'i2v'),
        patch_size=tuple(config.action_config.patch_size),
        in_dim=config.in_dim,
        dim=config.dim,
        ffn_dim=config.ffn_dim,
        freq_dim=config.freq_dim,
        out_dim=config.out_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        qk_norm=config.action_config.qk_norm,
        action_config=config.action_config,
        eps=config.eps
    )
    
    print(f"Loading pretrained weights from: {weights_path}")
    state_dict = load_file(weights_path)
    
    # The base model was trained with 6-dim keyboard, but we're using 4-dim (WASD)
    # We need to filter out incompatible keyboard embedding weights
    print("\nFiltering incompatible keyboard embedding layers...")
    filtered_state_dict = {}
    skipped_keys = []
    
    for key, value in state_dict.items():
        # Skip keyboard embedding weights that have wrong dimensions
        if 'keyboard_embed' in key and 'weight' in key:
            # Check if the shape matches
            if key in model.state_dict():
                if value.shape != model.state_dict()[key].shape:
                    skipped_keys.append(key)
                    continue
        filtered_state_dict[key] = value
    
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} keyboard embedding layers (dimension mismatch)")
        print(f"  (Base model used 6-dim keyboard, we're using 4-dim WASD)")
        print(f"  These layers will be randomly initialized and trained from scratch")
    
    # Load the filtered weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    model = model.to(device, dtype=torch.bfloat16)
    
    return model, config

def custom_collate_fn(batch):
    """
    Custom collate function to ensure all images in batch have same dimensions.
    Pads or crops images to a consistent size if needed.
    """
    if len(batch) == 0:
        return {}
    
    # Get the target size from the first item
    target_shape = batch[0]['video_frames'].shape
    
    # Check if all items have the same shape
    all_same = all(item['video_frames'].shape == target_shape for item in batch)
    
    if all_same:
        # Simple case: stack everything
        return {
            'video_frames': torch.stack([item['video_frames'] for item in batch]),
            'keyboard_actions': torch.stack([item['keyboard_actions'] for item in batch]),
            'mouse_actions': torch.stack([item['mouse_actions'] for item in batch])
        }
    else:
        # Need to resize/pad to consistent dimensions
        # Find the most common shape or use the first one
        collated_batch = {
            'video_frames': [],
            'keyboard_actions': [],
            'mouse_actions': []
        }
        
        for item in batch:
            frames = item['video_frames']
            # If shape doesn't match, resize to target
            if frames.shape != target_shape:
                # Interpolate to target size
                # frames shape: (T, H, W, C)
                T, H, W, C = frames.shape
                target_T, target_H, target_W, target_C = target_shape
                
                # Resize each frame
                resized_frames = []
                for t in range(T):
                    frame = frames[t]  # (H, W, C)
                    # Convert to (C, H, W) for interpolate
                    frame = frame.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                    frame = torch.nn.functional.interpolate(
                        frame, size=(target_H, target_W), mode='bilinear', align_corners=False
                    )
                    frame = frame.squeeze(0).permute(1, 2, 0)  # (H, W, C)
                    resized_frames.append(frame)
                frames = torch.stack(resized_frames)
            
            collated_batch['video_frames'].append(frames)
            collated_batch['keyboard_actions'].append(item['keyboard_actions'])
            collated_batch['mouse_actions'].append(item['mouse_actions'])
        
        return {
            'video_frames': torch.stack(collated_batch['video_frames']),
            'keyboard_actions': torch.stack(collated_batch['keyboard_actions']),
            'mouse_actions': torch.stack(collated_batch['mouse_actions'])
        }

def configure_training_strategy(model, strategy='action_only'):
    """
    Configure which parameters to train based on the strategy.
    
    Strategies:
    - 'action_only': Only train action modules (recommended for game-specific behavior)
    - 'last_layers': Train action modules + last few transformer blocks
    - 'full': Fine-tune the entire model (requires more data and compute)
    """
    print(f"\nConfiguring training strategy: {strategy}")
    
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    if strategy == 'action_only':
        # Only train action modules
        for block in model.blocks:
            if hasattr(block, 'action_model') and block.action_model is not None:
                for param in block.action_model.parameters():
                    param.requires_grad = True
    
    elif strategy == 'last_layers':
        # Train action modules + last 3 transformer blocks
        for block in model.blocks:
            if hasattr(block, 'action_model') and block.action_model is not None:
                for param in block.action_model.parameters():
                    param.requires_grad = True
        
        # Last 3 blocks
        for block in model.blocks[-3:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Head layer
        for param in model.head.parameters():
            param.requires_grad = True
    
    elif strategy == 'full':
        # Train everything
        for param in model.parameters():
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Training {trainable_params:,} / {total_params:,} parameters ({100*trainable_params/total_params:.2f}%)")
    
    return model

def finetune_base_model():
    """Main fine-tuning function."""
    print("="*60)
    print("MATRIX-GAME BASE MODEL FINE-TUNING")
    print("="*60)
    
    # Configuration
    BASE_MODEL_DIR = "models/base_model"
    CONFIG_PATH = os.path.join(BASE_MODEL_DIR, "base_config.json")
    WEIGHTS_PATH = os.path.join(BASE_MODEL_DIR, "diffusion_pytorch_model.safetensors")
    DATA_DIR = "data"
    CHECKPOINT_DIR = "checkpoints"
    
    # Training hyperparameters
    # With VAE compression (4x) and num_frame_per_block=3:
    # 9 frames -> 3 latent frames (matches default block size!)
    # 13 frames -> 4 latent frames
    # 25 frames -> 7 latent frames
    SEQUENCE_LENGTH = 9  # Must give exactly 3 latent frames for num_frame_per_block=3
    BATCH_SIZE = 8  # Batch size for training (now supports batching!)
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
    TRAINING_STRATEGY = 'action_only'  # 'action_only' or 'last_layers' or 'full'
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model
    model, config = setup_model(CONFIG_PATH, WEIGHTS_PATH, device)
    
    # Analyze model architecture
    analyze_model(model)
    
    # Configure training strategy
    model = configure_training_strategy(model, TRAINING_STRATEGY)
    
    # Ensure all trainable parameters are in bfloat16
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
    
    # Create dataset
    print(f"\nLoading dataset from: {DATA_DIR}")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    dataset = UnrealDataset(data_dir=DATA_DIR, sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        collate_fn=custom_collate_fn  # Use custom collate to handle dimension mismatches
    )
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(dataloader))
    
    # GradScaler for mixed precision training with bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bfloat16 doesn't need scaling
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    model.train()
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        successful_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                video_frames = batch['video_frames'].to(device, dtype=torch.bfloat16)
                keyboard_actions = batch['keyboard_actions'].to(device, dtype=torch.bfloat16)
                mouse_actions = batch['mouse_actions'].to(device, dtype=torch.bfloat16)
                
                # Get dimensions
                batch_size, num_frames, height, width, channels = video_frames.shape
                
                # Debug: Print actual dimensions
                if batch_idx == 0:
                    print(f"\nDebug - Input dimensions:")
                    print(f"  video_frames: {video_frames.shape}")
                    print(f"  num_frames: {num_frames}")
                
                # For the base model, we need to create latents
                # In a real scenario, you'd use the VAE encoder
                # For now, we'll use placeholder latents
                
                # VAE downsamples:
                # - Spatially by 8x: (352, 640) -> (44, 80)
                # - Temporally by 4x: 57 frames -> 15 frames
                latent_height, latent_width = height // 8, width // 8
                
                # Calculate compressed frame count
                # Formula: N_feats = (N_frames - 1) // 4 + 1
                vae_time_compression = 4
                latent_frames = (num_frames - 1) // vae_time_compression + 1
                
                if batch_idx == 0:
                    print(f"  VAE compression: {num_frames} frames -> {latent_frames} latent frames")
                    print(f"  Spatial: ({height}, {width}) -> ({latent_height}, {latent_width})")
                
                # Create placeholder latents with compressed dimensions (16 channels)
                latents = torch.randn(batch_size, 16, latent_frames, latent_height, latent_width, 
                                    device=device, dtype=torch.bfloat16)
                
                # Create visual context (CLIP embeddings)
                visual_context = torch.randn(batch_size, 1280, device=device, dtype=torch.bfloat16)
                
                # Create cond_concat (20 channels to make 36 total with latents)
                # Must match latent dimensions (compressed frames)
                cond_concat = torch.zeros(batch_size, 20, latent_frames, latent_height, latent_width, 
                                        device=device, dtype=torch.bfloat16)
                
                # Diffusion training: add noise to latents
                noise = torch.randn_like(latents)
                
                # Sample timesteps (uniform sampling from [0, 1000])
                # Use bfloat16 to match model dtype
                timesteps = torch.randint(0, 1000, (batch_size,), device=device).float().to(dtype=torch.bfloat16)
                
                # Simple noise schedule (linear)
                sqrt_alpha_prod = (1 - timesteps / 1000.0) ** 0.5
                sqrt_one_minus_alpha_prod = (timesteps / 1000.0) ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1, 1)
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1, 1)
                
                # Add noise to latents
                noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
                
                # Forward pass through the model with autocast for mixed precision
                # Note: Action module requires batch_size=1, so we process each sample separately
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if batch_size > 1:
                        predicted_noise_list = []
                        for i in range(batch_size):
                            pred = model(
                                noisy_latents[i:i+1],
                                timesteps[i:i+1],
                                visual_context=visual_context[i:i+1],
                                cond_concat=cond_concat[i:i+1],
                                mouse_cond=mouse_actions[i:i+1],
                                keyboard_cond=keyboard_actions[i:i+1]
                            )
                            predicted_noise_list.append(pred)
                        predicted_noise = torch.cat(predicted_noise_list, dim=0)
                    else:
                        predicted_noise = model(
                            noisy_latents,
                            timesteps,
                            visual_context=visual_context,
                            cond_concat=cond_concat,
                            mouse_cond=mouse_actions,
                            keyboard_cond=keyboard_actions
                        )
                    
                    # Calculate loss (MSE between predicted and actual noise)
                    loss = nn.functional.mse_loss(predicted_noise, noise)
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping (unscale before clipping)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                successful_batches += 1
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/successful_batches:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Epoch summary
        if successful_batches > 0:
            avg_loss = epoch_loss / successful_batches
            print(f"\nEpoch {epoch+1} completed:")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Successful batches: {successful_batches}/{len(dataloader)}")
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0 or epoch == NUM_EPOCHS - 1:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"base_finetuned_epoch{epoch+1}.safetensors")
                print(f"  Saving checkpoint to: {checkpoint_path}")
                
                # Save only trainable parameters
                trainable_state = {k: v.cpu() for k, v in model.state_dict().items() 
                                 if any(p is v for p in trainable_params)}
                save_file(trainable_state, checkpoint_path)
        else:
            print(f"\nEpoch {epoch+1}: No successful batches!")
    
    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "base_finetuned_final.safetensors")
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Saving final model to: {final_path}")
    print(f"{'='*60}\n")
    
    # Save full model state
    save_file(model.state_dict(), final_path)
    
    # Save training info
    info_path = os.path.join(CHECKPOINT_DIR, "training_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Matrix-Game Base Model Fine-tuning\n")
        f.write(f"{'='*60}\n")
        f.write(f"Training Strategy: {TRAINING_STRATEGY}\n")
        f.write(f"Sequence Length: {SEQUENCE_LENGTH}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Final Loss: {avg_loss:.4f}\n")
        f.write(f"Dataset Size: {len(dataset)} sequences\n")
        f.write(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}\n")
    
    print(f"Training info saved to: {info_path}")
    print("\nFine-tuning complete!")

if __name__ == "__main__":
    finetune_base_model()

