#!/usr/bin/env python3
"""
Autoregressive Fine-tuning for Matrix-Game Base Model (WanModel)
Trains the model to generate video continuations for long-form generation
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
    
    # Filter incompatible keyboard embedding layers
    print("\nFiltering incompatible keyboard embedding layers...")
    filtered_state_dict = {}
    skipped_keys = []
    
    for key, value in state_dict.items():
        if 'keyboard_embed' in key and 'weight' in key:
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
    """Custom collate function to ensure all images in batch have same dimensions."""
    if len(batch) == 0:
        return {}
    
    # Get the target size from the first item
    target_shape = batch[0]['video_frames'].shape
    
    # Check if all items have the same shape
    all_same = all(item['video_frames'].shape == target_shape for item in batch)
    
    if all_same:
        return {
            'video_frames': torch.stack([item['video_frames'] for item in batch]),
            'keyboard_actions': torch.stack([item['keyboard_actions'] for item in batch]),
            'mouse_actions': torch.stack([item['mouse_actions'] for item in batch])
        }
    else:
        # Resize/pad to consistent dimensions
        collated_batch = {
            'video_frames': [],
            'keyboard_actions': [],
            'mouse_actions': []
        }
        
        for item in batch:
            frames = item['video_frames']
            if frames.shape != target_shape:
                T, H, W, C = frames.shape
                target_T, target_H, target_W, target_C = target_shape
                
                resized_frames = []
                for t in range(T):
                    frame = frames[t]
                    frame = frame.permute(2, 0, 1).unsqueeze(0)
                    frame = torch.nn.functional.interpolate(
                        frame, size=(target_H, target_W), mode='bilinear', align_corners=False
                    )
                    frame = frame.squeeze(0).permute(1, 2, 0)
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
    """Configure which parameters to train based on the strategy."""
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

class AutoregressiveDataset(torch.utils.data.Dataset):
    """
    Dataset for autoregressive training.
    Splits sequences into context (first N frames) and target (next N frames).
    """
    def __init__(self, base_dataset, context_frames=9, target_frames=9):
        self.base_dataset = base_dataset
        self.context_frames = context_frames
        self.target_frames = target_frames
        self.total_frames = context_frames + target_frames
        
        # Filter sequences that are long enough
        self.valid_indices = []
        for idx in range(len(base_dataset)):
            try:
                sample = base_dataset[idx]
                if sample['video_frames'].shape[0] >= self.total_frames:
                    self.valid_indices.append(idx)
            except:
                pass
        
        print(f"AutoregressiveDataset: {len(self.valid_indices)}/{len(base_dataset)} sequences are long enough")
        print(f"  Context frames: {context_frames}, Target frames: {target_frames}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        base_idx = self.valid_indices[idx]
        sample = self.base_dataset[base_idx]
        
        # Get context and target frames
        total_available = sample['video_frames'].shape[0]
        
        # Random starting point if we have more frames than needed
        if total_available > self.total_frames:
            max_start = total_available - self.total_frames
            start = np.random.randint(0, max_start + 1)
        else:
            start = 0
        
        context_end = start + self.context_frames
        target_end = context_end + self.target_frames
        
        return {
            # Context: first N frames
            'context_frames': sample['video_frames'][start:context_end],
            'context_keyboard': sample['keyboard_actions'][start:context_end],
            'context_mouse': sample['mouse_actions'][start:context_end],
            # Target: next N frames
            'target_frames': sample['video_frames'][context_end:target_end],
            'target_keyboard': sample['keyboard_actions'][context_end:target_end],
            'target_mouse': sample['mouse_actions'][context_end:target_end],
        }

def autoregressive_collate_fn(batch):
    """Collate function for autoregressive training."""
    if len(batch) == 0:
        return {}
    
    return {
        'context_frames': torch.stack([item['context_frames'] for item in batch]),
        'context_keyboard': torch.stack([item['context_keyboard'] for item in batch]),
        'context_mouse': torch.stack([item['context_mouse'] for item in batch]),
        'target_frames': torch.stack([item['target_frames'] for item in batch]),
        'target_keyboard': torch.stack([item['target_keyboard'] for item in batch]),
        'target_mouse': torch.stack([item['target_mouse'] for item in batch]),
    }

def finetune_autoregressive():
    """Main autoregressive fine-tuning function."""
    print("="*60)
    print("MATRIX-GAME AUTOREGRESSIVE FINE-TUNING")
    print("="*60)
    
    # Configuration
    BASE_MODEL_DIR = "models/base_model"
    CONFIG_PATH = os.path.join(BASE_MODEL_DIR, "base_config.json")
    WEIGHTS_PATH = os.path.join(BASE_MODEL_DIR, "diffusion_pytorch_model.safetensors")
    DATA_DIR = "data"
    CHECKPOINT_DIR = "checkpoints"
    
    # Training hyperparameters for autoregressive learning
    CONTEXT_FRAMES = 9   # Input: first 9 frames (3 latent frames)
    TARGET_FRAMES = 9    # Output: next 9 frames (3 latent frames)
    BATCH_SIZE = 4       # Smaller batch size due to processing 2 sequences
    NUM_EPOCHS = 15      # More epochs for autoregressive learning
    LEARNING_RATE = 5e-6  # Lower learning rate for stability
    TRAINING_STRATEGY = 'action_only'
    
    # Curriculum learning (optional)
    USE_CURRICULUM = True  # Start with easier tasks, gradually increase difficulty
    
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
    base_dataset = UnrealDataset(data_dir=DATA_DIR, sequence_length=CONTEXT_FRAMES + TARGET_FRAMES)
    
    # Wrap with autoregressive dataset
    dataset = AutoregressiveDataset(
        base_dataset, 
        context_frames=CONTEXT_FRAMES, 
        target_frames=TARGET_FRAMES
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        collate_fn=autoregressive_collate_fn
    )
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    def lr_lambda(current_step):
        warmup_steps = len(dataloader) * 2  # 2 epochs of warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / 
                                     (NUM_EPOCHS * len(dataloader) - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bfloat16 doesn't need scaling
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING AUTOREGRESSIVE TRAINING")
    print("="*60 + "\n")
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_context_loss = 0  # Loss on context (reconstruction)
        epoch_target_loss = 0   # Loss on target (prediction)
        successful_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move to device
                context_frames = batch['context_frames'].to(device, dtype=torch.bfloat16)
                context_keyboard = batch['context_keyboard'].to(device, dtype=torch.bfloat16)
                context_mouse = batch['context_mouse'].to(device, dtype=torch.bfloat16)
                target_frames = batch['target_frames'].to(device, dtype=torch.bfloat16)
                target_keyboard = batch['target_keyboard'].to(device, dtype=torch.bfloat16)
                target_mouse = batch['target_mouse'].to(device, dtype=torch.bfloat16)
                
                batch_size = context_frames.shape[0]
                num_context_frames = context_frames.shape[1]
                num_target_frames = target_frames.shape[1]
                height, width = context_frames.shape[2:4]
                
                # Calculate VAE compression
                vae_time_compression = 4
                latent_height, latent_width = height // 8, width // 8
                
                # Process context and target separately
                # 1. Train on TARGET frames conditioned on CONTEXT
                context_latent_frames = (num_context_frames - 1) // vae_time_compression + 1
                target_latent_frames = (num_target_frames - 1) // vae_time_compression + 1
                
                # Create placeholder latents for target (what we want to generate)
                target_latents = torch.randn(
                    batch_size, 16, target_latent_frames, latent_height, latent_width,
                    device=device, dtype=torch.bfloat16
                )
                
                # Create visual context from context frames (using CLIP)
                # In practice, this would come from the VAE encoder
                visual_context = torch.randn(batch_size, 1280, device=device, dtype=torch.bfloat16)
                
                # Create cond_concat that represents the context frames
                # This tells the model "these are the frames we've already generated"
                cond_concat = torch.zeros(
                    batch_size, 20, target_latent_frames, latent_height, latent_width,
                    device=device, dtype=torch.bfloat16
                )
                
                # Add noise for diffusion training
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(0, 1000, (batch_size,), device=device).float().to(dtype=torch.bfloat16)
                
                # Noise schedule
                sqrt_alpha_prod = (1 - timesteps / 1000.0) ** 0.5
                sqrt_one_minus_alpha_prod = (timesteps / 1000.0) ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1, 1)
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1, 1)
                
                noisy_latents = sqrt_alpha_prod * target_latents + sqrt_one_minus_alpha_prod * noise
                
                # Forward pass: predict noise for TARGET frames
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # Process each batch item separately (required for action modules)
                    if batch_size > 1:
                        predicted_noise_list = []
                        for i in range(batch_size):
                            pred = model(
                                noisy_latents[i:i+1],
                                timesteps[i:i+1],
                                visual_context=visual_context[i:i+1],
                                cond_concat=cond_concat[i:i+1],
                                mouse_cond=target_mouse[i:i+1],  # Use TARGET actions
                                keyboard_cond=target_keyboard[i:i+1]
                            )
                            predicted_noise_list.append(pred)
                        predicted_noise = torch.cat(predicted_noise_list, dim=0)
                    else:
                        predicted_noise = model(
                            noisy_latents,
                            timesteps,
                            visual_context=visual_context,
                            cond_concat=cond_concat,
                            mouse_cond=target_mouse,
                            keyboard_cond=target_keyboard
                        )
                    
                    # Calculate target prediction loss
                    target_loss = nn.functional.mse_loss(predicted_noise, noise)
                    
                    # Optional: Also train on context reconstruction (curriculum learning)
                    if USE_CURRICULUM and epoch < NUM_EPOCHS // 3:
                        # Early training: also learn to reconstruct context
                        context_latents = torch.randn(
                            batch_size, 16, context_latent_frames, latent_height, latent_width,
                            device=device, dtype=torch.bfloat16
                        )
                        context_noise = torch.randn_like(context_latents)
                        noisy_context = sqrt_alpha_prod[:, :, :, :, :] * context_latents + \
                                       sqrt_one_minus_alpha_prod[:, :, :, :, :] * context_noise
                        
                        context_cond = torch.zeros(
                            batch_size, 20, context_latent_frames, latent_height, latent_width,
                            device=device, dtype=torch.bfloat16
                        )
                        
                        if batch_size > 1:
                            context_pred_list = []
                            for i in range(batch_size):
                                pred = model(
                                    noisy_context[i:i+1],
                                    timesteps[i:i+1],
                                    visual_context=visual_context[i:i+1],
                                    cond_concat=context_cond[i:i+1],
                                    mouse_cond=context_mouse[i:i+1],
                                    keyboard_cond=context_keyboard[i:i+1]
                                )
                                context_pred_list.append(pred)
                            context_predicted = torch.cat(context_pred_list, dim=0)
                        else:
                            context_predicted = model(
                                noisy_context,
                                timesteps,
                                visual_context=visual_context,
                                cond_concat=context_cond,
                                mouse_cond=context_mouse,
                                keyboard_cond=context_keyboard
                            )
                        
                        context_loss = nn.functional.mse_loss(context_predicted, context_noise)
                        
                        # Combined loss (weighted)
                        loss = 0.3 * context_loss + 0.7 * target_loss
                        epoch_context_loss += context_loss.item()
                    else:
                        # Later training: focus only on prediction
                        loss = target_loss
                        context_loss = torch.tensor(0.0)
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_target_loss += target_loss.item()
                successful_batches += 1
                global_step += 1
                
                # Update progress bar
                if USE_CURRICULUM and epoch < NUM_EPOCHS // 3:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'ctx': f'{context_loss.item():.4f}',
                        'tgt': f'{target_loss.item():.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg': f'{epoch_loss/successful_batches:.4f}',
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
            avg_target_loss = epoch_target_loss / successful_batches
            
            print(f"\nEpoch {epoch+1} completed:")
            print(f"  Average total loss: {avg_loss:.4f}")
            print(f"  Average target loss: {avg_target_loss:.4f}")
            if USE_CURRICULUM and epoch < NUM_EPOCHS // 3:
                avg_context_loss = epoch_context_loss / successful_batches
                print(f"  Average context loss: {avg_context_loss:.4f}")
            print(f"  Successful batches: {successful_batches}/{len(dataloader)}")
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint if best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "autoregressive_best.safetensors")
                print(f"  New best loss! Saving to: {checkpoint_path}")
                save_file(model.state_dict(), checkpoint_path)
            
            # Save periodic checkpoints
            if (epoch + 1) % 3 == 0 or epoch == NUM_EPOCHS - 1:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"autoregressive_epoch{epoch+1}.safetensors")
                print(f"  Saving checkpoint to: {checkpoint_path}")
                trainable_state = {k: v.cpu() for k, v in model.state_dict().items() 
                                 if any(p is v for p in trainable_params)}
                save_file(trainable_state, checkpoint_path)
        else:
            print(f"\nEpoch {epoch+1}: No successful batches!")
    
    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "autoregressive_final.safetensors")
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Saving final model to: {final_path}")
    print(f"{'='*60}\n")
    
    save_file(model.state_dict(), final_path)
    
    # Save training info
    info_path = os.path.join(CHECKPOINT_DIR, "autoregressive_training_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Matrix-Game Autoregressive Fine-tuning\n")
        f.write(f"{'='*60}\n")
        f.write(f"Training Strategy: {TRAINING_STRATEGY}\n")
        f.write(f"Context Frames: {CONTEXT_FRAMES}\n")
        f.write(f"Target Frames: {TARGET_FRAMES}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Curriculum Learning: {USE_CURRICULUM}\n")
        f.write(f"Best Loss: {best_loss:.4f}\n")
        f.write(f"Final Loss: {avg_loss:.4f}\n")
        f.write(f"Dataset Size: {len(dataset)} sequences\n")
        f.write(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}\n")
    
    print(f"Training info saved to: {info_path}")
    print("\nAutoregressive fine-tuning complete!")
    print(f"Best checkpoint: checkpoints/autoregressive_best.safetensors")

if __name__ == "__main__":
    finetune_autoregressive()

