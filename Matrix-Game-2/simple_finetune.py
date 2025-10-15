#!/usr/bin/env python3
"""
Simple Matrix-Game 2.0 Fine-tuning Script
Minimal code for quick experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from safetensors.torch import load_file, save_file
from omegaconf import OmegaConf

# Import Matrix-Game components
from wan.modules.causal_model import CausalWanModel
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from utils.wan_wrapper import WanDiffusionWrapper

class SimpleDataset(Dataset):
    """Simple dataset for quick experiments."""
    def __init__(self, num_samples=100, num_frames=50):
        self.num_samples = num_samples
        self.num_frames = num_frames
        
        # Generate dummy data for testing
        self.video_frames = []
        self.keyboard_actions = []
        self.mouse_actions = []
        
        for _ in range(num_samples):
            # Random video frames (T, H, W, C) - no batch dimension in dataset
            frames = torch.randn(num_frames, 352, 640, 3)
            self.video_frames.append(frames)
            
            # Random keyboard actions (T, 4) for universal mode
            keyboard = torch.randint(0, 2, (num_frames, 4)).float()
            self.keyboard_actions.append(keyboard)
            
            # Random mouse actions (T, 2)
            mouse = torch.randn(num_frames, 2) * 0.1
            self.mouse_actions.append(mouse)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'video_frames': self.video_frames[idx],
            'keyboard_actions': self.keyboard_actions[idx],
            'mouse_actions': self.mouse_actions[idx]
        }

def simple_finetune():
    """Simple fine-tuning function."""
    print("Starting simple fine-tuning experiment...")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model configuration
    config_path = "configs/distilled_model/universal/config.json"
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        # Default config
        config = OmegaConf.create({
            'dim': 1536,
            'ffn_dim': 8960,
            'num_heads': 12,
            'num_layers': 30,
            'action_config': {
                'enable_keyboard': True,
                'enable_mouse': True,
                'keyboard_dim_in': 4,
                'mouse_dim_in': 2,
                'hidden_size': 128,
                'img_hidden_size': 1536,
                'keyboard_hidden_dim': 1024,
                'mouse_hidden_dim': 1024,
                'blocks': list(range(15)),
                'heads_num': 16,
                'vae_time_compression_ratio': 4,
                'windows_size': 3,
                'patch_size': [1, 2, 2],
                'qk_norm': True,
                'qkv_bias': False,
                'rope_theta': 256,
                'mouse_qk_dim_list': [8, 28, 28],
                'rope_dim_list': [8, 28, 28]
            }
        })
    
    # Initialize model
    print("Loading model...")
    model = CausalWanModel(
        model_type='i2v',
        patch_size=tuple(config.action_config.patch_size),
        dim=config.dim,
        ffn_dim=config.ffn_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        action_config=config.action_config
    )
    
    # Load pretrained weights if available
    pretrained_path = "Matrix-Game-2.0/universal_distilled_model.safetensors"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = load_file(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No pretrained weights found, starting from scratch")
    
    model = model.to(device, dtype=torch.bfloat16)
    
    # Freeze most layers, only train action modules and last few layers
    print("Freezing backbone layers...")
    for name, param in model.named_parameters():
        if 'action_module' not in name and 'blocks.29' not in name and 'head' not in name:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = SimpleDataset(num_samples=50, num_frames=30)  # Small dataset for quick test
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(5):  # Just 5 epochs for quick test
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            video_frames = batch['video_frames'].to(device, dtype=torch.bfloat16)
            keyboard_actions = batch['keyboard_actions'].to(device, dtype=torch.bfloat16)
            mouse_actions = batch['mouse_actions'].to(device, dtype=torch.bfloat16)
            
            # Simple forward pass - just predict next frame
            batch_size, num_frames, height, width, channels = video_frames.shape
            
            # Reshape for model input
            video_tensor = video_frames.permute(0, 4, 1, 2, 3)  # B, C, T, H, W
            video_tensor = video_tensor.reshape(batch_size * channels, num_frames, height, width)
            
            # Create dummy latents (in real training, you'd use VAE)
            # The model expects 36 input channels total, so we need to adjust our latents
            latents = torch.randn(batch_size, 16, num_frames, 44, 80, device=device, dtype=torch.bfloat16)
            
            # Prepare conditioning
            visual_context = torch.randn(batch_size, 1280, device=device, dtype=torch.bfloat16)
            # cond_concat should be concatenated with latents to make 36 total channels
            # So we need 36 - 16 = 20 channels for cond_concat
            cond_concat = torch.zeros(batch_size, 20, num_frames, 44, 80, device=device, dtype=torch.bfloat16)
            
            # Add noise for diffusion training
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
            
            # Simple noise schedule
            sqrt_alpha_prod = (1 - timesteps / 1000.0) ** 0.5
            sqrt_one_minus_alpha_prod = (timesteps / 1000.0) ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1, 1).to(dtype=torch.bfloat16)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1, 1).to(dtype=torch.bfloat16)
            
            noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
            
            # Forward pass
            try:
                predicted_noise = model(
                    noisy_latents, 
                    timesteps, 
                    visual_context=visual_context,
                    cond_concat=cond_concat,
                    mouse_cond=mouse_actions,
                    keyboard_cond=keyboard_actions
                )
                loss = nn.functional.mse_loss(predicted_noise, noise)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save as safetensors
    model_state = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    save_file(model_state, "checkpoints/simple_finetuned.safetensors")
    
    print("Fine-tuning completed!")
    print("Saved model to: checkpoints/simple_finetuned.safetensors")

if __name__ == "__main__":
    simple_finetune()
