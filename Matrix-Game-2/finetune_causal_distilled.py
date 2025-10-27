"""
Finetune the causal distilled model (base_distill.safetensors) on your dataset.
This model will work with inference.py after finetuning.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from safetensors.torch import save_file, load_file

from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from omegaconf import OmegaConf
from convert_unreal_data import UnrealDataset
from einops import rearrange


def normalize_frames(frames):
    """Convert frames from [0, 1] to [-1, 1] for model input."""
    return frames * 2.0 - 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune causal distilled model")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing gameplay data (frames + input.csv)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--pretrained_checkpoint", type=str, 
                        default="models/base_distilled_model/base_distill.safetensors",
                        help="Path to pretrained distilled model")
    parser.add_argument("--config_path", type=str,
                        default="configs/inference_yaml/inference_universal.yaml",
                        help="Path to model config")
    parser.add_argument("--model_variant", type=str, choices=["universal", "gta_drive"], default="gta_drive",
                        help="Which distilled model variant to use for training")
    parser.add_argument("--keyboard_only", action="store_true",
                        help="Condition only on keyboard; omit mouse inputs during training")
    parser.add_argument("--sequence_length", type=int, default=9,
                        help="Number of frames per training sample (9 recommended)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--save_every", type=int, default=2,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("FINETUNING CAUSAL DISTILLED MODEL")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
    print(f"Model variant: {args.model_variant}")
    print(f"Keyboard only: {args.keyboard_only}")
    print(f"Data directory: {args.data_dir}")
    print(f"Frames per sample: {args.sequence_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load config
    print("\nLoading model config...")
    config = OmegaConf.load(args.config_path)
    # Switch model config directory based on selected variant
    if args.model_variant == "gta_drive":
        # Point to GTA distilled model config directory
        config.model_kwargs.model_config = "configs/distilled_model/gta_drive"
        # If not overridden, default to GTA distilled pretrained weights
        if args.pretrained_checkpoint == "models/base_distilled_model/base_distill.safetensors":
            args.pretrained_checkpoint = "models/gta_distilled_model/gta_keyboard2dim.safetensors"
    else:
        config.model_kwargs.model_config = "configs/distilled_model/universal"
    
    # Initialize model
    print("\nInitializing causal model...")
    model = WanDiffusionWrapper(
        **getattr(config, "model_kwargs", {}), 
        is_causal=True
    )
    
    # Load pretrained weights
    print(f"\nLoading pretrained checkpoint from {args.pretrained_checkpoint}...")
    state_dict = load_file(args.pretrained_checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights loaded successfully!")
    
    # Set num_frame_per_block to match our sequence length
    # Model expects to process num_frame_per_block frames at a time
    model.model.num_frame_per_block = config.get("num_frame_per_block", 3)
    print(f"Set model.model.num_frame_per_block = {model.model.num_frame_per_block}")
    
    model = model.to(device, dtype=torch.bfloat16)
    model.train()
    
    # Initialize VAE for encoding frames
    print("\nLoading VAE encoder...")
    vae = get_wanx_vae_wrapper("models/", torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    # Keep VAE on CPU to save GPU memory, move to GPU only when encoding
    vae = vae.to('cpu', torch.float16)
    
    # Initialize Flow Match scheduler
    print("\nInitializing Flow Match scheduler...")
    diffusion_scheduler = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True
    )
    diffusion_scheduler.set_timesteps(1000, training=True)
    
    # Initialize dataset and dataloader
    print("\nLoading dataset...")
    dataset = UnrealDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        fps=25
    )
    
    if len(dataset) == 0:
        print("ERROR: No sequences found in dataset!")
        print(f"Please check that {args.data_dir} contains:")
        print("  - frame_*.png files")
        print("  - input.csv file")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(dataloader),
        eta_min=args.learning_rate * 0.1
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                # UnrealDataset returns: video_frames, keyboard_actions, mouse_actions
                frames = batch['video_frames'].to(device)  # [B, T, H, W, C]
                keyboard_per_frame = batch['keyboard_actions'].to(device, dtype=torch.bfloat16)  # [B, T, 4]
                # Optionally drop mouse entirely for keyboard-only training
                mouse_per_frame = None if args.keyboard_only else batch['mouse_actions'].to(device, dtype=torch.bfloat16)

                # If training with GTA variant, reduce keyboard to 2D [W, S]
                if args.model_variant == "gta_drive":
                    # keyboard_per_frame layout is [W, A, S, D]; select W and S
                    keyboard_per_frame = keyboard_per_frame[..., [0, 2]]  # [B, T, 2]
                
                # Normalize frames to [-1, 1]
                frames = normalize_frames(frames)
                
                # Encode frames to latents using VAE
                with torch.no_grad():
                    # Move VAE to GPU temporarily
                    vae = vae.to(device, torch.float16)
                    
                    # Rearrange for VAE: [B, C, T, H, W]
                    frames_vae = frames.permute(0, 4, 1, 2, 3).to(dtype=torch.float16)
                    
                    # Encode with tiling
                    tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
                    latents = vae.encode(frames_vae, device=device, **tiler_kwargs)
                    latents = latents.to(device=device, dtype=torch.bfloat16)
                    
                    # Get visual context from CLIP
                    visual_context = vae.clip.encode_video(frames_vae).to(
                        device=device, dtype=torch.bfloat16
                    )
                    
                    # Move VAE back to CPU to free GPU memory
                    vae = vae.to('cpu', torch.float16)
                    torch.cuda.empty_cache()
                
                # Now match actions to latent temporal resolution
                # VAE compresses time by 4x: 9 video frames -> 3 latent frames (9/4 + 1)
                # Actions need to match: 1 action per video frame -> 4 actions per latent frame
                # So we upsample video-rate actions to match expected input format
                num_latent_frames = latents.shape[2]
                # Create action sequence matching the format: 1 + 4*(num_latent_frames - 1)
                num_action_steps = 1 + 4 * (num_latent_frames - 1)
                
                # Interpolate actions to match the expected action sequence length
                # Simple approach: repeat each action 4 times, then trim to match
                keyboard_expanded = keyboard_per_frame.repeat_interleave(4, dim=1)
                keyboard = keyboard_expanded[:, :num_action_steps]
                if mouse_per_frame is not None:
                    mouse_expanded = mouse_per_frame.repeat_interleave(4, dim=1)
                    mouse = mouse_expanded[:, :num_action_steps]
                else:
                    mouse = None
                
                # Prepare conditioning
                # For causal model, we use first frame as condition
                mask_cond = torch.ones_like(latents[:, :4])  # [B, 4, T, H, W]
                mask_cond[:, :, 1:] = 0  # Only first frame is real
                
                img_cond = latents.clone()
                cond_concat = torch.cat([mask_cond, img_cond], dim=1)  # [B, 8, T, H, W]
                
                # Sample random timesteps
                batch_size = latents.shape[0]
                num_latent_frames = latents.shape[2]
                timestep_indices = torch.randint(0, 1000, (batch_size,))  # CPU
                timesteps_base = diffusion_scheduler.timesteps[timestep_indices].to(
                    device=device, dtype=torch.bfloat16
                )
                # Expand timesteps to [B, F] format for the model
                timesteps = timesteps_base.unsqueeze(1).expand(batch_size, num_latent_frames)
                
                # Flatten timesteps for scheduler: [B, F] -> [B*F]
                timesteps_expanded = timesteps.flatten()
                
                # Reshape latents from [B, C, T, H, W] to [B*T, C, H, W] for scheduler
                latents_flat = rearrange(latents, 'b c t h w -> (b t) c h w')
                
                # Add noise to latents
                noise = torch.randn_like(latents_flat)
                noisy_latents_flat = diffusion_scheduler.add_noise(
                    latents_flat,
                    noise,
                    timesteps_expanded
                )
                
                # Reshape back to [B, C, T, H, W]
                noisy_latents = rearrange(noisy_latents_flat, '(b t) c h w -> b c t h w', b=batch_size, t=num_latent_frames)
                
                # Forward pass through model
                # Create conditional_dict for the model
                conditional_dict = {
                    "cond_concat": cond_concat,
                    "visual_context": visual_context,
                    "keyboard_cond": keyboard
                }
                if mouse is not None:
                    conditional_dict["mouse_cond"] = mouse
                
                predicted_velocity = model(
                    noisy_latents,
                    conditional_dict,
                    timesteps
                )
                
                # Handle tuple output (model may return (output, logits))
                if isinstance(predicted_velocity, tuple):
                    predicted_velocity = predicted_velocity[0]
                
                # Compute target velocity (use flattened version for scheduler)
                target_velocity_flat = diffusion_scheduler.training_target(
                    latents_flat,
                    noise,
                    timesteps_expanded
                )
                # Reshape back to [B, C, T, H, W]
                target_velocity = rearrange(target_velocity_flat, '(b t) c h w -> b c t h w', b=batch_size, t=num_latent_frames)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(
                    predicted_velocity.float(),
                    target_velocity.float(),
                    reduction='mean'
                )
                
                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights every N steps
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # Track loss
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Aggressive memory cleanup
                del noisy_latents, noisy_latents_flat, predicted_velocity, target_velocity, target_velocity_flat, loss
                del latents, latents_flat, visual_context, cond_concat, noise
                del frames, frames_vae, keyboard, conditional_dict
                if mouse is not None:
                    del mouse
                del keyboard_per_frame
                if mouse_per_frame is not None:
                    del mouse_per_frame
                del keyboard_expanded
                if 'mouse_expanded' in locals():
                    del mouse_expanded
                
                # Clear cache after every batch to prevent OOM
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                if batch_idx == 0:  # Only show detailed debug for first batch
                    import traceback
                    traceback.print_exc()
                    print("\nDEBUG INFO:")
                    if 'frames' in locals():
                        print(f"frames shape: {frames.shape}")
                    if 'latents' in locals():
                        print(f"latents shape: {latents.shape}")
                    if 'cond_concat' in locals():
                        print(f"cond_concat shape: {cond_concat.shape}")
                    if 'keyboard' in locals():
                        print(f"keyboard shape: {keyboard.shape}")
                    if 'mouse' in locals():
                        print(f"mouse shape: {mouse.shape}")
                    if 'noisy_latents' in locals():
                        print(f"noisy_latents shape: {noisy_latents.shape}")
                
                # Clean up any allocated tensors to prevent memory leaks on error
                torch.cuda.empty_cache()
                continue
        
        # Calculate average epoch loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"causal_distilled_epoch{epoch+1}.safetensors"
            )
            print(f"Saving checkpoint to {checkpoint_path}...")
            save_file(model.state_dict(), checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, "causal_distilled_best.safetensors")
            print(f"New best model! Saving to {best_path}...")
            save_file(model.state_dict(), best_path)
        
        torch.cuda.empty_cache()
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "causal_distilled_final.safetensors")
    print(f"\nTraining complete! Saving final model to {final_path}...")
    save_file(model.state_dict(), final_path)
    
    # Save training info
    info_path = os.path.join(args.checkpoint_dir, "causal_distilled_training_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Training Configuration\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Pretrained checkpoint: {args.pretrained_checkpoint}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Number of epochs: {args.num_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
        f.write(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Final loss: {avg_loss:.4f}\n")
        f.write(f"Best loss: {best_loss:.4f}\n")
    
    print(f"\nTraining info saved to {info_path}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nYou can now run inference with:")
    print(f"python inference.py \\")
    recommend_config = "configs/inference_yaml/inference_gta_drive.yaml" if args.model_variant == "gta_drive" else args.config_path
    print(f"    --config_path {recommend_config} \\")
    print(f"    --checkpoint_path {best_path} \\")
    print(f"    --img_path demo_images/universal/0000.png ")
    print(f"    --output_folder outputs \\")
    print(f"    --num_output_frames 150 \\")
    print(f"    --pretrained_model_path models/")


if __name__ == "__main__":
    main()

