"""
Finetune the causal distilled model with SELF-FORCING.
This addresses error accumulation by training the model to condition on its own predictions
rather than ground truth frames.
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


def denormalize_frames(frames):
    """Convert frames from [-1, 1] back to [0, 1]."""
    return (frames + 1.0) / 2.0


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune causal distilled model with self-forcing")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing gameplay data (frames + input.csv)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_self_forcing",
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
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate (lower than standard training due to self-forcing)")
    parser.add_argument("--save_every", type=int, default=2,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    
    # Self-forcing specific parameters
    parser.add_argument("--self_forcing_mode", type=str, 
                        choices=["scheduled", "full", "curriculum"], 
                        default="scheduled",
                        help="Self-forcing strategy: scheduled (mix GT and predictions), "
                             "full (always use predictions), curriculum (gradually shift)")
    parser.add_argument("--self_forcing_prob_start", type=float, default=0.0,
                        help="Initial probability of using model predictions (for scheduled/curriculum)")
    parser.add_argument("--self_forcing_prob_end", type=float, default=0.9,
                        help="Final probability of using model predictions (for curriculum)")
    parser.add_argument("--num_conditioning_frames", type=int, default=1,
                        help="Number of initial frames to condition on (ground truth)")
    parser.add_argument("--inference_steps", type=int, default=1,
                        help="Number of denoising steps for self-forcing generation (1 for distilled model)")
    
    return parser.parse_args()


@torch.no_grad()
def generate_frame_chunk(
    model, 
    vae,
    latent_cond,
    visual_context,
    keyboard_actions,
    mouse_actions,
    diffusion_scheduler,
    device,
    num_frames_to_generate=3,
    inference_steps=1
):
    """
    Generate a chunk of frames autoregressively using the model.
    This is used during self-forcing to get model predictions.
    
    Args:
        model: The diffusion model
        vae: VAE for decoding
        latent_cond: Conditioning latent (first frame) [B, C, 1, H, W]
        visual_context: CLIP context [B, D]
        keyboard_actions: Keyboard actions for the chunk
        mouse_actions: Mouse actions for the chunk (can be None)
        diffusion_scheduler: Flow match scheduler
        device: Device to run on
        num_frames_to_generate: Number of latent frames to generate (e.g., 3)
        inference_steps: Number of denoising steps (1 for distilled)
    
    Returns:
        generated_latents: [B, C, T, H, W]
    """
    batch_size = latent_cond.shape[0]
    c, h, w = latent_cond.shape[1], latent_cond.shape[3], latent_cond.shape[4]
    
    # Initialize latents with noise
    generated_latents = torch.randn(
        batch_size, c, num_frames_to_generate, h, w,
        device=device, dtype=torch.bfloat16
    )
    
    # Prepare conditioning
    mask_cond = torch.ones_like(latent_cond)  # [B, 4, 1, H, W]
    
    # Expand conditioning to match generated sequence length
    img_cond = torch.cat([latent_cond, generated_latents[:, :, :]], dim=2)
    mask_cond_full = torch.zeros_like(img_cond)
    mask_cond_full[:, :, :1] = 1  # Only first frame is conditioning
    
    cond_concat = torch.cat([mask_cond_full, img_cond], dim=1)  # [B, 8, T, H, W]
    
    # Set up inference timesteps
    diffusion_scheduler.set_timesteps(inference_steps, training=False)
    timesteps = diffusion_scheduler.timesteps.to(device)
    
    # Denoising loop
    for t_idx, timestep in enumerate(timesteps):
        # Prepare full sequence for model input (conditioning + generated)
        full_sequence = torch.cat([latent_cond, generated_latents], dim=2)
        
        # Prepare timestep tensor for FULL sequence
        timestep_tensor = torch.full(
            (batch_size, full_sequence.shape[2]), 
            timestep, 
            device=device, 
            dtype=torch.bfloat16
        )
        
        # Prepare conditional dict
        conditional_dict = {
            "cond_concat": cond_concat,
            "visual_context": visual_context,
            "keyboard_cond": keyboard_actions
        }
        if mouse_actions is not None:
            conditional_dict["mouse_cond"] = mouse_actions
        
        # Predict velocity for the FULL sequence
        predicted_velocity = model(
            full_sequence,
            conditional_dict,
            timestep_tensor
        )
        
        if isinstance(predicted_velocity, tuple):
            predicted_velocity = predicted_velocity[0]
        
        # Extract predicted velocity for generated frames only (skip conditioning frame)
        predicted_velocity_generated = predicted_velocity[:, :, 1:, :, :]
        
        # Update latents using scheduler
        # For single-step distilled model, this is essentially: x = x + predicted_velocity
        generated_latents = diffusion_scheduler.step(
            predicted_velocity_generated.reshape(batch_size * num_frames_to_generate, c, h, w),
            timestep,
            generated_latents.reshape(batch_size * num_frames_to_generate, c, h, w)
        ).reshape(batch_size, c, num_frames_to_generate, h, w)
        
        # Update cond_concat with new latents for next iteration
        img_cond = torch.cat([latent_cond, generated_latents], dim=2)
        cond_concat = torch.cat([mask_cond_full, img_cond], dim=1)
    
    return generated_latents


def compute_self_forcing_probability(epoch, total_epochs, mode, prob_start, prob_end):
    """
    Compute the probability of using self-forcing based on training progress.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        mode: 'scheduled', 'full', or 'curriculum'
        prob_start: Starting probability
        prob_end: Ending probability
    
    Returns:
        Probability of using model predictions instead of ground truth
    """
    if mode == "full":
        return 1.0
    elif mode == "scheduled":
        return prob_end  # Fixed probability throughout training
    elif mode == "curriculum":
        # Linear schedule from prob_start to prob_end
        progress = epoch / max(total_epochs - 1, 1)
        return prob_start + (prob_end - prob_start) * progress
    else:
        return prob_start


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("FINETUNING CAUSAL DISTILLED MODEL WITH SELF-FORCING")
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
    print(f"\n--- Self-Forcing Configuration ---")
    print(f"Mode: {args.self_forcing_mode}")
    print(f"Probability range: {args.self_forcing_prob_start} -> {args.self_forcing_prob_end}")
    print(f"Conditioning frames: {args.num_conditioning_frames}")
    print(f"Inference steps: {args.inference_steps}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load config
    print("\nLoading model config...")
    config = OmegaConf.load(args.config_path)
    if args.model_variant == "gta_drive":
        config.model_kwargs.model_config = "configs/distilled_model/gta_drive"
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
    
    model.model.num_frame_per_block = config.get("num_frame_per_block", 3)
    print(f"Set model.model.num_frame_per_block = {model.model.num_frame_per_block}")
    
    model = model.to(device, dtype=torch.bfloat16)
    
    # Initialize VAE for encoding/decoding frames
    print("\nLoading VAE encoder/decoder...")
    vae = get_wanx_vae_wrapper("models/", torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to('cpu', torch.float16)
    
    # Initialize Flow Match scheduler
    print("\nInitializing Flow Match scheduler...")
    diffusion_scheduler = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True
    )
    
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
    print("STARTING SELF-FORCING TRAINING")
    print("=" * 60)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Compute self-forcing probability for this epoch
        self_forcing_prob = compute_self_forcing_probability(
            epoch, args.num_epochs, args.self_forcing_mode,
            args.self_forcing_prob_start, args.self_forcing_prob_end
        )
        
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        print(f"Self-forcing probability: {self_forcing_prob:.2f}")
        
        epoch_loss = 0.0
        epoch_gt_samples = 0
        epoch_sf_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                frames = batch['video_frames'].to(device)  # [B, T, H, W, C]
                keyboard_per_frame = batch['keyboard_actions'].to(device, dtype=torch.bfloat16)
                mouse_per_frame = None if args.keyboard_only else batch['mouse_actions'].to(device, dtype=torch.bfloat16)
                
                # Handle GTA variant keyboard reduction
                if args.model_variant == "gta_drive":
                    keyboard_per_frame = keyboard_per_frame[..., [0, 2]]  # [B, T, 2]
                
                # Normalize frames
                frames = normalize_frames(frames)
                
                # Encode ALL frames to latents using VAE
                with torch.no_grad():
                    vae = vae.to(device, torch.float16)
                    
                    frames_vae = frames.permute(0, 4, 1, 2, 3).to(dtype=torch.float16)
                    
                    tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
                    latents_gt = vae.encode(frames_vae, device=device, **tiler_kwargs)
                    latents_gt = latents_gt.to(device=device, dtype=torch.bfloat16)
                    
                    visual_context = vae.clip.encode_video(frames_vae).to(
                        device=device, dtype=torch.bfloat16
                    )
                    
                    vae = vae.to('cpu', torch.float16)
                    torch.cuda.empty_cache()
                
                # Process actions to match latent temporal resolution
                num_latent_frames = latents_gt.shape[2]
                num_action_steps = 1 + 4 * (num_latent_frames - 1)
                
                keyboard_expanded = keyboard_per_frame.repeat_interleave(4, dim=1)
                keyboard = keyboard_expanded[:, :num_action_steps]
                if mouse_per_frame is not None:
                    mouse_expanded = mouse_per_frame.repeat_interleave(4, dim=1)
                    mouse = mouse_expanded[:, :num_action_steps]
                else:
                    mouse = None
                
                # --- SELF-FORCING LOGIC ---
                # Decide whether to use ground truth or model predictions
                use_self_forcing = (np.random.rand() < self_forcing_prob) and (epoch > 0)
                
                if use_self_forcing:
                    # Generate frames using the model (in eval mode to avoid affecting training)
                    model.eval()
                    with torch.no_grad():
                        # Use first frame(s) as conditioning
                        latent_cond = latents_gt[:, :, :args.num_conditioning_frames]
                        
                        # Generate remaining frames
                        num_frames_to_generate = num_latent_frames - args.num_conditioning_frames
                        
                        if num_frames_to_generate > 0:
                            # Split actions for conditioning and generation
                            num_cond_actions = 1 + 4 * (args.num_conditioning_frames - 1)
                            keyboard_gen = keyboard[:, num_cond_actions:]
                            mouse_gen = mouse[:, num_cond_actions:] if mouse is not None else None
                            
                            # Generate frames autoregressively
                            generated_latents = generate_frame_chunk(
                                model=model,
                                vae=vae,
                                latent_cond=latent_cond[:, :, -1:],  # Use last conditioning frame
                                visual_context=visual_context,
                                keyboard_actions=keyboard_gen,
                                mouse_actions=mouse_gen,
                                diffusion_scheduler=diffusion_scheduler,
                                device=device,
                                num_frames_to_generate=num_frames_to_generate,
                                inference_steps=args.inference_steps
                            )
                            
                            # Combine conditioning and generated latents
                            latents = torch.cat([latent_cond, generated_latents], dim=2)
                        else:
                            latents = latent_cond
                    
                    model.train()
                    epoch_sf_samples += 1
                else:
                    # Use ground truth latents
                    latents = latents_gt
                    epoch_gt_samples += 1
                
                # --- STANDARD FLOW MATCHING TRAINING ---
                # Now train on the latents (either GT or self-forcing)
                
                # Prepare conditioning
                mask_cond = torch.ones_like(latents[:, :4])
                mask_cond[:, :, 1:] = 0
                
                img_cond = latents.clone()
                cond_concat = torch.cat([mask_cond, img_cond], dim=1)
                
                # Sample random timesteps
                batch_size = latents.shape[0]
                diffusion_scheduler.set_timesteps(1000, training=True)
                timestep_indices = torch.randint(0, 1000, (batch_size,))
                timesteps_base = diffusion_scheduler.timesteps[timestep_indices].to(
                    device=device, dtype=torch.bfloat16
                )
                timesteps = timesteps_base.unsqueeze(1).expand(batch_size, num_latent_frames)
                timesteps_expanded = timesteps.flatten()
                
                # Reshape latents for scheduler
                latents_flat = rearrange(latents, 'b c t h w -> (b t) c h w')
                
                # Add noise
                noise = torch.randn_like(latents_flat)
                noisy_latents_flat = diffusion_scheduler.add_noise(
                    latents_flat,
                    noise,
                    timesteps_expanded
                )
                noisy_latents = rearrange(noisy_latents_flat, '(b t) c h w -> b c t h w', 
                                         b=batch_size, t=num_latent_frames)
                
                # Forward pass
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
                
                if isinstance(predicted_velocity, tuple):
                    predicted_velocity = predicted_velocity[0]
                
                # Compute target velocity
                # IMPORTANT: For self-forcing, we still compute target based on GT for supervision
                latents_target = latents_gt if use_self_forcing else latents
                target_flat = rearrange(latents_target, 'b c t h w -> (b t) c h w')
                target_velocity_flat = diffusion_scheduler.training_target(
                    target_flat,
                    noise,
                    timesteps_expanded
                )
                target_velocity = rearrange(target_velocity_flat, '(b t) c h w -> b c t h w',
                                           b=batch_size, t=num_latent_frames)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(
                    predicted_velocity.float(),
                    target_velocity.float(),
                    reduction='mean'
                )
                
                # Scale for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
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
                    'sf_prob': f"{self_forcing_prob:.2f}",
                    'sf_count': epoch_sf_samples,
                    'gt_count': epoch_gt_samples,
                    'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Cleanup
                del noisy_latents, noisy_latents_flat, predicted_velocity, target_velocity, target_velocity_flat
                del loss, latents, latents_flat, latents_gt, target_flat, visual_context, cond_concat, noise
                del frames, frames_vae, keyboard, conditional_dict
                if mouse is not None:
                    del mouse
                del keyboard_per_frame
                if mouse_per_frame is not None:
                    del mouse_per_frame
                del keyboard_expanded
                if 'mouse_expanded' in locals():
                    del mouse_expanded
                if 'generated_latents' in locals():
                    del generated_latents
                
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                if batch_idx == 0:
                    import traceback
                    traceback.print_exc()
                torch.cuda.empty_cache()
                continue
        
        # Calculate average epoch loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} completed.")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Ground truth samples: {epoch_gt_samples}")
        print(f"  Self-forcing samples: {epoch_sf_samples}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"causal_distilled_sf_epoch{epoch+1}.safetensors"
            )
            print(f"Saving checkpoint to {checkpoint_path}...")
            save_file(model.state_dict(), checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, "causal_distilled_sf_best.safetensors")
            print(f"New best model! Saving to {best_path}...")
            save_file(model.state_dict(), best_path)
        
        torch.cuda.empty_cache()
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "causal_distilled_sf_final.safetensors")
    print(f"\nTraining complete! Saving final model to {final_path}...")
    save_file(model.state_dict(), final_path)
    
    # Save training info
    info_path = os.path.join(args.checkpoint_dir, "training_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Self-Forcing Training Configuration\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Pretrained checkpoint: {args.pretrained_checkpoint}\n")
        f.write(f"Model variant: {args.model_variant}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Number of epochs: {args.num_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"\nSelf-Forcing Parameters:\n")
        f.write(f"  Mode: {args.self_forcing_mode}\n")
        f.write(f"  Probability range: {args.self_forcing_prob_start} -> {args.self_forcing_prob_end}\n")
        f.write(f"  Conditioning frames: {args.num_conditioning_frames}\n")
        f.write(f"  Inference steps: {args.inference_steps}\n")
        f.write(f"\nTraining Results:\n")
        f.write(f"  Final loss: {avg_loss:.4f}\n")
        f.write(f"  Best loss: {best_loss:.4f}\n")
    
    print(f"\nTraining info saved to {info_path}")
    print("\n" + "=" * 60)
    print("SELF-FORCING TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nKey improvements over standard training:")
    print(f"  - Model trained on its own predictions, not just ground truth")
    print(f"  - Addresses train/test mismatch and error accumulation")
    print(f"  - Should generate more stable long-horizon predictions")
    print(f"\nYou can now run inference with:")
    recommend_config = "configs/inference_yaml/inference_gta_drive.yaml" if args.model_variant == "gta_drive" else args.config_path
    print(f"  python inference.py --config_path {recommend_config} \\")
    print(f"      --checkpoint_path {best_path} \\")
    print(f"      --img_path demo_images/universal/0000.png --num_output_frames 150")


if __name__ == "__main__":
    main()

