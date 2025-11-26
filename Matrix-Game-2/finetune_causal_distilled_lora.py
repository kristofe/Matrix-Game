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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import get_peft_model_state_dict


def latent_cache_collate_fn(batch):
    """Custom collate function that handles mixed cached/uncached batches."""
    # Check if all items are cached
    all_cached = all(item.get('cached', False) for item in batch)

    # Build result dict with standard collation for most keys
    result = {}

    # Keys that should always be collated normally
    standard_keys = ['video_frames', 'keyboard_actions', 'mouse_actions', 'sequence_idx']
    for key in standard_keys:
        if key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values

    # Handle cached flag
    result['cached'] = torch.tensor([item.get('cached', False) for item in batch])

    # Only collate latents/visual_context if ALL items are cached
    if all_cached and 'latents' in batch[0] and batch[0]['latents'].numel() > 0:
        result['latents'] = torch.stack([item['latents'] for item in batch])
        result['visual_context'] = torch.stack([item['visual_context'] for item in batch])

    # Keep run_path for debugging if present
    if 'run_path' in batch[0]:
        result['run_path'] = [item['run_path'] for item in batch]

    return result

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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=4,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps (effective batch size)")

    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_strategy", type=str, default="full",
                       choices=["attention_only", "attention_ffn", "full", "action_focused"])

    return parser.parse_args()

def get_lora_target_modules(strategy):
    """Get target modules based on strategy."""
    
    strategies = {
        "attention_only": [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        ],
        "attention_ffn": [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
            "ffn.0", "ffn.2",
        ],
        "full": [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
            "ffn.0", "ffn.2",
            "action_model.mouse_mlp.0", "action_model.mouse_mlp.2",
            "action_model.keyboard_embed.0", "action_model.keyboard_embed.2",
            "action_model.proj_mouse", "action_model.proj_keyboard",
        ],
        "action_focused": [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "action_model.mouse_mlp.0", "action_model.mouse_mlp.2",
            "action_model.keyboard_embed.0", "action_model.keyboard_embed.2",
            "action_model.proj_mouse", "action_model.proj_keyboard",
        ],
    }
    
    return strategies[strategy]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable cudnn benchmark for faster convolutions
    torch.backends.cudnn.benchmark = True
    
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
    
    # 2. Configure LoRA
    print(f"\n2. Configuring LoRA (strategy: {args.lora_strategy})...")
    target_modules = get_lora_target_modules(args.lora_strategy)
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Store base model reference before PEFT wrapping (for key comparison after merging)
    base_model_ref = model
    original_state_dict_keys = set(model.state_dict().keys())
    print(f"Original model has {len(original_state_dict_keys)} parameters")
    
    model = get_peft_model(model, lora_config)
    model = model.to(device, dtype=torch.bfloat16)
    # torch.compile disabled - too many graph breaks from einops/dynamic shapes
    # provides minimal benefit with this model architecture
    model.train()
    
    # Verify that PEFT has frozen base model parameters and only LoRA adapters are trainable
    print("\nVerifying parameter freezing...")
    base_model_params = 0
    base_model_trainable = 0
    lora_params = 0
    lora_trainable = 0
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params += param.numel()
            if param.requires_grad:
                lora_trainable += param.numel()
        else:
            base_model_params += param.numel()
            if param.requires_grad:
                base_model_trainable += param.numel()
    
    print(f"  Base model parameters: {base_model_params:,} total, {base_model_trainable:,} trainable")
    print(f"  LoRA adapter parameters: {lora_params:,} total, {lora_trainable:,} trainable")
    
    if base_model_trainable > 0:
        print(f"  WARNING: {base_model_trainable:,} base model parameters are trainable (should be 0)")
        print(f"  This may indicate an issue with PEFT configuration!")
    else:
        print(f"  ✓ Base model parameters are correctly frozen")
    
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total trainable: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")
    
    # Initialize VAE for encoding frames
    print("\nLoading VAE encoder...")
    vae = get_wanx_vae_wrapper("models/", torch.float16)
    vae.requires_grad_(False)
    vae.eval()
    # Keep VAE on GPU - we have 90GB VRAM, no need to move it back and forth
    vae = vae.to(device, torch.float16)
    
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
        fps=25,
        cache_latents=True  # Enable latent caching for faster subsequent epochs
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
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=latent_cache_collate_fn  # Handle mixed cached/uncached batches
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

    # Check cache status
    if hasattr(dataset, '_count_cached_sequences'):
        cached = dataset._count_cached_sequences()
        total = len(dataset)
        if cached < total:
            print(f"Note: {total - cached} sequences need latent encoding (will be cached during first epoch)")
        else:
            print(f"All {total} sequences have cached latents - training will be faster!")
    
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

                # Check if we have cached latents (all items in batch must be cached)
                # The custom collate only adds 'latents' key if all items are cached
                use_cache = 'latents' in batch

                if use_cache:
                    # Use cached latents and visual context
                    latents = batch['latents'].to(device=device, dtype=torch.bfloat16)
                    visual_context = batch['visual_context'].to(device=device, dtype=torch.bfloat16)
                else:
                    # Normalize frames to [-1, 1]
                    frames_norm = normalize_frames(frames)

                    # Encode frames to latents using VAE
                    with torch.no_grad():
                        # Rearrange for VAE: [B, C, T, H, W]
                        frames_vae = frames_norm.permute(0, 4, 1, 2, 3).to(dtype=torch.float16)

                        # Encode with tiling
                        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
                        latents = vae.encode(frames_vae, device=device, **tiler_kwargs)
                        latents = latents.to(device=device, dtype=torch.bfloat16)

                        # Get visual context from CLIP
                        visual_context = vae.clip.encode_video(frames_vae).to(
                            device=device, dtype=torch.bfloat16
                        )

                    # Cache the latents for future epochs
                    if hasattr(dataset, 'save_cached_latents'):
                        sequence_indices = batch['sequence_idx']
                        for i, seq_idx in enumerate(sequence_indices):
                            # Handle both tensor and int cases
                            idx = seq_idx.item() if isinstance(seq_idx, torch.Tensor) else seq_idx
                            if not dataset.has_cached_latents(idx):
                                # Save without batch dimension - DataLoader will add it back
                                dataset.save_cached_latents(
                                    idx,
                                    latents[i],  # [C, T, H, W] not [1, C, T, H, W]
                                    visual_context[i]  # [T, D] not [1, T, D]
                                )
                
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

                # Compute target velocity (use flattened version for scheduler)
                target_velocity_flat = diffusion_scheduler.training_target(
                    latents_flat,
                    noise,
                    timesteps_expanded
                )
                # Reshape back to [B, C, T, H, W]
                target_velocity = rearrange(target_velocity_flat, '(b t) c h w -> b c t h w', b=batch_size, t=num_latent_frames)

                # Use autocast for mixed precision forward pass
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicted_velocity = model(
                        noisy_image_or_video=noisy_latents,
                        conditional_dict=conditional_dict,
                        timestep=timesteps
                    )

                    # Handle tuple output (model may return (output, logits))
                    if isinstance(predicted_velocity, tuple):
                        predicted_velocity = predicted_velocity[0]

                    # Compute loss
                    loss = torch.nn.functional.mse_loss(
                        predicted_velocity,
                        target_velocity,
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
                
                # Minimal memory cleanup (Python GC will handle most of this)
                # Only clear cache if we're actually running low on memory
                # torch.cuda.empty_cache() is slow and unnecessary with 90GB VRAM
                
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
                
                # Only clear cache on error in case there's a memory leak
                torch.cuda.empty_cache()
                continue
        
        # Calculate average epoch loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint (adapter weights only - for resuming training if needed)
        # NOTE: Periodic checkpoints save only adapter weights to preserve training state.
        # Use final/best models (merged) for inference with the regular inference script.
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"causal_distilled_lora_epoch{epoch+1}.safetensors"
            )
            print(f"Saving checkpoint adapter weights to {checkpoint_path}...")
            # Save only adapter weights (can't merge without breaking training state)
            if isinstance(model, PeftModel):
                adapter_state = get_peft_model_state_dict(model)
                save_file(adapter_state, checkpoint_path)
                print(f"  NOTE: This checkpoint contains only LoRA adapter weights.")
                print(f"  To use for inference, merge with base model first.")
            else:
                save_file(model.state_dict(), checkpoint_path)
        
        # Save best model (merged for inference)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, "causal_distilled_lora_best.safetensors")
            print(f"New best model! Saving merged model to {best_path} (ready for inference)...")
            if isinstance(model, PeftModel):
                # Merge adapters into base model for inference compatibility
                # merge_and_unload() computes: W_merged = W_base + (lora_B @ lora_A) * (alpha/rank)
                # It returns the base model (WanDiffusionWrapper) with merged weights
                merged_model = model.merge_and_unload()
                merged_state_dict = merged_model.state_dict()
                merged_keys = set(merged_state_dict.keys())
                
                # Verify keys match original model structure (for inference compatibility)
                if merged_keys == original_state_dict_keys:
                    print(f"  ✓ Merged model keys match original ({len(merged_keys)} parameters)")
                    save_file(merged_state_dict, best_path)
                else:
                    # Check for common prefix differences - try multiple prefix patterns
                    # PEFT merge_and_unload() may leave base_model.model.model. or base_model.model. prefix
                    fixed_state_dict = None
                    prefix_removed = None
                    
                    # Try removing base_model.model.model. -> model.
                    merged_no_prefix = {k.replace('base_model.model.model.', 'model.') if 'base_model.model.model.' in k else k for k in merged_keys}
                    if merged_no_prefix == original_state_dict_keys:
                        prefix_removed = 'base_model.model.model.'
                        fixed_state_dict = {k.replace('base_model.model.model.', 'model.'): v for k, v in merged_state_dict.items()}
                    else:
                        # Try removing base_model.model. -> model.
                        merged_no_prefix = {k.replace('base_model.model.', 'model.') if 'base_model.model.' in k else k for k in merged_keys}
                        if merged_no_prefix == original_state_dict_keys:
                            prefix_removed = 'base_model.model.'
                            fixed_state_dict = {k.replace('base_model.model.', 'model.'): v for k, v in merged_state_dict.items()}
                        else:
                            # Try removing base_model. -> (empty)
                            merged_no_prefix = {k.replace('base_model.', '') if k.startswith('base_model.') else k for k in merged_keys}
                            if merged_no_prefix == original_state_dict_keys:
                                prefix_removed = 'base_model.'
                                fixed_state_dict = {k.replace('base_model.', ''): v for k, v in merged_state_dict.items()}
                    
                    if fixed_state_dict is not None:
                        print(f"  ⚠ Merged model has '{prefix_removed}' prefix - removing for compatibility...")
                        save_file(fixed_state_dict, best_path)
                    else:
                        print(f"  ⚠ WARNING: Key mismatch detected!")
                        print(f"    Original keys: {len(original_state_dict_keys)}")
                        print(f"    Merged keys: {len(merged_keys)}")
                        print(f"    Missing in merged: {original_state_dict_keys - merged_keys}")
                        print(f"    Extra in merged: {merged_keys - original_state_dict_keys}")
                        print(f"    Saving anyway - may need manual key adjustment for inference")
                        save_file(merged_state_dict, best_path)
                
                # Re-create PEFT model to continue training (with fresh adapters - will continue from merged state)
                # Note: This means best model checkpoint breaks training continuity, but it's ready for inference
                model = get_peft_model(merged_model, lora_config)
                model = model.to(device, dtype=torch.bfloat16)
                model.train()
            else:
                save_file(model.state_dict(), best_path)
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "causal_distilled_lora_final.safetensors")
    print(f"\nTraining complete! Saving final model to {final_path}...")
    # Merge LoRA adapters into base model for inference compatibility
    if isinstance(model, PeftModel):
        # Use PEFT's built-in merge and unload for final save
        merged_model = model.merge_and_unload()
        merged_state_dict = merged_model.state_dict()
        merged_keys = set(merged_state_dict.keys())
        
        # Verify keys match original model structure
        if merged_keys == original_state_dict_keys:
            print(f"✓ Merged final model keys match original ({len(merged_keys)} parameters)")
            save_file(merged_state_dict, final_path)
        else:
            # Remove base_model prefix if present - try multiple prefix patterns
            # PEFT merge_and_unload() may leave base_model.model.model. or base_model.model. prefix
            fixed_state_dict = None
            prefix_removed = None
            
            # Try removing base_model.model.model. -> model.
            merged_no_prefix = {k.replace('base_model.model.model.', 'model.') if 'base_model.model.model.' in k else k for k in merged_keys}
            if merged_no_prefix == original_state_dict_keys:
                prefix_removed = 'base_model.model.model.'
                fixed_state_dict = {k.replace('base_model.model.model.', 'model.'): v for k, v in merged_state_dict.items()}
            else:
                # Try removing base_model.model. -> model.
                merged_no_prefix = {k.replace('base_model.model.', 'model.') if 'base_model.model.' in k else k for k in merged_keys}
                if merged_no_prefix == original_state_dict_keys:
                    prefix_removed = 'base_model.model.'
                    fixed_state_dict = {k.replace('base_model.model.', 'model.'): v for k, v in merged_state_dict.items()}
                else:
                    # Try removing base_model. -> (empty)
                    merged_no_prefix = {k.replace('base_model.', '') if k.startswith('base_model.') else k for k in merged_keys}
                    if merged_no_prefix == original_state_dict_keys:
                        prefix_removed = 'base_model.'
                        fixed_state_dict = {k.replace('base_model.', ''): v for k, v in merged_state_dict.items()}
            
            if fixed_state_dict is not None:
                print(f"  Removing '{prefix_removed}' prefix from keys for compatibility...")
                save_file(fixed_state_dict, final_path)
            else:
                print(f"  WARNING: Key structure differs - saving as-is")
                save_file(merged_state_dict, final_path)
    else:
        save_file(model.state_dict(), final_path)
    
    # Save training info
    info_path = os.path.join(args.checkpoint_dir, "causal_distilled_lora_training_info.txt")
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
    print("Make sure that the frame count is a multiple of 3 for the inference script to work.")
    print("=" * 60)
    print(f"\nYou can now run inference with:")
    print(f"python inference.py ", end="")
    recommend_config = "configs/inference_yaml/inference_gta_drive.yaml" if args.model_variant == "gta_drive" else args.config_path
    print(f"    --config_path {recommend_config} ", end="")
    print(f"    --checkpoint_path {best_path} ", end="")
    print(f"    --img_path data/frame_0100.png ", end="")
    print(f"    --output_folder outputs ", end="")
    print(f"    --num_output_frames 51 ", end="")
    print(f"    --pretrained_model_path models")


if __name__ == "__main__":
    main()

