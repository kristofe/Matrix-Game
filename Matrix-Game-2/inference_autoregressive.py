#!/usr/bin/env python3
"""
Autoregressive inference for finetuned WanModel
Generates long videos by chaining multiple generations together
"""

import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange
from safetensors.torch import load_file
from tqdm import tqdm

# Import Matrix-Game components
from wan.modules.model import WanModel
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="models/base_model/base_config.json")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/autoregressive_best.safetensors")
    parser.add_argument("--img_path", type=str, default="data/frame_0300.png")
    parser.add_argument("--output_folder", type=str, default="outputs/")
    parser.add_argument("--num_output_frames", type=int, default=81, help="Total frames to generate (will round to nearest 9)")
    parser.add_argument("--frames_per_generation", type=int, default=9, help="Frames generated per step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained_model_path", type=str, default="models/")
    parser.add_argument("--action_mode", type=str, default="random", choices=["random", "forward", "custom"])
    args = parser.parse_args()
    return args

class AutoregressiveInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.bfloat16
        
        print(f"Using device: {self.device}")
        
        self._init_config()
        self._init_models()
        
        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        """Load the base model configuration."""
        print(f"Loading config from: {self.args.config_path}")
        self.config = OmegaConf.load(self.args.config_path)
        print(f"Model type: {self.config.get('model_type', 'i2v')}")

    def _init_models(self):
        """Initialize the finetuned model and VAE components."""
        print("Initializing finetuned WanModel...")
        
        self.model = WanModel(
            model_type=self.config.get('model_type', 'i2v'),
            patch_size=tuple(self.config.action_config.patch_size),
            in_dim=self.config.in_dim,
            dim=self.config.dim,
            ffn_dim=self.config.ffn_dim,
            freq_dim=self.config.freq_dim,
            out_dim=self.config.out_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            qk_norm=self.config.action_config.qk_norm,
            action_config=self.config.action_config,
            eps=self.config.eps
        )
        
        # Load the finetuned weights
        print(f"Loading finetuned weights from: {self.args.checkpoint_path}")
        state_dict = load_file(self.args.checkpoint_path)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        self.model = self.model.to(self.device, dtype=self.weight_dtype)
        self.model.eval()
        
        # Initialize VAE components
        print("Initializing VAE components...")
        self.vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae = self.vae.to(self.device, self.weight_dtype)
        
        # Initialize VAE decoder
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(
            os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"), 
            map_location="cpu"
        )
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        
        self.vae_decoder = current_vae_decoder

    def _resizecrop(self, image, th, tw):
        """Resize and crop image to target dimensions."""
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    def generate_actions(self, num_frames, mode="random"):
        """Generate action sequences for the given number of frames."""
        if mode == "random":
            cond_data = Bench_actions_universal(num_frames)
        elif mode == "forward":
            # Generate actions that move forward
            keyboard = np.zeros((num_frames, 4))  # [W, A, S, D]
            keyboard[:, 0] = 1  # W key (forward)
            mouse = np.zeros((num_frames, 2))  # [dx, dy]
            cond_data = {
                'keyboard_condition': torch.from_numpy(keyboard).float(),
                'mouse_condition': torch.from_numpy(mouse).float()
            }
        else:
            # Default random
            cond_data = Bench_actions_universal(num_frames)
        
        return cond_data

    def denoise_latents(self, latents, visual_context, cond_concat, mouse_cond, keyboard_cond, num_steps=100):
        """
        Denoise latents using the diffusion model.
        
        Args:
            latents: Initial noisy latents
            visual_context: CLIP visual embeddings
            cond_concat: Conditioning concat (mask + context)
            mouse_cond: Mouse action conditioning
            keyboard_cond: Keyboard action conditioning
            num_steps: Number of denoising steps
        
        Returns:
            Denoised latents
        """
        # Ensure all inputs are in correct dtype
        latents = latents.to(dtype=self.weight_dtype)
        visual_context = visual_context.to(dtype=self.weight_dtype)
        cond_concat = cond_concat.to(dtype=self.weight_dtype)
        mouse_cond = mouse_cond.to(dtype=self.weight_dtype)
        keyboard_cond = keyboard_cond.to(dtype=self.weight_dtype)
        
        # Simple DDPM denoising
        for t in tqdm(range(1000, 0, -1000//num_steps), desc="Denoising", leave=False):
            timestep = torch.tensor([t], device=self.device, dtype=self.weight_dtype)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.model(
                    latents,
                    timestep,
                    visual_context=visual_context,
                    cond_concat=cond_concat,
                    mouse_cond=mouse_cond,
                    keyboard_cond=keyboard_cond
                )
            
            # Denoising step
            alpha = 1.0 - t / 1000.0
            beta = 1.0 - alpha
            latents = (latents - beta ** 0.5 * predicted_noise) / (alpha ** 0.5)
            latents = latents.to(dtype=self.weight_dtype)
        
        return latents

    def decode_latents_to_frames(self, latents):
        """Decode latents to video frames using VAE decoder."""
        with torch.no_grad():
            # Convert to float16 for VAE decoder
            latents_for_decode = latents.to(dtype=torch.float16)
            # Permute: B C T H W -> B T C H W
            latents_for_decode = latents_for_decode.permute(0, 2, 1, 3, 4)
            
            # Decode using VAE decoder
            decoded_video, _ = self.vae_decoder(latents_for_decode, *[None] * 32)
            
            # Convert: B T C H W -> B T H W C
            frames = rearrange(decoded_video, "B T C H W -> B T H W C")
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        
        return frames

    def generate_video_autoregressive(self):
        """Generate long video using autoregressive generation."""
        print("="*60)
        print("AUTOREGRESSIVE VIDEO GENERATION")
        print("="*60)
        
        # Load and prepare input image
        print("\nLoading input image...")
        image = load_image(self.args.img_path)
        image = self._resizecrop(image, 352, 640)
        image_tensor = self.frame_process(image)[None, :, None, :, :].to(
            dtype=self.weight_dtype, device=self.device
        )
        
        # Configuration
        frames_per_step = self.args.frames_per_generation
        total_frames_desired = self.args.num_output_frames
        
        # Round to nearest multiple
        num_generations = max(1, total_frames_desired // frames_per_step)
        total_frames = num_generations * frames_per_step
        
        print(f"\nGeneration plan:")
        print(f"  Frames per generation: {frames_per_step}")
        print(f"  Number of generations: {num_generations}")
        print(f"  Total frames: {total_frames}")
        
        # Setup for first generation
        vae_time_compression = 4
        latent_frames = (frames_per_step - 1) // vae_time_compression + 1
        latent_height, latent_width = 44, 80
        
        # Encode initial image
        print("\nEncoding initial image...")
        padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (frames_per_step - 1), 1, 1)
        img_cond = torch.concat([image_tensor, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        
        # Compress temporal dimension
        if img_cond.shape[2] != latent_frames:
            step = img_cond.shape[2] // latent_frames
            img_cond = img_cond[:, :, ::step, :, :][:, :, :latent_frames, :, :]
        
        # Get visual context (CLIP embedding)
        visual_context = self.vae.clip.encode_video(image_tensor)
        
        # Storage for all generated frames and actions
        all_frames = []
        all_keyboard_actions = []
        all_mouse_actions = []
        
        # Autoregressive generation loop
        print(f"\nGenerating {total_frames} frames autoregressively...")
        
        for gen_idx in range(num_generations):
            print(f"\n{'='*60}")
            print(f"Generation {gen_idx + 1}/{num_generations}")
            print(f"{'='*60}")
            
            # Generate actions for this step
            cond_data = self.generate_actions(frames_per_step, mode=self.args.action_mode)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(
                device=self.device, dtype=self.weight_dtype
            )
            keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(
                device=self.device, dtype=self.weight_dtype
            )
            
            # Store actions
            all_keyboard_actions.append(cond_data['keyboard_condition'])
            all_mouse_actions.append(cond_data['mouse_condition'])
            
            # Setup conditioning
            if gen_idx == 0:
                # First generation: use initial image as context
                mask_cond = torch.ones_like(img_cond)
                mask_cond[:, :, 1:] = 0  # Mask all but first frame
                cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            else:
                # Subsequent generations: use last generated latents as context
                # Create mask that conditions on previous frame
                mask_cond = torch.ones_like(prev_latents)
                mask_cond[:, :, 1:] = 0
                cond_concat = torch.cat([mask_cond[:, :4], prev_latents], dim=1)
            
            # Initialize noise
            sampled_noise = torch.randn(
                [1, 16, latent_frames, latent_height, latent_width],
                device=self.device,
                dtype=self.weight_dtype
            )
            
            # Denoise to generate new frames
            print(f"Generating frames {gen_idx * frames_per_step} to {(gen_idx + 1) * frames_per_step}...")
            generated_latents = self.denoise_latents(
                sampled_noise,
                visual_context,
                cond_concat,
                mouse_condition,
                keyboard_condition,
                num_steps=100
            )
            
            # Decode latents to frames
            print("Decoding latents...")
            frames = self.decode_latents_to_frames(generated_latents)
            all_frames.append(frames)
            
            # Use these latents as context for next generation
            prev_latents = generated_latents.detach()
            
            print(f"Generated frames shape: {frames.shape}")
        
        # Concatenate all generated frames
        print("\n" + "="*60)
        print("Combining all generated frames...")
        all_frames = np.concatenate(all_frames, axis=0)
        all_keyboard_actions = torch.cat(all_keyboard_actions, dim=0)
        all_mouse_actions = torch.cat(all_mouse_actions, dim=0)
        
        print(f"Total generated frames: {all_frames.shape[0]}")
        print(f"Frame dimensions: {all_frames.shape[1:3]}")
        
        # Save video
        print("\nSaving video...")
        os.makedirs(self.args.output_folder, exist_ok=True)
        
        mouse_icon = 'assets/images/mouse.png'
        config = (
            all_keyboard_actions.float().cpu().numpy(),
            all_mouse_actions.float().cpu().numpy()
        )
        
        # Save without icon overlay
        output_path_no_icon = os.path.join(self.args.output_folder, 'autoregressive_demo.mp4')
        process_video(
            all_frames.astype(np.uint8),
            output_path_no_icon,
            config,
            mouse_icon,
            mouse_scale=0.1,
            process_icon=False,
            mode='universal'
        )
        print(f"Video saved to: {output_path_no_icon}")
        
        # Save with icon overlay
        output_path_with_icon = os.path.join(self.args.output_folder, 'autoregressive_demo_icon.mp4')
        process_video(
            all_frames.astype(np.uint8),
            output_path_with_icon,
            config,
            mouse_icon,
            mouse_scale=0.1,
            process_icon=True,
            mode='universal'
        )
        print(f"Video with icons saved to: {output_path_with_icon}")
        
        print("\n" + "="*60)
        print("Autoregressive generation complete!")
        print("="*60)

def main():
    """Main entry point for autoregressive video generation."""
    args = parse_args()
    set_seed(args.seed)
    
    print("="*60)
    print("AUTOREGRESSIVE INFERENCE")
    print("="*60)
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Input image: {args.img_path}")
    print(f"Target frames: {args.num_output_frames}")
    print(f"Action mode: {args.action_mode}")
    print("="*60)
    
    pipeline = AutoregressiveInference(args)
    pipeline.generate_video_autoregressive()

if __name__ == "__main__":
    main()

