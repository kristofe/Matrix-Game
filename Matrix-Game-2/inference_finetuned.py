#!/usr/bin/env python3
"""
Custom inference script for finetuned WanModel (not CausalWanModel)
This script is designed to work with the base model architecture that was finetuned.
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
    parser.add_argument("--config_path", type=str, default="models/base_model/base_config.json", help="Path to the base model config")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/base_finetuned_final.safetensors", help="Path to the finetuned checkpoint")
    parser.add_argument("--img_path", type=str, default="data/frame_0300.png", help="Path to the input image")
    parser.add_argument("--output_folder", type=str, default="outputs/", help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=150, help="Number of output frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pretrained_model_path", type=str, default="models/", help="Path to the VAE model folder")
    args = parser.parse_args()
    return args

class FinetunedModelInference:
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
        
        # Create the base model with the same architecture as training
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
        
        # Load the state dict with strict=False to handle any missing keys
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} (these will use random initialization)")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)} (these will be ignored)")
        
        self.model = self.model.to(self.device, dtype=self.weight_dtype)
        self.model.eval()
        
        # Initialize VAE components
        print("Initializing VAE components...")
        self.vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae = self.vae.to(self.device, self.weight_dtype)
        
        # Initialize VAE decoder (don't compile it for testing)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        # Don't compile for now to avoid dimension issues
        # current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        
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

    def generate_video(self):
        """Generate video using the finetuned model."""
        print("Loading input image...")
        image = load_image(self.args.img_path)
        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        
        print("Encoding input image...")
        # Use the same sequence length as training for consistency
        sequence_length = 9  # Same as training
        # Calculate the compressed frame count for latents
        latent_frames = (sequence_length - 1) // 4 + 1  # VAE time compression (9 -> 3)
        
        # Encode the input image as the first latent
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (sequence_length - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        
        print(f"VAE encoded image shape: {img_cond.shape}")
        print(f"Expected latent frames: {latent_frames}")
        
        # The VAE is not compressing temporal dimension as expected
        # Manually compress the temporal dimension to match latent frames
        if img_cond.shape[2] != latent_frames:
            print(f"Compressing temporal dimension from {img_cond.shape[2]} to {latent_frames}")
            # Sample every nth frame to match latent dimensions
            step = img_cond.shape[2] // latent_frames
            img_cond = img_cond[:, :, ::step, :, :][:, :, :latent_frames, :, :]
        
        # Create mask_cond with the same frame count as latents
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        
        print(f"Final cond_concat shape: {cond_concat.shape}") 
        visual_context = self.vae.clip.encode_video(image)
        
        print("Generating random actions...")
        # The action module expects frames to be compatible with vae_time_compression_ratio=4
        # Use 9 frames (which works: (9-1)+4 = 12, 12%4 = 0)
        action_frames = 9  # This satisfies the assertion: ((9-1)+4)%4 = 0
        cond_data = Bench_actions_universal(action_frames)
        mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        
        print(f"Action conditions shape: mouse={mouse_condition.shape}, keyboard={keyboard_condition.shape}")
        
        print(f"Generating {sequence_length} frames (same as training)...")
        print(f"Latent frames after VAE compression: {latent_frames}")
        # Initialize noise with correct dimensions for the model
        # The model expects latents with specific frame dimensions
        sampled_noise = torch.randn(
            [1, 16, latent_frames, 44, 80], device=self.device, dtype=self.weight_dtype
        )
        
        # Simple diffusion sampling (DDPM)
        with torch.no_grad():
            # Start with pure noise
            latents = sampled_noise
            
            # Ensure all tensors are in the same dtype as the model
            latents = latents.to(dtype=self.weight_dtype)
            cond_concat = cond_concat.to(dtype=self.weight_dtype)
            visual_context = visual_context.to(dtype=self.weight_dtype)
            mouse_condition = mouse_condition.to(dtype=self.weight_dtype)
            keyboard_condition = keyboard_condition.to(dtype=self.weight_dtype)
            
            # Simple denoising loop (you can make this more sophisticated)
            for t in tqdm(range(1000, 0, -10), desc="Denoising"):
                timestep = torch.tensor([t], device=self.device, dtype=self.weight_dtype)
                
                # Ensure latents remain in correct dtype after update
                latents = latents.to(dtype=self.weight_dtype)
                
                # Forward pass through the model with correct action dimensions
                predicted_noise = self.model(
                    latents,
                    timestep,
                    visual_context=visual_context,
                    cond_concat=cond_concat,
                    mouse_cond=mouse_condition,
                    keyboard_cond=keyboard_condition
                )
                
                # Simple denoising step (keep in bfloat16)
                alpha = 1.0 - t / 1000.0
                latents = latents - 0.01 * predicted_noise.to(dtype=self.weight_dtype)
        
        print("Decoding latents to video...")
        # Decode latents to video frames
        with torch.no_grad():
            # The latents need to be in the correct format for VAE decoder
            # VAEDecoderWrapper expects: [batch, frames, channels, height, width]
            # Our latents are: [batch, channels, frames, height, width]
            # Need to permute: B C T H W -> B T C H W
            
            # Convert to float16 for VAE decoder and permute dimensions
            latents_for_decode = latents.to(dtype=torch.float16)
            latents_for_decode = latents_for_decode.permute(0, 2, 1, 3, 4)  # B C T H W -> B T C H W
            
            print(f"Latents shape for decoder: {latents_for_decode.shape}")
            
            # Decode using VAE decoder - it returns (decoded_video, feat_cache)
            # The decoder expects feat_cache as *args, pass empty tensors for initial cache
            # Based on vae_block3.py, we need to pass cache tensors as separate arguments
            decoded_video, _ = self.vae_decoder(latents_for_decode, *[None] * 32)
            
            # decoded_video is [B, T, C, H, W], convert to [B, T, H, W, C]
            videos = rearrange(decoded_video, "B T C H W -> B T H W C")
            videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        
        print("Saving video...")
        # Save the video
        video = np.ascontiguousarray(videos)
        mouse_icon = 'assets/images/mouse.png'
        config = (
            keyboard_condition[0].float().cpu().numpy(),
            mouse_condition[0].float().cpu().numpy()
        )
        
        os.makedirs(self.args.output_folder, exist_ok=True)
        process_video(video.astype(np.uint8), 
                     os.path.join(self.args.output_folder, 'finetuned_demo.mp4'), 
                     config, mouse_icon, mouse_scale=0.1, process_icon=False, mode='universal')
        process_video(video.astype(np.uint8), 
                     os.path.join(self.args.output_folder, 'finetuned_demo_icon.mp4'), 
                     config, mouse_icon, mouse_scale=0.1, process_icon=True, mode='universal')
        
        print(f"Video saved to {self.args.output_folder}/finetuned_demo.mp4")
        print("Done!")

def main():
    """Main entry point for video generation."""
    args = parse_args()
    set_seed(args.seed)
    
    print("="*60)
    print("FINETUNED MODEL INFERENCE")
    print("="*60)
    print(f"Config: {args.config_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Input image: {args.img_path}")
    print(f"Output frames: {args.num_output_frames}")
    print("="*60)
    
    pipeline = FinetunedModelInference(args)
    pipeline.generate_video()

if __name__ == "__main__":
    main()
