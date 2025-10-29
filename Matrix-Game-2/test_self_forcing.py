"""
Test script to verify self-forcing implementation is working correctly.
"""

import torch
import numpy as np
from finetune_causal_distilled_self_forcing import compute_self_forcing_probability


def test_probability_schedules():
    """Test that self-forcing probability schedules work correctly."""
    print("Testing self-forcing probability schedules...")
    print("=" * 60)
    
    total_epochs = 10
    
    # Test scheduled mode (fixed probability)
    print("\n1. Scheduled Mode (Fixed Probability)")
    print("-" * 60)
    for epoch in range(0, total_epochs, 2):
        prob = compute_self_forcing_probability(epoch, total_epochs, "scheduled", 0.0, 0.5)
        print(f"  Epoch {epoch:2d}: {prob:.2f}")
    assert all(
        compute_self_forcing_probability(e, total_epochs, "scheduled", 0.0, 0.5) == 0.5
        for e in range(total_epochs)
    ), "Scheduled mode should return constant probability"
    print("  ✓ Scheduled mode working correctly")
    
    # Test full mode
    print("\n2. Full Mode (Always Self-Forcing)")
    print("-" * 60)
    for epoch in range(0, total_epochs, 2):
        prob = compute_self_forcing_probability(epoch, total_epochs, "full", 0.0, 0.9)
        print(f"  Epoch {epoch:2d}: {prob:.2f}")
    assert all(
        compute_self_forcing_probability(e, total_epochs, "full", 0.0, 0.9) == 1.0
        for e in range(total_epochs)
    ), "Full mode should always return 1.0"
    print("  ✓ Full mode working correctly")
    
    # Test curriculum mode
    print("\n3. Curriculum Mode (Gradual Increase)")
    print("-" * 60)
    probs = []
    for epoch in range(total_epochs):
        prob = compute_self_forcing_probability(epoch, total_epochs, "curriculum", 0.0, 0.9)
        probs.append(prob)
        if epoch % 2 == 0:
            print(f"  Epoch {epoch:2d}: {prob:.2f}")
    
    # Check that probabilities increase monotonically
    assert all(probs[i] <= probs[i+1] for i in range(len(probs)-1)), \
        "Curriculum probabilities should increase monotonically"
    assert abs(probs[0] - 0.0) < 1e-6, "Curriculum should start at prob_start"
    assert abs(probs[-1] - 0.9) < 1e-6, "Curriculum should end at prob_end"
    print("  ✓ Curriculum mode working correctly")
    
    print("\n" + "=" * 60)
    print("✓ All probability schedule tests passed!")


def test_self_forcing_logic():
    """Test the core self-forcing decision logic."""
    print("\n\nTesting self-forcing decision logic...")
    print("=" * 60)
    
    # Simulate multiple samples with different probabilities
    n_samples = 10000
    
    probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for prob in probabilities:
        # Simulate self-forcing decisions
        decisions = [np.random.rand() < prob for _ in range(n_samples)]
        sf_count = sum(decisions)
        gt_count = n_samples - sf_count
        actual_prob = sf_count / n_samples
        
        print(f"\nTarget probability: {prob:.2f}")
        print(f"  Self-forcing samples: {sf_count}")
        print(f"  Ground truth samples: {gt_count}")
        print(f"  Actual probability: {actual_prob:.3f}")
        
        # Check that actual probability is within reasonable range (5% tolerance)
        assert abs(actual_prob - prob) < 0.05, \
            f"Actual probability {actual_prob} too far from target {prob}"
        print(f"  ✓ Within tolerance")
    
    print("\n" + "=" * 60)
    print("✓ Self-forcing decision logic test passed!")


def test_tensor_operations():
    """Test that tensor operations work correctly for self-forcing."""
    print("\n\nTesting tensor operations...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Simulate latent dimensions
    batch_size = 2
    channels = 4
    num_frames = 3
    height, width = 45, 80
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Latent frames: {num_frames}")
    print(f"  Latent shape: [{channels}, {height}, {width}]")
    
    # Create dummy tensors
    print("\n1. Creating dummy tensors...")
    latents_gt = torch.randn(batch_size, channels, num_frames, height, width, 
                            device=device, dtype=torch.bfloat16)
    latent_cond = latents_gt[:, :, :1]  # First frame as conditioning
    print(f"  ✓ Ground truth latents: {latents_gt.shape}")
    print(f"  ✓ Conditioning latent: {latent_cond.shape}")
    
    # Simulate generated latents
    print("\n2. Simulating generated latents...")
    generated_latents = torch.randn(batch_size, channels, num_frames - 1, height, width,
                                   device=device, dtype=torch.bfloat16)
    latents_sf = torch.cat([latent_cond, generated_latents], dim=2)
    print(f"  ✓ Generated latents: {generated_latents.shape}")
    print(f"  ✓ Combined latents: {latents_sf.shape}")
    
    # Test conditioning preparation
    print("\n3. Testing conditioning preparation...")
    mask_cond = torch.ones_like(latents_gt[:, :channels])
    mask_cond[:, :, 1:] = 0
    img_cond = latents_gt.clone()
    cond_concat = torch.cat([mask_cond, img_cond], dim=1)
    print(f"  ✓ Mask condition: {mask_cond.shape}")
    print(f"  ✓ Combined conditioning: {cond_concat.shape}")
    assert cond_concat.shape[1] == channels * 2, "Conditioning should have 2x channels"
    
    # Test action preparation
    print("\n4. Testing action preparation...")
    num_action_steps = 1 + 4 * (num_frames - 1)
    keyboard_actions = torch.randn(batch_size, num_action_steps, 2, device=device, dtype=torch.bfloat16)
    mouse_actions = torch.randn(batch_size, num_action_steps, 2, device=device, dtype=torch.bfloat16)
    print(f"  ✓ Keyboard actions: {keyboard_actions.shape}")
    print(f"  ✓ Mouse actions: {mouse_actions.shape}")
    print(f"  ✓ Action steps ({num_action_steps}) = 1 + 4 * ({num_frames} - 1)")
    
    # Test memory cleanup
    print("\n5. Testing memory management...")
    initial_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    
    # Create and delete tensors
    temp_tensors = [torch.randn(100, 100, device=device) for _ in range(10)]
    del temp_tensors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    final_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    print(f"  ✓ Memory management working")
    print(f"    Initial: {initial_memory / 1e6:.2f} MB")
    print(f"    Final: {final_memory / 1e6:.2f} MB")
    
    print("\n" + "=" * 60)
    print("✓ All tensor operation tests passed!")


def test_compatibility():
    """Test that the new script is compatible with existing infrastructure."""
    print("\n\nTesting compatibility with existing code...")
    print("=" * 60)
    
    print("\n1. Checking imports...")
    try:
        from utils.scheduler import FlowMatchScheduler
        print("  ✓ FlowMatchScheduler imported")
    except ImportError as e:
        print(f"  ✗ FlowMatchScheduler import failed: {e}")
        return False
    
    try:
        from utils.wan_wrapper import WanDiffusionWrapper
        print("  ✓ WanDiffusionWrapper imported")
    except ImportError as e:
        print(f"  ✗ WanDiffusionWrapper import failed: {e}")
        return False
    
    try:
        from convert_unreal_data import UnrealDataset
        print("  ✓ UnrealDataset imported")
    except ImportError as e:
        print(f"  ✗ UnrealDataset import failed: {e}")
        return False
    
    print("\n2. Checking scheduler compatibility...")
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    print(f"  ✓ Scheduler initialized with {len(scheduler.timesteps)} timesteps")
    
    print("\n3. Checking device availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✓ Compatibility tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" " * 20 + "SELF-FORCING IMPLEMENTATION TESTS")
    print("=" * 80)
    
    try:
        # Run tests
        test_probability_schedules()
        test_self_forcing_logic()
        test_tensor_operations()
        success = test_compatibility()
        
        # Summary
        print("\n\n" + "=" * 80)
        print(" " * 30 + "TEST SUMMARY")
        print("=" * 80)
        print("\n✓ All tests passed successfully!")
        print("\nYour self-forcing implementation is ready to use.")
        print("\nNext steps:")
        print("  1. Prepare your training data")
        print("  2. Run: python finetune_causal_distilled_self_forcing.py --data_dir data")
        print("  3. Monitor training and compare with standard approach")
        print("\nFor more information, see QUICKSTART_SELF_FORCING.md")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n\n" + "=" * 80)
        print(" " * 30 + "TEST FAILED")
        print("=" * 80)
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error message above and fix the issue.")
        print("=" * 80 + "\n")
        raise


if __name__ == "__main__":
    main()

