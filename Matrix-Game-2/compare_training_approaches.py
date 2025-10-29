"""
Comparison utility to visualize the difference between standard training
and self-forcing training approaches.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches


def plot_standard_training():
    """Visualize standard (teacher forcing) training."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Title
    ax.text(0.5, 0.95, 'Standard Training (Teacher Forcing)', 
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    
    # Ground truth frames
    gt_frames = ['GT\nFrame 1', 'GT\nFrame 2', 'GT\nFrame 3', 'GT\nFrame 4']
    for i, frame in enumerate(gt_frames):
        box = FancyBboxPatch((i*0.2 + 0.05, 0.6), 0.15, 0.2,
                             boxstyle="round,pad=0.01", 
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
        ax.add_patch(box)
        ax.text(i*0.2 + 0.125, 0.7, frame, ha='center', va='center', fontsize=10)
    
    # Model predictions (conditioned on GT)
    pred_frames = ['Pred\nFrame 2', 'Pred\nFrame 3', 'Pred\nFrame 4', 'Pred\nFrame 5']
    for i, frame in enumerate(pred_frames):
        box = FancyBboxPatch((i*0.2 + 0.05, 0.3), 0.15, 0.2,
                             boxstyle="round,pad=0.01", 
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
        ax.add_patch(box)
        ax.text(i*0.2 + 0.125, 0.4, frame, ha='center', va='center', fontsize=10)
        
        # Arrow from GT to prediction
        arrow = FancyArrowPatch((i*0.2 + 0.125, 0.6), (i*0.2 + 0.125, 0.52),
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Add labels
    ax.text(0.02, 0.7, 'Ground Truth\n(Conditioning)', ha='left', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(0.02, 0.4, 'Model Output\n(Training)', ha='left', va='center', 
            fontsize=11, fontweight='bold')
    
    # Problem annotation
    ax.text(0.5, 0.1, '❌ Problem: Model never learns to handle its own predictions!\n'
                      'At inference, errors accumulate because model conditions on imperfect predictions.',
            ha='center', va='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8),
            transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    return fig


def plot_self_forcing_training():
    """Visualize self-forcing training."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Title
    ax.text(0.5, 0.95, 'Self-Forcing Training', 
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    
    # First frame is GT (conditioning)
    box = FancyBboxPatch((0.05, 0.6), 0.15, 0.2,
                         boxstyle="round,pad=0.01", 
                         edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box)
    ax.text(0.125, 0.7, 'GT\nFrame 1', ha='center', va='center', fontsize=10)
    
    # Generated frames (from model's distribution)
    gen_frames = ['Gen\nFrame 2', 'Gen\nFrame 3', 'Gen\nFrame 4']
    for i, frame in enumerate(gen_frames):
        box = FancyBboxPatch(((i+1)*0.2 + 0.05, 0.6), 0.15, 0.2,
                             boxstyle="round,pad=0.01", 
                             edgecolor='orange', facecolor='lightyellow', linewidth=2)
        ax.add_patch(box)
        ax.text((i+1)*0.2 + 0.125, 0.7, frame, ha='center', va='center', fontsize=10)
        
        # Arrow from previous to current
        if i == 0:
            start_x = 0.125
        else:
            start_x = i*0.2 + 0.125
        arrow = FancyArrowPatch((start_x + 0.075, 0.7), ((i+1)*0.2 + 0.05, 0.7),
                              arrowstyle='->', mutation_scale=20, linewidth=2, 
                              color='orange', linestyle='--')
        ax.add_patch(arrow)
    
    # Model predictions (trained on generated frames)
    pred_frames = ['Pred\nFrame 2', 'Pred\nFrame 3', 'Pred\nFrame 4', 'Pred\nFrame 5']
    for i, frame in enumerate(pred_frames):
        box = FancyBboxPatch((i*0.2 + 0.05, 0.3), 0.15, 0.2,
                             boxstyle="round,pad=0.01", 
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
        ax.add_patch(box)
        ax.text(i*0.2 + 0.125, 0.4, frame, ha='center', va='center', fontsize=10)
        
        # Arrow from generated to prediction
        arrow = FancyArrowPatch((i*0.2 + 0.125, 0.6), (i*0.2 + 0.125, 0.52),
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Add labels
    ax.text(0.02, 0.7, 'Generated\n(Model Dist)', ha='left', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(0.02, 0.4, 'Model Output\n(Training)', ha='left', va='center', 
            fontsize=11, fontweight='bold')
    
    # Benefit annotation
    ax.text(0.5, 0.1, '✓ Solution: Model learns to handle its own predictions!\n'
                      'Training distribution matches inference distribution → less error accumulation.',
            ha='center', va='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
            transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    return fig


def plot_error_accumulation():
    """Plot error accumulation comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Simulate error accumulation
    frames = np.arange(0, 100, 1)
    
    # Standard training: exponential error growth
    error_standard = 0.01 * (1.05 ** frames) - 0.01
    
    # Self-forcing: much slower error growth
    error_self_forcing = 0.01 * (1.01 ** frames) - 0.01
    
    # Plot
    ax.plot(frames, error_standard, 'r-', linewidth=3, label='Standard Training (Teacher Forcing)')
    ax.plot(frames, error_self_forcing, 'g-', linewidth=3, label='Self-Forcing Training')
    
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, 
               label='Quality Degradation Threshold')
    
    ax.set_xlabel('Frame Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accumulated Error', fontsize=13, fontweight='bold')
    ax.set_title('Error Accumulation: Standard vs Self-Forcing Training', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Annotations
    ax.annotate('Standard training:\nRapid error growth', 
                xy=(50, error_standard[50]), xytext=(50, 2.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    ax.annotate('Self-forcing:\nControlled error growth', 
                xy=(80, error_self_forcing[80]), xytext=(60, 0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 5)
    
    return fig


def plot_training_schedule():
    """Plot curriculum learning schedule."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    epochs = np.arange(0, 20, 1)
    
    # Different schedules
    scheduled = np.ones_like(epochs) * 0.5
    curriculum = np.linspace(0, 0.9, len(epochs))
    full = np.ones_like(epochs)
    full[0] = 0  # Start with GT in epoch 0
    
    ax.plot(epochs, scheduled, 'b-', linewidth=3, marker='o', label='Scheduled (Fixed 0.5)')
    ax.plot(epochs, curriculum, 'g-', linewidth=3, marker='s', label='Curriculum (0→0.9)')
    ax.plot(epochs, full, 'r-', linewidth=3, marker='^', label='Full (Always)')
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Self-Forcing Probability', fontsize=13, fontweight='bold')
    ax.set_title('Self-Forcing Training Schedules', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim(0, 19)
    ax.set_ylim(-0.05, 1.05)
    
    # Add shaded regions
    ax.axhspan(0, 0.3, alpha=0.1, color='red', label='_nolegend_')
    ax.text(10, 0.15, 'Conservative\n(Safer training)', ha='center', fontsize=10, color='red')
    
    ax.axhspan(0.7, 1.0, alpha=0.1, color='green', label='_nolegend_')
    ax.text(10, 0.85, 'Aggressive\n(Better for inference)', ha='center', fontsize=10, color='green')
    
    return fig


def main():
    """Generate all comparison plots."""
    print("Generating comparison visualizations...")
    
    # Create output directory
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    # Generate plots
    fig1 = plot_standard_training()
    fig1.savefig("visualizations/standard_training.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/standard_training.png")
    
    fig2 = plot_self_forcing_training()
    fig2.savefig("visualizations/self_forcing_training.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/self_forcing_training.png")
    
    fig3 = plot_error_accumulation()
    fig3.savefig("visualizations/error_accumulation.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/error_accumulation.png")
    
    fig4 = plot_training_schedule()
    fig4.savefig("visualizations/training_schedules.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/training_schedules.png")
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Standard training conditions on GT → train/test mismatch")
    print("2. Self-forcing conditions on model predictions → matches inference")
    print("3. Result: Significantly reduced error accumulation")
    print("4. Curriculum learning provides stable training progression")
    
    plt.close('all')


if __name__ == "__main__":
    main()

