# Using the New Data Format

## Quick Start

The new data format uses **steering** and **throttle** controls instead of discrete WASD keys.

### Option 1: Use the Existing Script with a Simple Change

Just modify the import in any fine-tuning script:

**Before:**
```python
from convert_unreal_data import UnrealDataset
```

**After:**
```python
from process_new_data import NewDataset as UnrealDataset
```

That's it! The `NewDataset` class has the same interface as `UnrealDataset`, so all existing fine-tuning scripts will work.

### Option 2: Run the Test Script

Test that your data is loading correctly:

```bash
python process_new_data.py
```

This will:
- Find all runs across all session folders
- Create training sequences
- Show sample conversions
- Display statistics

### Example: Fine-tune with LoRA

```bash
# Fine-tune the causal distilled model with your new data
python finetune_causal_distilled_lora.py \
    --data_dir /media/kristofe/eight/data/ \
    --sequence_length 9 \
    --batch_size 1 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --lora_rank 16 \
    --checkpoint_dir checkpoints_new_data
```

But first, modify line 20 of `finetune_causal_distilled_lora.py`:
```python
from process_new_data import NewDataset as UnrealDataset
```

### Data Format Conversion

Your new format converts like this:

| Input Format | Range | WASD Output | Value |
|-------------|-------|-------------|-------|
| `steering = -1` | (full left) | `A` | `1.0` |
| `steering = -0.5` | (half left) | `A` | `0.5` |
| `steering = 0` | (neutral) | `A`, `D` | `0, 0` |
| `steering = 0.5` | (half right) | `D` | `0.5` |
| `steering = 1` | (full right) | `D` | `1.0` |
| `throttle = 0.5` | | `W` | `0.5` |
| `throttle = 1.0` | | `W` | `1.0` |
| `brake = 0.5` | (if exists) | `S` | `0.5` |

### Data Location

- **New data:** `/media/kristofe/eight/data/`
- **Old data:** `./data/`

### Dataset Statistics

From your new data:
- **Total runs:** 596
- **Total sequences (30 frames):** 11,324
- **Session folders:** 3 (timestamped folders)
- **Data types:** steering (float, -1 to 1) and throttle (float, 0 to 1)

### Files

- `process_new_data.py` - Data loader for new format
- `convert_unreal_data.py` - Old data loader (discrete WASD)
- `finetune_causal_distilled_lora.py` - LoRA fine-tuning script
- `finetune_causal_distilled.py` - Full fine-tuning script
- `finetune_base_model.py` - Base model fine-tuning script

### Tips

1. **Sequence Length:** Use 9 frames for distilled model, 30-57 for base model
2. **Batch Size:** Start with 1 if you have GPU memory constraints
3. **Learning Rate:** 1e-5 is recommended for fine-tuning
4. **LoRA Rank:** 16 is a good balance (lower = faster, higher = more capacity)

### Troubleshooting

**Issue:** "No sequences found"
- Check that `/media/kristofe/eight/data/` is accessible
- Verify Run_XXXXXX folders contain `input.csv` and `frame_*.png` files

**Issue:** "Out of memory"
- Reduce `--batch_size` to 1
- Reduce `--sequence_length`
- Use LoRA instead of full fine-tuning

**Issue:** "Wrong keyboard dimension"
- The script automatically handles 4-dim WASD format
- Make sure you're using `process_new_data.py` not `convert_unreal_data.py`
