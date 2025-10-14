# Matrix-Game 2.0 Data Preparation Checklist

This checklist will help you prepare training data that gives your fine-tuning the best chance of success, even if your data looks different from the model's training data.

## 📋 Pre-Training Checklist

### 1. Data Requirements Assessment
- [ ] **Video Quality**: Minimum 720p resolution, stable footage
- [ ] **Frame Rate**: 25-30 FPS (preferably 25 FPS to match model)
- [ ] **Video Length**: 5-60 seconds per video (shorter is better for training)
- [ ] **Total Duration**: At least 1-2 hours of video data
- [ ] **Action Diversity**: Multiple different action sequences per video

### 2. Data Collection Strategy
- [ ] **Consistent Environment**: Similar lighting, camera angle, and setting
- [ ] **Smooth Actions**: Avoid jerky or unrealistic movements
- [ ] **Action Variety**: Include all action types you want the model to learn
- [ ] **Edge Cases**: Include some challenging scenarios (rapid direction changes, etc.)
- [ ] **Quality Control**: Remove blurry, corrupted, or unusable videos

## 🎯 Data Formatting Checklist

### 3. Video Preprocessing
- [ ] **Resolution**: Resize to 352×640 pixels (height × width)
- [ ] **Aspect Ratio**: Maintain 16:9 aspect ratio if possible
- [ ] **Color Space**: Convert to RGB if needed
- [ ] **Frame Rate**: Ensure consistent 25 FPS
- [ ] **Video Codec**: Use MP4 with H.264 codec for compatibility

### 4. Action Data Preparation
- [ ] **Action Mapping**: Define clear mapping between your actions and model actions
- [ ] **Temporal Alignment**: Ensure actions are perfectly aligned with video frames
- [ ] **Action Encoding**: Use the correct format for your mode:
  - **Universal**: 4-dim keyboard + 2-dim mouse
  - **GTA Drive**: 2-dim keyboard + 2-dim mouse  
  - **Temple Run**: 7-dim keyboard only

### 5. Data Organization
- [ ] **File Structure**: Organize videos and actions in clear directory structure
- [ ] **Naming Convention**: Use consistent naming (e.g., `video_001.mp4`, `video_001.json`)
- [ ] **Action Files**: Create JSON files with proper action sequences
- [ ] **Metadata**: Include video length, frame count, and action count

## 🔧 Technical Preparation

### 6. Environment Setup
- [ ] **Python Environment**: Set up with all required dependencies
- [ ] **Data Paths**: Ensure all file paths are correct and accessible
- [ ] **Storage Space**: At least 50GB free space for processed data
- [ ] **GPU Memory**: Ensure sufficient VRAM for data loading

### 7. Data Validation
- [ ] **Video Loading**: Test that all videos can be loaded without errors
- [ ] **Action Parsing**: Verify action files are correctly formatted
- [ ] **Temporal Sync**: Check that actions match video frames exactly
- [ ] **Data Integrity**: Run through the data preparation script successfully

## 📊 Data Quality Guidelines

### 8. Visual Quality Standards
- [ ] **Sharpness**: Videos should be clear and in focus
- [ ] **Stability**: Minimize camera shake and jitter
- [ ] **Lighting**: Consistent, well-lit scenes
- [ ] **Background**: Avoid overly cluttered or distracting backgrounds
- [ ] **Motion**: Smooth, natural movements

### 9. Action Quality Standards
- [ ] **Realism**: Actions should be realistic and achievable
- [ ] **Consistency**: Similar actions should produce similar results
- [ ] **Completeness**: All actions should be properly recorded
- [ ] **Smoothness**: Avoid sudden, jerky action changes
- [ ] **Diversity**: Include various action combinations

## 🚀 Optimization Tips

### 10. Data Augmentation (Optional)
- [ ] **Temporal Jittering**: Slightly vary action timing
- [ ] **Action Smoothing**: Apply smoothing to action sequences
- [ ] **Frame Sampling**: Use different frame sampling strategies
- [ ] **Color Jittering**: Slight color variations (if needed)

### 11. Training Data Splits
- [ ] **Train/Validation Split**: 80/20 or 90/10 split
- [ ] **Balanced Splits**: Ensure both splits have similar action distributions
- [ ] **Representative Samples**: Validation set should represent overall data quality

## 📝 Step-by-Step Tutorial

### Step 1: Collect Your Data
1. **Record videos** of your desired interactions
2. **Ensure consistent quality** and environment
3. **Record corresponding actions** (keyboard/mouse inputs)
4. **Aim for 50-100 video sequences** minimum

### Step 2: Preprocess Videos
```bash
# Example using FFmpeg to resize and standardize
ffmpeg -i input_video.mp4 -vf "scale=640:352" -r 25 output_video.mp4
```

### Step 3: Create Action Files
Create JSON files with this structure:
```json
{
  "video_001": {
    "keyboard": ["w", "w", "a", "d", "q", ...],
    "mouse": ["i", "i", "j", "l", "u", ...]
  }
}
```

### Step 4: Use Data Preparation Script
```bash
python prepare_data.py \
    --mode universal \
    --data_type video_files \
    --data_path ./your_videos \
    --action_file ./your_actions.json \
    --output_dir ./prepared_data \
    --output_name training_data
```

### Step 5: Validate Prepared Data
```bash
# Check the prepared data
python -c "
import numpy as np
data = np.load('./prepared_data/training_data.npz')
print(f'Videos: {len(data[\"video_frames\"])}')
print(f'Keyboard actions: {len(data[\"keyboard_actions\"])}')
print(f'Mouse actions: {len(data[\"mouse_actions\"])}')
"
```

## ⚠️ Common Pitfalls to Avoid

### Data Quality Issues
- ❌ **Blurry or low-quality videos**
- ❌ **Inconsistent frame rates**
- ❌ **Misaligned actions and video frames**
- ❌ **Too few training examples**
- ❌ **Actions that don't match video content**

### Technical Issues
- ❌ **Wrong image dimensions** (not 352×640)
- ❌ **Incorrect action encoding**
- ❌ **Missing action files**
- ❌ **Corrupted video files**
- ❌ **Insufficient disk space**

## 🎯 Success Indicators

Your data is ready when you can answer "Yes" to all these questions:

- [ ] **Can I load all videos without errors?**
- [ ] **Do actions perfectly match video frames?**
- [ ] **Is the data diverse enough to represent my use case?**
- [ ] **Are the videos high quality and consistent?**
- [ ] **Do I have enough data (50+ videos minimum)?**
- [ ] **Are the action mappings correct for my mode?**
- [ ] **Does the data preparation script run successfully?**

## 🔄 Iterative Improvement

### After First Training Attempt
1. **Analyze results**: Check generated videos for quality
2. **Identify issues**: Look for patterns in poor generations
3. **Improve data**: Add more examples of problematic cases
4. **Refine actions**: Ensure action mappings are correct
5. **Retrain**: Run fine-tuning again with improved data

### Continuous Improvement
- **Collect more data** based on what the model struggles with
- **Refine action mappings** based on results
- **Improve video quality** where needed
- **Add edge cases** that are important for your use case

## 📞 Getting Help

If you encounter issues:

1. **Check the logs** from the data preparation script
2. **Validate your data format** against the examples
3. **Test with a small subset** first
4. **Check file permissions** and paths
5. **Ensure sufficient resources** (disk space, memory)

Remember: **Quality over quantity**. It's better to have 50 high-quality, well-aligned videos than 200 poor-quality ones.

---

**Next Steps**: Once your data is prepared, run the `simple_finetune.py` script to start training!
