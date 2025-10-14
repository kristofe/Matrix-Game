# Unreal Engine 5 Data Generation Guide for Matrix-Game 2.0

This guide is specifically tailored for creating high-quality training data using Unreal Engine 5, which is actually ideal for Matrix-Game fine-tuning since the original model was trained on UE-generated data.

## 🎮 Why Unreal Engine 5 is Perfect

- **High-quality visuals** that match the model's training data
- **Precise control** over camera movements and actions
- **Consistent lighting** and rendering
- **Easy action recording** and synchronization
- **Scalable data generation** with procedural content

## 🛠️ UE5 Setup Checklist

### 1. Project Configuration
- [ ] **UE5 Version**: Use UE5.1+ for best performance
- [ ] **Rendering Settings**: Set to High/Epic quality
- [ ] **Resolution**: Configure for 640×352 output (16:9 aspect ratio)
- [ ] **Frame Rate**: Set to 25 FPS (matches Matrix-Game training)
- [ ] **Anti-aliasing**: Enable TAA for smooth edges

### 2. Scene Setup
- [ ] **Environment**: Create or use a detailed 3D environment
- [ ] **Lighting**: Use consistent, realistic lighting (avoid harsh shadows)
- [ ] **Camera**: Set up a first-person or third-person camera
- [ ] **Character/Vehicle**: Add controllable character or vehicle
- [ ] **Physics**: Enable realistic physics for natural movement

## 🎯 Data Generation Strategy

### 3. Action Recording System
Create a Blueprint system to record actions:

```cpp
// Blueprint Action Recording Component
UCLASS()
class MATRIXGAMEDATA_API UActionRecorder : public UActorComponent
{
    GENERATED_BODY()

public:
    // Record keyboard input
    UFUNCTION(BlueprintCallable)
    void RecordKeyboardAction(FString Action, float Value);
    
    // Record mouse input
    UFUNCTION(BlueprintCallable)
    void RecordMouseAction(FVector2D MouseDelta);
    
    // Save actions to JSON
    UFUNCTION(BlueprintCallable)
    void SaveActionsToFile(FString FileName);
    
private:
    TArray<FString> KeyboardActions;
    TArray<FVector2D> MouseActions;
    TArray<float> Timestamps;
};
```

### 4. Camera Control System
Set up precise camera control:

- [ ] **Smooth Movement**: Use interpolation for camera movements
- [ ] **Action Mapping**: Map WASD to movement, mouse to camera rotation
- [ ] **Sensitivity**: Set appropriate mouse sensitivity (not too fast/slow)
- [ ] **Constraints**: Limit camera movement to realistic ranges
- [ ] **Stabilization**: Add slight smoothing to avoid jittery movements

## 📹 Video Recording Setup

### 5. Rendering Configuration
- [ ] **Output Resolution**: 640×352 pixels
- [ ] **Frame Rate**: 25 FPS (exactly)
- [ ] **Video Length**: 5-30 seconds per sequence
- [ ] **Codec**: H.264 for compatibility
- [ ] **Quality**: High bitrate (10+ Mbps)

### 6. Recording Workflow
1. **Start Recording**: Begin video capture
2. **Perform Actions**: Execute planned action sequence
3. **Stop Recording**: End video capture
4. **Save Actions**: Export action data to JSON
5. **Validate**: Check video and action alignment

## 🎮 Action Mapping for UE5

### 7. Universal Mode Actions
Map UE5 inputs to Matrix-Game actions:

```cpp
// Keyboard Actions (4-dimensional)
W Key → [1, 0, 0, 0]  // Forward
S Key → [0, 1, 0, 0]  // Backward  
A Key → [0, 0, 1, 0]  // Left
D Key → [0, 0, 0, 1]  // Right
No Input → [0, 0, 0, 0]  // No movement

// Mouse Actions (2-dimensional)
Mouse X → Camera Yaw (horizontal rotation)
Mouse Y → Camera Pitch (vertical rotation)
```

### 8. GTA Drive Mode Actions
For driving scenarios:

```cpp
// Keyboard Actions (2-dimensional)
W Key → [1, 0]  // Accelerate
S Key → [0, 1]  // Brake/Reverse
No Input → [0, 0]  // Coast

// Mouse Actions (2-dimensional)
A Key → [0, -0.1]  // Turn Left
D Key → [0, 0.1]   // Turn Right
No Input → [0, 0]  // Straight
```

### 9. Temple Run Mode Actions
For endless runner scenarios:

```cpp
// Keyboard Actions (7-dimensional)
W Key → [0, 1, 0, 0, 0, 0, 0]  // Jump
S Key → [0, 0, 1, 0, 0, 0, 0]  // Slide
A Key → [0, 0, 0, 0, 0, 1, 0]  // Left
D Key → [0, 0, 0, 0, 0, 0, 1]  // Right
Z Key → [0, 0, 0, 1, 0, 0, 0]  // Turn Left
C Key → [0, 0, 0, 0, 1, 0, 0]  // Turn Right
No Input → [1, 0, 0, 0, 0, 0, 0]  // Run Straight
```

## 🔧 UE5 Implementation Tips

### 10. Blueprint Implementation
Create a comprehensive action recording system:

```cpp
// In your PlayerController or Pawn Blueprint
Event Tick:
├── Get Input Values
├── Convert to Matrix-Game Format
├── Store in Action Arrays
└── Update Timestamp

Event Save Data:
├── Create JSON Object
├── Add Keyboard Actions
├── Add Mouse Actions
├── Add Timestamps
└── Save to File
```

### 11. Data Quality Controls
- [ ] **Action Smoothing**: Apply slight smoothing to avoid jerky movements
- [ ] **Dead Zone**: Add small dead zone for mouse input
- [ ] **Action Validation**: Check that actions match visual movement
- [ ] **Temporal Sync**: Ensure actions are perfectly aligned with frames
- [ ] **Quality Checks**: Validate video quality before saving

## 📊 Data Generation Workflow

### 12. Systematic Data Collection
1. **Plan Sequences**: Design 50-100 different action sequences
2. **Vary Scenarios**: Include different environments, lighting, weather
3. **Action Diversity**: Mix simple and complex action combinations
4. **Edge Cases**: Include rapid direction changes, stops, starts
5. **Consistency**: Maintain similar camera angles and movement styles

### 13. Automated Data Generation
Create a system for automated data collection:

```cpp
// Automated Data Generation Blueprint
For Each Planned Sequence:
├── Load Environment
├── Set Camera Position
├── Start Recording
├── Execute Action Sequence
├── Stop Recording
├── Save Video + Actions
└── Move to Next Sequence
```

## 🎯 UE5-Specific Optimizations

### 14. Rendering Optimizations
- [ ] **LOD Settings**: Optimize for consistent frame rate
- [ ] **Shadow Quality**: Use appropriate shadow settings
- [ ] **Post-Processing**: Apply consistent post-processing effects
- [ ] **Texture Quality**: Use high-quality textures
- [ ] **Lighting**: Use baked lighting for consistency

### 15. Performance Considerations
- [ ] **Frame Rate Stability**: Maintain consistent 25 FPS
- [ ] **Memory Management**: Clear unused assets between recordings
- [ ] **Disk Space**: Ensure sufficient space for video files
- [ ] **Batch Processing**: Process multiple sequences in batches

## 📁 File Organization

### 16. Directory Structure
```
UE5_Data/
├── videos/
│   ├── sequence_001.mp4
│   ├── sequence_002.mp4
│   └── ...
├── actions/
│   ├── sequence_001.json
│   ├── sequence_002.json
│   └── ...
└── metadata/
    ├── sequence_info.json
    └── generation_log.txt
```

### 17. Action File Format
```json
{
  "sequence_001": {
    "keyboard": ["w", "w", "a", "d", "q", ...],
    "mouse": ["0.1,0.0", "0.0,0.1", "-0.1,0.0", ...],
    "timestamps": [0.0, 0.04, 0.08, 0.12, ...],
    "video_length": 5.0,
    "frame_count": 125
  }
}
```

## 🚀 Advanced UE5 Features

### 18. Procedural Content Generation
- [ ] **Random Environments**: Use procedural generation for variety
- [ ] **Dynamic Lighting**: Vary lighting conditions
- [ ] **Weather Effects**: Add rain, fog, different times of day
- [ ] **Object Placement**: Randomly place objects for variety
- [ ] **Path Generation**: Create varied movement paths

### 19. Data Augmentation
- [ ] **Camera Angles**: Slight variations in camera position
- [ ] **Lighting Changes**: Different lighting conditions
- [ ] **Weather Effects**: Rain, fog, different times
- [ ] **Object Variations**: Different objects in scenes
- [ ] **Action Timing**: Slight variations in action timing

## ✅ UE5 Data Quality Checklist

### 20. Pre-Generation Checks
- [ ] **Scene Quality**: High-quality, detailed environment
- [ ] **Lighting**: Consistent, realistic lighting
- [ ] **Camera Setup**: Proper camera positioning and movement
- [ ] **Action Mapping**: Correct input-to-action mapping
- [ ] **Recording Setup**: Proper video recording configuration

### 21. Post-Generation Validation
- [ ] **Video Quality**: Clear, stable videos at 25 FPS
- [ ] **Action Alignment**: Actions match visual movement
- [ ] **File Integrity**: All files saved correctly
- [ ] **Data Consistency**: Consistent format across all files
- [ ] **Temporal Sync**: Perfect alignment between video and actions

## 🎯 Success Metrics

Your UE5 data is ready when:
- [ ] **Videos are high quality** and visually appealing
- [ ] **Actions are perfectly aligned** with visual movement
- [ ] **Data is diverse** with various scenarios and actions
- [ ] **Files are properly formatted** and organized
- [ ] **Temporal synchronization** is perfect
- [ ] **Data preparation script** runs without errors

## 🔄 Iterative Improvement

### 22. Continuous Refinement
1. **Generate initial dataset** (50-100 sequences)
2. **Test with Matrix-Game** fine-tuning
3. **Analyze results** and identify issues
4. **Improve UE5 setup** based on results
5. **Generate more data** focusing on problem areas
6. **Repeat** until satisfied with results

---

**Next Steps**: Once your UE5 data is generated, use the data preparation script to convert it to Matrix-Game format, then run the fine-tuning script!

**Pro Tip**: UE5's high-quality rendering and precise control make it ideal for generating training data that will work well with Matrix-Game 2.0's fine-tuning process.
