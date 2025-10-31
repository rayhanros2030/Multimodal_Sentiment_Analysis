# Demonstration Guide

This guide explains how to create a demonstration video of the multimodal sentiment analysis system for your MIT Slideroom submission.

## Quick Demo Script

The `demo_single_sample.py` script demonstrates the architecture on a single sample from CMU-MOSI dataset, showing:
- Feature extraction from video, audio, and text
- Model inference
- Visualized results

## Setup for Demo

1. **Prepare Dataset:**
   - Ensure CMU-MOSI dataset is at: `C:\Users\PC\Downloads\CMU-MOSI Dataset`
   - Should contain folders: `MOSI-Videos`, `MOSI-Audios`, `MOSI-Transcript`, and `labels.json`

2. **Install Additional Dependencies:**
   ```bash
   pip install opencv-python
   ```

3. **Train or Load Model:**
   - Either train using `train_mosei_only.py` first
   - Or the demo will use random weights (clearly marked)

## Running the Demo

```bash
python demo_single_sample.py
```

The script will:
1. Find a sample from CMU-MOSI
2. Extract features from video, audio, and transcript
3. Run inference using your trained model
4. Generate two visualizations:
   - `demo_visualization.png` - Detailed technical view
   - `demo_simple.png` - Clean presentation view

## Creating a Demo Video

### Option 1: Screen Recording

1. **Prepare:**
   - Open the video file from CMU-MOSI
   - Run `demo_single_sample.py`
   - Have `demo_simple.png` ready

2. **Record:**
   - Show the input video playing (left side)
   - Show the transcript text (middle)
   - Show the prediction result (right side)
   - Display `demo_simple.png` with prediction highlighted

3. **Narration Script:**
   ```
   "This is a demonstration of our multimodal sentiment analysis system. 
   The model processes three modalities: visual features from the video, 
   audio features from the speech, and text features from the transcript.
   
   Here we see a sample from the CMU-MOSI dataset. The model extracts
   713-dimensional visual features, 74-dimensional audio features, and 
   300-dimensional text features.
   
   These features are encoded through separate modality-specific encoders,
   then fused using cross-modal attention. The final prediction is a 
   sentiment score ranging from -3 (very negative) to +3 (very positive).
   
   As you can see, the model predicted [X.XXX], indicating [sentiment label]."
   ```

### Option 2: Animated Visualization

1. Create a presentation showing:
   - Input modalities (video frame, audio waveform, transcript text)
   - Feature extraction process (show dimensions)
   - Model architecture diagram
   - Prediction result with confidence

2. Use tools like:
   - PowerPoint/Keynote for slides
   - Screen recording software (OBS, Camtasia)
   - Python animations (matplotlib)

### Option 3: Interactive Demo

Show live inference:
- Run the demo script
- Show console output
- Display the visualization
- Explain each step

## What to Highlight in Your Video

1. **Multimodal Nature:**
   - Emphasize that all three modalities (visual, audio, text) are used
   - Show how they're combined

2. **Architecture:**
   - Mention cross-modal attention
   - Explain feature fusion

3. **Results:**
   - Show prediction accuracy if available
   - Display sentiment on scale
   - Compare with ground truth if shown

4. **Technical Details:**
   - Model parameters
   - Feature dimensions
   - Training methodology

## Tips for MIT Slideroom Submission

1. **Keep it concise:** 2-3 minutes maximum
2. **Show results:** Actual predictions on real data
3. **Explain clearly:** What the model does and why it's effective
4. **Highlight innovation:** Cross-modal attention, proper regularization
5. **Show metrics:** Mention correlation and MAE if available

## Output Files

After running the demo:
- `demo_visualization.png` - Technical detailed view
- `demo_simple.png` - Clean presentation view (best for video)
- Console output showing the prediction process

## Example Video Structure

```
[0:00-0:15] Introduction
- Project overview
- Problem statement

[0:15-0:45] Architecture Overview
- Three modalities
- Model structure
- Feature dimensions

[0:45-1:30] Live Demo
- Show sample video
- Extract features
- Run inference
- Display results

[1:30-2:00] Results & Conclusion
- Performance metrics
- Key innovations
- Future work
```

## Troubleshooting

- **Model not found:** Train a model first using `train_mosei_only.py`
- **Video file issues:** Check if files are unzipped or in correct format
- **OpenCV errors:** Install `opencv-python` package
- **Feature extraction slow:** This is normal for the first run

