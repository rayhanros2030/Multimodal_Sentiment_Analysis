# Multimodal Sentiment Analysis

A PyTorch implementation of multimodal sentiment analysis using CMU-MOSEI and IEMOCAP datasets. This project combines visual, audio, and text modalities to predict sentiment scores with improved correlation and MAE metrics.

## Features

- **Multimodal Fusion**: Combines visual (OpenFace2), audio (COVAREP), and text (GloVe) features
- **Cross-Modal Attention**: Uses attention mechanisms for better feature interaction
- **Robust Training**: Includes regularization techniques (dropout, batch normalization, weight decay)
- **Visualization**: Generates training curves for MAE and Correlation metrics
- **Data Quality**: Proper data cleaning and normalization to handle missing/NaN values
- **No Data Leakage**: Scalers are fitted only on training data and applied to validation/test sets

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

See `requirements.txt` for full dependencies.

## Live Demo

**Interactive Demo Available:**
- **[View Demo in Browser](demo.html)** - Click to open interactive demonstration
- **Demo Visualization:** [demo_video_frame.png](demo_video_frame.png)

**Live Video Demonstrations:**
- **`live_demo_display.py`** - Real-time display showing CMU-MOSI video with live predictions (perfect for screen recording!)
- **`create_demo_video_file.py`** - Creates an actual MP4 video file showing the processing pipeline

The demo shows:
- Feature extraction from all three modalities
- Model architecture visualization
- Sentiment prediction on a -3 to +3 scale
- Real-time processing of actual CMU-MOSI videos

**Run Live Demo (for screen recording):**
```bash
python live_demo_display.py
```
This opens a window showing the video playing with real-time sentiment predictions overlaid. Perfect for recording with OBS or screen recording software!

**Run Single Sample Demo:**
```bash
python demo_single_sample.py
```

This will process a sample from CMU-MOSI and show the complete pipeline in action.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rayhanros2030/Multimodal_Sentiment_Analysis.git
cd Multimodal_Sentiment_Analysis

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

#### CMU-MOSEI
Download CMU-MOSEI dataset and organize it as:
```
CMU-MOSEI/
  ├── visuals/
  │   └── CMU_MOSEI_VisualOpenFace2.csd
  ├── acoustics/
  │   └── CMU_MOSEI_COVAREP.csd
  ├── languages/
  │   └── CMU_MOSEI_TimestampedWordVectors.csd
  └── labels/
      └── CMU_MOSEI_Labels.csd
```

#### IEMOCAP (Optional)
For combined training, extract IEMOCAP dataset to:
```
IEMOCAP_Extracted/
  └── IEMOCAP_full_release/
      ├── Session1/
      ├── Session2/
      └── ...
```

### Training

#### CMU-MOSEI Only
```bash
python train_mosei_only.py
```

This will:
- Train for 100 epochs (no early stopping)
- Generate `mosei_training_metrics.png` with MAE and Correlation plots
- Save `best_mosei_model.pth` (best model weights)
- Save `mosei_results.json` (final test results)
- Save `mosei_metrics_history.json` (full training history)

#### Combined CMU-MOSEI + IEMOCAP
```bash
python train_combined_final.py
```

## Model Architecture

### Input Dimensions
- **Visual**: 713 dimensions (OpenFace2 features)
- **Audio**: 74 dimensions (COVAREP features)
- **Text**: 300 dimensions (GloVe word vectors)

### Architecture Overview

```
Input Features:
    Visual (713)    Audio (74)    Text (300)
         |              |              |
         v              v              v
    Encoder         Encoder         Encoder
    (192→96)        (192→96)        (192→96)
         |              |              |
         +--------------+--------------+
                        |
                        v
              Cross-Modal Attention
                   (4 heads)
                        |
                        v
                  Fusion Layers
              (288→192→96→1)
                        |
                        v
              Sentiment Score (-3 to +3)
```

### Key Components
- **Modality-Specific Encoders**: Each modality (visual, audio, text) is processed through separate encoders with BatchNorm and dropout
- **Cross-Modal Attention**: Multi-head attention mechanism (4 heads) enables the model to learn relationships between different modalities
- **Fusion Layers**: Hierarchical fusion with residual connections combines the attended features
- **Output**: Single sentiment score regression in range [-3, +3]

### Hyperparameters
- Hidden dimension: 192
- Embedding dimension: 96
- Dropout: 0.7
- Learning rate: 0.001 (with ReduceLROnPlateau scheduler)
- Weight decay: 0.05
- Batch size: 32

### Architecture Diagram

![Model Architecture](architecture_diagram.png)

The diagram above illustrates the complete architecture flow from input modalities through encoding, cross-modal attention, and fusion to the final sentiment prediction.

## Results

### Performance Metrics

After training on CMU-MOSEI dataset for 100 epochs:

- **Test Loss**: 0.6112
- **Test Correlation**: 0.4113 (Target: >0.3990) - Achieved
- **Test MAE**: 0.5984 (Target: <0.6) - Achieved
- **Best Validation Correlation**: 0.4799

### Training Statistics

- **Training Dataset**: CMU-MOSEI (3,292 samples)
- **Model Parameters**: 777,345
- **Training Epochs**: 100
- **Final Model**: Best model selected based on validation correlation

### Model Performance Analysis

The model successfully achieves both target metrics:
- Test correlation of 0.4113 exceeds the target of 0.3990, demonstrating strong alignment between predicted and actual sentiment scores
- Test MAE of 0.5984 meets the target of under 0.6, showing accurate sentiment prediction
- The model shows good generalization with validation correlation reaching 0.4799 during training

Training progress is tracked and visualized with MAE and Correlation plots showing the evolution of both training and validation metrics over 100 epochs.

## Project Structure

```
multimodal-sentiment-analysis/
├── train_mosei_only.py          # Main training script (CMU-MOSEI only)
├── train_combined_final.py      # Combined CMU-MOSEI + IEMOCAP training
├── demo_single_sample.py        # Single sample demonstration script
├── create_demo_video.py         # Generate demo content
├── demo.html                     # Interactive HTML demo
├── demo_video_frame.png         # Demo visualization frame
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── DEMO_GUIDE.md                 # Demo video creation guide
└── .gitignore                   # Git ignore rules
```

## Key Features

### Data Quality
- Cleans NaN/Inf values instead of filtering samples
- RobustScaler normalization fitted on training data only
- Proper handling of missing features

### Training
- 100 epochs (no early stopping for `train_mosei_only.py`)
- Correlation-focused loss function (40% MSE/MAE, 60% correlation loss)
- Gradient clipping for stability
- Learning rate scheduling

### Visualization
- Real-time training curves
- Separate plots for MAE and Correlation
- Train vs Validation comparison

## Output Files

After training, you'll get:
- `best_mosei_model.pth` - Best model weights (saved based on validation correlation)
- `mosei_training_metrics.png` - Training curves visualization showing MAE and Correlation over epochs
- `mosei_results.json` - Final test metrics including loss, correlation, and MAE
- `mosei_metrics_history.json` - Full epoch-by-epoch history for analysis

### Training Visualization

### Training Visualization

![Training Metrics](mosei_training_metrics.png)

The training process generates comprehensive visualizations:
- **MAE Plot**: Shows training and validation Mean Absolute Error over 100 epochs, demonstrating the model's learning progression
- **Correlation Plot**: Displays Pearson Correlation progression for both training and validation sets, showing alignment with ground truth
- **Training Curves**: Visual representation of model learning progress, validation plateau at correlation 0.48, and test performance achieving 0.4113 correlation with 0.5984 MAE

The graphs show that while training metrics continue to improve throughout 100 epochs, validation metrics plateau early, indicating the model achieves optimal generalization early in training.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- CMU-MOSEI dataset: [CMU Multimodal SDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
- IEMOCAP dataset: [IEMOCAP](https://sail.usc.edu/iemocap/)

## Contact

For questions or issues, please open an issue on GitHub.
