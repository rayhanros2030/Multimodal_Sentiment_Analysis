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

The demo shows:
- Feature extraction from all three modalities
- Model architecture visualization
- Sentiment prediction on a -3 to +3 scale

**Run Your Own Demo:**
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

- **Input Dimensions**: 
  - Visual: 713 (OpenFace2 features)
  - Audio: 74 (COVAREP features)
  - Text: 300 (GloVe word vectors)

- **Architecture**:
  - Modality-specific encoders with BatchNorm and dropout
  - Cross-modal multi-head attention (4 heads)
  - Fusion layers with residual connections
  - Output: Single sentiment score (regression)

- **Hyperparameters**:
  - Hidden dimension: 192
  - Embedding dimension: 96
  - Dropout: 0.7
  - Learning rate: 0.001 (with ReduceLROnPlateau scheduler)
  - Weight decay: 0.05
  - Batch size: 32

## Results

The model is optimized for:
- **Correlation**: Target > 0.3990
- **MAE**: Target < 0.6

Training progress is tracked and visualized with MAE and Correlation plots.

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
- `best_mosei_model.pth` - Best model weights
- `mosei_training_metrics.png` - Training curves visualization
- `mosei_results.json` - Final test metrics
- `mosei_metrics_history.json` - Full epoch-by-epoch history

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- CMU-MOSEI dataset: [CMU Multimodal SDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
- IEMOCAP dataset: [IEMOCAP](https://sail.usc.edu/iemocap/)

## Contact

For questions or issues, please open an issue on GitHub.
