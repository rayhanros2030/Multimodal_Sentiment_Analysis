# Project Setup Instructions

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Datasets**
   - Download CMU-MOSEI dataset
   - Organize as specified in README.md
   - (Optional) Download IEMOCAP for combined training

3. **Run Training**
   ```bash
   # CMU-MOSEI only
   python train_mosei_only.py
   
   # Or combined datasets
   python train_combined_final.py
   ```

## File Structure

```
multimodal-sentiment-analysis/
├── train_mosei_only.py       # Main training script (CMU-MOSEI only)
├── train_combined_final.py   # Combined CMU-MOSEI + IEMOCAP training
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
├── GITHUB_SETUP.md         # GitHub upload guide
└── SETUP_INSTRUCTIONS.md   # This file
```

## Key Features

- ✅ No data leakage (scalers fitted on train only)
- ✅ Full 100 epochs training
- ✅ Visualization plots (MAE & Correlation)
- ✅ Real feature extraction from datasets
- ✅ Robust error handling

## Output Files

After training, you'll get:
- `best_mosei_model.pth` - Model weights
- `mosei_training_metrics.png` - Training curves
- `mosei_results.json` - Final results
- `mosei_metrics_history.json` - Full history

## Need Help?

Check README.md for detailed documentation or open an issue on GitHub.

