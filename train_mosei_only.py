#!/usr/bin/env python3
"""
CMU-MOSEI Only Training Script with Visualization
==================================================

This script trains the multimodal architecture on CMU-MOSEI only and generates
plots for MAE and Correlation over training epochs.
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import time
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== MODEL ARCHITECTURE ====================

class RegularizedMultimodalModel(nn.Module):
    """Regularized multimodal sentiment analysis model"""
    
    def __init__(self, visual_dim=713, audio_dim=74, text_dim=300, 
                 hidden_dim=192, embed_dim=96, dropout=0.7, num_layers=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Modality encoders
        self.visual_encoder = self._create_encoder(visual_dim, hidden_dim, embed_dim, num_layers, dropout)
        self.audio_encoder = self._create_encoder(audio_dim, hidden_dim, embed_dim, num_layers, dropout)
        self.text_encoder = self._create_encoder(text_dim, hidden_dim, embed_dim, num_layers, dropout)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=min(dropout + 0.1, 0.8), batch_first=True)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def _create_encoder(self, input_dim, hidden_dim, embed_dim, num_layers, dropout):
        """Create encoder with batch normalization"""
        layers = []
        current_dim = input_dim
        
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_dim = hidden_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        layers.append(nn.Linear(hidden_dim, embed_dim))
        layers.append(nn.BatchNorm1d(embed_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, visual, audio, text):
        v_enc = self.visual_encoder(visual)
        a_enc = self.audio_encoder(audio)
        t_enc = self.text_encoder(text)
        
        features = torch.stack([v_enc, a_enc, t_enc], dim=1)
        attended_features, _ = self.cross_attention(features, features, features)
        attended_features = attended_features.reshape(attended_features.size(0), -1)
        
        concat_features = torch.cat([v_enc, a_enc, t_enc], dim=-1)
        output = self.fusion_layers(concat_features)
        
        return output.squeeze(-1)

class ImprovedCorrelationLoss(nn.Module):
    """Improved loss function"""
    
    def __init__(self, alpha=0.4, beta=0.6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def pearson_correlation_loss(self, pred, target):
        """Stable Pearson correlation loss"""
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        
        numerator = (pred_centered * target_centered).mean()
        pred_std = torch.sqrt((pred_centered ** 2).mean() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).mean() + 1e-8)
        denominator = pred_std * target_std
        
        correlation = numerator / denominator
        return 1 - correlation
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        corr_loss = self.pearson_correlation_loss(pred, target)
        
        total_loss = self.alpha * (mse_loss + mae_loss) / 2 + self.beta * corr_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(),
            'corr': corr_loss.item()
        }

# ==================== DATA LOADING ====================

class MOSEIDataset(Dataset):
    """CMU-MOSEI Dataset Loader"""
    
    def __init__(self, mosei_dir: str, max_samples: int = None):
        self.mosei_dir = Path(mosei_dir)
        self.max_samples = max_samples
        
        print(f"Loading CMU-MOSEI from: {self.mosei_dir}")
        self.samples = self._load_mosei_data()
        print(f"Loaded {len(self.samples)} MOSEI samples")
        # IMPORTANT: Do NOT normalize here to avoid data leakage.
        # Normalization will be fit on train split and applied to val/test.
    
    def _load_mosei_data(self) -> List[Dict]:
        """Load MOSEI data"""
        samples = []
        
        visual_path = self.mosei_dir / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd'
        audio_path = self.mosei_dir / 'acoustics' / 'CMU_MOSEI_COVAREP.csd'
        text_path = self.mosei_dir / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd'
        labels_path = self.mosei_dir / 'labels' / 'CMU_MOSEI_Labels.csd'
        
        if not all([visual_path.exists(), audio_path.exists(), text_path.exists(), labels_path.exists()]):
            print(f"ERROR: MOSEI files not found!")
            return []
        
        print("  Loading visual features...")
        visual_data = self._load_csd_file(visual_path, 'OpenFace_2') or self._load_csd_file(visual_path, 'Visual')
        print("  Loading audio features...")
        audio_data = self._load_csd_file(audio_path, 'COVAREP') or self._load_csd_file(audio_path, 'Audio')
        print("  Loading text features...")
        text_data = self._load_csd_file(text_path, 'glove_vectors') or self._load_csd_file(text_path, 'Text')
        print("  Loading labels...")
        labels_data = self._load_csd_file(labels_path, 'All Labels') or self._load_csd_file(labels_path, 'Sentiment')
        
        common_ids = set(visual_data.keys()) & set(audio_data.keys()) & set(text_data.keys()) & set(labels_data.keys())
        print(f"Found {len(common_ids)} common video IDs")
        
        total_attempted = 0
        skipped = 0
        for vid_id in list(common_ids)[:self.max_samples] if self.max_samples else common_ids:
            total_attempted += 1
            try:
                visual_feat = self._extract_features(visual_data[vid_id], 713)
                audio_feat = self._extract_features(audio_data[vid_id], 74)
                text_feat = self._extract_features(text_data[vid_id], 300)
                sentiment = self._extract_sentiment(labels_data[vid_id])
                
                visual_feat = self._clean_features(visual_feat)
                audio_feat = self._clean_features(audio_feat)
                text_feat = self._clean_features(text_feat)
                sentiment = self._clean_sentiment(sentiment)
                
                if (np.all(visual_feat == 0) and np.all(audio_feat == 0) and np.all(text_feat == 0)):
                    skipped += 1
                    continue
                
                samples.append({
                    'audio': audio_feat,
                    'visual': visual_feat,
                    'text': text_feat,
                    'sentiment': sentiment
                })
            except Exception as e:
                skipped += 1
                continue
        
        print(f"Successfully created {len(samples)} valid samples out of {total_attempted} attempted ({skipped} skipped)")
        return samples
    
    def _clean_features(self, features: np.ndarray) -> np.ndarray:
        """Clean features"""
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = np.clip(features, -1000, 1000)
        return features
    
    def _clean_sentiment(self, sentiment: float) -> float:
        """Clean sentiment value"""
        if np.isnan(sentiment) or np.isinf(sentiment):
            return 0.0
        return float(np.clip(sentiment, -3.0, 3.0))
    
    def _load_csd_file(self, path: Path, key: str) -> Dict:
        """Load .csd file"""
        data = {}
        try:
            with h5py.File(path, 'r') as f:
                keys_to_try = [key, key.lower(), key.upper(), 'data']
                found_key = None
                
                for k in keys_to_try:
                    if k in f:
                        found_key = k
                        break
                
                if not found_key:
                    found_key = list(f.keys())[0] if len(f.keys()) > 0 else None
                
                if found_key:
                    feature_group = f[found_key]
                    if 'data' in feature_group:
                        data_group = feature_group['data']
                        for video_id in data_group.keys():
                            try:
                                video_group = data_group[video_id]
                                if 'features' in video_group:
                                    features = video_group['features'][:]
                                    data[video_id] = {'features': features}
                            except Exception:
                                continue
        except Exception as e:
            print(f"  Error loading {path.name}: {e}")
        return data
    
    def _extract_features(self, data: Dict, target_dim: int) -> np.ndarray:
        """Extract and pad features"""
        if data is None or 'features' not in data:
            return np.zeros(target_dim, dtype=np.float32)
        
        features = data['features']
        
        if len(features.shape) > 1:
            features = np.mean(features, axis=0) if features.shape[0] > 1 else features[0]
        
        features = features.flatten()
        
        if len(features) > target_dim:
            features = features[:target_dim]
        elif len(features) < target_dim:
            features = np.pad(features, (0, target_dim - len(features)), mode='constant', constant_values=0)
        
        return features.astype(np.float32)
    
    def _extract_sentiment(self, data: Dict) -> float:
        """Extract sentiment score"""
        if data is None or 'features' not in data:
            return 0.0
        
        features = data['features']
        
        try:
            if len(features.shape) > 1:
                sentiment = float(features[0, 0])
            else:
                sentiment = float(features[0]) if len(features) > 0 else 0.0
        except:
            try:
                sentiment = float(np.mean(features))
            except:
                sentiment = 0.0
        
        return sentiment
    
    def _normalize_features(self):
        """Normalize features"""
        if not self.samples:
            return
        
        audio_features = np.array([s['audio'] for s in self.samples])
        visual_features = np.array([s['visual'] for s in self.samples])
        text_features = np.array([s['text'] for s in self.samples])
        
        audio_features = np.nan_to_num(audio_features, nan=0.0, posinf=10, neginf=-10)
        visual_features = np.nan_to_num(visual_features, nan=0.0, posinf=10, neginf=-10)
        text_features = np.nan_to_num(text_features, nan=0.0, posinf=10, neginf=-10)
        
        audio_features = np.clip(audio_features, -1000, 1000)
        visual_features = np.clip(visual_features, -1000, 1000)
        text_features = np.clip(text_features, -1000, 1000)
        
        self.audio_scaler = RobustScaler()
        self.visual_scaler = RobustScaler()
        self.text_scaler = RobustScaler()
        
        audio_norm = self.audio_scaler.fit_transform(audio_features)
        visual_norm = self.visual_scaler.fit_transform(visual_features)
        text_norm = self.text_scaler.fit_transform(text_features)
        
        audio_norm = np.nan_to_num(audio_norm, nan=0.0, posinf=3, neginf=-3)
        visual_norm = np.nan_to_num(visual_norm, nan=0.0, posinf=3, neginf=-3)
        text_norm = np.nan_to_num(text_norm, nan=0.0, posinf=3, neginf=-3)
        
        for i, sample in enumerate(self.samples):
            sample['audio'] = audio_norm[i]
            sample['visual'] = visual_norm[i]
            sample['text'] = text_norm[i]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'audio': torch.FloatTensor(sample['audio']),
            'visual': torch.FloatTensor(sample['visual']),
            'text': torch.FloatTensor(sample['text']),
            'sentiment': torch.FloatTensor([sample['sentiment']])
        }

class TransformedSubset(torch.utils.data.Subset):
    """Wrap a Subset to apply scalers on-the-fly without leakage."""
    def __init__(self, subset, audio_scaler, visual_scaler, text_scaler):
        super().__init__(subset.dataset, subset.indices)
        self.audio_scaler = audio_scaler
        self.visual_scaler = visual_scaler
        self.text_scaler = text_scaler
        self._base_subset = subset

    def __getitem__(self, idx):
        item = self._base_subset[idx]
        # Convert to numpy, transform, back to torch
        audio = item['audio'].numpy().reshape(1, -1)
        visual = item['visual'].numpy().reshape(1, -1)
        text = item['text'].numpy().reshape(1, -1)
        audio_t = torch.from_numpy(self.audio_scaler.transform(audio).astype(np.float32)).squeeze(0)
        visual_t = torch.from_numpy(self.visual_scaler.transform(visual).astype(np.float32)).squeeze(0)
        text_t = torch.from_numpy(self.text_scaler.transform(text).astype(np.float32)).squeeze(0)
        return {
            'audio': audio_t,
            'visual': visual_t,
            'text': text_t,
            'sentiment': item['sentiment']
        }

# ==================== TRAINING ====================

def calculate_metrics(predictions, targets):
    """Calculate MAE and Correlation"""
    pred_np = predictions.flatten()
    target_np = targets.flatten()
    
    mae = np.mean(np.abs(pred_np - target_np))
    
    if len(pred_np) > 1:
        corr = np.corrcoef(pred_np, target_np)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
    
    return mae, corr

def plot_metrics(train_maes, val_maes, train_corrs, val_corrs, save_path='training_metrics.png'):
    """Plot MAE and Correlation over epochs"""
    epochs = range(len(train_maes))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot MAE
    ax1.plot(epochs, train_maes, 'b-', label='Train MAE', linewidth=2)
    ax1.plot(epochs, val_maes, 'r-', label='Validation MAE', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_title('MAE over Training Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Correlation
    ax2.plot(epochs, train_corrs, 'b-', label='Train Correlation', linewidth=2)
    ax2.plot(epochs, val_corrs, 'r-', label='Validation Correlation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Pearson Correlation', fontsize=12)
    ax2.set_title('Correlation over Training Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.2, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {save_path}")
    plt.close()

def train_model():
    """Main training function"""
    
    print("=" * 80)
    print("CMU-MOSEI ONLY TRAINING WITH VISUALIZATION")
    print("=" * 80)
    
    mosei_dir = r"C:\Users\PC\Downloads\CMU-MOSEI"
    
    print("\nLoading dataset...")
    mosei_dataset = MOSEIDataset(mosei_dir, max_samples=None)
    
    if len(mosei_dataset) == 0:
        print("ERROR: No data loaded!")
        return None, None
    
    print(f"Total samples: {len(mosei_dataset)}")
    
    # Split data
    total_size = len(mosei_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        mosei_dataset, [train_size, val_size, test_size]
    )
    
    # Fit scalers on TRAIN ONLY to avoid leakage
    print("\nFitting scalers on train split...")
    train_audio = []
    train_visual = []
    train_text = []
    for idx in train_dataset.indices:
        s = mosei_dataset[idx]
        train_audio.append(s['audio'].numpy())
        train_visual.append(s['visual'].numpy())
        train_text.append(s['text'].numpy())
    train_audio = np.vstack(train_audio)
    train_visual = np.vstack(train_visual)
    train_text = np.vstack(train_text)
    audio_scaler = RobustScaler().fit(train_audio)
    visual_scaler = RobustScaler().fit(train_visual)
    text_scaler = RobustScaler().fit(train_text)

    # Wrap subsets with transforming views
    train_dataset = TransformedSubset(train_dataset, audio_scaler, visual_scaler, text_scaler)
    val_dataset = TransformedSubset(val_dataset, audio_scaler, visual_scaler, text_scaler)
    test_dataset = TransformedSubset(test_dataset, audio_scaler, visual_scaler, text_scaler)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = RegularizedMultimodalModel(
        visual_dim=713, audio_dim=74, text_dim=300,
        hidden_dim=192, embed_dim=96, dropout=0.7, num_layers=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = ImprovedCorrelationLoss(alpha=0.4, beta=0.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7)
    
    best_correlation = -1.0
    num_epochs = 100
    
    # Track metrics for plotting
    train_maes = []
    val_maes = []
    train_corrs = []
    val_corrs = []
    
    print(f"\nStarting training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for batch in train_loader:
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            sentiment = batch['sentiment'].to(device)
            
            optimizer.zero_grad()
            predictions = model(visual, audio, text)
            loss, _ = criterion(predictions, sentiment.squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.extend(predictions.detach().cpu().numpy())
            train_targets.extend(sentiment.squeeze().cpu().numpy())
        
        # Calculate training metrics
        train_preds = np.array(train_predictions)
        train_tgts = np.array(train_targets)
        train_mae, train_corr = calculate_metrics(train_preds, train_tgts)
        train_maes.append(train_mae)
        train_corrs.append(train_corr)
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                visual = batch['visual'].to(device)
                text = batch['text'].to(device)
                sentiment = batch['sentiment'].to(device)
                
                predictions = model(visual, audio, text)
                loss, _ = criterion(predictions, sentiment.squeeze())
                
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(sentiment.squeeze().cpu().numpy())
        
        # Calculate validation metrics
        val_preds = np.array(val_predictions)
        val_tgts = np.array(val_targets)
        val_mae, val_corr = calculate_metrics(val_preds, val_tgts)
        val_maes.append(val_mae)
        val_corrs.append(val_corr)
        
        # Step scheduler based on validation correlation
        scheduler.step(val_corr)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if val_corr > best_correlation:
            best_correlation = val_corr
            torch.save(model.state_dict(), 'best_mosei_model.pth')
        
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f}, Train MAE: {train_mae:.4f}, Train Corr: {train_corr:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}, Val Corr: {val_corr:.4f} | Best: {best_correlation:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_metrics(train_maes, val_maes, train_corrs, val_corrs, save_path='mosei_training_metrics.png')
    
    # Test evaluation
    print(f"\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_mosei_model.pth'))
    model.eval()
    
    test_loss = 0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            sentiment = batch['sentiment'].to(device)
            
            predictions = model(visual, audio, text)
            loss, _ = criterion(predictions, sentiment.squeeze())
            
            test_loss += loss.item()
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(sentiment.squeeze().cpu().numpy())
    
    test_preds = np.array(test_predictions)
    test_tgts = np.array(test_targets)
    test_mae, test_corr = calculate_metrics(test_preds, test_tgts)
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Best Validation Correlation: {best_correlation:.4f}")
    print(f"{'='*80}")
    
    # Save results
    results = {
        'best_validation_correlation': float(best_correlation),
        'test_correlation': float(test_corr),
        'test_mae': float(test_mae),
        'test_loss': float(avg_test_loss),
        'total_samples': len(mosei_dataset),
        'training_epochs': epoch + 1,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset)
    }
    
    with open('mosei_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics history (convert numpy types to native Python types)
    metrics_history = {
        'train_mae': [float(x) for x in train_maes],
        'val_mae': [float(x) for x in val_maes],
        'train_correlation': [float(x) for x in train_corrs],
        'val_correlation': [float(x) for x in val_corrs]
    }
    
    with open('mosei_metrics_history.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"\nResults saved to: mosei_results.json")
    print(f"Metrics history saved to: mosei_metrics_history.json")
    
    return test_corr, test_mae

if __name__ == "__main__":
    print("CMU-MOSEI Only Training with Visualization")
    print("=" * 80)
    
    correlation, mae = train_model()
    
    if correlation is not None:
        print(f"\nTraining completed!")
        print(f"Final Test Correlation: {correlation:.4f}")
        print(f"Final Test MAE: {mae:.4f}")
        
        if correlation > 0.3990:
            print("✅ SUCCESS: Correlation above target!")
        else:
            print("⚠️  WARNING: Correlation below target (0.3990)")
    else:
        print("❌ Training failed")

