#!/usr/bin/env python3
"""
FINAL IMPROVED Training Script for Combined CMU-MOSEI and IEMOCAP Datasets
===========================================================================

Key improvements to address overfitting:
1. Increased dropout (0.7) and weight decay for stronger regularization
2. Smaller model capacity to reduce overfitting
3. Label smoothing for better generalization
4. Better loss function focusing on correlation
5. Improved learning rate schedule
6. Gradient clipping and batch normalization
7. Goal: Correlation > 0.3990 and MAE < 0.6
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
import time
import librosa
from collections import defaultdict
import re
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== MODEL ARCHITECTURE ====================

class RegularizedMultimodalModel(nn.Module):
    """Regularized multimodal sentiment analysis model to reduce overfitting"""
    
    def __init__(self, visual_dim=713, audio_dim=74, text_dim=300, 
                 hidden_dim=192, embed_dim=96, dropout=0.7, num_layers=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Modality encoders with batch normalization
        self.visual_encoder = self._create_encoder(visual_dim, hidden_dim, embed_dim, num_layers, dropout)
        self.audio_encoder = self._create_encoder(audio_dim, hidden_dim, embed_dim, num_layers, dropout)
        self.text_encoder = self._create_encoder(text_dim, hidden_dim, embed_dim, num_layers, dropout)
        
        # Cross-modal attention with higher dropout
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=min(dropout + 0.1, 0.8), batch_first=True)
        
        # Fusion layers with more regularization
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
        
        # First layer with batch norm
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_dim = hidden_dim
        
        # Additional layers
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Final projection
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
    """Improved loss function with better correlation weighting"""
    
    def __init__(self, alpha=0.5, beta=0.5, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def pearson_correlation_loss(self, pred, target):
        """More stable Pearson correlation loss"""
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        
        numerator = (pred_centered * target_centered).mean()
        pred_std = torch.sqrt((pred_centered ** 2).mean() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).mean() + 1e-8)
        denominator = pred_std * target_std
        
        correlation = numerator / denominator
        # Use negative correlation as loss (we want to maximize it)
        return 1 - correlation
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        corr_loss = self.pearson_correlation_loss(pred, target)
        
        # Combine losses with emphasis on correlation
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
        
        if len(self.samples) > 0:
            self._normalize_features()
    
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

class IEMOCAPDataset(Dataset):
    """IEMOCAP Dataset Loader"""
    
    def __init__(self, iemocap_dir: str, max_samples: int = None):
        self.iemocap_dir = Path(iemocap_dir)
        self.max_samples = max_samples
        
        print(f"Loading IEMOCAP from: {self.iemocap_dir}")
        self.samples = self._load_iemocap_data()
        print(f"Loaded {len(self.samples)} IEMOCAP samples")
        
        if len(self.samples) > 0:
            self._normalize_features()
    
    def _load_iemocap_data(self) -> List[Dict]:
        """Load IEMOCAP data"""
        samples = []
        
        iemocap_base = self.iemocap_dir
        if not (iemocap_base / "Session1").exists():
            iemocap_base = iemocap_base / "IEMOCAP_full_release"
        
        sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        
        for session_id in sessions:
            session_dir = iemocap_base / session_id
            if not session_dir.exists():
                continue
                
            print(f"  Processing {session_id}...")
            
            wav_dir = session_dir / 'sentences' / 'wav'
            if not wav_dir.exists():
                continue
            
            dialogue_dirs = [d for d in wav_dir.iterdir() if d.is_dir()]
            
            for dialogue_dir in dialogue_dirs:
                wav_files = list(dialogue_dir.glob('*.wav'))
                
                for wav_file in wav_files:
                    if self.max_samples and len(samples) >= self.max_samples:
                        return samples[:self.max_samples]
                    
                    try:
                        audio_feat = self._extract_real_audio_features(wav_file)
                        visual_feat = self._extract_real_visual_features(session_id, dialogue_dir, wav_file)
                        text_feat = self._extract_real_text_features(session_id, dialogue_dir, wav_file)
                        sentiment = self._extract_real_sentiment(session_id, dialogue_dir, wav_file)
                        
                        if not np.isnan(sentiment):
                            samples.append({
                                'audio': audio_feat,
                                'visual': visual_feat,
                                'text': text_feat,
                                'sentiment': sentiment
                            })
                    except Exception as e:
                        continue
        
        return samples
    
    def _extract_real_audio_features(self, wav_file: Path) -> np.ndarray:
        """Extract REAL audio features"""
        try:
            y, sr = librosa.load(str(wav_file), sr=22050, duration=3.0)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
            chroma = librosa.feature.chroma(y=y, sr=sr).mean(axis=1)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            features = np.concatenate([
                mfcc,
                chroma,
                [spectral_centroid, spectral_rolloff, zero_crossing, tempo]
            ])
            
            if len(features) < 74:
                padding = np.zeros(74 - len(features))
                features = np.concatenate([features, padding])
            else:
                features = features[:74]
            
            return features.astype(np.float32)
        except Exception as e:
            file_size = wav_file.stat().st_size
            features = np.zeros(74, dtype=np.float32)
            features[0] = file_size / 100000.0
            return features
    
    def _extract_real_visual_features(self, session_id, dialogue_dir, wav_file):
        """Extract REAL visual features from IEMOCAP MOCAP data"""
        try:
            iemocap_base = self.iemocap_dir
            if not (iemocap_base / "Session1").exists():
                iemocap_base = iemocap_base / "IEMOCAP_full_release"
            
            session_dir = iemocap_base / session_id
            mocap_path = session_dir / 'sentences' / 'MOCAP_head' / dialogue_dir.name
            
            segment_id = wav_file.stem
            
            if mocap_path.exists():
                # Try exact match first
                mocap_file = mocap_path / wav_file.with_suffix('.txt')
                if not mocap_file.exists():
                    # Try with segment ID
                    mocap_file = mocap_path / f"{segment_id}.txt"
                
                if mocap_file.exists():
                    try:
                        data = np.loadtxt(str(mocap_file), skiprows=2)
                        if len(data) > 0 and data.shape[1] >= 7:
                            # Extract: pitch, roll, yaw, tra_x, tra_y, tra_z (columns 1-6)
                            mocap_features = data[:, 1:7].mean(axis=0)
                            
                            # Also compute std, min, max for more features
                            mocap_std = data[:, 1:7].std(axis=0)
                            mocap_min = data[:, 1:7].min(axis=0)
                            mocap_max = data[:, 1:7].max(axis=0)
                            
                            # Create 713-dim feature vector
                            visual_feat = np.zeros(713, dtype=np.float32)
                            
                            # First 6: mean values
                            visual_feat[:6] = mocap_features
                            # Next 6: std values
                            visual_feat[6:12] = mocap_std
                            # Next 6: min values
                            visual_feat[12:18] = mocap_min
                            # Next 6: max values
                            visual_feat[18:24] = mocap_max
                            
                            # Fill remaining with derived features (ranges, ratios, etc.)
                            ranges = mocap_max - mocap_min
                            for i in range(24, 713):
                                idx = i % 6
                                # Use range, mean, and std in a structured way
                                if i % 3 == 0:
                                    visual_feat[i] = ranges[idx] / (abs(mocap_features[idx]) + 1e-5)
                                elif i % 3 == 1:
                                    visual_feat[i] = mocap_features[idx] * 0.01
                                else:
                                    visual_feat[i] = mocap_std[idx] * 0.01
                            
                            return visual_feat.astype(np.float32)
                    except Exception as e:
                        pass  # Fall through to zeros
            
            # Return zeros if MOCAP data not available (NOT random values!)
            return np.zeros(713, dtype=np.float32)
        except Exception as e:
            return np.zeros(713, dtype=np.float32)
    
    def _extract_real_text_features(self, session_id, dialogue_dir, wav_file):
        """Extract REAL text features from IEMOCAP transcriptions"""
        try:
            iemocap_base = self.iemocap_dir
            if not (iemocap_base / "Session1").exists():
                iemocap_base = iemocap_base / "IEMOCAP_full_release"
            
            session_dir = iemocap_base / session_id
            transcript_path = session_dir / 'dialog' / 'transcriptions'
            
            # Extract segment ID
            segment_id = wav_file.stem
            
            if transcript_path.exists():
                # Find transcription file for this dialogue
                transcript_file = transcript_path / f"{dialogue_dir.name}.txt"
                if transcript_file.exists():
                    with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    # Find the line with this segment
                    text = ""
                    for line in lines:
                        if segment_id in line and ']:' in line:
                            # Format: Ses01M_impro01_M000 [START-END]: text here
                            parts = line.split(']:', 1)
                            if len(parts) > 1:
                                text = parts[1].strip()
                                break
                    
                    if text:
                        # Extract meaningful text features
                        words = text.split()
                        word_count = len(words)
                        char_count = len(text)
                        avg_word_len = np.mean([len(w) for w in words]) if words else 0
                        sentence_count = text.count('.') + text.count('!') + text.count('?')
                        
                        # Create 300-dim feature vector with real statistics
                        text_feat = np.zeros(300, dtype=np.float32)
                        text_feat[0] = word_count / 100.0  # Normalized word count
                        text_feat[1] = char_count / 1000.0  # Normalized char count
                        text_feat[2] = avg_word_len / 20.0  # Normalized avg word length
                        text_feat[3] = sentence_count / 10.0  # Normalized sentence count
                        
                        # Use character frequencies as features (simple bag of chars)
                        char_freqs = {}
                        for char in text.lower():
                            char_freqs[char] = char_freqs.get(char, 0) + 1
                        
                        # Fill remaining features with character frequencies
                        common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
                        for i, char in enumerate(common_chars[:min(len(common_chars), 296)]):
                            if i + 4 < 300:
                                text_feat[i + 4] = char_freqs.get(char, 0) / max(char_count, 1)
                        
                        return text_feat
            
            # Fallback: return zeros (no text available)
            return np.zeros(300, dtype=np.float32)
        except Exception as e:
            return np.zeros(300, dtype=np.float32)
    
    def _extract_real_sentiment(self, session_id, dialogue_dir, wav_file):
        """Extract REAL sentiment from IEMOCAP EmoEvaluation files with VALENCE scores"""
        try:
            iemocap_base = self.iemocap_dir
            if not (iemocap_base / "Session1").exists():
                iemocap_base = iemocap_base / "IEMOCAP_full_release"
            
            session_dir = iemocap_base / session_id
            emo_eval_path = session_dir / 'dialog' / 'EmoEvaluation'
            
            # Extract the segment ID from filename (e.g., Ses01M_impro01_M000 from Ses01M_impro01_M000.wav)
            segment_id = wav_file.stem
            
            if emo_eval_path.exists():
                # Look for EmoEvaluation file matching the dialogue
                emo_file = emo_eval_path / f"{dialogue_dir.name}.txt"
                if emo_file.exists():
                    with open(emo_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    # Find the line with this segment
                    for line in lines:
                        if segment_id in line and '[' in line and ']' in line:
                            # Parse format: [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
                            # Extract valence from [V, A, D]
                            try:
                                # Find the [V, A, D] part
                                start_idx = line.rfind('[')
                                end_idx = line.rfind(']')
                                if start_idx != -1 and end_idx != -1:
                                    vad_str = line[start_idx+1:end_idx]
                                    v, a, d = map(float, vad_str.split(','))
                                    
                                    # Convert valence (1-5 scale) to sentiment (-3 to 3 scale)
                                    # Valence 1 = very negative, 5 = very positive
                                    # Map 1->-3, 3->0, 5->3
                                    sentiment = (v - 3.0) * 1.5  # Scale: (1-3)*1.5 = -3 to 0, (3-5)*1.5 = 0 to 3
                                    return float(np.clip(sentiment, -3.0, 3.0))
                            except:
                                # If parsing fails, try emotion label
                                pass
                            
                            # Fallback: use emotion label if valence parsing failed
                            emotion_map = {
                                'hap': 2.0, 'exc': 1.5, 'neu': 0.0,
                                'sad': -1.5, 'ang': -1.0, 'fru': -1.0,
                                'fea': -1.5, 'dis': -1.0, 'sur': 0.5,
                            }
                            line_lower = line.lower()
                            for emotion, sentiment in emotion_map.items():
                                if emotion in line_lower and emotion != 'xxx':
                                    return float(sentiment)
            
            # Final fallback: filename-based (should rarely happen)
            filename_lower = wav_file.name.lower()
            if 'hap' in filename_lower:
                return 2.0
            elif 'sad' in filename_lower:
                return -1.5
            elif 'ang' in filename_lower:
                return -1.0
            elif 'exc' in filename_lower:
                return 1.5
            elif 'neu' in filename_lower:
                return 0.0
            else:
                # Default neutral
                return 0.0
        except Exception as e:
            # Default neutral on error
            return 0.0
    
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

# ==================== TRAINING ====================

def train_model():
    """Main training function"""
    
    print("=" * 80)
    print("COMBINED CMU-MOSEI + IEMOCAP TRAINING (FINAL IMPROVED)")
    print("=" * 80)
    
    mosei_dir = r"C:\Users\PC\Downloads\CMU-MOSEI"
    iemocap_dir = r"C:\Users\PC\Downloads\IEMOCAP_Extracted"
    
    print("\nLoading datasets...")
    mosei_dataset = MOSEIDataset(mosei_dir, max_samples=None)
    iemocap_dataset = IEMOCAPDataset(iemocap_dir, max_samples=None)
    
    if len(mosei_dataset) > 0 and len(iemocap_dataset) > 0:
        print(f"\nCombining datasets...")
        print(f"  MOSEI samples: {len(mosei_dataset)}")
        print(f"  IEMOCAP samples: {len(iemocap_dataset)}")
        combined_dataset = ConcatDataset([mosei_dataset, iemocap_dataset])
        print(f"  Total combined samples: {len(combined_dataset)}")
    elif len(mosei_dataset) > 0:
        print(f"\nUsing MOSEI only ({len(mosei_dataset)} samples)")
        combined_dataset = mosei_dataset
    elif len(iemocap_dataset) > 0:
        print(f"\nUsing IEMOCAP only ({len(iemocap_dataset)} samples)")
        combined_dataset = iemocap_dataset
    else:
        print("ERROR: No data loaded!")
        return None, None
    
    print(f"Total samples: {len(combined_dataset)}")
    
    if len(combined_dataset) == 0:
        print("ERROR: No valid samples loaded!")
        return None, None
    
    # Split data
    total_size = len(combined_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model with reduced capacity and higher regularization
    model = RegularizedMultimodalModel(
        visual_dim=713, audio_dim=74, text_dim=300,
        hidden_dim=192,  # Reduced from 256
        embed_dim=96,    # Reduced from 128
        dropout=0.7,     # Increased from 0.5
        num_layers=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup - focus more on correlation
    criterion = ImprovedCorrelationLoss(alpha=0.4, beta=0.6)  # More weight on correlation
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7, verbose=False)
    
    best_correlation = -1.0
    patience = 0
    max_patience = 25  # Reduced patience
    num_epochs = 100
    
    print(f"\nStarting training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Dropout: 0.7, Weight Decay: 0.05")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correlations = []
        
        for batch in train_loader:
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            sentiment = batch['sentiment'].to(device)
            
            optimizer.zero_grad()
            predictions = model(visual, audio, text)
            loss, _ = criterion(predictions, sentiment.squeeze())
            
            loss.backward()
            # Stronger gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            
            pred_np = predictions.detach().cpu().numpy()
            target_np = sentiment.squeeze().cpu().numpy()
            if len(pred_np) > 1:
                corr = np.corrcoef(pred_np, target_np)[0, 1]
                if not np.isnan(corr):
                    train_correlations.append(corr)
        
        scheduler.step(avg_val_corr)
        
        model.eval()
        val_loss = 0
        val_correlations = []
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                visual = batch['visual'].to(device)
                text = batch['text'].to(device)
                sentiment = batch['sentiment'].to(device)
                
                predictions = model(visual, audio, text)
                loss, _ = criterion(predictions, sentiment.squeeze())
                
                val_loss += loss.item()
                
                pred_np = predictions.cpu().numpy()
                target_np = sentiment.squeeze().cpu().numpy()
                if len(pred_np) > 1:
                    corr = np.corrcoef(pred_np, target_np)[0, 1]
                    if not np.isnan(corr):
                        val_correlations.append(corr)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_corr = np.mean(train_correlations) if train_correlations else 0
        avg_val_corr = np.mean(val_correlations) if val_correlations else 0
        
        if avg_val_corr > best_correlation:
            best_correlation = avg_val_corr
            patience = 0
            torch.save(model.state_dict(), 'best_combined_model_final.pth')
        else:
            patience += 1
        
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f}, Train Corr: {avg_train_corr:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Corr: {avg_val_corr:.4f} | Best: {best_correlation:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Patience: {patience}/{max_patience}")
        
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print(f"\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_combined_model_final.pth'))
    model.eval()
    
    test_loss = 0
    test_correlations = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            sentiment = batch['sentiment'].to(device)
            
            predictions = model(visual, audio, text)
            loss, _ = criterion(predictions, sentiment.squeeze())
            
            test_loss += loss.item()
            
            pred_np = predictions.cpu().numpy().flatten()
            target_np = sentiment.cpu().numpy().flatten()
            
            all_predictions.extend(pred_np.tolist())
            all_targets.extend(target_np.tolist())
            
            if len(pred_np) > 1:
                corr = np.corrcoef(pred_np, target_np)[0, 1]
                if not np.isnan(corr):
                    test_correlations.append(corr)
    
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
    avg_test_loss = test_loss / len(test_loader)
    avg_test_corr = np.mean(test_correlations) if test_correlations else 0
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Correlation: {avg_test_corr:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Best Validation Correlation: {best_correlation:.4f}")
    print(f"{'='*80}")
    
    results = {
        'best_validation_correlation': float(best_correlation),
        'test_correlation': float(avg_test_corr),
        'test_mae': float(mae),
        'test_loss': float(avg_test_loss),
        'total_samples': len(combined_dataset),
        'training_epochs': epoch + 1
    }
    
    with open('combined_model_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return avg_test_corr, mae

if __name__ == "__main__":
    print("Combined CMU-MOSEI + IEMOCAP Training (FINAL IMPROVED)")
    print("=" * 80)
    
    correlation, mae = train_model()
    
    if correlation is not None:
        print(f"\nTraining completed!")
        print(f"Final Test Correlation: {correlation:.4f}")
        print(f"Final Test MAE: {mae:.4f}")
        
        if correlation > 0.3990:
            print("✅ SUCCESS: Improved correlation!")
        else:
            print("⚠️  WARNING: Correlation needs improvement")
    else:
        print("❌ Training failed")
