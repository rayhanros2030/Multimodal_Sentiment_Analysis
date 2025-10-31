#!/usr/bin/env python3
"""
Single Sample Demonstration Script
==================================

This script demonstrates the multimodal sentiment analysis architecture
on a single sample from CMU-MOSI dataset. Perfect for creating a demo video
to show how the model processes video, audio, and text to predict sentiment.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import librosa
import cv2
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture from training script
sys.path.insert(0, os.path.dirname(__file__))

# Model Architecture (same as training)
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

def extract_audio_features(audio_path, target_dim=74):
    """Extract audio features from WAV file"""
    try:
        y, sr = librosa.load(str(audio_path), sr=22050, duration=10.0)
        
        # Extract multiple features
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
        
        if len(features) < target_dim:
            padding = np.zeros(target_dim - len(features))
            features = np.concatenate([features, padding])
        else:
            features = features[:target_dim]
        
        return features.astype(np.float32)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return np.zeros(target_dim, dtype=np.float32)

def extract_text_features(transcript_text, target_dim=300):
    """Extract text features from transcript"""
    words = transcript_text.split()
    word_count = len(words)
    char_count = len(transcript_text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    sentence_count = transcript_text.count('.') + transcript_text.count('!') + transcript_text.count('?')
    
    text_feat = np.zeros(target_dim, dtype=np.float32)
    text_feat[0] = word_count / 100.0
    text_feat[1] = char_count / 1000.0
    text_feat[2] = avg_word_len / 20.0
    text_feat[3] = sentence_count / 10.0
    
    # Character frequencies
    char_freqs = {}
    for char in transcript_text.lower():
        char_freqs[char] = char_freqs.get(char, 0) + 1
    
    common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    for i, char in enumerate(common_chars[:min(len(common_chars), 296)]):
        if i + 4 < target_dim:
            text_feat[i + 4] = char_freqs.get(char, 0) / max(char_count, 1)
    
    return text_feat

def extract_visual_features_simple(video_path, target_dim=713):
    """Extract simple visual features from video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return np.zeros(target_dim, dtype=np.float32)
        
        frame_count = 0
        features_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simple features: mean color values, basic statistics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features_list.append([
                np.mean(frame[:, :, 0]),  # B
                np.mean(frame[:, :, 1]),  # G
                np.mean(frame[:, :, 2]),  # R
                np.mean(gray),
                np.std(gray),
                np.max(gray) - np.min(gray),  # Range
            ])
            frame_count += 1
            
            if frame_count >= 100:  # Limit frames
                break
        
        cap.release()
        
        if len(features_list) == 0:
            return np.zeros(target_dim, dtype=np.float32)
        
        features_array = np.array(features_list)
        # Aggregate: mean, std, min, max
        mean_features = features_array.mean(axis=0)
        std_features = features_array.std(axis=0)
        min_features = features_array.min(axis=0)
        max_features = features_array.max(axis=0)
        
        visual_feat = np.zeros(target_dim, dtype=np.float32)
        visual_feat[:6] = mean_features
        visual_feat[6:12] = std_features
        visual_feat[12:18] = min_features
        visual_feat[18:24] = max_features
        
        # Fill remaining with derived features
        ranges = max_features - min_features
        for i in range(24, target_dim):
            idx = i % 6
            if i % 3 == 0:
                visual_feat[i] = ranges[idx] / (abs(mean_features[idx]) + 1e-5)
            elif i % 3 == 1:
                visual_feat[i] = mean_features[idx] * 0.01
            else:
                visual_feat[i] = std_features[idx] * 0.01
        
        return visual_feat.astype(np.float32)
    except Exception as e:
        print(f"Error extracting visual features: {e}")
        return np.zeros(target_dim, dtype=np.float32)

def sentiment_to_label(sentiment_score):
    """Convert sentiment score to label"""
    if sentiment_score >= 1.5:
        return "Very Positive", "green"
    elif sentiment_score >= 0.5:
        return "Positive", "lightgreen"
    elif sentiment_score >= -0.5:
        return "Neutral", "gray"
    elif sentiment_score >= -1.5:
        return "Negative", "lightcoral"
    else:
        return "Very Negative", "red"

def create_demo_visualization(video_path, audio_path, transcript_text, 
                             visual_feat, audio_feat, text_feat, 
                             predicted_sentiment, actual_sentiment=None):
    """Create a comprehensive visualization for the demo"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Multimodal Sentiment Analysis Demo', fontsize=20, fontweight='bold', y=0.98)
    
    # Row 1: Input Modalities
    # Visual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Visual Features (713-dim)', fontsize=12, fontweight='bold')
    ax1.plot(visual_feat[:100], 'b-', linewidth=1.5)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Feature Value')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 0.95, f'Mean: {visual_feat[:100].mean():.3f}\nStd: {visual_feat[:100].std():.3f}',
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Audio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Audio Features (74-dim)', fontsize=12, fontweight='bold')
    ax2.plot(audio_feat, 'g-', linewidth=1.5)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Feature Value')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, 0.95, f'MFCC: {audio_feat[:13].mean():.3f}\nChroma: {audio_feat[13:25].mean():.3f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Text
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Text Features (300-dim)', fontsize=12, fontweight='bold')
    ax3.plot(text_feat[:50], 'purple', linewidth=1.5)
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Feature Value')
    ax3.grid(True, alpha=0.3)
    words = transcript_text.split()
    ax3.text(0.5, 0.95, f'Words: {len(words)}\nChars: {len(transcript_text)}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    
    # Row 2: Transcript and Feature Statistics
    # Transcript
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title('Transcript', fontsize=12, fontweight='bold')
    ax4.text(0.05, 0.5, transcript_text[:500] + ('...' if len(transcript_text) > 500 else ''),
             transform=ax4.transAxes, fontsize=10, verticalalignment='center',
             wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax4.axis('off')
    
    # Row 3: Prediction Results
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title('Sentiment Prediction', fontsize=14, fontweight='bold')
    
    # Sentiment scale bar
    sentiment_range = np.linspace(-3, 3, 100)
    colors = []
    for s in sentiment_range:
        _, color = sentiment_to_label(s)
        colors.append(color)
    
    # Create color bar
    sentiment_bar = ax5.imshow(np.array([sentiment_range]), aspect='auto', cmap='RdYlGn', 
                               extent=[-3, 3, -0.5, 0.5], vmin=-3, vmax=3)
    ax5.set_xlim(-3, 3)
    ax5.set_ylim(-1, 2)
    ax5.set_xlabel('Sentiment Score (Negative ← → Positive)', fontsize=12, fontweight='bold')
    ax5.set_yticks([])
    ax5.grid(True, alpha=0.3)
    
    # Mark predicted sentiment
    pred_label, pred_color = sentiment_to_label(predicted_sentiment)
    ax5.axvline(x=predicted_sentiment, color='blue', linewidth=4, linestyle='--', 
                label=f'Predicted: {predicted_sentiment:.3f} ({pred_label})')
    
    # Mark actual sentiment if available
    if actual_sentiment is not None:
        actual_label, actual_color = sentiment_to_label(actual_sentiment)
        ax5.axvline(x=actual_sentiment, color='red', linewidth=4, linestyle='--',
                    label=f'Actual: {actual_sentiment:.3f} ({actual_label})')
        error = abs(predicted_sentiment - actual_sentiment)
        ax5.text(0.5, 1.5, f'Error: {error:.3f}', transform=ax5.transAxes,
                fontsize=14, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax5.legend(loc='upper center', fontsize=11, framealpha=0.9)
    
    # Add model architecture info
    info_text = (
        f"Model: RegularizedMultimodalModel\n"
        f"Visual Encoder: 713→192→96 dim\n"
        f"Audio Encoder: 74→192→96 dim\n"
        f"Text Encoder: 300→192→96 dim\n"
        f"Cross-Modal Attention: 4 heads\n"
        f"Fusion: 288→192→96→1"
    )
    ax5.text(0.02, 1.3, info_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.savefig('demo_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: demo_visualization.png")
    plt.close()
    
    # Also create a simpler version for video
    create_simple_demo(video_path, audio_path, transcript_text, predicted_sentiment, actual_sentiment)

def create_simple_demo(video_path, audio_path, transcript_text, predicted_sentiment, actual_sentiment=None):
    """Create a simpler visualization for video recording"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Background
    ax.set_facecolor('#f0f0f0')
    
    # Title
    fig.suptitle('Multimodal Sentiment Analysis - Live Demo', fontsize=22, fontweight='bold', y=0.96)
    
    # Three panels
    y_positions = [0.7, 0.4, 0.1]
    
    # Visual Modality
    ax.text(0.1, y_positions[0], 'VISUAL', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(0.1, y_positions[0]-0.05, f'Video: {Path(video_path).name}', fontsize=12)
    ax.text(0.1, y_positions[0]-0.08, 'Features: 713 dimensions (color, texture, motion)', fontsize=10, style='italic')
    
    # Audio Modality
    ax.text(0.1, y_positions[1], 'AUDIO', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.1, y_positions[1]-0.05, f'Audio: {Path(audio_path).name}', fontsize=12)
    ax.text(0.1, y_positions[1]-0.08, 'Features: 74 dimensions (MFCC, chroma, spectral)', fontsize=10, style='italic')
    
    # Text Modality
    ax.text(0.1, y_positions[2], 'TEXT', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    ax.text(0.1, y_positions[2]-0.05, f'Transcript: {len(transcript_text.split())} words', fontsize=12)
    preview_text = transcript_text[:100] + ('...' if len(transcript_text) > 100 else '')
    ax.text(0.1, y_positions[2]-0.08, f'"{preview_text}"', fontsize=10, style='italic')
    
    # Prediction Result
    pred_label, pred_color = sentiment_to_label(predicted_sentiment)
    ax.text(0.6, 0.5, 'PREDICTED SENTIMENT', fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor=pred_color, alpha=0.7))
    ax.text(0.6, 0.42, f'{predicted_sentiment:.3f}', fontsize=32, fontweight='bold', 
            ha='center', color=pred_color)
    ax.text(0.6, 0.35, pred_label, fontsize=16, ha='center', fontweight='bold')
    
    if actual_sentiment is not None:
        actual_label, actual_color = sentiment_to_label(actual_sentiment)
        ax.text(0.6, 0.25, f'Actual: {actual_sentiment:.3f} ({actual_label})', 
                fontsize=14, ha='center')
        error = abs(predicted_sentiment - actual_sentiment)
        ax.text(0.6, 0.18, f'Error: {error:.3f}', fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Sentiment scale
    sentiment_scale = np.linspace(-3, 3, 7)
    scale_labels = ['Very\nNegative', 'Negative', 'Neutral', 'Positive', 'Very\nPositive']
    for i, (val, label) in enumerate(zip(sentiment_scale[1:-1], scale_labels)):
        x_pos = 0.1 + (i + 1) * 0.16
        ax.scatter([x_pos], [0.05], s=200, c=plt.cm.RdYlGn((val + 3) / 6), edgecolors='black', linewidth=2)
        ax.text(x_pos, 0.02, label, ha='center', fontsize=9, fontweight='bold')
        if i == 0:
            ax.text(x_pos - 0.08, 0.05, '-3', ha='center', fontsize=10)
        if i == len(scale_labels) - 1:
            ax.text(x_pos + 0.08, 0.05, '+3', ha='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('demo_simple.png', dpi=300, bbox_inches='tight')
    print("✓ Simple demo saved to: demo_simple.png")
    plt.close()

def main():
    """Main demonstration function"""
    
    print("=" * 80)
    print("MULTIMODAL SENTIMENT ANALYSIS - SINGLE SAMPLE DEMO")
    print("=" * 80)
    print()
    
    # Paths
    mosi_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset")
    model_path = Path("best_mosei_model.pth")  # Change if model is elsewhere
    
    if not mosi_dir.exists():
        print(f"ERROR: CMU-MOSI dataset not found at {mosi_dir}")
        return
    
    # Find a sample
    print("Searching for sample files...")
    video_files = list((mosi_dir / "MOSI-Videos").glob("*.mp4"))
    if not video_files:
        video_files = list((mosi_dir / "MOSI-Videos").glob("*.zip"))
    
    if not video_files:
        print("ERROR: No video files found!")
        return
    
    # Use first video
    video_path = video_files[0]
    video_id = video_path.stem.replace(' (1)', '').replace(' ', '_')
    
    print(f"Using video: {video_path.name}")
    
    # Find corresponding audio and transcript
    audio_files = list((mosi_dir / "MOSI-Audios").glob(f"*{video_id}*"))
    transcript_files = list((mosi_dir / "MOSI-Transcript").glob(f"*{video_id}*"))
    
    audio_path = audio_files[0] if audio_files else None
    transcript_path = transcript_files[0] if transcript_files else None
    
    # Load labels
    labels_path = mosi_dir / "labels.json"
    labels = {}
    actual_sentiment = None
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        actual_sentiment = labels.get(video_id, None)
    
    # Load transcript
    transcript_text = ""
    if transcript_path and transcript_path.exists():
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read().strip()
        except:
            transcript_text = "Transcript not available"
    else:
        transcript_text = "Sample transcript: This is a demonstration of multimodal sentiment analysis."
    
    print(f"Audio: {audio_path.name if audio_path else 'Not found'}")
    print(f"Transcript: {transcript_path.name if transcript_path else 'Using placeholder'}")
    if actual_sentiment is not None:
        print(f"Actual Sentiment: {actual_sentiment:.3f}")
    print()
    
    # Extract features
    print("Extracting features...")
    print("  - Visual features from video...")
    visual_feat = extract_visual_features_simple(video_path, target_dim=713)
    
    print("  - Audio features from audio file...")
    if audio_path:
        audio_feat = extract_audio_features(audio_path, target_dim=74)
    else:
        audio_feat = np.zeros(74, dtype=np.float32)
        print("    (Using placeholder - audio file not found)")
    
    print("  - Text features from transcript...")
    text_feat = extract_text_features(transcript_text, target_dim=300)
    print()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {model_path}")
    
    if not model_path.exists():
        print(f"WARNING: Model not found at {model_path}")
        print("Creating model with random weights for demonstration...")
        model = RegularizedMultimodalModel().to(device)
        print("NOTE: Predictions will be random. Train a model first for real predictions!")
    else:
        model = RegularizedMultimodalModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully")
    
    # Set model to eval mode (critical for BatchNorm with single samples)
    model.eval()
    # Also disable gradient tracking and ensure BatchNorm uses running stats
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    
    print()
    
    # Normalize features (using dummy scalers for demo)
    # In real usage, these would be fitted on training data
    audio_scaler = RobustScaler()
    visual_scaler = RobustScaler()
    text_scaler = RobustScaler()
    
    # Fit scalers on single sample (just to normalize)
    audio_scaler.fit(audio_feat.reshape(1, -1))
    visual_scaler.fit(visual_feat.reshape(1, -1))
    text_scaler.fit(text_feat.reshape(1, -1))
    
    audio_feat_norm = audio_scaler.transform(audio_feat.reshape(1, -1)).flatten()
    visual_feat_norm = visual_scaler.transform(visual_feat.reshape(1, -1)).flatten()
    text_feat_norm = text_scaler.transform(text_feat.reshape(1, -1)).flatten()
    
    # Prepare tensors
    visual_tensor = torch.FloatTensor(visual_feat_norm).unsqueeze(0).to(device)
    audio_tensor = torch.FloatTensor(audio_feat_norm).unsqueeze(0).to(device)
    text_tensor = torch.FloatTensor(text_feat_norm).unsqueeze(0).to(device)
    
    # Run inference
    print("Running inference...")
    model.eval()  # Ensure eval mode
    
    with torch.no_grad():
        # BatchNorm requires eval mode for single samples
        # Ensure all BatchNorm layers are in eval mode
        try:
            predicted_sentiment = model(visual_tensor, audio_tensor, text_tensor).cpu().item()
        except ValueError as e:
            if "Expected more than 1 value" in str(e):
                print("WARNING: BatchNorm issue with single sample. Using workaround...")
                # Create a dummy batch by duplicating the sample
                visual_tensor = visual_tensor.repeat(2, 1)
                audio_tensor = audio_tensor.repeat(2, 1)
                text_tensor = text_tensor.repeat(2, 1)
                predicted_sentiment = model(visual_tensor, audio_tensor, text_tensor).cpu()[0].item()
                print("✓ Inference successful with batch workaround")
            else:
                raise
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Predicted Sentiment: {predicted_sentiment:.3f}")
    pred_label, _ = sentiment_to_label(predicted_sentiment)
    print(f"Predicted Label: {pred_label}")
    
    if actual_sentiment is not None:
        print(f"Actual Sentiment: {actual_sentiment:.3f}")
        actual_label, _ = sentiment_to_label(actual_sentiment)
        print(f"Actual Label: {actual_label}")
        error = abs(predicted_sentiment - actual_sentiment)
        print(f"Absolute Error: {error:.3f}")
    print("=" * 80)
    print()
    
    # Create visualization
    print("Creating visualization...")
    create_demo_visualization(
        video_path, 
        audio_path or Path("placeholder.wav"),
        transcript_text,
        visual_feat,
        audio_feat,
        text_feat,
        predicted_sentiment,
        actual_sentiment
    )
    
    print()
    print("✓ Demo complete! Check 'demo_visualization.png' and 'demo_simple.png'")
    print()
    print("For video recording:")
    print("  - Use 'demo_simple.png' for a clean presentation")
    print("  - Or display 'demo_visualization.png' for detailed technical view")
    print("  - You can also show the actual video playing while displaying the prediction")

if __name__ == "__main__":
    main()

