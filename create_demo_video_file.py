#!/usr/bin/env python3
"""
Create Actual Demo Video from CMU-MOSI
=======================================

This script creates an actual video file showing a CMU-MOSI sample being processed
with visualizations of feature extraction and sentiment prediction.
Perfect for MIT Slideroom submission!
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import librosa
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Import model architecture
class RegularizedMultimodalModel(nn.Module):
    """Regularized multimodal sentiment analysis model"""
    
    def __init__(self, visual_dim=713, audio_dim=74, text_dim=300, 
                 hidden_dim=192, embed_dim=96, dropout=0.7, num_layers=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        self.visual_encoder = self._create_encoder(visual_dim, hidden_dim, embed_dim, num_layers, dropout)
        self.audio_encoder = self._create_encoder(audio_dim, hidden_dim, embed_dim, num_layers, dropout)
        self.text_encoder = self._create_encoder(text_dim, hidden_dim, embed_dim, num_layers, dropout)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=min(dropout + 0.1, 0.8), batch_first=True)
        
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
    """Extract audio features"""
    try:
        y, sr = librosa.load(str(audio_path), sr=22050, duration=10.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        chroma = librosa.feature.chroma(y=y, sr=sr).mean(axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        features = np.concatenate([mfcc, chroma, [spectral_centroid, spectral_rolloff, zero_crossing, tempo]])
        
        if len(features) < target_dim:
            padding = np.zeros(target_dim - len(features))
            features = np.concatenate([features, padding])
        else:
            features = features[:target_dim]
        
        return features.astype(np.float32)
    except:
        return np.zeros(target_dim, dtype=np.float32)

def extract_text_features(text, target_dim=300):
    """Extract text features"""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    
    text_feat = np.zeros(target_dim, dtype=np.float32)
    text_feat[0] = word_count / 100.0
    text_feat[1] = char_count / 1000.0
    text_feat[2] = avg_word_len / 20.0
    
    char_freqs = {}
    for char in text.lower():
        char_freqs[char] = char_freqs.get(char, 0) + 1
    
    common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    for i, char in enumerate(common_chars[:min(len(common_chars), 296)]):
        if i + 4 < target_dim:
            text_feat[i + 4] = char_freqs.get(char, 0) / max(char_count, 1)
    
    return text_feat

def extract_frame_features(frame):
    """Extract simple features from a single frame"""
    if frame is None:
        return np.zeros(713, dtype=np.float32)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features_list = [
        np.mean(frame[:, :, 0]),  # B
        np.mean(frame[:, :, 1]),  # G
        np.mean(frame[:, :, 2]),  # R
        np.mean(gray),
        np.std(gray),
        np.max(gray) - np.min(gray),
    ]
    
    # Create 713-dim feature vector
    visual_feat = np.zeros(713, dtype=np.float32)
    visual_feat[:6] = features_list
    
    # Fill remaining with derived features
    for i in range(6, 713):
        idx = i % 6
        visual_feat[i] = features_list[idx] * (0.1 + 0.1 * np.sin(i * 0.01))
    
    return visual_feat

def create_demo_video(video_path, audio_path, transcript_text, model_path, output_video="demo_video.mp4"):
    """Create an actual demo video file"""
    
    print("=" * 80)
    print("CREATING DEMO VIDEO FROM CMU-MOSI")
    print("=" * 80)
    print()
    
    # Check inputs
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        return False
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        print(f"WARNING: Model not found. Creating with random weights for demo...")
        model = RegularizedMultimodalModel().to(device)
    else:
        model = RegularizedMultimodalModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    
    # Extract audio and text features once
    print("Extracting audio features...")
    audio_feat = extract_audio_features(audio_path) if audio_path and Path(audio_path).exists() else np.zeros(74, dtype=np.float32)
    
    print("Extracting text features...")
    text_feat = extract_text_features(transcript_text)
    
    # Normalize
    audio_scaler = RobustScaler()
    visual_scaler = RobustScaler()
    text_scaler = RobustScaler()
    
    audio_feat_norm = audio_scaler.fit_transform(audio_feat.reshape(1, -1)).flatten()
    text_feat_norm = text_scaler.fit_transform(text_feat.reshape(1, -1)).flatten()
    
    # Open video
    print(f"Opening video: {Path(video_path).name}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Output video setup
    # Create a larger frame to show video + visualizations
    output_width = 1920
    output_height = 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))
    
    print(f"\nProcessing video frames...")
    print(f"This may take a few minutes...")
    
    frame_count = 0
    visual_features_accumulated = []
    predictions_history = []
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract visual features from frame
        visual_feat = extract_frame_features(frame)
        visual_features_accumulated.append(visual_feat)
        
        # Normalize visual features
        visual_feat_norm = visual_scaler.fit_transform(visual_feat.reshape(1, -1)).flatten()
        
        # Prepare tensors
        visual_tensor = torch.FloatTensor(visual_feat_norm).unsqueeze(0).to(device)
        audio_tensor = torch.FloatTensor(audio_feat_norm).unsqueeze(0).to(device)
        text_tensor = torch.FloatTensor(text_feat_norm).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            try:
                pred = model(visual_tensor, audio_tensor, text_tensor).cpu().item()
            except:
                # Workaround for batch size 1
                visual_tensor = visual_tensor.repeat(2, 1)
                audio_tensor = audio_tensor.repeat(2, 1)
                text_tensor = text_tensor.repeat(2, 1)
                pred = model(visual_tensor, audio_tensor, text_tensor).cpu()[0].item()
        
        predictions_history.append(pred)
        
        # Create composite frame
        composite_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        composite_frame.fill(30)  # Dark background
        
        # Resize video frame to fit on left side
        video_width = output_width // 2
        video_height = output_height
        frame_resized = cv2.resize(frame, (video_width, video_height))
        composite_frame[:, :video_width] = frame_resized
        
        # Create visualization panel on right side
        vis_panel = np.zeros((output_height, output_width - video_width, 3), dtype=np.uint8)
        vis_panel.fill(40)
        
        # Convert to RGB for matplotlib
        vis_panel_rgb = cv2.cvtColor(vis_panel, cv2.COLOR_BGR2RGB)
        
        # Create matplotlib figure
        fig = plt.figure(figsize=((output_width - video_width)/100, output_height/100), dpi=100)
        fig.patch.set_facecolor('#1a1a1a')
        
        gs = plt.GridSpec(3, 1, figure=fig, hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Title
        fig.suptitle('Multimodal Sentiment Analysis - Live Processing', 
                     fontsize=18, fontweight='bold', color='white', y=0.98)
        
        # Panel 1: Features visualization
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#2a2a2a')
        
        # Show feature plots
        ax1.plot(visual_feat_norm[:50], 'cyan', linewidth=2, label='Visual (713-dim)', alpha=0.8)
        ax1.plot([50, 50+len(audio_feat_norm)], [audio_feat_norm.mean()] * 2, 'lime', linewidth=3, label='Audio (74-dim)', alpha=0.8)
        ax1.plot([60, 60+len(text_feat_norm[:20])], [text_feat_norm[:20].mean()] * 2, 'magenta', linewidth=3, label='Text (300-dim)', alpha=0.8)
        
        ax1.set_title('Feature Extraction', fontsize=14, fontweight='bold', color='white', pad=10)
        ax1.set_xlabel('Feature Index', color='white', fontsize=10)
        ax1.set_ylabel('Feature Value', color='white', fontsize=10)
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.2, color='white')
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['right'].set_color('white')
        ax1.spines['left'].set_color('white')
        
        # Panel 2: Prediction over time
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor('#2a2a2a')
        
        if len(predictions_history) > 1:
            frames = range(len(predictions_history))
            ax2.plot(frames, predictions_history, 'yellow', linewidth=2, marker='o', markersize=3)
            ax2.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
            ax2.fill_between(frames, predictions_history, 0, alpha=0.3, color='yellow')
        
        ax2.set_title('Sentiment Prediction Over Time', fontsize=14, fontweight='bold', color='white', pad=10)
        ax2.set_xlabel('Frame Number', color='white', fontsize=10)
        ax2.set_ylabel('Sentiment Score', color='white', fontsize=10)
        ax2.set_ylim(-3, 3)
        ax2.set_xlim(0, max(100, len(predictions_history)))
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.2, color='white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.spines['left'].set_color('white')
        
        # Panel 3: Current prediction
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.set_facecolor('#2a2a2a')
        ax3.axis('off')
        
        # Sentiment scale
        sentiment_range = np.linspace(-3, 3, 100)
        cmap = plt.cm.RdYlGn
        for i, s in enumerate(sentiment_range):
            c = cmap((s + 3) / 6)
            ax3.barh(0.5, 6/100, left=s-3, height=0.3, color=c, edgecolor='none')
        
        # Current prediction marker
        pred_label = "Very Positive" if pred >= 1.5 else "Positive" if pred >= 0.5 else "Neutral" if pred >= -0.5 else "Negative" if pred >= -1.5 else "Very Negative"
        pred_color = 'lime' if pred > 0 else 'red' if pred < 0 else 'gray'
        
        ax3.scatter([pred], [0.5], s=400, color='blue', marker='^', 
                   edgecolors='white', linewidth=3, zorder=10)
        ax3.plot([pred, pred], [0.3, 0.7], 'blue', linewidth=3, linestyle='--', alpha=0.7)
        
        ax3.set_xlim(-3.2, 3.2)
        ax3.set_ylim(0, 1)
        
        # Prediction text
        ax3.text(0.5, 0.9, 'CURRENT PREDICTION', ha='center', va='top',
                fontsize=16, fontweight='bold', color='white', transform=ax3.transAxes)
        ax3.text(0.5, 0.15, f'{pred:.3f}', ha='center', va='bottom',
                fontsize=32, fontweight='bold', color='yellow', transform=ax3.transAxes)
        ax3.text(0.5, 0.05, pred_label, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=pred_color, transform=ax3.transAxes)
        
        ax3.set_xlabel('Sentiment Scale (-3: Very Negative → +3: Very Positive)', 
                      fontsize=12, fontweight='bold', color='white')
        ax3.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax3.set_xticklabels(['-3', '-2', '-1', '0', '+1', '+2', '+3'], color='white', fontsize=10)
        
        # Save figure to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        vis_panel_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        vis_panel_resized = cv2.resize(vis_panel_bgr, (output_width - video_width, output_height))
        
        # Combine frames
        composite_frame[:, video_width:] = vis_panel_resized
        
        # Add frame counter
        cv2.putText(composite_frame, f'Frame: {frame_count}/{total_frames}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(composite_frame)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    out.release()
    
    print()
    print("=" * 80)
    print(f"SUCCESS! Demo video created: {output_video}")
    print("=" * 80)
    print(f"Duration: {frame_count/fps:.1f} seconds")
    print(f"Resolution: {output_width}x{output_height}")
    print(f"FPS: {fps}")
    print()
    print("You can now use this video for your MIT Slideroom submission!")
    
    return True

def main():
    """Main function"""
    
    mosi_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset")
    model_path = Path("best_mosei_model.pth")
    
    if not mosi_dir.exists():
        print(f"ERROR: CMU-MOSI dataset not found at {mosi_dir}")
        return
    
    # Find a sample
    print("Searching for CMU-MOSI sample...")
    video_files = list((mosi_dir / "MOSI-Videos").glob("*.mp4"))
    
    if not video_files:
        # Try zip files (might need extraction)
        video_files = list((mosi_dir / "MOSI-Videos").glob("*.zip"))
        if video_files:
            print(f"Found zip file: {video_files[0].name}")
            print("Note: You may need to extract the video first")
            return
    
    if not video_files:
        print("ERROR: No video files found!")
        return
    
    video_path = video_files[0]
    video_id = video_path.stem.replace(' (1)', '').replace(' ', '_')
    
    print(f"Using video: {video_path.name}")
    
    # Find audio and transcript
    audio_files = list((mosi_dir / "MOSI-Audios").glob(f"*{video_id}*"))
    transcript_files = list((mosi_dir / "MOSI-Transcript").glob(f"*{video_id}*"))
    
    audio_path = audio_files[0] if audio_files else None
    transcript_path = transcript_files[0] if transcript_files else None
    
    # Load transcript
    transcript_text = ""
    if transcript_path and transcript_path.exists():
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read().strip()
        except:
            transcript_text = "Sample transcript for demonstration."
    else:
        transcript_text = "This is a demonstration of multimodal sentiment analysis processing video, audio, and text features."
    
    print(f"Audio: {audio_path.name if audio_path else 'Using placeholder'}")
    print(f"Transcript: {transcript_path.name if transcript_path else 'Using placeholder'}")
    print()
    
    # Create demo video
    output_file = "demo_live_processing.mp4"
    success = create_demo_video(
        video_path,
        audio_path,
        transcript_text,
        model_path,
        output_file
    )
    
    if success:
        print()
        print("Next steps:")
        print(f"  1. Check the video: {output_file}")
        print("  2. Upload to GitHub (if under 100MB) or use GitHub releases")
        print("  3. Or upload to YouTube/Vimeo and link in README")

if __name__ == "__main__":
    main()

