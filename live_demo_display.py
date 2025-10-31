#!/usr/bin/env python3
"""
Live Demo Display - CMU-MOSI Real-Time Processing
==================================================

This script displays a CMU-MOSI video playing with real-time feature extraction
and sentiment prediction overlays. Perfect for screen recording!
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
import librosa
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
    """Extract features from a single frame"""
    if frame is None:
        return np.zeros(713, dtype=np.float32)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features_list = [
        np.mean(frame[:, :, 0]),
        np.mean(frame[:, :, 1]),
        np.mean(frame[:, :, 2]),
        np.mean(gray),
        np.std(gray),
        np.max(gray) - np.min(gray),
    ]
    
    visual_feat = np.zeros(713, dtype=np.float32)
    visual_feat[:6] = features_list
    
    for i in range(6, 713):
        idx = i % 6
        visual_feat[i] = features_list[idx] * (0.1 + 0.1 * np.sin(i * 0.01))
    
    return visual_feat

def draw_prediction_overlay(frame, pred, frame_num, total_frames):
    """Draw prediction overlay on frame"""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    
    # Prediction box (top right)
    box_width = 400
    box_height = 200
    box_x = w - box_width - 20
    box_y = 20
    
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Prediction value
    pred_label = "Very Positive" if pred >= 1.5 else "Positive" if pred >= 0.5 else "Neutral" if pred >= -0.5 else "Negative" if pred >= -1.5 else "Very Negative"
    pred_color = (0, 255, 0) if pred > 0.5 else (0, 165, 255) if pred > 0 else (128, 128, 128) if pred > -0.5 else (0, 0, 255)
    
    cv2.putText(frame, f"Sentiment: {pred:.3f}", (box_x + 10, box_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 2)
    cv2.putText(frame, pred_label, (box_x + 10, box_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)
    
    # Sentiment bar
    bar_width = box_width - 40
    bar_x = box_x + 20
    bar_y = box_y + 120
    
    # Draw sentiment scale bar
    for i in range(bar_width):
        val = -3 + (i / bar_width) * 6
        ratio = (val + 3) / 6
        b = int(255 * (1 - ratio) if ratio < 0.5 else 0)
        g = int(255 * ratio if ratio < 0.5 else 255 * (1 - (ratio - 0.5) * 2))
        r = int(255 * (ratio - 0.5) * 2 if ratio > 0.5 else 0)
        color = (b, g, r)
        cv2.line(frame, (bar_x + i, bar_y), (bar_x + i, bar_y + 30), color, 1)
    
    # Prediction marker
    marker_x = bar_x + int((pred + 3) / 6 * bar_width)
    cv2.circle(frame, (marker_x, bar_y + 15), 8, (255, 255, 255), -1)
    cv2.circle(frame, (marker_x, bar_y + 15), 8, (0, 0, 255), 2)
    
    # Labels
    cv2.putText(frame, "-3", (bar_x, bar_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "0", (bar_x + bar_width // 2 - 5, bar_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "+3", (bar_x + bar_width - 20, bar_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Frame counter
    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Modality indicators
    cv2.putText(frame, "Visual", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "Audio", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Text", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    return frame

def main():
    """Main function - Live demo display"""
    
    print("=" * 80)
    print("LIVE DEMO - CMU-MOSI Real-Time Processing")
    print("=" * 80)
    print()
    print("Press 'q' to quit, 'p' to pause")
    print()
    
    mosi_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset")
    model_path = Path("best_mosei_model.pth")
    
    if not mosi_dir.exists():
        print(f"ERROR: CMU-MOSI dataset not found at {mosi_dir}")
        return
    
    # Find sample
    video_files = list((mosi_dir / "MOSI-Videos").glob("*.mp4"))
    if not video_files:
        video_files = list((mosi_dir / "MOSI-Videos").glob("*.zip"))
        if video_files:
            print(f"Found zip file: {video_files[0].name}")
            print("Please extract the video first")
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
        transcript_text = "This is a demonstration of multimodal sentiment analysis."
    
    # Extract audio and text features (once)
    print("Extracting audio features...")
    audio_feat = extract_audio_features(audio_path) if audio_path else np.zeros(74, dtype=np.float32)
    
    print("Extracting text features...")
    text_feat = extract_text_features(transcript_text)
    
    # Normalize
    audio_scaler = RobustScaler()
    text_scaler = RobustScaler()
    
    audio_feat_norm = audio_scaler.fit_transform(audio_feat.reshape(1, -1)).flatten()
    text_feat_norm = text_scaler.fit_transform(text_feat.reshape(1, -1)).flatten()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {model_path}")
    
    if not model_path.exists():
        print("WARNING: Model not found. Using random weights for demo...")
        model = RegularizedMultimodalModel().to(device)
    else:
        model = RegularizedMultimodalModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    
    # Open video
    print(f"\nOpening video: {Path(video_path).name}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} fps, {total_frames} frames")
    print("\nStarting live demo...")
    print("Window will show video with real-time predictions")
    print()
    
    # Create window
    window_name = "Multimodal Sentiment Analysis - Live Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    frame_count = 0
    predictions_history = []
    visual_scaler = RobustScaler()
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract visual features
            visual_feat = extract_frame_features(frame)
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
                    visual_tensor = visual_tensor.repeat(2, 1)
                    audio_tensor = audio_tensor.repeat(2, 1)
                    text_tensor = text_tensor.repeat(2, 1)
                    pred = model(visual_tensor, audio_tensor, text_tensor).cpu()[0].item()
            
            predictions_history.append(pred)
            frame_count += 1
        
        # Draw overlay
        display_frame = draw_prediction_overlay(frame.copy(), pred, frame_count, total_frames)
        
        # Show transcript (bottom)
        h, w = display_frame.shape[:2]
        transcript_preview = transcript_text[:80] + "..." if len(transcript_text) > 80 else transcript_text
        cv2.rectangle(display_frame, (10, h - 60), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(display_frame, 0.8, display_frame, 0.2, 0)
        cv2.putText(display_frame, f"Transcript: {transcript_preview}", (20, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow(window_name, display_frame)
        
        # Controls
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            # Restart video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            predictions_history = []
            print("Restarted video")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=" * 80)
    print("Demo finished!")
    print(f"Processed {frame_count} frames")
    if predictions_history:
        avg_pred = np.mean(predictions_history)
        print(f"Average sentiment: {avg_pred:.3f}")
    print()
    print("Tip: Use screen recording software (OBS, Camtasia) to record this window")
    print("     for your MIT Slideroom submission video!")

if __name__ == "__main__":
    main()

