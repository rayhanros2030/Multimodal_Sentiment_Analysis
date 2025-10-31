#!/usr/bin/env python3
"""
Create Demo Video Content
=========================

This script helps create demonstration content for your repository.
It generates visualizations and demo frames that can be used to create a video.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2

def create_video_demo_frame():
    """Create a static demo frame showing the system in action"""
    
    fig = plt.figure(figsize=(16, 9))  # 16:9 aspect ratio for video
    fig.patch.set_facecolor('#1a1a1a')
    
    # Create grid layout
    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3, 
                      left=0.05, right=0.95, top=0.93, bottom=0.07)
    
    # Title
    fig.suptitle('Multimodal Sentiment Analysis - Live Demonstration', 
                 fontsize=24, fontweight='bold', color='white', y=0.98)
    
    # Sample data for visualization
    np.random.seed(42)
    
    # Visual Features Panel
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#2a2a2a')
    visual_feat = np.random.randn(100) * 0.5
    ax1.plot(visual_feat, 'cyan', linewidth=2, alpha=0.8)
    ax1.fill_between(range(len(visual_feat)), visual_feat, alpha=0.3, color='cyan')
    ax1.set_title('VISUAL FEATURES', fontsize=14, fontweight='bold', color='cyan', pad=10)
    ax1.set_xlabel('Feature Dimension (713 total)', color='white', fontsize=10)
    ax1.set_ylabel('Feature Value', color='white', fontsize=10)
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.text(0.5, 0.95, 'OpenFace2 Features\nVideo Frame Analysis',
             transform=ax1.transAxes, ha='center', color='white', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.7))
    
    # Audio Features Panel
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#2a2a2a')
    audio_feat = np.random.randn(74) * 0.3
    ax2.plot(audio_feat, 'lime', linewidth=2, alpha=0.8)
    ax2.fill_between(range(len(audio_feat)), audio_feat, alpha=0.3, color='lime')
    ax2.set_title('AUDIO FEATURES', fontsize=14, fontweight='bold', color='lime', pad=10)
    ax2.set_xlabel('Feature Dimension (74 total)', color='white', fontsize=10)
    ax2.set_ylabel('Feature Value', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.text(0.5, 0.95, 'COVAREP Features\nSpeech Analysis',
             transform=ax2.transAxes, ha='center', color='white', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.7))
    
    # Text Features Panel
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#2a2a2a')
    text_feat = np.random.randn(50) * 0.4
    ax3.plot(text_feat, 'magenta', linewidth=2, alpha=0.8)
    ax3.fill_between(range(len(text_feat)), text_feat, alpha=0.3, color='magenta')
    ax3.set_title('TEXT FEATURES', fontsize=14, fontweight='bold', color='magenta', pad=10)
    ax3.set_xlabel('Feature Dimension (300 total)', color='white', fontsize=10)
    ax3.set_ylabel('Feature Value', color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='white')
    ax3.spines['bottom'].set_color('white')
    ax3.spines['top'].set_color('white')
    ax3.spines['right'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.text(0.5, 0.95, 'GloVe Word Vectors\nTranscript Analysis',
             transform=ax3.transAxes, ha='center', color='white', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.7))
    
    # Architecture Diagram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#2a2a2a')
    ax4.axis('off')
    
    # Draw architecture
    y_positions = [0.8, 0.5, 0.2]
    colors = ['cyan', 'lime', 'magenta']
    labels = ['Visual Encoder', 'Audio Encoder', 'Text Encoder']
    
    for i, (y, color, label) in enumerate(zip(y_positions, colors, labels)):
        # Encoder box
        rect = mpatches.FancyBboxPatch((0.1, y-0.1), 0.3, 0.15, 
                                       boxstyle="round,pad=0.02", 
                                       facecolor=color, edgecolor='white', linewidth=2)
        ax4.add_patch(rect)
        ax4.text(0.25, y-0.025, label, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Arrow to fusion
        ax4.arrow(0.45, y-0.025, 0.15, 0, head_width=0.03, head_length=0.02,
                 fc='white', ec='white', linewidth=2)
    
    # Fusion box
    fusion_rect = mpatches.FancyBboxPatch((0.65, 0.25), 0.25, 0.5,
                                          boxstyle="round,pad=0.02",
                                          facecolor='yellow', edgecolor='white', linewidth=2)
    ax4.add_patch(fusion_rect)
    ax4.text(0.775, 0.7, 'Cross-Modal', ha='center', va='center',
            fontsize=12, fontweight='bold', color='black')
    ax4.text(0.775, 0.6, 'Attention', ha='center', va='center',
            fontsize=12, fontweight='bold', color='black')
    ax4.text(0.775, 0.4, 'Fusion', ha='center', va='center',
            fontsize=12, fontweight='bold', color='black')
    
    # Arrow to output
    ax4.arrow(0.95, 0.5, 0.03, 0, head_width=0.05, head_length=0.02,
             fc='white', ec='white', linewidth=2)
    
    # Output
    output_rect = mpatches.FancyBboxPatch((0.98, 0.45), 0.01, 0.1,
                                          boxstyle="round,pad=0.01",
                                          facecolor='red', edgecolor='white', linewidth=2)
    ax4.add_patch(output_rect)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.text(0.5, 0.95, 'MODEL ARCHITECTURE', ha='center', va='top',
            fontsize=14, fontweight='bold', color='white')
    
    # Prediction Result
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_facecolor('#2a2a2a')
    
    # Sentiment scale
    sentiment_range = np.linspace(-3, 3, 100)
    
    # Create color gradient
    cmap = plt.cm.RdYlGn
    colors_gradient = [cmap((s + 3) / 6) for s in sentiment_range]
    
    # Draw gradient bar
    for i, (s, c) in enumerate(zip(sentiment_range, colors_gradient)):
        ax5.barh(0, 6/100, left=s-3, height=0.3, color=c, edgecolor='none')
    
    # Markers
    predicted = 1.85
    actual = 1.94
    
    # Predicted marker
    ax5.scatter([predicted], [0], s=500, color='blue', marker='^', 
               edgecolors='white', linewidth=3, zorder=10,
               label=f'Predicted: {predicted:.2f}')
    ax5.plot([predicted, predicted], [-0.3, 0.3], 'blue', linewidth=3, linestyle='--', alpha=0.7)
    
    # Actual marker
    ax5.scatter([actual], [0], s=500, color='red', marker='v',
               edgecolors='white', linewidth=3, zorder=10,
               label=f'Actual: {actual:.2f}')
    ax5.plot([actual, actual], [-0.3, 0.3], 'red', linewidth=3, linestyle='--', alpha=0.7)
    
    ax5.set_xlim(-3.2, 3.2)
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_xlabel('Sentiment Score (-3: Very Negative → +3: Very Positive)', 
                   fontsize=14, fontweight='bold', color='white')
    ax5.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax5.set_xticklabels(['-3', '-2', '-1', '0', '+1', '+2', '+3'], color='white', fontsize=12)
    ax5.tick_params(colors='white')
    ax5.spines['bottom'].set_color('white')
    ax5.spines['top'].set_color('white')
    ax5.spines['right'].set_color('white')
    ax5.spines['left'].set_color('white')
    ax5.set_yticks([])
    ax5.grid(True, alpha=0.3, color='white', axis='x')
    
    # Title and info
    ax5.text(0.5, 1.1, 'SENTIMENT PREDICTION', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='white', transform=ax5.transAxes)
    
    error = abs(predicted - actual)
    ax5.text(0.5, 0.7, f'Error: {error:.3f}', ha='center', va='center',
            fontsize=18, fontweight='bold', color='yellow',
            transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8, edgecolor='yellow', linewidth=2))
    
    ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2,
              fontsize=12, framealpha=0.9, facecolor='#1a1a1a', edgecolor='white')
    
    # Footer
    fig.text(0.5, 0.02, 'Multimodal Sentiment Analysis | CMU-MOSEI Dataset | MIT Submission',
            ha='center', fontsize=11, color='white', style='italic')
    
    # Save
    plt.savefig('demo_video_frame.png', dpi=300, facecolor='#1a1a1a', 
                bbox_inches='tight', pad_inches=0.1)
    print("[OK] Demo frame saved to: demo_video_frame.png")
    plt.close()

def create_demo_html():
    """Create an HTML page that can be used as a demo"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Sentiment Analysis - Demo</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(0,0,0,0.3);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 30px;
            opacity: 0.9;
        }
        .demo-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .modality-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid rgba(255,255,255,0.3);
        }
        .modality-card h3 {
            margin-top: 0;
            color: #ffd700;
        }
        .modality-card.visual { border-color: cyan; }
        .modality-card.audio { border-color: lime; }
        .modality-card.text { border-color: magenta; }
        .prediction-box {
            background: rgba(255,215,0,0.2);
            border: 3px solid gold;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin: 30px 0;
        }
        .sentiment-score {
            font-size: 4em;
            font-weight: bold;
            color: #ffd700;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }
        .sentiment-label {
            font-size: 1.5em;
            margin-top: 10px;
        }
        .sentiment-scale {
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }
        .scale-point {
            text-align: center;
            padding: 10px;
            flex: 1;
        }
        .instructions {
            background: rgba(0,0,0,0.4);
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
        }
        .code-block {
            background: rgba(0,0,0,0.5);
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            margin: 10px 0;
        }
        a {
            color: #ffd700;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Multimodal Sentiment Analysis</h1>
        <p class="subtitle">Live Demonstration - CMU-MOSEI Dataset</p>
        
        <div class="demo-section">
            <div class="modality-card visual">
                <h3>📹 Visual Modality</h3>
                <p><strong>Features:</strong> 713 dimensions</p>
                <p><strong>Source:</strong> OpenFace2 facial analysis</p>
                <p><strong>Extraction:</strong> Frame-by-frame video analysis</p>
            </div>
            
            <div class="modality-card audio">
                <h3>🔊 Audio Modality</h3>
                <p><strong>Features:</strong> 74 dimensions</p>
                <p><strong>Source:</strong> COVAREP acoustic features</p>
                <p><strong>Extraction:</strong> MFCC, chroma, spectral features</p>
            </div>
            
            <div class="modality-card text">
                <h3>📝 Text Modality</h3>
                <p><strong>Features:</strong> 300 dimensions</p>
                <p><strong>Source:</strong> GloVe word vectors</p>
                <p><strong>Extraction:</strong> Transcript analysis</p>
            </div>
        </div>
        
        <div class="prediction-box">
            <h2>Predicted Sentiment</h2>
            <div class="sentiment-score">+1.85</div>
            <div class="sentiment-label">Positive</div>
            
            <div class="sentiment-scale">
                <div class="scale-point" style="background: rgba(255,0,0,0.5);">Very Negative<br>-3</div>
                <div class="scale-point" style="background: rgba(255,100,100,0.5);">Negative<br>-1.5</div>
                <div class="scale-point" style="background: rgba(128,128,128,0.5);">Neutral<br>0</div>
                <div class="scale-point" style="background: rgba(100,255,100,0.5);">Positive<br>+1.5</div>
                <div class="scale-point" style="background: rgba(0,255,0,0.5);">Very Positive<br>+3</div>
            </div>
        </div>
        
        <div class="instructions">
            <h2>🚀 Run Your Own Demo</h2>
            <p>To create a demonstration with your own data:</p>
            <div class="code-block">
python demo_single_sample.py
            </div>
            <p>This will:</p>
            <ul>
                <li>Load a sample from CMU-MOSI dataset</li>
                <li>Extract features from video, audio, and transcript</li>
                <li>Run inference using the trained model</li>
                <li>Generate visualization images</li>
            </ul>
            
            <h3>📹 Creating a Demo Video</h3>
            <p>See <a href="DEMO_GUIDE.md">DEMO_GUIDE.md</a> for complete instructions on creating a demonstration video for your MIT Slideroom submission.</p>
            
            <h3>🔗 Repository Links</h3>
            <ul>
                <li><a href="README.md">📖 README</a> - Complete project documentation</li>
                <li><a href="train_mosei_only.py">🔬 Training Script</a> - Main training code</li>
                <li><a href="demo_single_sample.py">🎬 Demo Script</a> - Single sample demonstration</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    with open('demo.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("[OK] Demo HTML page saved to: demo.html")
    print("  You can open this in a browser for an interactive demo")

def main():
    """Main function"""
    print("=" * 80)
    print("CREATING DEMO CONTENT FOR GITHUB")
    print("=" * 80)
    print()
    
    print("1. Creating demo video frame...")
    create_video_demo_frame()
    print()
    
    print("2. Creating interactive HTML demo...")
    create_demo_html()
    print()
    
    print("=" * 80)
    print("[SUCCESS] Demo content created successfully!")
    print("=" * 80)
    print()
    print("Files created:")
    print("  - demo_video_frame.png - Static frame for video (16:9 aspect ratio)")
    print("  - demo.html - Interactive HTML demo page")
    print()
    print("Next steps:")
    print("  1. Use demo_video_frame.png as a reference for your video")
    print("  2. Run 'python demo_single_sample.py' to create actual demo")
    print("  3. Record your screen showing the demo in action")
    print("  4. See DEMO_GUIDE.md for video creation tips")

if __name__ == "__main__":
    main()

