# GitHub Repository Setup Guide

## Step 1: Initialize Git Repository

Open PowerShell/Command Prompt in the project directory and run:

```bash
cd "C:\Users\PC\Downloads\New_Multimodal_Sentiment_Analysis"

# Initialize git repository
git init

# Add all files (respects .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: Multimodal Sentiment Analysis with CMU-MOSEI"
```

## Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **+** icon in the top right → **New repository**
3. Repository name: `multimodal-sentiment-analysis` (or your preferred name)
4. Description: `Multimodal sentiment analysis using CMU-MOSEI and IEMOCAP datasets`
5. Choose **Public** or **Private**
6. **DO NOT** initialize with README, .gitignore, or license (we already have them)
7. Click **Create repository**

## Step 3: Connect and Push

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/multimodal-sentiment-analysis.git

# Rename default branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Verify

1. Go to your GitHub repository page
2. You should see:
   - README.md with project description
   - train_mosei_only.py
   - train_combined_final.py
   - requirements.txt
   - .gitignore
   - LICENSE

## Optional: Add a Badge

You can add badges to your README.md. Here's an example to add at the top:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

## File Organization Recommendations

If you want to organize better, you could create:

```
multimodal-sentiment-analysis/
├── src/
│   ├── train_mosei_only.py
│   ├── train_combined_final.py
│   └── models/
│       └── (model architecture code)
├── docs/
│   └── (documentation files)
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

But the current flat structure is fine for most projects!

## Tips

- **Commit frequently**: Make commits for each significant change
- **Write clear commit messages**: e.g., "Add visualization plots", "Fix data normalization"
- **Use branches**: Create branches for new features
- **Update README**: Keep README.md updated with latest results/changes

## Common Git Commands

```bash
# Check status
git status

# Add specific files
git add train_mosei_only.py

# Commit changes
git commit -m "Update training script"

# Push changes
git push

# Pull latest changes (if working from multiple machines)
git pull
```

