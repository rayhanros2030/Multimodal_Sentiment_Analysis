# Git Repository Initialization Script
# Run this script to initialize git and prepare for GitHub upload

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Initializing Git Repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Initialize git repository
Write-Host "Initializing git repository..." -ForegroundColor Yellow
git init

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to initialize git repository" -ForegroundColor Red
    exit 1
}

# Add all files
Write-Host "Adding files to git..." -ForegroundColor Yellow
git add .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Files added to staging area" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to add files" -ForegroundColor Red
    exit 1
}

# Create initial commit
Write-Host "Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Multimodal Sentiment Analysis with CMU-MOSEI"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Initial commit created" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to create commit" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Repository Ready!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Create a new repository on GitHub" -ForegroundColor White
Write-Host "2. Run these commands:" -ForegroundColor White
Write-Host ""
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/your-repo-name.git" -ForegroundColor Cyan
Write-Host "   git branch -M main" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "See GITHUB_SETUP.md for detailed instructions!" -ForegroundColor Yellow

