# Push to GitHub Script
# This script helps you push your repository to GitHub

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Upload Helper" -ForegroundColor Cyan
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

Write-Host ""
Write-Host "STEP 1: Create a GitHub Repository" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow
Write-Host "1. Go to https://github.com and sign in" -ForegroundColor White
Write-Host "2. Click the '+' icon → New repository" -ForegroundColor White
Write-Host "3. Repository name: multimodal-sentiment-analysis" -ForegroundColor White
Write-Host "4. Description: Multimodal sentiment analysis using CMU-MOSEI and IEMOCAP datasets" -ForegroundColor White
Write-Host "5. Choose Public or Private" -ForegroundColor White
Write-Host "6. DO NOT initialize with README, .gitignore, or license" -ForegroundColor White
Write-Host "7. Click 'Create repository'" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "STEP 2: Enter your GitHub repository URL (e.g., https://github.com/username/multimodal-sentiment-analysis.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "✗ Repository URL is required!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Connecting to GitHub..." -ForegroundColor Yellow

# Remove existing remote if it exists
git remote remove origin 2>$null

# Add remote
git remote add origin $repoUrl

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Remote repository added" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to add remote repository" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "(This may prompt for your GitHub credentials)" -ForegroundColor Gray

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ SUCCESS! Repository uploaded to GitHub" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your repository is now live at:" -ForegroundColor Cyan
    Write-Host $repoUrl -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ Push failed. Possible issues:" -ForegroundColor Red
    Write-Host "  - Repository URL might be incorrect" -ForegroundColor Yellow
    Write-Host "  - Authentication required (GitHub username/password or token)" -ForegroundColor Yellow
    Write-Host "  - If using 2FA, you need a Personal Access Token instead of password" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "For authentication help:" -ForegroundColor Cyan
    Write-Host "  https://docs.github.com/en/authentication" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To retry manually:" -ForegroundColor Cyan
    Write-Host "  git push -u origin main" -ForegroundColor Yellow
}

