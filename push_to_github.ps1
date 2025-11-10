# PowerShell script to push project to GitHub
# Repository: https://github.com/subham2023/Emotion_stress_detection.git

Write-Host "Initializing git repository and pushing to GitHub..." -ForegroundColor Green

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check if .git directory exists
if (Test-Path .git) {
    Write-Host "Git repository already initialized." -ForegroundColor Yellow
} else {
    Write-Host "Initializing git repository..." -ForegroundColor Cyan
    git init
}

# Add remote if it doesn't exist
$remoteUrl = "https://github.com/subham2023/Emotion_stress_detection.git"
$remoteExists = git remote get-url origin 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Adding remote origin..." -ForegroundColor Cyan
    git remote add origin $remoteUrl
} else {
    Write-Host "Remote origin already exists: $remoteExists" -ForegroundColor Yellow
    Write-Host "Updating remote URL..." -ForegroundColor Cyan
    git remote set-url origin $remoteUrl
}

# Add all files
Write-Host "Adding all files to staging..." -ForegroundColor Cyan
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    Write-Host "Committing changes..." -ForegroundColor Cyan
    git commit -m "Initial commit: Emotion & Stress Detection System with live tracking"
} else {
    Write-Host "No changes to commit." -ForegroundColor Yellow
}

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
Write-Host "Note: You may need to authenticate with GitHub." -ForegroundColor Yellow
git branch -M main
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "Repository: $remoteUrl" -ForegroundColor Green
} else {
    Write-Host "Push failed. You may need to:" -ForegroundColor Red
    Write-Host "1. Set up GitHub authentication (Personal Access Token)" -ForegroundColor Yellow
    Write-Host "2. Or use SSH instead of HTTPS" -ForegroundColor Yellow
    Write-Host "3. Check your internet connection" -ForegroundColor Yellow
}

