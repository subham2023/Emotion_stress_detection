@echo off
echo Initializing git repository and pushing to GitHub...
echo.

REM Check if git is available
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/download/win
    pause
    exit /b 1
)

echo Git found!
echo.

REM Initialize git if needed
if not exist .git (
    echo Initializing git repository...
    git init
)

REM Add remote
echo Adding/updating remote origin...
git remote remove origin 2>nul
git remote add origin https://github.com/subham2023/Emotion_stress_detection.git

REM Add all files
echo Adding all files to staging...
git add .

REM Commit
echo Committing changes...
git commit -m "Initial commit: Emotion & Stress Detection System with live tracking"

REM Push
echo.
echo Pushing to GitHub...
echo Note: You may need to authenticate with GitHub.
git branch -M main
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Successfully pushed to GitHub!
    echo Repository: https://github.com/subham2023/Emotion_stress_detection.git
) else (
    echo.
    echo Push failed. Please check:
    echo 1. GitHub authentication (Personal Access Token)
    echo 2. Internet connection
    echo 3. Repository permissions
)

pause

