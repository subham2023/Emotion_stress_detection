# Instructions to Push to GitHub

## Repository URL
https://github.com/subham2023/Emotion_stress_detection.git

## Option 1: Using PowerShell Script (Recommended)

1. **Install Git** (if not already installed):
   - Download from: https://git-scm.com/download/win
   - Install with default settings

2. **Run the push script**:
   ```powershell
   .\push_to_github.ps1
   ```

3. **If authentication is required**:
   - You'll need a GitHub Personal Access Token
   - Create one at: https://github.com/settings/tokens
   - Use the token as your password when prompted

## Option 2: Manual Git Commands

If you prefer to run commands manually:

```powershell
# Initialize git repository (if not already done)
git init

# Add remote repository
git remote add origin https://github.com/subham2023/Emotion_stress_detection.git
# Or update if it already exists:
git remote set-url origin https://github.com/subham2023/Emotion_stress_detection.git

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Emotion & Stress Detection System with live tracking"

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

## Option 3: Using GitHub Desktop

1. Install GitHub Desktop: https://desktop.github.com/
2. Open GitHub Desktop
3. File → Add Local Repository
4. Select this folder
5. Publish repository → Choose the repository name
6. Click "Publish repository"

## Authentication

If you encounter authentication issues:

### Using Personal Access Token (Recommended)
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control of private repositories)
4. Copy the token
5. When git asks for password, paste the token instead

### Using SSH (Alternative)
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add SSH key to GitHub: https://github.com/settings/keys
3. Change remote URL: `git remote set-url origin git@github.com:subham2023/Emotion_stress_detection.git`

## What Gets Pushed

The `.gitignore` file ensures these are NOT pushed:
- `node_modules/`
- `.env` files
- Build outputs (`dist/`, `build/`)
- Log files
- IDE files
- Database files

All source code, configuration files, and documentation will be pushed.

## Troubleshooting

- **"Git is not recognized"**: Install Git from https://git-scm.com/download/win
- **"Authentication failed"**: Use Personal Access Token instead of password
- **"Repository not found"**: Make sure the repository exists and you have write access
- **"Permission denied"**: Check your GitHub account permissions

