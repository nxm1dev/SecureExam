# SecureExam Desktop - Python Dependencies Setup
# Run this script once after installing SecureExam

$ErrorActionPreference = "Stop"

Write-Host "SecureExam - Installing Python dependencies..." -ForegroundColor Cyan

# Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Python not found. Please install Python 3.11 or 3.12 from https://python.org" -ForegroundColor Red
    exit 1
}

$version = python --version 2>&1
Write-Host "Found: $version"

# Install backend dependencies
Write-Host "`nInstalling backend dependencies..." -ForegroundColor Yellow
$backendDir = Join-Path $PSScriptRoot "backend"
if (Test-Path $backendDir) {
    python -m pip install -r "$backendDir\requirements.txt" --quiet
    Write-Host "  Backend dependencies installed"
}

# Install AI service dependencies
Write-Host "Installing AI service dependencies..." -ForegroundColor Yellow
$aiServiceDir = Join-Path $PSScriptRoot "ai-service"
if (Test-Path $aiServiceDir) {
    python -m pip install -r "$aiServiceDir\requirements.txt" --quiet
    Write-Host "  AI service dependencies installed"
}

# Download InsightFace models
Write-Host "`nDownloading InsightFace models (first run may take a few minutes)..." -ForegroundColor Yellow
$modelCacheDir = Join-Path $PSScriptRoot "ai-service\models_cache"
python -c @"
import os
os.makedirs(r'$modelCacheDir', exist_ok=True)
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_sc', root=r'$modelCacheDir')
    app.prepare(ctx_id=-1)
    print('Models downloaded successfully!')
except Exception as e:
    print(f'Model download warning: {e}')
"@

Write-Host "`nSetup complete! You can now run SecureExam." -ForegroundColor Green
Write-Host "Run: npm run dev (for development) or use the installed .exe"