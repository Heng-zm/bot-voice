# ==============================================================================
# 🪟 BOT VOICE — LOCAL WINDOWS RUN SCRIPT (PowerShell)
# ==============================================================================

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " 🤖 Starting Bot Voice on Windows..." -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 1. Check Python installation
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "❌ Error: Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# 2. Check virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

# 3. Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 4. Install / Update dependencies
Write-Host "📦 Verifying dependencies (requirements.txt)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# 5. Check .env file
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "⚠️ No .env file found. Creating .env from .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "📝 Please edit .env with your credentials (TELEGRAM_BOT_TOKEN, etc.)!" -ForegroundColor Magenta
    }
}

# 6. Start the application
Write-Host "🚀 Launching Bot Voice server..." -ForegroundColor Green
python main.py
