param(
    [string]$VenvPath = ".venv",
    [string]$PythonVersion = "3.11",
    [string]$CacheDir = ".uv-cache"
)

$ErrorActionPreference = "Stop"
$env:UV_CACHE_DIR = (Join-Path (Get-Location) $CacheDir)

Write-Host "Creating virtual environment at $VenvPath with Python $PythonVersion"
uv venv $VenvPath --python $PythonVersion --clear

$python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found at $python"
}

Write-Host "Installing CPU PyTorch"
uv pip install --python $python torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing project dependencies"
uv pip install --python $python -r requirements-local.txt

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate with: .\$VenvPath\Scripts\Activate.ps1"
Write-Host "Run demo with: $python demo.py"
