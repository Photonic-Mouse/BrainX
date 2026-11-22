# BrainX PowerShell Launcher
Set-Location -Path $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  BrainX: Multi-Model Brain MRI Analysis System" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  [1/2] Launching Backend & Loading Checkpoints (CUDA/CPU)..." -ForegroundColor Yellow
Write-Host "  [2/2] Opening Web Dashboard at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host ""
Write-Host "  To stop the server, press Ctrl+C or close this window." -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

Start-Process "http://127.0.0.1:8000"
& ".\venv\Scripts\python.exe" run_app.py
