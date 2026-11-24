@echo off
title BrainScope-XAI: Real-Time PyTorch Tensor & Matrix Monitor
cd /d "%~dp0"

echo ======================================================================
echo   BRAINSCOPE-XAI: REAL-TIME PYTORCH TENSOR & MATRIX MONITOR
echo   Streams raw logit vectors, softmax distributions, and Grad-CAM math
echo ======================================================================
echo.

powershell -NoExit -Command "Set-Location '%~dp0'; $logPath = 'results\app_runs\live_tensor_stream.log'; if (!(Test-Path $logPath)) { New-Item -ItemType File -Path $logPath -Force | Out-Null }; Write-Host '=====================================================================' -ForegroundColor Cyan; Write-Host '  [ACTIVE] Real-Time PyTorch Tensor & Linear Algebra Stream Active   ' -ForegroundColor Green; Write-Host '=====================================================================' -ForegroundColor Cyan; Write-Host 'Waiting for MRI scan analysis dispatch from browser...' -ForegroundColor Yellow; Get-Content -Path $logPath -Wait -Tail 40"
