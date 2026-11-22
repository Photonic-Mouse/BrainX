@echo off
title BrainX - Shutdown Utility
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe stop.py
) else (
    python stop.py
)

echo.
pause
