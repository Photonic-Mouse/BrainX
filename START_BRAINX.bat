@echo off
setlocal enabledelayedexpansion
title BrainX - Multi-Model MRI Analysis Launcher
cd /d "%~dp0"

echo ======================================================================
echo   BRAINX: MULTI-MODEL BRAIN MRI ANALYSIS SYSTEM
echo   Automated Demonstration & Production Launcher
echo ======================================================================
echo.

:: 1. Verify Project Root
if not exist "run_app.py" (
    echo [ERROR] Not running from BrainX project root!
    echo Current folder: %CD%
    echo Missing required file: run_app.py
    echo.
    echo Please make sure START_BRAINX.bat is located in the BrainX directory.
    echo.
    pause
    exit /b 1
)

if not exist "src\models\resnet.py" (
    echo [ERROR] Project source files missing! Could not locate src\models\resnet.py.
    echo.
    pause
    exit /b 1
)

:: 2. Verify Virtual Environment
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment missing!
    echo Could not locate Python interpreter at: venv\Scripts\python.exe
    echo.
    echo Please create/configure the virtual environment before launching.
    echo.
    pause
    exit /b 1
)

:: 3. Verify Model Checkpoints
if not exist "checkpoints\resnet18_baseline.pth" (
    echo [ERROR] Missing checkpoint: checkpoints\resnet18_baseline.pth
    echo.
    pause
    exit /b 1
)

if not exist "checkpoints\resnet18_he.pth" (
    echo [ERROR] Missing checkpoint: checkpoints\resnet18_he.pth
    echo.
    pause
    exit /b 1
)

if not exist "checkpoints\resnet18_clahe.pth" (
    echo [ERROR] Missing checkpoint: checkpoints\resnet18_clahe.pth
    echo.
    pause
    exit /b 1
)

:: 4. Launch through Python Orchestrator
.\venv\Scripts\python.exe launcher.py
set LAUNCHER_EXIT_CODE=%ERRORLEVEL%

if %LAUNCHER_EXIT_CODE% neq 0 (
    echo.
    echo ======================================================================
    echo   [ERROR] Launcher exited with code %LAUNCHER_EXIT_CODE%.
    echo   Please inspect the diagnostic messages above.
    echo ======================================================================
    echo.
    pause
    exit /b %LAUNCHER_EXIT_CODE%
)

exit /b 0
