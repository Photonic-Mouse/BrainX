@echo off
title BrainX - Shutdown Utility
cd /d "%~dp0"

echo ======================================================================
echo   BRAINX: CLEAN SHUTDOWN UTILITY
echo ======================================================================
echo.

if exist "venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe -c "
import socket, os, subprocess, sys

port = 8000
try:
    cmd = 'netstat -ano | findstr :8000'
    out = subprocess.check_output(cmd, shell=True, text=True)
    pids = set()
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 5 and 'LISTENING' in line:
            pids.add(parts[-1])
    
    if pids:
        print(f'Found BrainX server process(es) on port 8000: {pids}')
        for pid in pids:
            print(f'Terminating process PID {pid}...')
            subprocess.run(f'taskkill /F /PID {pid}', shell=True)
        print('BrainX server stopped cleanly.')
    else:
        print('No active process found listening on port 8000.')
except Exception as e:
    print('Port 8000 is not active or already stopped.')
"
) else (
    echo Terminating any process on port 8000...
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
        echo Killing PID %%a...
        taskkill /F /PID %%a >nul 2>&1
    )
    echo BrainX server stopped.
)

echo.
echo ======================================================================
pause
