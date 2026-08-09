@echo off
REM Hermes Voice Services - Start Script
REM Usage: start.bat [device]
REM Example: start.bat cuda:0

set PORT=7860
set DEVICE=%1
if "%DEVICE%"=="" set DEVICE=auto

echo ========================================
echo Hermes Voice Services
echo ========================================
echo Port: %PORT%
echo Device: %DEVICE%
echo ========================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

REM Activate venv if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run server
python hermes_voice_server.py --device %DEVICE% --port %PORT%

echo.
echo Server stopped.
pause
