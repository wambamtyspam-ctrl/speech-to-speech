@echo off
REM Hermes Voice Services - Stop Script
REM Usage: stop.bat

echo Stopping Hermes Voice Services...

REM Find and kill process on port 7860
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do (
    echo Killing PID: %%a
    taskkill /F /PID %%a
)

echo Done.
