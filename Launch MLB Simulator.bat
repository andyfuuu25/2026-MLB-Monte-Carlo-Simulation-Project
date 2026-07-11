@echo off
rem =====================================================================
rem  MLB Hybrid Simulator - one-click launcher
rem  Double-click this file. It checks Python, installs any missing
rem  dependencies on first run, starts the local server, and opens the
rem  dashboard in your browser. Close this window to stop the app.
rem =====================================================================
title MLB Hybrid Simulator
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Python was not found on this computer.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   ^(check "Add python.exe to PATH" during install^), then
    echo   double-click this launcher again.
    echo.
    pause
    exit /b 1
)

python -c "import flask, pandas, numpy, sklearn, matplotlib, requests, pybaseball" >nul 2>nul
if errorlevel 1 (
    echo First run: installing dependencies ^(one-time, ~1 minute^)...
    python -m pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   Dependency install failed - check your internet connection
        echo   and try again.
        echo.
        pause
        exit /b 1
    )
)

echo Starting MLB Hybrid Simulator... your browser will open shortly.
echo (First launch fetches league data and takes a few extra seconds.)
echo.
python app.py
if errorlevel 1 pause
