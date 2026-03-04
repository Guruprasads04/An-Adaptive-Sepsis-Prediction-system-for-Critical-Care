@echo off
REM Sepsis Prediction API - Quick Start Script (Windows)
REM Usage: run_api.bat

setlocal enabledelayedexpansion

echo ======================================================================
echo   SEPSIS PREDICTION SYSTEM - API QUICK START
echo ======================================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] pip upgraded

REM Install dependencies
echo [INFO] Installing dependencies from requirements_api.txt...
pip install -r requirements_api.txt >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
) else (
    echo [OK] Dependencies installed
)

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo [INFO] Creating .env file...
    copy .env.example .env >nul 2>&1
    echo [OK] .env file created
) else (
    echo [OK] .env file already exists
)

REM Create logs directory if it doesn't exist
if not exist "logs" (
    mkdir logs
    echo [OK] Logs directory created
)

REM Display start options
echo.
echo ======================================================================
echo Choose how to start the API:
echo ======================================================================
echo.
echo 1) Development mode (with auto-reload)
echo 2) Production mode (no reload)
echo 3) Run tests
echo 4) Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo [INFO] Starting API in development mode...
    echo [INFO] Access at: http://localhost:8000
    echo [INFO] Docs at: http://localhost:8000/docs
    echo.
    python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
) else if "%choice%"=="2" (
    echo [INFO] Starting API in production mode...
    echo [INFO] Access at: http://localhost:8000
    echo [INFO] Docs at: http://localhost:8000/docs
    echo.
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
) else if "%choice%"=="3" (
    echo [INFO] Running tests...
    python -m pytest test_app.py -v
) else if "%choice%"=="4" (
    echo [INFO] Exiting...
    exit /b 0
) else (
    echo [WARNING] Invalid choice. Exiting...
    exit /b 1
)

echo.
echo ======================================================================
echo Setup complete!
echo ======================================================================
pause
