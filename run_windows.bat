@echo off
REM =========================================================
REM DeepfakeDetector - One-Click Launcher (Windows)
REM Double-click this file to set up (first time) and run the app.
REM =========================================================

cd /d "%~dp0"

IF NOT EXIST venv (
    echo [Setup] Creating virtual environment for the first time...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo Trying "py" instead of "python"...
        py -m venv venv
    )
)

echo [Setup] Activating virtual environment...
call venv\Scripts\activate.bat

echo [Setup] Installing/updating dependencies (this may take a few minutes the first time)...
pip install -r requirements.txt

echo.
echo =========================================================
echo  Starting DeepfakeDetector...
echo  Once you see "Running on http://127.0.0.1:5000",
echo  open that link in your browser.
echo  Press CTRL+C in this window to stop the server.
echo =========================================================
echo.

python app.py

pause
