@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 py -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo Unable to create a virtual environment. Install Python 3.10 or 3.11.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. See README.md.
    pause
    exit /b 1
)
python -m streamlit run dashboard.py
pause
