@echo off
REM ─────────────────────────────────────────────────────────────
REM  VegeVision — Start Backend Server
REM  Run this from the project root: s:\college-sem6\mini-project-DL
REM ─────────────────────────────────────────────────────────────

echo.
echo ============================================
echo   VegeVision AI — Vegetation Analysis System
echo ============================================
echo.

REM Activate the existing venv if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [WARN] No venv found. Using system Python.
)

REM Install / upgrade backend dependencies
echo.
echo [INFO] Installing backend dependencies...
pip install -r backend\requirements.txt --quiet

echo.
echo [INFO] Starting FastAPI server on http://localhost:8000
echo [INFO] Frontend is served at http://localhost:8000/static/index.html
echo [INFO] API docs at http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server.
echo.

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
