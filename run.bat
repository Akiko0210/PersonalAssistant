@echo off
REM ============================================================
REM  Voice Notes Agent - one-click launcher for Windows
REM  Double-click this file, or run "run.bat" from a terminal.
REM  First run sets everything up; later runs just start the app.
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- 1. Create the virtual environment on first run ---------
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [error] Could not create venv. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

REM --- 2. Install the app (first run, or whenever deps change) -
REM Reinstall if there is no marker, or if pyproject.toml is newer than it.
set "NEED_INSTALL="
if not exist ".venv\.installed" set "NEED_INSTALL=1"
if not defined NEED_INSTALL (
    for /f "delims=" %%i in ('dir /b /o-d "pyproject.toml" ".venv\.installed" 2^>nul') do (
        if not defined _NEWEST set "_NEWEST=%%i"
    )
    if /i "%_NEWEST%"=="pyproject.toml" set "NEED_INSTALL=1"
)
if defined NEED_INSTALL (
    echo [setup] Installing dependencies. This can take a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e ".[conversation,embeddings]"
    if errorlevel 1 (
        echo [error] Install failed. See the messages above.
        pause
        exit /b 1
    )
    echo done > ".venv\.installed"
)

REM --- 3. Create config.yaml from the example if missing ------
if not exist "config.yaml" (
    echo [setup] Creating config.yaml from config.example.yaml...
    copy /y "config.example.yaml" "config.yaml" >nul
)

REM --- 4. Load API keys from .env (KEY=value per line) --------
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
) else (
    echo [warn] No .env file found. Put your API keys in a .env file:
    echo        ANTHROPIC_API_KEY=your-key
    echo        DEEPGRAM_API_KEY=your-key
)

REM --- 5. Run the app ----------------------------------------
echo [run] Starting Voice Notes Agent...
".venv\Scripts\voice-notes-agent.exe"

echo.
echo [done] Agent stopped.
pause
