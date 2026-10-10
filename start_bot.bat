@echo off
rem === blurface - start Telegram bot ===
rem Double-click this file: bot starts in an open terminal window.
rem To stop the bot - close this window (X button) or press Ctrl+C.
setlocal
cd /d "%~dp0"

echo ============================================
echo    blurface - Telegram face-blur bot
echo ============================================
echo.

rem --- 1. Virtual environment ---
if not exist ".venv\Scripts\python.exe" (
    echo [X] Virtual environment not found.
    echo     Run setup.bat first.
    pause
    exit /b 1
)

rem --- 2. Docker and local Bot API server ---
docker version >nul 2>nul
if errorlevel 1 (
    echo [!] Docker is not running - bot will not work.
    echo     Start Docker Desktop and run this file again.
    echo.
    pause
    exit /b 1
)

docker ps --format "{{.Names}}" | findstr /i "blurface-telegram-bot-api" >nul
if errorlevel 1 (
    echo [~] Local Bot API server is down, starting it...
    docker compose up -d
    if errorlevel 1 (
        echo [X] Failed to start container. Check docker-compose.yml
        pause
        exit /b 1
    )
    timeout /t 5 /nobreak >nul
)
echo [OK] Local Bot API server is running.
echo.

rem --- 3. Start the bot (window stays open) ---
echo Starting bot... Close this window to stop it.
echo --------------------------------------------
".venv\Scripts\python.exe" -X utf8 main.py bot

echo.
echo --------------------------------------------
echo Bot stopped.
pause
