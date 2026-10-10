@echo off
rem === Универсальный запуск blurface ===
rem Примеры:
rem   run.bat process --input video.mp4 --output result.mp4
rem   run.bat bot
rem   run.bat check
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [X] Виртуальное окружение не найдено. Сначала запустите setup.bat
    exit /b 1
)

".venv\Scripts\python.exe" main.py %*
exit /b %ERRORLEVEL%
