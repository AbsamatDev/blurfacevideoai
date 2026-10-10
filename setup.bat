@echo off
rem === Настройка окружения blurface (запускать один раз) ===
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [X] Python не найден. Установите Python 3.10+ с python.org и повторите.
    exit /b 1
)

if not exist ".venv" (
    echo Создаю виртуальное окружение...
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [X] Не удалось установить зависимости.
    exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Создан .env из примера. Вставьте туда токен бота, когда он появится.
)

echo.
echo Готово. Проверка окружения:
python main.py check
pause
