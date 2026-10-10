"""Проверка окружения: Python, OpenCV, модель YuNet, FFmpeg/ffprobe, NVENC, токен.

Запуск: python main.py check   (или python check_env.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK = "[OK]"
FAIL = "[НЕТ]"
WARN = "[WARN]"


def run_checks() -> int:
    from src.config import Settings
    from src.video.ffmpeg_tools import FFmpegTool, FFmpegNotFoundError

    failures = 0
    lines: list[str] = []

    # --- Python ---
    v = sys.version_info
    if v >= (3, 10):
        lines.append(f"{OK} Python {v.major}.{v.minor}.{v.micro}")
    else:
        lines.append(f"{FAIL} Python {v.major}.{v.minor} — нужен 3.10+")
        failures += 1

    settings = Settings.load()

    # --- OpenCV ---
    try:
        import cv2
        import numpy
        has_yunet = hasattr(cv2, "FaceDetectorYN")
        lines.append(f"{OK} OpenCV {cv2.__version__}, NumPy {numpy.__version__}")
        if not has_yunet:
            lines.append(f"{FAIL} OpenCV собран без FaceDetectorYN — обновите opencv-python")
            failures += 1
    except ImportError as exc:
        lines.append(f"{FAIL} OpenCV/NumPy не установлены: {exc}")
        failures += 1

    # --- Модель детектора ---
    try:
        from src.face_detection.detector import FaceDetector, DetectorError

        FaceDetector(settings.model_path)
        lines.append(f"{OK} Модель YuNet: {settings.model_path}")
    except Exception as exc:
        lines.append(f"{FAIL} Модель YuNet не загружается: {exc}")
        failures += 1

    # --- FFmpeg / ffprobe ---
    try:
        ff = FFmpegTool(settings.ffmpeg_path, settings.ffprobe_path)
        lines.append(f"{OK} ffmpeg: {ff.ffmpeg}")
        lines.append(f"{OK} {ff.version()}")
        lines.append(f"{OK} ffprobe: {ff.ffprobe}")
    except FFmpegNotFoundError as exc:
        lines.append(f"{FAIL} {exc}")
        failures += 1
        for line in lines:
            print(line)
        return 1

    # --- NVENC ---
    if ff.nvenc_available():
        lines.append(f"{OK} GPU-кодирование NVENC доступно (H.264 через видеокарту)")
    else:
        lines.append(
            f"{WARN} NVENC недоступен — будет использован кодировщик CPU (libx264). "
            "Для GPU-ускорения установите драйвер NVIDIA и свежий FFmpeg."
        )

    # --- Токен бота (не обязателен для локального режима) ---
    if settings.bot_token:
        lines.append(f"{OK} Токен Telegram-бота задан (.env)")
    else:
        lines.append(f"{WARN} Токен Telegram-бота не задан — локальный режим (process) работает и без него")

    # --- Временный каталог ---
    try:
        settings.ensure_temp_dir()
        probe_file = settings.temp_dir / ".write_probe"
        probe_file.write_text("ok")
        probe_file.unlink()
        lines.append(f"{OK} Каталог временных файлов доступен: {settings.temp_dir}")
    except OSError as exc:
        lines.append(f"{FAIL} Не доступен каталог временных файлов: {exc}")
        failures += 1

    print()
    print("Проверка окружения blurface")
    print("=" * 60)
    for line in lines:
        print(line)
    print("=" * 60)
    if failures:
        print(f"Результат: {failures} проблема(ы). Исправьте пункты [НЕТ].")
        return 1
    print("Результат: всё в порядке.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_checks())