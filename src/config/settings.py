"""Настройки приложения.

Значения берутся из переменных окружения / файла ``.env`` в корне проекта.
Все параметры имеют безопасные значения по умолчанию, подходящие для
Windows 11 + NVIDIA GPU.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

# Загружаем .env один раз при импорте модуля.
load_dotenv(PROJECT_ROOT / ".env")

BLUR_MODES = ("normal", "strong", "pixelate")
ENCODER_MODES = ("auto", "nvenc", "x264")


def _str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on", "да")


def _floats(name: str, default: str) -> tuple[float, ...]:
    raw = _str(name, default)
    out = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.append(float(part))
            except ValueError:
                continue
    return tuple(out) or (1.0,)


def _path(name: str, default: str) -> Path:
    raw = _str(name, default)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class Settings:
    """Все настраиваемые параметры приложения."""

    # --- Telegram ---
    bot_token: str = ""
    # Лимит скачивания Bot API = 20 МБ, поэтому по умолчанию чуть меньше.
    max_input_size_mb: int = 19
    # Лимит отправки Bot API = 50 МБ (видео и документы).
    max_send_size_mb: int = 50
    # Origin локального Bot API сервера (docker-compose.yml), например
    # "http://localhost:8081". Пусто = облачный api.telegram.org (лимиты 20/50 МБ).
    # Локальный сервер снимает лимиты: приём до ~2 ГБ, отправка до 2000 МБ.
    telegram_api_base: str = ""
    # Каталог тома данных сервера со стороны бота (запасной способ чтения файлов,
    # если HTTP-выдача файлов сервером недоступна).
    telegram_local_files_dir: Optional[Path] = None
    # Префикс путей файлов на диске сервера (как в file_path ответа getFile).
    telegram_server_files_prefix: str = "/var/lib/telegram-bot-api"

    # --- Файлы ---
    temp_dir: Path = PROJECT_ROOT / "tmp"
    max_duration_sec: float = 600.0  # 10 минут; 0 = без ограничения
    keep_temp: bool = False

    # --- Размытие ---
    blur_mode: str = "strong"
    blur_strength: float = 1.0
    face_padding: float = 0.30

    # --- Детекция лиц ---
    model_path: Path = PROJECT_ROOT / "models" / "face_yunet_2023mar.onnx"
    detect_max_side: int = 960
    detect_scales: tuple = (1.0, 0.5)
    detect_score_threshold: float = 0.6
    detect_nms_threshold: float = 0.3
    detect_nms_iou: float = 0.35

    # --- Кодирование ---
    encoder: str = "auto"  # auto | nvenc | x264
    nvenc_cq: int = 23
    x264_crf: int = 18
    video_bitrate: str = ""  # пусто = режим качества (CQ/CRF); иначе, напр. "8M"
    verify_max_duration_sec: float = 300.0  # полная проверка декодируемости для видео короче этого; 0 = выкл.
    convert_hdr: bool = True  # конвертировать HDR/BT.2020 в SDR BT.709

    # --- Очередь / прогресс ---
    max_parallel_jobs: int = 1
    progress_interval_sec: float = 3.0

    # --- Статистика ---
    stats_file: Path = PROJECT_ROOT / "tmp" / "stats.jsonl"
    admin_user_id: int = 0  # Telegram user id владельца; 0 = команда /stats отключена

    # --- Пути к FFmpeg (пусто = автоопределение) ---
    ffmpeg_path: Optional[Path] = None
    ffprobe_path: Optional[Path] = None

    # --- Переопределения для CLI ---
    overrides: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def load(cls, **overrides) -> "Settings":
        s = cls(
            bot_token=_str("TELEGRAM_BOT_TOKEN", ""),
            max_input_size_mb=_int("MAX_INPUT_SIZE_MB", 19),
            max_send_size_mb=_int("MAX_SEND_SIZE_MB", 50),
            telegram_api_base=_str("TELEGRAM_API_BASE", "").strip().rstrip("/"),
            telegram_local_files_dir=_path("TELEGRAM_LOCAL_FILES_DIR", "") if os.getenv("TELEGRAM_LOCAL_FILES_DIR") else None,
            telegram_server_files_prefix=_str("TELEGRAM_SERVER_FILES_PREFIX", "/var/lib/telegram-bot-api"),
            temp_dir=_path("TEMP_DIR", str(PROJECT_ROOT / "tmp")),
            max_duration_sec=_float("MAX_DURATION_SEC", 600.0),
            keep_temp=_bool("KEEP_TEMP", False),
            blur_mode=_str("BLUR_MODE", "strong").strip().lower(),
            blur_strength=_float("BLUR_STRENGTH", 1.0),
            face_padding=_float("FACE_PADDING", 0.30),
            model_path=_path("MODEL_PATH", str(PROJECT_ROOT / "models" / "face_yunet_2023mar.onnx")),
            detect_max_side=_int("DET_MAX_SIDE", 960),
            detect_scales=_floats("DET_SCALES", "1.0,0.5"),
            detect_score_threshold=_float("DET_SCORE_THRESHOLD", 0.6),
            detect_nms_threshold=_float("DET_NMS_THRESHOLD", 0.3),
            detect_nms_iou=_float("DET_NMS_IOU", 0.35),
            encoder=_str("ENCODER", "auto").strip().lower(),
            nvenc_cq=_int("NVENC_CQ", 23),
            x264_crf=_int("X264_CRF", 18),
            video_bitrate=_str("VIDEO_BITRATE", ""),
            verify_max_duration_sec=_float("VERIFY_MAX_DURATION_SEC", 300.0),
            convert_hdr=_bool("CONVERT_HDR", True),
            max_parallel_jobs=max(1, _int("MAX_PARALLEL_JOBS", 1)),
            progress_interval_sec=_float("PROGRESS_INTERVAL_SEC", 3.0),
            stats_file=_path("STATS_FILE", str(PROJECT_ROOT / "tmp" / "stats.jsonl")),
            admin_user_id=_int("ADMIN_USER_ID", 0),
            ffmpeg_path=Path(os.getenv("FFMPEG_PATH")) if os.getenv("FFMPEG_PATH") else None,
            ffprobe_path=Path(os.getenv("FFPROBE_PATH")) if os.getenv("FFPROBE_PATH") else None,
        )
        if s.blur_mode not in BLUR_MODES:
            s.blur_mode = "strong"
        if s.encoder not in ENCODER_MODES:
            s.encoder = "auto"
        if s.blur_strength <= 0:
            s.blur_strength = 1.0
        if s.telegram_api_base and s.max_input_size_mb == 19 and s.max_send_size_mb == 50:
            # Подключён локальный Bot API сервер, а лимиты оставлены облачные
            # (20/50 МБ) — поднимаем их до возможностей локального сервера.
            # Явно заданные в .env значения не трогаем.
            s.max_input_size_mb = 1900
            s.max_send_size_mb = 1900
            logger.info(
                "Локальный Bot API сервер: лимиты файлов подняты до %d/%d МБ.",
                s.max_input_size_mb, s.max_send_size_mb,
            )
        s.overrides = dict(overrides)
        for key, value in overrides.items():
            if hasattr(s, key) and value is not None:
                setattr(s, key, value)
        return s

    @property
    def max_input_size_bytes(self) -> int:
        return self.max_input_size_mb * 1024 * 1024

    @property
    def max_send_size_bytes(self) -> int:
        return self.max_send_size_mb * 1024 * 1024

    def ensure_temp_dir(self) -> Path:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        return self.temp_dir