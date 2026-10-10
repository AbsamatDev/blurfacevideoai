"""Тесты настроек и вспомогательных функций."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def test_defaults():
    from src.config import Settings

    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_API_BASE", "MAX_INPUT_SIZE_MB", "BLUR_MODE",
                 "MAX_DURATION_SEC", "DET_SCALES", "ENCODER",
                 "FACE_PADDING", "BLUR_STRENGTH"):
        os.environ.pop(name, None)
    s = Settings.load()
    assert s.blur_mode == "strong"
    assert s.face_padding == 0.30
    assert s.detect_scales == (1.0, 0.5)
    assert s.encoder == "auto"
    assert s.max_parallel_jobs >= 1
    assert s.max_input_size_mb <= 20


def test_env_overrides(monkeypatch):
    from src.config import Settings

    monkeypatch.setenv("BLUR_MODE", "pixelate")
    monkeypatch.setenv("MAX_DURATION_SEC", "120")
    monkeypatch.setenv("DET_SCALES", "1.0, 0.5, 0.25")
    monkeypatch.setenv("FACE_PADDING", "0.45")
    s = Settings.load()
    assert s.blur_mode == "pixelate"
    assert s.max_duration_sec == 120.0
    assert s.detect_scales == (1.0, 0.5, 0.25)
    assert s.face_padding == 0.45


def test_cli_overrides_win():
    from src.config import Settings

    s = Settings.load(blur_mode="normal", max_duration_sec=42)
    assert s.blur_mode == "normal"
    assert s.max_duration_sec == 42


def test_invalid_values_fallback(monkeypatch):
    from src.config import Settings

    monkeypatch.setenv("BLUR_MODE", "супер")
    monkeypatch.setenv("BLUR_STRENGTH", "-3")
    monkeypatch.setenv("ENCODER", "quantum")
    s = Settings.load()
    assert s.blur_mode == "strong"
    assert s.blur_strength == 1.0
    assert s.encoder == "auto"


def test_result_path_always_mp4(tmp_path):
    """Регрессия: Telegram не всегда передаёт имя файла, и итог задания
    должен всегда иметь расширение .mp4 (иначе ffmpeg не может выбрать muxer)."""
    from src.video.tempfiles import result_path, safe_input_name

    for name in (None, "video", "IMG_7826.MOV", "my clip.mkx"):
        inp = tmp_path / safe_input_name(name)
        outp = result_path(tmp_path, inp)
        assert outp.suffix == ".mp4", f"для входа {name!r} выход без .mp4: {outp}"
        assert outp.exists() is False


def test_safe_input_name():
    from src.video.tempfiles import safe_input_name

    name = safe_input_name("мое видео (1).mp4")
    assert name.endswith(".mp4")
    assert " " not in name and "(" not in name
    assert safe_input_name("../../../etc/passwd").endswith(".mp4") or "input_" in safe_input_name("../../../etc/passwd")
    assert safe_input_name(None).startswith("input_")


def test_ffmpeg_found():
    from src.video.ffmpeg_tools import FFmpegTool

    ff = FFmpegTool()
    assert ff.ffmpeg.exists()
    assert "ffmpeg" in ff.version().lower() or "version" in ff.version().lower()


def test_probe_garbage(tmp_path):
    from src.video.ffmpeg_tools import FFmpegTool, ProbeError

    ff = FFmpegTool()
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(np.random.default_rng(0).integers(0, 255, 4096, dtype=np.uint8).tobytes())
    try:
        ff.probe(bad)
        raised = False
    except ProbeError:
        raised = True
    assert raised