"""Общие фикстуры и утилиты тестов: короткие синтетические видео."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.video.ffmpeg_tools import FFmpegTool  # noqa: E402
from src.video import synth  # noqa: E402


@pytest.fixture(scope="session")
def ff() -> FFmpegTool:
    return FFmpegTool()


@pytest.fixture(scope="session")
def ff_tool(ff) -> FFmpegTool:
    return ff


@pytest.fixture()
def settings(tmp_path) -> Settings:
    s = Settings.load(
        temp_dir=tmp_path / "tmp",
        keep_temp=False,
        max_duration_sec=120,
        verify_max_duration_sec=0,  # тесты не гоняют полную проверку декодирования
        progress_interval_sec=999,
    )
    s.ensure_temp_dir()
    return s


@pytest.fixture(scope="session")
def videos(tmp_path_factory, ff) -> dict:
    """Синтетические видео, генерируются один раз за сессию тестов."""
    base = tmp_path_factory.mktemp("videos")

    def make(w=640, h=360, fps=15, n_frames=30):
        return dict(w=w, h=h, fps=fps, n_frames=n_frames)

    cfg = make()
    made = {}
    made["no_face"] = synth.make_no_face_video(base / "no_face.mp4", **cfg, ff=ff)
    made["one_face"] = synth.make_face_video(base / "one_face.mp4", n_faces=1, **cfg, ff=ff)
    made["two_faces"] = synth.make_face_video(base / "two_faces.mp4", n_faces=2, **cfg, ff=ff)
    made["moving"] = synth.make_face_video(base / "moving.mp4", moving=True, **cfg, ff=ff)
    made["with_audio"] = synth.add_sine_audio(made["one_face"], base / "with_audio.mp4", ff)
    made["hd720"] = synth.make_face_video(base / "hd720.mp4", n_faces=1, w=1280, h=720, fps=15, n_frames=30, ff=ff)
    made["rotated"] = synth.make_portrait_rotated_video(base / "rotated90.mp4", n_frames=cfg["n_frames"], fps=cfg["fps"], ff=ff)
    made["corrupted"] = synth.make_corrupted_video(base / "corrupted.mp4", ff)
    made["garbage"] = synth.make_garbage_file(base / "garbage.mp4")
    return made


def frame_diff_region(img_a, img_b, x1, y1, x2, y2) -> float:
    """Средняя абсолютная разность каналов в области."""
    import numpy as np

    a = img_a[y1:y2, x1:x2].astype(np.float32)
    b = img_b[y1:y2, x1:x2].astype(np.float32)
    return float(np.mean(np.abs(a - b)))