"""Интеграционные тесты полного цикла обработки видео.

Покрывают сценарии из ТЗ:
1) видео без лиц; 2) одно лицо; 3) несколько лиц; 4) движущееся лицо;
5) видео со звуком; 6) другое разрешение (1280x720); 7) повреждённый файл;
8) очистка временных файлов после ошибки; 9) сохранение разрешения
и длительности; 10) сохранение аудио и его длительности;
11) лицо в результате действительно размыто (а фон не тронут);
12) файл без аудиодорожки.
Плюс: видео с rotation metadata.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.video.blur import expand_box
from src.video.ffmpeg_tools import ProbeError
from src.video.pipeline import VideoProcessingError, process_video
from src.video.tempfiles import job_directory
from tests.conftest import frame_diff_region


def _extract(ff, video_path, t: float, png_path):
    assert ff.extract_frame(video_path, t, png_path), f"не удалось извлечь кадр из {video_path}"
    import cv2

    img = cv2.imread(str(png_path))
    assert img is not None
    return img


def _assert_resolution_and_duration(ff, in_path, out_path, tol=0.15):
    a = ff.probe(in_path)
    b = ff.probe(out_path)
    assert (b.display_width, b.display_height) == (a.display_width, a.display_height)
    assert abs(b.duration - a.duration) <= max(1.0, tol * a.duration)
    return a, b


# 1. Видео без лиц --------------------------------------------------------
def test_video_without_faces(videos, tmp_path, settings, ff_tool):
    out = tmp_path / "out_no_face.mp4"
    result = process_video(videos["no_face"], out, settings)
    assert result.stats.frames_processed == 30
    assert result.stats.faces_detected == 0
    a, b = _assert_resolution_and_duration(ff_tool, videos["no_face"], out)
    assert a.has_audio is False and b.has_audio is False


# 2. Видео с одним лицом --------------------------------------------------
def test_video_with_one_face(videos, tmp_path, settings, ff_tool):
    out = tmp_path / "out_one_face.mp4"
    result = process_video(videos["one_face"], out, settings)
    assert result.stats.faces_detected > 0
    assert result.stats.frames_with_faces > 10
    _assert_resolution_and_duration(ff_tool, videos["one_face"], out)


# 3. Несколько лиц --------------------------------------------------------
def test_video_with_two_faces(videos, tmp_path, settings, ff_tool):
    out = tmp_path / "out_two_faces.mp4"
    result = process_video(videos["two_faces"], out, settings)
    # в кадре два лица — суммарно должно быть заметно больше лиц, чем кадров
    assert result.stats.faces_detected >= 2 * result.stats.frames_processed * 0.5


# 4. Лицо перемещается ----------------------------------------------------
def test_moving_face(videos, tmp_path, settings):
    out = tmp_path / "out_moving.mp4"
    result = process_video(videos["moving"], out, settings)
    # лицо должно находиться почти в каждом кадре
    assert result.stats.frames_with_faces >= 0.8 * result.stats.frames_processed


# 5. Видео со звуком ------------------------------------------------------
def test_audio_preserved(videos, tmp_path, settings, ff_tool):
    out = tmp_path / "out_audio.mp4"
    result = process_video(videos["with_audio"], out, settings)
    a = ff_tool.probe(videos["with_audio"])
    b = ff_tool.probe(out)
    assert a.has_audio and b.has_audio
    assert abs((b.audio_duration or 0) - (a.audio_duration or 0)) <= 1.0


# 6. Другое разрешение (1280x720) -----------------------------------------
def test_hd_720p(videos, tmp_path, settings, ff_tool):
    out = tmp_path / "out_hd720.mp4"
    result = process_video(videos["hd720"], out, settings)
    assert result.in_width == 1280 and result.in_height == 720
    assert result.out_width == 1280 and result.out_height == 720
    assert result.stats.faces_detected > 0


# 7. Некорректный/повреждённый файл --------------------------------------
@pytest.mark.parametrize("key", ["corrupted", "garbage"])
def test_corrupted_file_raises(videos, tmp_path, settings, key):
    out = tmp_path / f"out_{key}.mp4"
    with pytest.raises(VideoProcessingError):
        process_video(videos[key], out, settings)
    assert not out.exists(), "частичный итоговый файл должен быть удалён"


def test_not_a_video_raises(tmp_path, settings):
    bad = tmp_path / "text.mp4"
    bad.write_text("это не видео, просто текст")
    with pytest.raises(VideoProcessingError):
        process_video(bad, tmp_path / "out.mp4", settings)


# 8. Очистка временных файлов после ошибки --------------------------------
def test_job_directory_cleanup_on_error(settings):
    with pytest.raises(RuntimeError):
        with job_directory(settings.temp_dir) as job_dir:
            (job_dir / "input.mp4").write_bytes(b"x" * 1024)
            raise RuntimeError("имитация сбоя")
    assert not job_dir.exists()
    assert not any(settings.temp_dir.glob("job_*"))


def test_job_directory_cleanup_stale(settings):
    # осиротевший каталог удаляется при уборке
    orphan = settings.temp_dir / "job_orphan"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "junk.bin").write_bytes(b"junk")
    from src.video.tempfiles import cleanup_stale_jobs

    cleanup_stale_jobs(settings.temp_dir)
    assert not orphan.exists()


# 9. Сохранение разрешения и длительности ---------------------------------
def test_resolution_duration_preserved(videos, tmp_path, settings, ff_tool):
    out = tmp_path / "out_preserve.mp4"
    process_video(videos["one_face"], out, settings)
    _assert_resolution_and_duration(ff_tool, videos["one_face"], out)


# 10. Аудио: наличие и длительность (см. также test_audio_preserved) ------
def test_audio_duration_tolerance(videos, tmp_path, settings, ff_tool):
    out = tmp_path / "out_audio2.mp4"
    process_video(videos["with_audio"], out, settings)
    b = ff_tool.probe(out)
    a = ff_tool.probe(videos["with_audio"])
    assert abs((b.audio_duration or 0) - (a.audio_duration or 0)) <= 1.5


# 11. Лица реально размыты, фон не тронут ---------------------------------
def test_face_region_blurred_background_intact(videos, tmp_path, settings, ff_tool):
    src = videos["one_face"]
    out = tmp_path / "out_diff.mp4"
    result = process_video(src, out, settings)
    assert result.stats.faces_detected > 0

    import cv2

    frame_in = _extract(ff_tool, src, 1.0, tmp_path / "in.png")
    frame_out = _extract(ff_tool, out, 1.0, tmp_path / "out.png")
    # детектируем лицо на исходном кадре
    from src.config import PROJECT_ROOT
    from src.face_detection.detector import FaceDetector

    det = FaceDetector(PROJECT_ROOT / "models" / "face_yunet_2023mar.onnx",
                       score_threshold=settings.detect_score_threshold)
    boxes, _ = det.detect(frame_in)
    assert boxes, "лицо должно находиться на кадре теста"

    # зона глаз: высококонтрастная область, которую должно «съесть» размытие
    bx, by, bw, bh = boxes[0]
    cx, cy = bx + bw / 2, by + bh / 2
    s = bw / 320.0
    ex1, ex2 = int(cx - 70 * s), int(cx + 70 * s)
    ey1, ey2 = int(cy - 55 * s), int(cy - 28 * s)

    d_face = frame_diff_region(frame_in, frame_out, ex1, ey1, ex2, ey2)
    # фон — правый верхний угол, заведомо без лица
    d_bg = frame_diff_region(frame_in, frame_out, frame_in.shape[1] - 100, 10,
                             frame_in.shape[1] - 10, 90)
    assert d_face > 12, f"область лица должна сильно отличаться (diff={d_face:.1f})"
    assert d_bg < 6, f"фон не должен заметно отличаться (diff={d_bg:.1f})"


# 12. Файл без аудиодорожки ------------------------------------------------
def test_file_without_audio(videos, tmp_path, settings, ff_tool):
    out = tmp_path / "out_no_audio.mp4"
    process_video(videos["one_face"], out, settings)  # one_face генерируется без аудио
    b = ff_tool.probe(out)
    assert b.has_audio is False


# Поворот (rotation metadata, видео «с телефона») ---------------------------
def test_rotated_video_orientation(videos, tmp_path, settings, ff_tool):
    src = videos["rotated"]
    a = ff_tool.probe(src)
    assert a.rotation in (90, -90)
    assert (a.display_width, a.display_height) == (360, 640)
    out = tmp_path / "out_rotated.mp4"
    result = process_video(src, out, settings)
    b = ff_tool.probe(out)
    # результат: кадры уже повёрнуты, метка поворота не нужна
    assert (b.display_width, b.display_height) == (360, 640)
    assert result.stats.faces_detected > 0
    assert abs(b.duration - a.duration) <= max(1.0, 0.1 * a.duration)


# Прогресс-колбэк ----------------------------------------------------------
def test_progress_callback(videos, tmp_path, settings):
    seen = []

    def cb(done, total):
        seen.append((done, total))

    process_video(videos["one_face"], tmp_path / "out_prog.mp4", settings, progress=cb)
    assert seen, "прогресс-колбэк должен вызываться"
    assert seen[-1][0] == seen[-1][1] == 30  # финальный вызов: всё обработано


# Ограничение длительности --------------------------------------------------
def test_duration_limit(videos, tmp_path):
    from src.config import Settings

    s = Settings.load(temp_dir=tmp_path / "t", max_duration_sec=0.5,
                      verify_max_duration_sec=0)
    with pytest.raises(VideoProcessingError, match="слишком длинное"):
        process_video(videos["one_face"], tmp_path / "out_limit.mp4", s)


# Режимы размытия через настройки -------------------------------------------
@pytest.mark.parametrize("mode", ["normal", "pixelate"])
def test_blur_modes_pipeline(videos, tmp_path, mode):
    from src.config import Settings

    s = Settings.load(temp_dir=tmp_path / "t", blur_mode=mode,
                      verify_max_duration_sec=0)
    out = tmp_path / f"out_{mode}.mp4"
    result = process_video(videos["one_face"], out, s)
    assert result.stats.faces_detected > 0