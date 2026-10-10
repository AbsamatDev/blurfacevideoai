"""Тесты размытия областей."""
from __future__ import annotations

import numpy as np
import pytest

from src.video.blur import blur_boxes, blur_roi, expand_box
from src.face_detection.synface import make_face_frame


@pytest.fixture()
def frame():
    return make_face_frame(320, 240, noise=0.0)


BOX = (110, 60, 100, 120)


# Зона глаз синтетического лица 320x240 (высококонтрастная деталь).
# Лицо рисуется с центром (160, 120) и шириной 148: глаза в полосе y 90..112, x 125..195.
EYE_PATCH = (slice(90, 112), slice(125, 195))


def _region_change(orig, out, box, padding=0.30):
    """(1) MAD в зоне глаз (высококонтрастные детали лица),
    (2) MAD в области заведомо вне лица (правый верхний угол)."""
    d_in = np.abs(orig[EYE_PATCH].astype(np.int32) - out[EYE_PATCH].astype(np.int32)).mean()
    d_out = np.abs(orig[0:40, 280:320].astype(np.int32) - out[0:40, 280:320].astype(np.int32)).mean()
    return d_in, d_out


@pytest.mark.parametrize("mode", ["normal", "strong", "pixelate"])
def test_blur_changes_face_region_only(frame, mode):
    out = blur_boxes(frame, [BOX], mode=mode, strength=1.0, padding=0.0)
    d_in, d_out = _region_change(frame, out, BOX, padding=0.0)
    assert d_in > 20, f"детали лица (зона глаз) должны заметно измениться (получено {d_in:.1f})"
    assert d_out == 0, "вне области кадр не должен меняться"


def test_strong_blur_hides_details(frame):
    out = blur_boxes(frame, [BOX], mode="strong", strength=1.0, padding=0.0)
    x1, y1, x2, y2 = expand_box(BOX, 0.0, frame.shape[1], frame.shape[0])
    roi = out[y1:y2, x1:x2].astype(np.float32)
    # сильное размытие: локальный контраст (градиент между соседними пикселями) почти исчезает
    grad = np.abs(np.diff(roi, axis=0)).mean() + np.abs(np.diff(roi, axis=1)).mean()
    assert grad < 3.0, f"после сильного размытия градиент должен быть минимален (получено {grad:.2f})"


def test_pixelate_blocks(frame):
    out = blur_boxes(frame, [BOX], mode="pixelate", strength=1.0, padding=0.0)
    x1, y1, x2, y2 = expand_box(BOX, 0.0, frame.shape[1], frame.shape[0])
    roi = out[y1:y2, x1:x2]
    # пикселизация: большие однотонные квадраты — соседние пиксели почти равны
    flat = roi[:, :-1, :].astype(np.int32) - roi[:, 1:, :].astype(np.int32)
    assert np.abs(flat).mean() < 1.5


def test_padding_expands_blur(frame):
    small_out = blur_boxes(frame, [BOX], mode="strong", strength=1.0, padding=0.0)
    big_out = blur_boxes(frame, [BOX], mode="strong", strength=1.0, padding=0.5)
    # кольцо вокруг бокса: при padding=0.5 оно тоже размыто
    x1, y1, x2, y2 = expand_box(BOX, 0.35, frame.shape[1], frame.shape[0])
    sx1, sy1, sx2, sy2 = expand_box(BOX, 0.0, frame.shape[1], frame.shape[0])
    ring_diff = np.abs(
        big_out[y1:sy1, x1:x2].astype(np.int32) - frame[y1:sy1, x1:x2].astype(np.int32)
    ).mean()
    small_ring_diff = np.abs(
        small_out[y1:sy1, x1:x2].astype(np.int32) - frame[y1:sy1, x1:x2].astype(np.int32)
    ).mean()
    assert ring_diff > small_ring_diff


def test_invalid_mode_raises(frame):
    from src.video.blur import BlurModeError

    with pytest.raises(BlurModeError):
        blur_roi(frame, "unknown", 1.0)


def test_box_clamped_to_frame(frame):
    x1, y1, x2, y2 = expand_box((-500, -500, 100, 100), 0.3, frame.shape[1], frame.shape[0])
    assert x1 >= 0 and y1 >= 0
    assert x2 <= frame.shape[1] and y2 <= frame.shape[0]
    assert x2 > x1 and y2 > y1