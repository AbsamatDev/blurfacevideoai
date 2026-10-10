"""Тесты детектора лиц (YuNet) на синтетических изображениях."""
from __future__ import annotations

import numpy as np
import pytest

from src.face_detection.detector import DetectorError, FaceDetector, nms
from src.face_detection.synface import make_face_frame

MODEL = None


@pytest.fixture(scope="module")
def detector():
    from src.config import PROJECT_ROOT

    return FaceDetector(
        PROJECT_ROOT / "models" / "face_yunet_2023mar.onnx",
        score_threshold=0.5,
        max_side=960,
        scales=(1.0, 0.5),
    )


def test_model_missing_raises(tmp_path):
    with pytest.raises(DetectorError):
        FaceDetector(tmp_path / "missing.onnx")


def test_detects_synthetic_face(detector):
    frame = make_face_frame(320, 320)
    boxes, raw = detector.detect(frame)
    assert len(boxes) >= 1
    assert raw >= 1
    x, y, w, h = boxes[0]
    # Лицо занимает большую часть кадра.
    assert w > 120 and h > 150
    assert x >= 0 and y >= 0


def test_no_false_positive_on_noise(detector):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    boxes, raw = detector.detect(frame)
    assert boxes == [] and raw == 0


def test_multiscale_finds_large_face(detector):
    """Крупный план (лицо шире 300 px — предел одиночного прохода) — должен
    ловиться дополнительным проходом в масштабе 0.5."""
    frame = make_face_frame(640, 640, positions=[(320, 300, 480)])
    boxes, raw = detector.detect(frame)
    assert len(boxes) >= 1
    assert boxes[0][2] > 300  # найденное лицо действительно крупное


def test_nms_suppresses_overlaps():
    items = [
        [0, 0, 100, 100, 0.9],
        [5, 5, 100, 100, 0.8],   # почти дубль
        [300, 300, 100, 100, 0.7],
    ]
    kept = nms(items, iou_threshold=0.35)
    assert len(kept) == 2
    assert kept[0][4] == pytest.approx(0.9)
    assert kept[1][4] == pytest.approx(0.7)