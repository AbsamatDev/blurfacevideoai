"""Процедурная генерация синтетического «лица» для тестов.

Лицо рисуется полностью программно (градиентная кожа, глаза, брови, нос,
рот, волосы, лёгкий шум) и детектируется YuNet — это позволяет тестировать
полный цикл без загрузки фотографий реальных людей.
"""
from __future__ import annotations

import numpy as np
import cv2


def draw_face(img: np.ndarray, cx: int, cy: int, face_w: int) -> None:
    """Рисует синтетическое лицо с центром (cx, cy) и шириной face_w прямо на img (BGR)."""
    h, w = img.shape[:2]
    s = face_w / 320.0
    fh = int(face_w * 1.25)  # высота лица

    # Кожа с радиальным градиентом (мягкие края).
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - cx) / (face_w / 2.0)) ** 2 + ((yy - cy) / (fh / 2.0)) ** 2)
    mask = np.clip(1.10 - d, 0, 1)
    shade = 1.0 - 0.25 * np.clip((yy - cy) / (fh / 2), 0, 1)
    for c, base_color in ((0, 205.0), (1, 170.0), (2, 150.0)):
        layer = np.where(mask > 0, base_color * shade * (0.85 + 0.3 * mask), img[:, :, c])
        img[:, :, c] = np.where(mask > 0, layer, img[:, :, c])

    # Волосы.
    cv2.ellipse(img, (cx, cy - int(fh * 0.42)),
                (int(face_w * 0.56), int(fh * 0.28)), 0, 0, 360,
                (35, 25, 20), -1)

    # Глаза: белки, радужка, зрачок, брови.
    for ex in (cx - int(48 * s), cx + int(48 * s)):
        cv2.ellipse(img, (ex, cy - int(42 * s)), (int(24 * s), int(13 * s)), 0, 0, 360,
                    (255, 255, 250), -1)
        cv2.circle(img, (ex, cy - int(40 * s)), int(8 * s), (70, 60, 90), -1)
        cv2.circle(img, (ex, cy - int(40 * s)), int(4 * s), (20, 15, 25), -1)
        cv2.ellipse(img, (ex, cy - int(60 * s)), (int(26 * s), int(6 * s)), 0, 0, 180,
                    (60, 45, 40), max(1, int(6 * s)))

    # Нос.
    cv2.ellipse(img, (cx, cy), (int(12 * s), int(20 * s)), 0, 0, 360,
                (160, 125, 108), -1)
    cv2.circle(img, (cx - int(10 * s), cy + int(18 * s)), max(1, int(4 * s)), (120, 90, 75), -1)
    cv2.circle(img, (cx + int(10 * s), cy + int(18 * s)), max(1, int(4 * s)), (120, 90, 75), -1)

    # Рот.
    cv2.ellipse(img, (cx, cy + int(50 * s)), (int(30 * s), int(14 * s)), 0, 0, 180,
                (110, 50, 55), max(1, int(6 * s)))


def make_face_frame(width: int, height: int, seed: int = 42, positions=None, noise: float = 6.0) -> np.ndarray:
    """Кадр с фоном и лицом (лицами). ``positions`` — список (cx, cy, face_w) или None для центра."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), np.uint8)
    # Спокойный фон с вертикальным градиентом и шумом.
    for row in range(height):
        t = row / max(1, height - 1)
        img[row, :, 0] = int(90 + 40 * t)
        img[row, :, 1] = int(110 + 30 * t)
        img[row, :, 2] = int(140 + 20 * t)
    if positions is None:
        positions = [(width // 2, height // 2, int(height * 0.62))]
    for cx, cy, fw in positions:
        draw_face(img, int(cx), int(cy), int(fw))
    if noise > 0:
        n = rng.normal(0, noise, (height, width, 1)).repeat(3, axis=2)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def face_path_at_t(width: int, height: int, t: float) -> tuple[int, int, int]:
    """Траектория движения лица: слева направо по кадру."""
    margin = height * 0.35
    xs = margin + (width - 2 * margin) * (0.5 - 0.5 * np.cos(2 * np.pi * t))  # плавный проход туда-обратно
    return int(xs), height // 2, int(height * 0.55)