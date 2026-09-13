# -*- coding: utf-8 -*-
"""底色估计与前景软掩膜。

设计要点：**不做二值化**。抗锯齿、缩放插值会把字符边缘变成一堆中间色，
先二值化就把这些信息丢掉了，而它们恰恰是缩放后还能认出来的关键。
所以这里输出的是 0~1 的连续强度图。
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def load_rgb(image_bytes: bytes) -> np.ndarray:
    """读入为 (H, W, 3) 的 uint8 数组。"""
    return np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def estimate_background(rgb: np.ndarray) -> np.ndarray:
    """出现次数最多的颜色即平坦底色。

    验证码底色基本都是一整块平涂，所以取众数非常稳；深色底、白底、随机色底都能应对。
    """
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    return colors[int(counts.argmax())]


def soft_foreground(image_bytes: bytes, bg: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(强度图, 底色)``。

    强度 = 与底色的最大通道差，再按 95 分位归一化到 0~1。
    用分位数归一化是为了**对对比度免疫**：浅灰字配白底、纯红字配深底，
    归一化后得到的强度分布是一样的。
    """
    rgb = load_rgb(image_bytes)
    if bg is None:
        bg = estimate_background(rgb)
    diff = np.abs(rgb.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
    positive = diff[diff > 1e-6]
    if positive.size == 0:
        return np.zeros(diff.shape, dtype=np.float32), bg
    scale = float(np.percentile(positive, 95))
    return np.clip(diff / max(scale, 1e-6), 0.0, 1.0).astype(np.float32), bg


def content_bbox(soft: np.ndarray, threshold: float = 0.35):
    """前景内容的紧致外框，返回 ``(top, bottom, left, right)``（含端点）或 None。"""
    mask = soft > threshold
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def to_binary(soft: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return soft > threshold
