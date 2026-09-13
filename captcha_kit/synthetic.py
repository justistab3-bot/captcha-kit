# -*- coding: utf-8 -*-
"""合成验证码生成器：让整个仓库自给自足，不依赖任何第三方站点数据。

内置一套自绘的 5x7 点阵字体（数字 0-9 与 ``+ - = x``），可以按任意倍率渲染、
换底色、加噪点，用来复现"固定点阵字体 + 平坦底色"这一类验证码。
测试与示例全部基于它，所以跑测试不需要联网、也不需要任何真实验证码样本。
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image

# 5 宽 7 高的点阵字体
FONT_5X7: dict[str, list[str]] = {
    "0": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    "+": [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "=": [".....", ".....", "#####", ".....", "#####", ".....", "....."],
    "x": [".....", ".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
}

FONT_H = 7
FONT_W = 5


def glyph(ch: str) -> np.ndarray:
    """单个字符的位图（含空白边），形状 ``(7, 5)``。"""
    return np.array([[c == "#" for c in row] for row in FONT_5X7[ch]], dtype=bool)


def render(
    text: str,
    *,
    scale: int = 3,
    bg=(255, 255, 255),
    fg=(20, 20, 20),
    gap: int = 1,
    pad: int = 4,
    noise: int = 0,
    seed: int = 0,
) -> bytes:
    """把文本渲染成验证码图片，返回 PNG 字节。

    所有字形都顶对齐在同一条基线上 —— 通道 1 的精确匹配依赖这个前提。
    ``noise`` 是随机背景噪点数（用与底色相近的浅色，模拟真实验证码的干扰点）。
    """
    glyphs = [glyph(ch) for ch in text]
    width = sum(g.shape[1] for g in glyphs) + gap * max(0, len(glyphs) - 1)
    band = np.zeros((FONT_H, width), dtype=bool)
    x = 0
    for g in glyphs:
        band[:, x:x + g.shape[1]] = g
        x += g.shape[1] + gap

    big = np.kron(band, np.ones((scale, scale), dtype=bool))
    height = big.shape[0] + 2 * pad
    wid = big.shape[1] + 2 * pad
    img = np.zeros((height, wid, 3), dtype=np.uint8)
    img[:, :] = np.asarray(bg, dtype=np.uint8)
    img[pad:pad + big.shape[0], pad:pad + big.shape[1]][big] = np.asarray(fg, dtype=np.uint8)

    if noise > 0:
        rng = np.random.default_rng(seed)
        bg_arr = np.asarray(bg, dtype=np.int16)
        for _ in range(noise):
            y = int(rng.integers(0, height))
            xx = int(rng.integers(0, wid))
            # 只落在背景上，颜色靠近底色，避免破坏字符结构
            if pad <= y < pad + big.shape[0] and pad <= xx < pad + big.shape[1] \
                    and big[y - pad, xx - pad]:
                continue
            jitter = rng.integers(-40, 40)
            img[y, xx] = np.clip(bg_arr + jitter, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, "PNG")
    return buf.getvalue()


def random_text(rng: random.Random, lo: int = 1, hi: int = 20) -> str:
    return f"{rng.randint(lo, hi)}+{rng.randint(lo, hi)}="


# 几种典型的「平坦底色 + 高对比字色」配色
DEFAULT_PALETTE = [
    ((255, 255, 255), (20, 20, 20)),      # 白底黑字
    ((12, 18, 40), (235, 235, 235)),      # 深色底白字
    ((40, 40, 28), (230, 220, 120)),      # 暗底亮黄字
    ((95, 30, 35), (250, 250, 250)),      # 暗红底白字
    ((232, 240, 250), (180, 20, 30)),     # 浅底红字
]


def make_dataset(n: int = 60, *, seed: int = 0, lo: int = 1, hi: int = 20,
                 palette=None, scale: int = 3, noise: int = 0):
    """生成 ``[(图片字节, 标准答案), ...]``，可直接喂给
    :meth:`captcha_kit.templates.TemplateSet.from_labeled`。"""
    palette = palette or DEFAULT_PALETTE
    rng = random.Random(seed)
    out = []
    for i in range(n):
        text = random_text(rng, lo, hi)
        bg, fg = palette[i % len(palette)]
        out.append((render(text, scale=scale, bg=bg, fg=fg, noise=noise, seed=i), text))
    return out
