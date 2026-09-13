# -*- coding: utf-8 -*-
"""字形模板集：构建、保存、加载。

固定点阵字体的验证码有个很强的性质：**同一个字符在任意两张图里逐像素完全相同**。
只要验证了这个性质，识别就可以退化成精确查表，不需要任何机器学习。

这个模块负责：
* 从一批「图片 + 标准答案」里把字形抠出来、聚类、校验；
* 校验字体确实是固定的（一个字符只对应一个位图，一个位图只对应一个字符）；
* 存成 JSON 供识别器加载。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .color import content_bbox, soft_foreground, to_binary


# ------------------------------------------------------------------ 切分
def column_segments(mask: np.ndarray, gap: int = 2) -> list[np.ndarray]:
    """按列投影把前景切成独立的字形位图（各自裁到紧致外框）。

    ``gap`` 是判定字形边界的连续空白列数。注意：**这个方法只在原图尺度下可靠**。
    图片一旦被缩放插值，空隙会消失或断裂，那时请用 :mod:`captcha_kit.stripe`。
    """
    has = mask.any(axis=0)
    spans: list[tuple[int, int]] = []
    start, blank = None, 0
    for x, on in enumerate(has):
        if on:
            if start is None:
                start = x
            blank = 0
        elif start is not None:
            blank += 1
            if blank >= gap:
                spans.append((start, x - blank))
                start = None
    if start is not None:
        spans.append((start, len(has) - 1))

    out = []
    for x0, x1 in spans:
        sub = mask[:, x0:x1 + 1]
        rows = np.where(sub.any(axis=1))[0]
        if rows.size:
            out.append(sub[int(rows.min()):int(rows.max()) + 1, :])
    return out


def bitmap_key(bits: np.ndarray) -> tuple:
    """位图的精确身份，用于聚类。"""
    return (bits.shape[1], bits.shape[0], bits.tobytes())


def render_bitmap(bits: np.ndarray, on: str = "#", off: str = ".") -> str:
    return "\n".join("".join(on if v else off for v in row) for row in bits)


# ------------------------------------------------------------------ 模板集
@dataclass
class TemplateSet:
    """一套字形模板。

    ``glyphs``   : 字符 -> 0/1 位图（各自紧致外框）
    ``offsets``  : 字符 -> 允许的首行偏移元组（相对内容带顶行）
    ``band_height``: 内容带高度 = max(offset + 字形高度)
    """

    glyphs: dict[str, np.ndarray]
    offsets: dict[str, tuple[int, ...]]
    band_height: int
    meta: dict = field(default_factory=dict)

    @property
    def charset(self) -> str:
        return "".join(sorted(self.glyphs))

    def __repr__(self) -> str:  # pragma: no cover
        return (f"TemplateSet({len(self.glyphs)} 个字形, "
                f"band_height={self.band_height}, charset={self.charset!r})")

    # -- 序列化 ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "band_height": self.band_height,
            "glyphs": {c: render_bitmap(g).split("\n") for c, g in self.glyphs.items()},
            "offsets": {c: list(o) for c, o in self.offsets.items()},
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateSet":
        glyphs = {
            c: np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
            for c, rows in data["glyphs"].items()
        }
        offsets = {c: tuple(int(v) for v in o) for c, o in data["offsets"].items()}
        return cls(glyphs=glyphs, offsets=offsets,
                   band_height=int(data["band_height"]), meta=data.get("meta", {}))

    @classmethod
    def load(cls, path: str | Path) -> "TemplateSet":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- 构建 --------------------------------------------------------
    @classmethod
    def from_labeled(cls, samples, threshold: float = 0.5, gap: int = 2,
                     strict: bool = True):
        """从 ``[(图片字节, 标准答案), ...]`` 构建模板。

        会校验字体确实是固定的：每个位图只能对应一个字符，每个字符只能对应一个位图。
        不满足就说明这不是固定点阵字体（或者标注有错），此时 ``strict=True`` 会直接报错。

        返回 ``(TemplateSet, 报告字典)``。
        """
        by_key: dict[tuple, np.ndarray] = {}
        key_label: dict[tuple, Counter] = defaultdict(Counter)
        label_offsets: dict[str, set[int]] = defaultdict(set)
        report = {"used": 0, "skipped": [], "segments": Counter()}

        for image_bytes, label in samples:
            soft, _bg = soft_foreground(image_bytes)
            box = content_bbox(soft)
            if box is None:
                report["skipped"].append((label, "没有前景内容"))
                continue
            top, bottom, left, right = box
            canvas = to_binary(soft, threshold)[top:bottom + 1, left:right + 1]

            segs = column_segments(canvas, gap=gap)
            if len(segs) != len(label):
                report["skipped"].append(
                    (label, f"切出 {len(segs)} 段，标注 {len(label)} 个字符"))
                continue

            # 重新按列定位以取得相对内容带顶行的偏移
            x = 0
            for ch, seg in zip(label, segs):
                w = seg.shape[1]
                while x < canvas.shape[1] and not canvas[:, x].any():
                    x += 1
                sub = canvas[:, x:x + w]
                rows = np.where(sub.any(axis=1))[0]
                key = bitmap_key(seg)
                by_key[key] = seg
                key_label[key][ch] += 1
                label_offsets[ch].add(int(rows.min()))
                x += w
            report["used"] += 1
            report["segments"][len(segs)] += 1

        if not by_key:
            raise ValueError("没有任何样本可用，检查图片内容和标注是否对齐")

        # 校验：位图 <-> 字符 必须一一对应
        glyphs, offsets = {}, {}
        problems = []
        key_to_char = {}
        for key, counter in key_label.items():
            if len(counter) > 1:
                problems.append(f"同一个位图被标成多个字符: {dict(counter)}")
            key_to_char[key] = counter.most_common(1)[0][0]
        char_to_key = {}
        for key, ch in key_to_char.items():
            if ch in char_to_key:
                problems.append(f"字符 {ch!r} 对应了多个不同位图（字体不固定？）")
            char_to_key[ch] = key
        if problems and strict:
            raise ValueError("字体不是固定点阵，或标注有误：\n  - " + "\n  - ".join(problems))
        if problems:
            report["problems"] = problems

        for ch, key in char_to_key.items():
            glyphs[ch] = by_key[key]
            offsets[ch] = tuple(sorted(label_offsets[ch]))

        band = max(off + glyphs[ch].shape[0]
                   for ch in glyphs for off in offsets[ch])
        meta = {
            "samples": report["used"],
            "glyph_sizes": {c: list(g.shape) for c, g in glyphs.items()},
        }
        return cls(glyphs=glyphs, offsets=offsets, band_height=band, meta=meta), report


def discover(images, threshold: float = 0.5, gap: int = 2, top: int = 40):
    """无标注地探查：把一批图里的字形聚成簇，用于先看看字体是不是固定的。

    返回 ``[(出现次数, 位图), ...]``（按次数降序，最多 ``top`` 个）。
    """
    counts: Counter = Counter()
    reps: dict[tuple, np.ndarray] = {}
    for image_bytes in images:
        soft, _ = soft_foreground(image_bytes)
        box = content_bbox(soft)
        if box is None:
            continue
        t, b, l, r = box
        canvas = to_binary(soft, threshold)[t:b + 1, l:r + 1]
        for seg in column_segments(canvas, gap=gap):
            k = bitmap_key(seg)
            counts[k] += 1
            reps.setdefault(k, seg)
    return [(n, reps[k]) for k, n in counts.most_common(top)]
