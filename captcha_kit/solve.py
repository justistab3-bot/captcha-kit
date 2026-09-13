# -*- coding: utf-8 -*-
"""统一入口：自动在两条通道之间选择。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import content_bbox, soft_foreground, to_binary
from .exact import ExactResult, parse_exact
from .stripe import StripeResult, read_stripe
from .templates import TemplateSet


@dataclass
class Result:
    text: str | None
    mode: str                 # "exact" | "scaled" | "fail"
    confidence: float
    covered: int = 0
    total: int = 0

    @property
    def ok(self) -> bool:
        return self.text is not None

    @property
    def provable(self) -> bool:
        """精确通道解出来的结果带有像素级完备性证明。"""
        return self.mode == "exact"

    def __str__(self) -> str:  # pragma: no cover
        if not self.ok:
            return "识别失败"
        if self.provable:
            return f"{self.text}  [精确 覆盖 {self.covered}/{self.total}]"
        return f"{self.text}  [缩放 相似度 {self.confidence:.3f}]"


def read(image_bytes: bytes, ts: TemplateSet, *, validate=None, min_agree: float = 0.65
         ) -> Result:
    """识别一张验证码，返回 :class:`Result`。

    先走通道 1（像素精确，带完备性证明）；解不出来再走通道 2（归一化 + DP，概率性）。
    """
    exact = _try_exact(image_bytes, ts)
    if exact.ok:
        return Result(exact.text, "exact", 1.0, exact.covered, exact.total)

    scaled = read_stripe(image_bytes, ts, validate=validate, min_agree=min_agree)
    if scaled.ok:
        return Result(scaled.text, "scaled", scaled.confidence)

    return Result(None, "fail", 0.0)


def _try_exact(image_bytes: bytes, ts: TemplateSet) -> ExactResult:
    try:
        soft, _bg = soft_foreground(image_bytes)
        box = content_bbox(soft)
        if box is None:
            return ExactResult(None, 0, 0, 0)
        top, bottom, left, right = box
        height = bottom - top + 1
        # 高度不等于内容带高度，说明图片被缩放过，通道 1 不适用
        if height != ts.band_height:
            return ExactResult(None, 0, int((soft > 0.5).sum()), 0)
        canvas = to_binary(soft, 0.5)[top:bottom + 1, left:right + 1]
        return parse_exact(canvas, ts)
    except Exception:
        return ExactResult(None, 0, 0, 0)
