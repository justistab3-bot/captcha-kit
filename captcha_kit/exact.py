# -*- coding: utf-8 -*-
"""通道 1：像素精确模板匹配（结果可证明）。

思路：既然字体是逐像素固定的，那正确解只有一个 —— 把模板在条带上摆好之后，
**所有前景像素恰好被模板覆盖，一个不多一个不少**。所以这里不设相似度阈值，
而是要求逐位相等 + 完全覆盖，解出来即为证明。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .templates import TemplateSet


@dataclass
class ExactResult:
    text: str | None
    covered: int
    total: int
    nodes: int
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.text is not None

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0


def parse_exact(canvas: np.ndarray, ts: TemplateSet, max_nodes: int = 200_000) -> ExactResult:
    """在二值条带 ``canvas``（``band_height × W``）上求唯一精确解。

    ``text`` 为 None 表示无解（不是固定字体、有噪点、条带被截断等）。
    """
    H, W = canvas.shape
    if H != ts.band_height:
        raise ValueError(f"条带高度 {H} 与模板集 band_height {ts.band_height} 不符")

    target = int(canvas.sum())
    items = sorted(ts.glyphs.items(), key=lambda kv: -kv[1].shape[1])
    state = {"nodes": 0, "truncated": False, "text": None, "covered": 0}
    cover = np.zeros_like(canvas)

    def rec(x: int, out: list[str], covered: int) -> None:
        if state["text"] is not None:
            return
        state["nodes"] += 1
        if state["nodes"] > max_nodes:
            state["truncated"] = True
            return
        # 空白列直接跳过；剩下第一个有内容的列，必然是下一个字形的起始列
        # （模板集保证每个字形的首列/末列都有前景像素）
        while x < W and not canvas[:, x].any():
            x += 1
        if x >= W:
            if covered == target and np.array_equal(cover, canvas):
                state["text"] = "".join(out)
                state["covered"] = covered
            return

        for ch, tpl in items:
            h, w = tpl.shape
            if x + w > W:
                continue
            for y0 in ts.offsets[ch]:
                if y0 + h > H:
                    continue
                win = canvas[y0:y0 + h, x:x + w]
                if not np.array_equal(win, tpl):
                    continue
                cover[y0:y0 + h, x:x + w] |= tpl     # 只 OR 模板里的前景像素
                out.append(ch)
                rec(x + w, out, covered + int(tpl.sum()))
                out.pop()
                cover[y0:y0 + h, x:x + w] &= ~tpl
                if state["text"] is not None:
                    return

    rec(0, [], 0)
    return ExactResult(state["text"], state["covered"], target,
                       state["nodes"], state["truncated"])


def check_edge_columns(ts: TemplateSet) -> list[str]:
    """校验每个字形首列/末列都有前景像素 —— 精确匹配的推进逻辑依赖这个性质。"""
    bad = []
    for ch, tpl in ts.glyphs.items():
        if not (tpl[:, 0].any() and tpl[:, -1].any()):
            bad.append(ch)
    return bad
