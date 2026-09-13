# -*- coding: utf-8 -*-
"""通道 2：缩放 / 抗锯齿图片的识别（归一化 + DP 联合切分识别）。

什么时候需要这条通道
-------------------
浏览器把 100×24 的验证码显示成 ``width:139px;height:56px`` 之类，是按 1.39×2.33
**非等比**放大并插值的；截图还会有系统 DPI 带来的另一套倍率。字符边缘被抹成一堆
中间色，"逐像素相等" 的前提就没了，通道 1 必然解不出来。

为什么不能靠"列空隙"切字
-----------------------
实测过：放大后一个字会被抗锯齿切成好几段（``6+5=`` 切成 7 段），缩小后相邻字又会
粘在一起（``10+14=`` 并成 4 段）。空隙法在这种输入上切字错误率极高，而认字错误率
却是 0 —— 也就是说瓶颈完全在切分，不在识别。

所以这里的做法是**把切分和识别一起解**：

* 纵向：内容带高度是已知常数（``TemplateSet.band_height``），直接归一化掉纵向缩放；
* 横向：枚举"规范总宽度 W"，把整条内容缩到 ``(band_height, W)``，用 DP 在整条上求
  最优模板序列，取契合度最高的 W。

不需要知道原始缩放比，也不需要猜任何空隙阈值。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from .color import content_bbox, soft_foreground
from .templates import TemplateSet


@dataclass
class StripeResult:
    text: str | None
    confidence: float
    width: int

    @property
    def ok(self) -> bool:
        return self.text is not None


def _agreement_row(can: np.ndarray, tpl: np.ndarray, y0: int) -> np.ndarray | None:
    """模板放在每个 x 位置上的契合度（一次算完整行，避免逐像素循环）。

    用的是**前景模糊 IoU**：``Σ(c·t) / (Σc + Σt − Σ(c·t))``。

    这一点很关键。最初我用的是 ``1 − mean|c − t|``，结果 0.5 倍缩小时 10/10 全错 ——
    因为字模大部分区域是背景，逐像素一致率把背景也算进去，错的字形照样能得 0.8+。
    换成 IoU 之后，背景不贡献任何分数，只有前景真正重叠才有分。

    两个求和都可以用向量化方式一次算完：
    ``Σc`` 是窗口和（前缀和滑窗），``Σ(c·t)`` 是模板前景点处的相关（位移相加）。
    """
    h, w = tpl.shape
    H, W = can.shape
    if y0 + h > H or w > W:
        return None

    band = can[y0:y0 + h, :]
    pref = np.zeros((h, W + 1), dtype=np.float64)
    np.cumsum(band, axis=1, out=pref[:, 1:])
    window_sum = (pref[:, w:] - pref[:, :-w]).sum(axis=0)        # Σc，长度 W-w+1

    overlap = np.zeros(W - w + 1, dtype=np.float64)
    for i, j in zip(*np.nonzero(tpl)):
        overlap += band[i, j:j + (W - w + 1)]                    # Σ(c·t)

    tpl_sum = float(tpl.sum())
    return (overlap / (window_sum + tpl_sum - overlap + 1e-6)).astype(np.float32)


def _agreement_table(can: np.ndarray, ts: TemplateSet) -> dict:
    table = {}
    for ch, tpl in ts.glyphs.items():
        for y0 in ts.offsets[ch]:
            row = _agreement_row(can, tpl, y0)
            if row is not None:
                table[(ch, y0)] = (row, tpl.shape[1])
    return table


def _dp(can: np.ndarray, table: dict, min_agree: float, skip_penalty: float = 0.35):
    """DP 求最优模板序列，返回 ``(文本, 字形数, 总契合度)``。"""
    H, W = can.shape
    NEG = -1e9
    best = np.full(W + 1, NEG, dtype=np.float64)
    back: list = [None] * (W + 1)
    best[0] = 0.0
    col_has = can.max(axis=0) > 0.35

    for x in range(W):
        if best[x] == NEG:
            continue
        # 跳过一列：空白列免费，有内容才罚分（容忍抗锯齿毛刺）
        pen = 0.0 if not col_has[x] else skip_penalty
        if best[x] - pen > best[x + 1]:
            best[x + 1] = best[x] - pen
            back[x + 1] = (x, None)
        # 在此放一个模板
        for (ch, _y0), (row, w) in table.items():
            if x + w > W:
                continue
            a = float(row[x])
            if a < min_agree:
                continue
            if best[x] + a > best[x + w]:
                best[x + w] = best[x] + a
                back[x + w] = (x, ch)

    if best[W] == NEG:
        return None, 0, 0.0
    chars, cur, n = [], W, 0
    while cur > 0:
        prev, ch = back[cur]
        if ch is not None:
            chars.append(ch)
            n += 1
        cur = prev
    return "".join(reversed(chars)), n, float(best[W])


def read_stripe(
    image_bytes: bytes,
    ts: TemplateSet,
    *,
    width_range=range(20, 160),
    min_agree: float = 0.65,
    coarse_step: int = 3,
    skip_penalty: float = 0.35,
    validate=None,
) -> StripeResult:
    """识别缩放/模糊图片。

    ``min_agree`` 是单字形的最低 IoU。实测基准（合成数据，见 README）：0.65 时
    0.8~2.75 倍缩放全部满分，而 0.5 倍时干净拒绝、零错答；调到 0.75 会让轻度模糊的
    图开始漏判，调到 0.55 则会在信息已丢失的图上开始乱猜。

    ``validate`` 是个可选回调，用来在多个候选宽度里挑选语义合法的解
    （例如算术验证码可以要求结果符合 ``\\d+\\+\\d+=``）。
    """
    try:
        soft, _bg = soft_foreground(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError):
        return StripeResult(None, 0.0, 0)      # 图片根本解不开，交给调用方当失败处理
    hard = soft > 0.35
    if int(hard.sum()) < 15:
        return StripeResult(None, 0.0, 0)
    box = content_bbox(soft)
    if box is None:
        return StripeResult(None, 0.0, 0)
    top, bottom, left, right = box
    if bottom - top < 3 or right - left < 4:
        return StripeResult(None, 0.0, 0)

    crop = soft[top:bottom + 1, left:right + 1]
    src = Image.fromarray(crop.astype(np.float32), "F")

    def evaluate(width: int):
        if width not in width_range:
            return None
        can = np.asarray(src.resize((width, ts.band_height), Image.BILINEAR),
                         dtype=np.float32)
        text, n, score = _dp(can, _agreement_table(can, ts), min_agree, skip_penalty)
        if not text or n < 2:
            return None
        if validate is not None and not validate(text):
            return None
        return (score / n, text, width)

    candidates = []
    for width in width_range:
        if (width - width_range.start) % coarse_step:
            continue
        got = evaluate(width)
        if got:
            candidates.append(got)
            if got[0] >= 0.99:          # 已经近乎逐像素吻合，不必再搜
                return StripeResult(got[1], float(got[0]), int(got[2]))
    if not candidates:
        return StripeResult(None, 0.0, 0)

    candidates.sort(key=lambda c: -c[0])
    best = candidates[0]
    # 在最优宽度附近细化
    for _conf, _text, w0 in candidates[:2]:
        for width in range(w0 - coarse_step + 1, w0 + coarse_step):
            got = evaluate(width)
            if got and got[0] > best[0]:
                best = got

    conf, text, width = best
    return StripeResult(text, float(conf), int(width))
