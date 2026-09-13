# -*- coding: utf-8 -*-
"""captcha-kit：固定点阵字体验证码的识别工具箱。

适用于「同一字符每张图逐像素完全相同」的验证码。两条通道：

* :mod:`captcha_kit.exact`  —— 像素精确模板匹配，结果带完备性证明；
* :mod:`captcha_kit.stripe` —— 缩放/抗锯齿图片的归一化 + DP 联合切分识别。

典型用法::

    from captcha_kit import TemplateSet, read, arithmetic

    ts = TemplateSet.from_labeled(samples)[0]      # samples = [(图片字节, 答案)]
    result = read(图片字节, ts, validate=arithmetic.is_arithmetic)
    print(result.text, arithmetic.evaluate(result.text))
"""
from .arithmetic import evaluate, extract_expression, is_arithmetic, safe_eval
from .color import (content_bbox, estimate_background, load_rgb, soft_foreground,
                    to_binary)
from .exact import ExactResult, parse_exact
from .solve import Result, read
from .stripe import StripeResult, read_stripe
from .templates import TemplateSet, column_segments, discover, render_bitmap
from .synthetic import FONT_5X7, glyph, make_dataset, render

__version__ = "0.1.0"

__all__ = [
    "TemplateSet", "column_segments", "discover", "render_bitmap",
    "read", "Result", "parse_exact", "ExactResult", "read_stripe", "StripeResult",
    "soft_foreground", "content_bbox", "estimate_background", "load_rgb", "to_binary",
    "safe_eval", "evaluate", "extract_expression", "is_arithmetic",
    "render", "make_dataset", "glyph", "FONT_5X7",
    "__version__",
]
