# -*- coding: utf-8 -*-
"""端到端演示：合成一套验证码 -> 建模板 -> 两条通道识别 -> 求算术答案。

    python examples/demo.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):          # Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image

from captcha_kit import TemplateSet, arithmetic, read
from captcha_kit.synthetic import make_dataset, render


def line(title: str = "") -> None:
    print(("── " + title + " ").ljust(74, "─") if title else "─" * 74)


def main() -> int:
    line("1. 生成合成验证码（仓库自带 5x7 点阵字体，不依赖任何真实样本）")
    samples = make_dataset(60, seed=1)
    print(f"   {len(samples)} 张，示例答案: {samples[0][1]!r} {samples[1][1]!r} {samples[2][1]!r}")

    line("2. 从标注样本构建模板集")
    ts, report = TemplateSet.from_labeled(samples)
    print(f"   {ts}")
    print(f"   可用样本 {report['used']}/{len(samples)}，"
          f"字形数分布 {dict(report['segments'])}")
    print(f"   字符      : {ts.charset}")
    print(f"   垂直偏移  : {ts.offsets}")

    line("3. 通道 1：原图（像素精确，结果可证明）")
    ok = 0
    t0 = time.perf_counter()
    for img, label in samples:
        r = read(img, ts, validate=arithmetic.is_arithmetic)
        ok += (r.text == label and r.provable)
    dt = time.perf_counter() - t0
    print(f"   {ok}/{len(samples)} 通过，{dt:.2f}s（{dt / len(samples) * 1000:.1f} ms/张）")

    line("4. 通道 2：缩放/抗锯齿截图（模拟浏览器显示后截图）")
    for sx, sy in [(1.68, 2.75), (1.39, 2.33), (1.5, 1.5), (2.0, 2.0), (1.2, 1.6)]:
        good, modes = 0, {"exact": 0, "scaled": 0, "fail": 0}
        t0 = time.perf_counter()
        for img, label in samples:
            im = Image.open(io.BytesIO(img))
            buf = io.BytesIO()
            im.resize((round(im.width * sx), round(im.height * sy)),
                      Image.BILINEAR).save(buf, "PNG")
            r = read(buf.getvalue(), ts, validate=arithmetic.is_arithmetic)
            good += (r.text == label)
            modes[r.mode] += 1
        print(f"   {sx}x{sy:<5} 正确 {good}/{len(samples)}  走通道 {modes}  "
              f"{time.perf_counter() - t0:.1f}s")

    line("5. 单张完整流程")
    img = render("17+25=", scale=3, bg=(12, 18, 40), fg=(235, 235, 235))
    r = read(img, ts, validate=arithmetic.is_arithmetic)
    expr, answer = arithmetic.evaluate(r.text)
    print(f"   识别文本 : {r.text!r}")
    print(f"   通道     : {r.mode}（provable={r.provable}）")
    print(f"   表达式   : {expr}")
    print(f"   答案     : {answer}")
    print(f"   校验     : {expr} = {answer}  "
          f"{'✓' if answer == 42 else '✗'}")

    line("6. 失败时的行为：给一张噪声图，应当拒绝作答而不是硬猜")
    bad = bytes([0]) * 64
    r = read(bad, ts, validate=arithmetic.is_arithmetic)
    print(f"   非法输入 -> text={r.text!r} mode={r.mode!r}")

    line()
    print("演示结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
