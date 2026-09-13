# -*- coding: utf-8 -*-
"""命令行工具：``captcha-kit``。

三个子命令对应实际工作流：

    discover  无标注地探查字形分布（先看字体是不是固定的）
    build     从「图片 + 标准答案」构建模板集
    read      用模板集识别图片
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .arithmetic import evaluate, is_arithmetic
from .solve import read
from .templates import TemplateSet, discover, render_bitmap

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")


def _images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _load_labels(csv_path: Path | None, files: list[Path]) -> dict[str, str]:
    """标注来源：优先 CSV（``文件名,答案``），否则从 ``答案__xxx.png`` 文件名取答案。"""
    labels: dict[str, str] = {}
    if csv_path:
        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) >= 2 and row[0].strip():
                    labels[row[0].strip()] = row[1].strip()
        return labels
    for f in files:
        stem = f.stem
        labels[f.name] = stem.split("__")[0] if "__" in stem else stem
    return labels


# ------------------------------------------------------------------ 子命令
def cmd_discover(args) -> int:
    files = _images(Path(args.path))
    if not files:
        print(f"没有找到图片: {args.path}", file=sys.stderr)
        return 2
    clusters = discover((f.read_bytes() for f in files), top=args.top)

    total = sum(n for n, _ in clusters)
    print(f"扫描 {len(files)} 张图，共切出 {total} 个字形，聚成 {len(clusters)} 个不同位图")
    print(f"（如果簇数接近字符种类数、且每簇出现次数可观，说明字体是固定的）\n")
    for i, (n, bits) in enumerate(clusters, 1):
        print(f"--- 簇{i}  {bits.shape[1]}x{bits.shape[0]}  出现 {n} 次 ---")
        print(render_bitmap(bits))
        print()
    return 0


def cmd_build(args) -> int:
    root = Path(args.path)
    files = _images(root)
    if not files:
        print(f"没有找到图片: {args.path}", file=sys.stderr)
        return 2
    labels = _load_labels(Path(args.labels) if args.labels else None, files)
    samples = []
    missing = []
    for f in files:
        label = labels.get(f.name)
        if not label:
            missing.append(f.name)
            continue
        samples.append((f.read_bytes(), label))
    if missing:
        print(f"有 {len(missing)} 张图没有标注，已跳过（前几个: {missing[:3]}）", file=sys.stderr)
    if not samples:
        print("没有可用的「图片 + 答案」样本", file=sys.stderr)
        return 2

    try:
        ts, report = TemplateSet.from_labeled(samples, strict=not args.loose)
    except ValueError as exc:
        print(f"建模失败：{exc}", file=sys.stderr)
        return 3

    for label, why in report["skipped"][:5]:
        print(f"  跳过 {label!r}: {why}", file=sys.stderr)
    if len(report["skipped"]) > 5:
        print(f"  ...另外还有 {len(report['skipped']) - 5} 条", file=sys.stderr)

    ts.save(args.output)
    print(f"已写出 {args.output}")
    print(f"  {ts}")
    print(f"  可用样本 {report['used']}/{len(samples)}")
    print(f"  字形尺寸 {json.dumps(ts.meta.get('glyph_sizes', {}), ensure_ascii=False)}")
    return 0


def cmd_read(args) -> int:
    ts = TemplateSet.load(args.templates)
    files = _images(Path(args.path))
    if not files:
        print(f"没有找到图片: {args.path}", file=sys.stderr)
        return 2

    validate = is_arithmetic if args.arithmetic else None
    rc = 0
    for f in files:
        try:
            r = read(f.read_bytes(), ts, validate=validate, min_agree=args.min_agree)
        except Exception as exc:
            print(f"读取失败 {f.name}: {exc}", file=sys.stderr)
            rc = 2
            continue

        if args.clean:
            # 纯净模式：stdout 只有结果本身；失败时完全无输出（错误一律走 stderr）
            if r.ok:
                if args.arithmetic:
                    _expr, answer = evaluate(r.text)
                    if answer is not None:
                        print(answer)
                else:
                    print(r.text)
            else:
                rc = 3
            continue
        if args.json:
            expr, answer = evaluate(r.text) if args.arithmetic else (None, None)
            print(json.dumps({"file": f.name, "text": r.text, "mode": r.mode,
                              "confidence": round(r.confidence, 4),
                              "provable": r.provable, "expression": expr,
                              "answer": answer}, ensure_ascii=False))
            continue
        if not r.ok:
            print(f"!! 识别失败   [{f.name}]")
            rc = 3
            continue
        if args.arithmetic:
            expr, answer = evaluate(r.text)
            print(f"OK {r.text}   [{f.name}]  通道={r.mode}  "
                  f"{'答案=' + str(answer) if answer is not None else '（非算术式）'}")
        else:
            print(f"OK {r.text!r}   [{f.name}]  通道={r.mode}  "
                  f"置信度={r.confidence:.3f}")
        rc = rc or (0 if r.ok else 3)
    return rc


# ------------------------------------------------------------------ 入口
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="captcha-kit",
        description="固定点阵字体验证码识别工具箱")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("discover", help="探查字形分布（建模板前的第一步）")
    p.add_argument("path", help="图片文件或目录")
    p.add_argument("--top", type=int, default=40, help="最多打印多少个簇")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("build", help="从标注样本构建模板集")
    p.add_argument("path", help="图片目录")
    p.add_argument("-l", "--labels", help="CSV 标注文件（文件名,答案）")
    p.add_argument("-o", "--output", default="templates.json", help="输出路径")
    p.add_argument("--loose", action="store_true",
                   help="字体不严格固定时不报错，只记录问题")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("read", help="识别图片")
    p.add_argument("path", help="图片文件或目录")
    p.add_argument("-t", "--templates", default="templates.json", help="模板集路径")
    p.add_argument("-a", "--arithmetic", action="store_true", help="按算术验证码求答案")
    p.add_argument("-c", "--clean", action="store_true",
                   help="纯净模式：stdout 只输出结果本身")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.add_argument("--min-agree", type=float, default=0.65,
                   help="通道 2 的单字形最低 IoU（默认 0.65）")
    p.set_defaults(func=cmd_read)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
