# -*- coding: utf-8 -*-
"""captcha-kit 测试套件。

全部基于仓库自带的合成验证码生成器，不依赖任何真实站点样本，也不需要联网。
"""
from __future__ import annotations

import io
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from captcha_kit import (TemplateSet, arithmetic, column_segments, discover,
                         estimate_background, read, soft_foreground)
from captcha_kit.color import content_bbox, load_rgb, to_binary
from captcha_kit.exact import check_edge_columns, parse_exact
from captcha_kit.synthetic import FONT_5X7, glyph, make_dataset, render

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return make_dataset(60, seed=1)


@pytest.fixture(scope="module")
def templates(dataset):
    ts, _report = TemplateSet.from_labeled(dataset)
    return ts


def rescale(image_bytes: bytes, sx: float, sy: float) -> bytes:
    im = Image.open(io.BytesIO(image_bytes))
    buf = io.BytesIO()
    im.resize((round(im.width * sx), round(im.height * sy)), Image.BILINEAR).save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ 颜色
def test_background_detection():
    img = render("7+3=", bg=(12, 18, 40), fg=(235, 235, 235))
    assert tuple(estimate_background(load_rgb(img))) == (12, 18, 40)


def test_soft_foreground_is_contrast_invariant():
    """极端对比度和低对比度应当得到同样的强度分布。"""
    high = soft_foreground(render("12+3=", bg=(255, 255, 255), fg=(0, 0, 0)))[0]
    low = soft_foreground(render("12+3=", bg=(200, 200, 200), fg=(120, 120, 120)))[0]
    assert high.max() == pytest.approx(1.0)
    assert low.max() == pytest.approx(1.0)
    assert abs(float(high.mean()) - float(low.mean())) < 0.05


def test_content_bbox_trims_padding():
    img = render("5+5=", pad=9)
    soft, _ = soft_foreground(img)
    top, bottom, left, right = content_bbox(soft)
    full = to_binary(soft)
    assert top > 0 and left > 0
    assert bottom < full.shape[0] - 1 and right < full.shape[1] - 1


# ------------------------------------------------------------------ 切分与探查
def test_column_segments_splits_glyphs():
    mask = to_binary(soft_foreground(render("18+1="))[0])
    segs = column_segments(mask, gap=2)
    assert len(segs) == 5


def test_discover_clusters_to_charset(dataset):
    clusters = discover((img for img, _ in dataset[:40]))
    # 合成字体用到 12 个字形（0-9, +, =）
    assert len(clusters) == 12
    total = sum(n for n, _ in clusters)
    assert total > 40 * 4


# ------------------------------------------------------------------ 模板构建
def test_build_templates_basic(templates):
    assert set(templates.charset) == set("0123456789+=")
    assert templates.band_height == 7 * 3
    assert not check_edge_columns(templates), "每个字形首列/末列都必须有前景像素"


def test_build_reports_unusable_samples(dataset):
    # 故意把一个标注写错，该样本应被跳过而不是污染模板集
    broken = dataset[:20] + [(dataset[0][0], "0+0=")]
    ts, report = TemplateSet.from_labeled(broken)
    assert report["used"] == 20
    assert len(report["skipped"]) == 1


def test_build_rejects_non_fixed_font():
    """字体不固定（同一字符两种形状）时应当直接报错。"""
    same_text = [(render("1+1=", scale=3), "1+1="), (render("1+1=", scale=4), "1+1=")]
    with pytest.raises(ValueError):
        TemplateSet.from_labeled(same_text)


def test_template_roundtrip(tmp_path, templates):
    path = tmp_path / "t.json"
    templates.save(path)
    loaded = TemplateSet.load(path)
    assert loaded.charset == templates.charset
    assert loaded.band_height == templates.band_height
    assert loaded.offsets == templates.offsets
    for ch in templates.charset:
        assert np.array_equal(loaded.glyphs[ch], templates.glyphs[ch])


# ------------------------------------------------------------------ 通道 1
def test_exact_channel_all_correct(dataset, templates):
    for img, label in dataset:
        r = read(img, templates, validate=arithmetic.is_arithmetic)
        assert r.mode == "exact"
        assert r.text == label
        assert r.provable
        assert r.covered == r.total


def test_exact_channel_proves_coverage(templates):
    img = render("13+15=", scale=3)
    soft, _ = soft_foreground(img)
    top, bottom, left, right = content_bbox(soft)
    canvas = to_binary(soft)[top:bottom + 1, left:right + 1]
    res = parse_exact(canvas, templates)
    assert res.ok and res.text == "13+15="
    assert res.covered == res.total


def test_exact_channel_rejects_unknown_glyph():
    """字符不在模板集里时，通道 1 必须解不出来，而不是硬凑一个错误答案。"""
    subset = [(render(f"{a}+{b}=", scale=3), f"{a}+{b}=")
              for a in "0123" for b in "0123"]
    ts, _ = TemplateSet.from_labeled(subset)
    assert set(ts.charset) == set("0123+=")

    r = read(render("7+1=", scale=3), ts)
    assert r.text != "7+1="


def test_exact_channel_requires_full_coverage(templates):
    """完备性校验：多出一个不落在任何模板上的前景像素，就必须判定无解。"""
    img = render("12+3=", scale=3)
    soft, _ = soft_foreground(img)
    top, bottom, left, right = content_bbox(soft)
    canvas = to_binary(soft)[top:bottom + 1, left:right + 1].copy()

    assert parse_exact(canvas, templates).ok

    canvas[0, 0] = True                     # 一个不属于任何字形的游离像素
    assert not parse_exact(canvas, templates).ok


# ------------------------------------------------------------------ 通道 2
@pytest.mark.parametrize("sx,sy", [(1.68, 2.75), (1.39, 2.33), (1.5, 1.5),
                                   (2.0, 2.0), (1.2, 1.6)])
def test_scaled_channel(dataset, templates, sx, sy):
    """缩放截图通道的正确率。阈值留一点余量，避免受浮点差异影响。"""
    good = 0
    for img, label in dataset[:20]:
        r = read(rescale(img, sx, sy), templates, validate=arithmetic.is_arithmetic)
        good += (r.text == label)
    assert good >= 19, f"{sx}x{sy} 只对了 {good}/20"


def test_scaled_channel_reports_mode(templates):
    img = render("17+25=", scale=3)
    r = read(rescale(img, 1.68, 2.75), templates)
    assert r.mode == "scaled"
    assert not r.provable
    assert 0.0 < r.confidence <= 1.0


def test_heavily_downscaled_refuses(templates):
    """缩小到 0.5 倍时字形只剩几个像素，应当拒绝作答而不是乱猜。"""
    dataset = make_dataset(10, seed=3)
    wrong = 0
    for img, label in dataset:
        r = read(rescale(img, 0.5, 0.5), templates, validate=arithmetic.is_arithmetic)
        if r.ok and r.text != label:
            wrong += 1
    assert wrong == 0


def test_read_never_raises_on_garbage(templates):
    """非法输入应当返回 fail，而不是抛异常 —— 库不该把解码错误甩给调用方。"""
    for bad in [b"", b"\x00" * 100, b"definitely not an image", b"\x89PNG\r\n\x1a\n"]:
        r = read(bad, templates)
        assert r.text is None
        assert r.mode == "fail"
        assert not r.provable


# ------------------------------------------------------------------ 求值
def test_arithmetic_evaluate():
    assert arithmetic.evaluate("13+15=") == ("13+15", 28)
    assert arithmetic.evaluate("7+12=") == ("7+12", 19)
    assert arithmetic.evaluate("3x4=") == ("3*4", 12)
    assert arithmetic.evaluate("9-4=") == ("9-4", 5)


def test_arithmetic_rejects_garbage():
    for bad in [None, "", "abc", "++", "1+2+3"]:
        assert arithmetic.evaluate(bad) == (None, None)


def test_safe_eval_blocks_code_execution():
    for evil in ["__import__('os').system('echo pwned')", "1;2", "open('/etc/passwd')"]:
        with pytest.raises(ValueError):
            arithmetic.safe_eval(evil)


def test_is_arithmetic_predicate():
    assert arithmetic.is_arithmetic("13+15=")
    assert not arithmetic.is_arithmetic("abc")


# ------------------------------------------------------------------ 合成器
def test_font_is_rectangular():
    for ch, rows in FONT_5X7.items():
        assert len(rows) == 7, ch
        assert all(len(r) == 5 for r in rows), ch
        assert glyph(ch).shape == (7, 5)


def test_render_scale_and_size():
    small = render("1+1=", scale=1, pad=0)
    big = render("1+1=", scale=4, pad=0)
    a, b = Image.open(io.BytesIO(small)), Image.open(io.BytesIO(big))
    assert (b.width, b.height) == (a.width * 4, a.height * 4)


def test_make_dataset_is_deterministic():
    assert [t for _, t in make_dataset(5, seed=9)] == [t for _, t in make_dataset(5, seed=9)]


# ------------------------------------------------------------------ 集成
def test_integration_with_noise_uses_scaled_channel(templates):
    """带噪点走通道 2 依然应当读对（噪声颜色贴近底色，不破坏字形）。"""
    good = 0
    for img, label in make_dataset(20, seed=11, noise=120):
        r = read(img, templates, validate=arithmetic.is_arithmetic)
        good += (r.text == label)
    assert good >= 18


def test_cli_build_and_read(tmp_path):
    dataset = make_dataset(40, seed=5)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    labels = []
    for i, (raw, label) in enumerate(dataset):
        name = f"{i:03d}.png"
        (img_dir / name).write_bytes(raw)
        labels.append(f"{name},{label}")
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("\n".join(labels), encoding="utf-8")
    tpl = tmp_path / "templates.json"

    env = {"PYTHONPATH": str(ROOT)}
    build = subprocess.run(
        [sys.executable, "-m", "captcha_kit.cli", "build", str(img_dir),
         "-l", str(csv_path), "-o", str(tpl)],
        capture_output=True, text=True, env={**env, "PATH": ""}, cwd=str(ROOT))
    assert build.returncode == 0, build.stderr
    assert tpl.exists()

    read_run = subprocess.run(
        [sys.executable, "-m", "captcha_kit.cli", "read", str(img_dir / "000.png"),
         "-t", str(tpl), "-a", "-c"],
        capture_output=True, text=True, env={**env, "PATH": ""}, cwd=str(ROOT))
    assert read_run.returncode == 0, read_run.stderr
    assert read_run.stdout.strip().isdigit()


def test_cli_clean_mode_keeps_stdout_pure(tmp_path):
    dataset = make_dataset(40, seed=6)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    labels = []
    for i, (raw, label) in enumerate(dataset):
        (img_dir / f"{i:03d}.png").write_bytes(raw)
        labels.append(f"{i:03d}.png,{label}")
    (tmp_path / "labels.csv").write_text("\n".join(labels), encoding="utf-8")
    tpl = tmp_path / "t.json"
    env = {**{"PYTHONPATH": str(ROOT)}, "PATH": ""}

    subprocess.run([sys.executable, "-m", "captcha_kit.cli", "build", str(img_dir),
                    "-l", str(tmp_path / "labels.csv"), "-o", str(tpl)],
                   capture_output=True, text=True, env=env, cwd=str(ROOT), check=True)

    good = subprocess.run(
        [sys.executable, "-m", "captcha_kit.cli", "read", str(img_dir), "-t", str(tpl), "-c"],
        capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert good.returncode == 0
    lines = good.stdout.strip().splitlines()
    assert len(lines) == len(dataset)
    assert all(line.strip() for line in lines)

    bad = subprocess.run(
        [sys.executable, "-m", "captcha_kit.cli", "read", str(tmp_path / "nope.png"),
         "-t", str(tpl), "-c"],
        capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert bad.returncode == 2
    assert bad.stdout == ""
    assert bad.stderr != ""
