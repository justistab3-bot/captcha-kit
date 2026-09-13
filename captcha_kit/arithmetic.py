# -*- coding: utf-8 -*-
"""算术验证码：把识别出的文本解析成表达式并求值。

用 AST 白名单求值，而不是 ``eval()``。
"""
from __future__ import annotations

import ast
import operator as _op
import re

_BIN = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod,
}
_UNARY = {ast.UAdd: _op.pos, ast.USub: _op.neg}
_SAFE = re.compile(r"[\d+\-*/().]+")
# 形如 "12+7=" / "3x4=" 的算术式（末尾等号可有可无）
PATTERN = re.compile(r"^(\d+)\s*([+\-*x×])\s*(\d+)\s*=?\s*\??$")


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("非法字面量")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError(f"不允许的表达式节点: {type(node).__name__}")


def safe_eval(expression: str):
    """白名单求值，只允许数字与 ``+ - * / ( )``。"""
    if not _SAFE.fullmatch(expression):
        raise ValueError(f"表达式含非法字符: {expression!r}")
    value = _eval(ast.parse(expression, mode="eval"))
    if isinstance(value, float) and abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def extract_expression(text: str) -> str:
    """从识别文本里取出等号之前的算术表达式。"""
    text = re.sub(r"[xX×*]", "*", text)
    cleaned = re.sub(r"[^\d+\-*/().=]", "", text.replace(" ", ""))
    return cleaned.split("=")[0] if "=" in cleaned else cleaned


def evaluate(text: str):
    """返回 ``(表达式, 答案)``；解析不了则 ``(None, None)``。

    只接受形如 ``A op B =`` 的二元式，**刻意不做通用表达式求值**：
    验证码一旦看错一个字符，通用求值会把错误结果算成一个看似合理的数字，
    静默地给出错答案。宁可返回 None 让调用方知道这次没读懂。
    """
    if text is None:
        return None, None
    match = PATTERN.match(text.strip())
    if not match:
        return None, None
    a, op, b = match.group(1), match.group(2), match.group(3)
    op = "*" if op in "x×" else op
    expression = f"{a}{op}{b}"
    try:
        return expression, safe_eval(expression)
    except Exception:
        return None, None


def is_arithmetic(text: str | None) -> bool:
    """给 :func:`captcha_kit.read` 当 ``validate`` 用的谓词。"""
    return evaluate(text)[0] is not None
