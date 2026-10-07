"""Chuẩn hoá chữ tiếng Việt trước khi đọc (số -> chữ, bỏ ký tự lạ)."""
from __future__ import annotations

import re

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _three(n: int, full: bool) -> str:
    tr, ch, dv = n // 100, (n // 10) % 10, n % 10
    out = []
    if tr or full:
        out += [DIGITS[tr], "trăm"]
    if ch == 0:
        if dv and (tr or full):
            out.append("lẻ")
    elif ch == 1:
        out.append("mười")
    else:
        out += [DIGITS[ch], "mươi"]
    if dv:
        if dv == 1 and ch > 1:
            out.append("mốt")
        elif dv == 5 and ch > 0:
            out.append("lăm")
        elif dv == 4 and ch > 1:
            out.append("tư")
        else:
            out.append(DIGITS[dv])
    return " ".join(out)


def number_to_words(n: int) -> str:
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + number_to_words(-n)
    units = ["", "nghìn", "triệu", "tỷ"]
    parts = []
    i = 0
    while n > 0 and i < len(units):
        parts.append(n % 1000)
        n //= 1000
        i += 1
    words = []
    for idx in range(len(parts) - 1, -1, -1):
        g = parts[idx]
        if g == 0:
            continue
        full = idx < len(parts) - 1
        words.append(_three(g, full))
        if units[idx]:
            words.append(units[idx])
    return " ".join(words)


def normalize_vi(text: str) -> str:
    t = text or ""
    t = t.replace("%", " phần trăm").replace("&", " và ")
    t = re.sub(r"\$\s*(\d+)", r"\1 đô", t)
    t = re.sub(r"(\d)\s*\$", r"\1 đô", t)
    for u, w in (("kg", "ký"), ("km", "cây số"), ("cm", "xăng ti mét"), ("mm", "mi li mét"), ("ml", "mi li lít"),
                 ("đ", "đồng"), ("vnd", "đồng"), ("usd", "đô")):
        t = re.sub(r"(\d)\s*" + u + r"\b", r"\1 " + w, t, flags=re.I)
    while re.search(r"\d\.\d{3}\b", t):  # 1.500.000 -> 1500000
        t = re.sub(r"(\d)\.(\d{3})\b", r"\1\2", t)
    t = re.sub(r"(\d+)[.,](\d+)", lambda m: f"{number_to_words(int(m.group(1)))} phẩy {number_to_words(int(m.group(2)))}", t)
    t = re.sub(r"(\d+)\s*[kK]\b", lambda m: f"{number_to_words(int(m.group(1)))} nghìn", t)
    t = re.sub(r"(\d+)\s*(?:tr|triệu)\b", lambda m: f"{number_to_words(int(m.group(1)))} triệu", t)
    t = re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))) if len(m.group(0)) <= 12 else m.group(0), t)
    t = re.sub(r"[\"“”«»()\[\]{}<>#*_~^|\\/@]", " ", t)
    t = t.replace("…", ", ").replace("...", ", ")
    t = re.sub(r"\s+", " ", t).strip(" ,;:-")
    if t and t[-1] not in ".!?":
        t += "."
    return t
