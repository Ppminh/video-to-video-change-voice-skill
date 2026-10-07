"""Gọi Gemini API (miễn phí) với danh sách model dự phòng.

Cách hoạt động: thử model đầu danh sách (Gemini Flash-Lite: nhanh, nhiều lượt miễn phí nhất).
Nếu hết lượt (lỗi 429) hoặc model không có (404) thì tự chuyển sang model kế tiếp.
Tối ưu tốc độ:
- Tắt "suy nghĩ" (thinking_level=minimal): model trả lời ngay, nhanh hơn nhiều lần (Gemma 4 31B có
  suy nghĩ mất ~190 giây, tắt đi còn vài chục giây). Bản dịch vẫn tốt vì đề bài đã rất chi tiết.
- Yêu cầu trả về JSON thuần (response_mime_type) -> không bị lỗi định dạng.
Đếm số lượt dùng mỗi ngày để theo dõi.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date

from . import paths
from .utils import log, read_json, write_json

QUOTA_FILE = paths.APPHOME / "quota.json"
_DEAD: set[str] = set()
_NO_THINK_CFG: set[str] = set()   # model không nhận tùy chọn tắt suy nghĩ
_NO_JSON_CFG: set[str] = set()    # model không nhận tùy chọn JSON


def _config(types, model: str, temperature: float, max_tokens: int, want_json: bool):
    kw = {"temperature": temperature, "max_output_tokens": max_tokens}
    if want_json and model.startswith("gemini") and model not in _NO_JSON_CFG:
        kw["response_mime_type"] = "application/json"
    if model not in _NO_THINK_CFG:
        try:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")
        except Exception:
            try:
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
    return types.GenerateContentConfig(**kw)


def _count(model: str) -> None:
    q = read_json(QUOTA_FILE, {}) or {}
    today = date.today().isoformat()
    q = {today: q.get(today, {})}
    q[today][model] = q[today].get(model, 0) + 1
    try:
        write_json(QUOTA_FILE, q)
    except Exception:
        pass


def usage_today() -> dict:
    q = read_json(QUOTA_FILE, {}) or {}
    return q.get(date.today().isoformat(), {})


def parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        t = t[a:b + 1]
    try:
        return json.loads(t)
    except Exception:
        from json_repair import repair_json
        return json.loads(repair_json(t))


def generate(prompt: str, temperature: float = 0.4, max_tokens: int = 8192, want_json: bool = True):
    from google import genai
    from google.genai import types

    key = paths.gemini_key()
    if not key:
        raise RuntimeError("Chưa có GEMINI_API_KEY trong config/.env")
    client = genai.Client(api_key=key)
    models = [m for m in paths.load_settings().get("translate_models", []) if m not in _DEAD]
    last_err = None
    for attempt in range(2):
        for model in models:
            if model in _DEAD:
                continue
            try:
                t0 = time.time()
                try:
                    cfg = _config(types, model, temperature, max_tokens, want_json)
                    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                except Exception as e:   # model không hỗ trợ tùy chọn -> bỏ tùy chọn, thử lại ngay
                    m = str(e).lower()
                    if ("think" in m or "mime" in m or "json" in m) and ("400" in m or "invalid" in m
                                                                         or "not supported" in m):
                        if "think" in m:
                            _NO_THINK_CFG.add(model)
                        else:
                            _NO_JSON_CFG.add(model)
                        cfg = _config(types, model, temperature, max_tokens, want_json)
                        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                    else:
                        raise
                _count(model)
                text = resp.text or ""
                if not text.strip():
                    raise RuntimeError("model trả về rỗng")
                log(f"LLM {model}: {len(text)} ký tự, {time.time() - t0:.1f}s")
                if not want_json:
                    return text, model
                return parse_json(text), model
            except Exception as e:  # chuyển model khác
                msg = str(e)
                last_err = e
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                log(f"LLM {model} lỗi: {msg[:200]}")
                if code == 404 or "not found" in msg.lower() or "is not supported" in msg.lower():
                    _DEAD.add(model)
                continue
        if attempt == 0:
            log("Tất cả model đều lỗi, chờ 60 giây rồi thử lại...")
            time.sleep(60)
    raise RuntimeError(f"Không gọi được Gemini API: {last_err}")
