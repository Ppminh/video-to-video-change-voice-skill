"""Chẩn đoán: thử từng model dịch với 1 câu ngắn, ghi lỗi chi tiết vào logs/diag.txt."""
from __future__ import annotations

import time

from . import llm, paths


def main():
    from google import genai
    from google.genai import types
    out = []
    client = genai.Client(api_key=paths.gemini_key())
    try:
        names = [m.name for m in client.models.list()]
        out.append("Model khả dụng: " + ", ".join(n for n in names if "gemma" in n or "flash" in n))
    except Exception as e:
        out.append(f"Không liệt kê được model: {e}")
    for m in paths.load_settings()["translate_models"]:
        t0 = time.time()
        try:
            r = client.models.generate_content(model=m, contents="Dịch sang tiếng Việt: 你好",
                                               config=llm._config(types, m, 0.4, 256, False))
            out.append(f"[OK ] {m} ({time.time() - t0:.1f}s): {(r.text or '').strip()[:80]}")
        except Exception as e:
            out.append(f"[LỖI] {m} ({time.time() - t0:.1f}s): {type(e).__name__}: {str(e)[:400]}")
    (paths.LOGS / "diag.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
