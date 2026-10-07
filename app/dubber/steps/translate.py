"""B5. Dịch + Việt hóa + giới hạn độ dài câu.

Cách hoạt động: gửi CẢ kịch bản (mọi câu, ai nói, thời lượng, chữ trên hình) cho LLM trong
một lượt để giữ mạch và xưng hô nhất quán. Mỗi câu có 'max_syll' = số âm tiết tối đa vừa
khung thời gian. Tiếng Việt mỗi âm tiết cách nhau dấu cách nên đếm âm tiết = đếm số từ.
"""
from __future__ import annotations

import math

from .. import llm
from ..utils import Job, fmt_ts, log, vi_syllables

GENRES = ["co_trang", "drama", "hai", "ban_hang", "review_phim", "kien_thuc", "vlog", "khac"]

RULES = """Bạn là đạo diễn lồng tiếng phim chuyên nghiệp, dịch video Douyin (tiếng Trung) sang tiếng Việt GIỌNG MIỀN NAM để diễn viên lồng tiếng diễn xuất.
Quy tắc:
1. Dịch Việt hóa giàu cảm xúc, tự nhiên và kịch tính như phim truyền hình thực thụ. Dùng từ cảm thán tự nhiên của tiếng Việt (Trời ơi, Hả, Không xong rồi, Chết tiệt, Sao lại thế, Mau lên...) và dấu câu diễn xuất (dấu '...' để ngập ngừng/lắng đọng/thở dài, dấu '!' cho câu phẫn nộ/cao trào).
2. Xưng hô đúng quan hệ và bối cảnh (anh-em, chị-em, ba-con, mẹ-con, vợ-chồng, sếp-nhân viên; cổ trang dùng ta-ngươi, thần-bệ hạ, công tử, cô nương...). Giữ xưng hô nhất quán suốt video.
3. Tên riêng tiếng Trung đổi sang âm Hán-Việt (李云 -> Lý Vân). Tên thương hiệu nước ngoài giữ nguyên.
4. Mỗi câu "vi" PHẢI có số âm tiết (số từ cách nhau bởi dấu cách) KHÔNG VƯỢT "max_syll" của câu đó. Nếu dài thì nói gọn, bỏ ý phụ, giữ ý chính.
5. Viết số và ký hiệu thành chữ để đọc được (2025 -> hai không hai lăm, 50% -> năm mươi phần trăm, 3kg -> ba ký). Không dùng emoji, ngoặc, ký tự đặc biệt.
6. Chữ nhận dạng giọng nói ("zh") có thể nghe nhầm. Nếu có "ocr" (chữ phụ đề trên màn hình) thì ưu tiên tin "ocr" để hiểu đúng câu.
7. Nếu một câu chỉ là tiếng ồn/vô nghĩa/hát thì để "vi" là chuỗi rỗng "".
8. Gán nhãn "emotion" cho mỗi câu phù hợp ngữ cảnh diễn xuất: một trong "neutral" (bình thường), "happy" (vui vẻ, cười), "angry" (tức giận, phẫn nộ, gắt gỏng), "sad" (buồn bã, khóc, đau đớn), "excited" (hào hứng, ngạc nhiên), "whisper" (thì thầm, tâm sự), "panicked" (hoảng hốt, sợ hãi), "sarcastic" (mỉa mai, khinh khỉnh), "dramatic" (hùng hồn, trang nghiêm).
9. Chỉ trả về JSON hợp lệ, không giải thích gì thêm."""

SCHEMA_FULL = """Trả về JSON đúng dạng:
{"genre": "<một trong: co_trang|drama|hai|ban_hang|review_phim|kien_thuc|vlog|khac>",
 "title_vi": "<tiêu đề tiếng Việt hấp dẫn, ngắn>",
 "summary_vi": "<tóm tắt 1-2 câu>",
 "speakers": {"S0": {"name_vi": "<tên hoặc vai>", "gender": "male|female", "age": "child|young|adult|old", "role": "<vai trò ngắn>"}},
 "lines": [{"id": 1, "vi": "<câu tiếng Việt>", "emotion": "<neutral|happy|angry|sad|excited|whisper|panicked|sarcastic|dramatic>"}]}"""

SCHEMA_LINES = """Trả về JSON đúng dạng: {"lines": [{"id": 1, "vi": "<câu tiếng Việt>", "emotion": "<neutral|happy|angry|sad|excited|whisper|panicked|sarcastic|dramatic>"}]}"""


def _max_syll(lines, i, duration, rate, borrow, cuts=None):
    ln = lines[i]
    nxt = lines[i + 1]["start"] if i + 1 < len(lines) else duration
    if cuts:
        valid_cuts = [c for c in cuts if ln["end"] - 0.1 <= c <= nxt + 0.05]
        if valid_cuts:
            nxt = min(nxt, max(ln["start"] + 0.5, min(valid_cuts) - 0.05))
    avail = min(nxt - ln["start"], (ln["end"] - ln["start"]) + borrow)
    avail = max(0.6, avail - 0.05)
    return max(2, int(math.floor(avail * rate * 0.95)))


def _ocr_for(caps, st, en):
    txt = [c["text"] for c in caps if c["end"] >= st - 0.6 and c["start"] <= en + 0.6]
    out = []
    for t in txt:
        if t not in out:
            out.append(t)
    return " / ".join(out)[:120]


def build_items(job: Job):
    tr = job.read("transcript.json")
    ocr = job.read("ocr.json", {}) or {}
    s = job.settings
    rate = float(s.get("speech_rate_sps", 4.8))
    # nếu đã đo được tốc độ đọc thật của các giọng ở video trước -> dùng số đo đó
    from .tts import RATES_FILE
    from ..utils import read_json
    measured = list((read_json(RATES_FILE, {}) or {}).values())
    if measured:
        rate = min(6.5, max(3.5, sum(measured) / len(measured)))
    borrow = float(s.get("borrow_gap_s", 1.5))
    lines = tr["lines"]
    caps = ocr.get("captions", [])
    try:
        from .scene import get_scene_cuts
        cuts = get_scene_cuts(job)
    except Exception:
        cuts = []
    items = []
    for i, ln in enumerate(lines):
        it = {"id": ln["id"], "spk": ln["spk"], "t": fmt_ts(ln["start"]),
              "dur": round(ln["end"] - ln["start"], 1),
              "max_syll": _max_syll(lines, i, tr["duration"], rate, borrow, cuts=cuts), "zh": ln["zh"]}
        o = _ocr_for(caps, ln["start"], ln["end"])
        if o:
            it["ocr"] = o
        items.append(it)
    return tr, items


def _compact(items):
    import json
    return "\n".join(json.dumps(x, ensure_ascii=False) for x in items)


def translate_items(meta, speakers, items, known=None):
    import json
    spk_info = {k: {"gender_by_voice": v["gender_voice"], "pitch_hz": v["f0"], "talk_seconds": v["talk_time"]}
                for k, v in speakers.items()}
    head = f"{RULES}\n\nMô tả video gốc: {meta.get('desc', '')[:300]}\nNgười nói (đoán từ giọng): {json.dumps(spk_info, ensure_ascii=False)}\n"
    if known:
        head += f"\nĐã xác định từ phần trước (giữ nhất quán): {json.dumps(known, ensure_ascii=False)}\n"
        schema = SCHEMA_LINES
    else:
        schema = SCHEMA_FULL
    prompt = f"{head}\nCác câu (mỗi dòng một JSON):\n{_compact(items)}\n\n{schema}"
    data, model = llm.generate(prompt, temperature=0.4)
    return data, model


def run(job: Job) -> None:
    meta = job.read("meta.json", {})
    tr, items = build_items(job)
    if not items:
        job.write("translation.json", {"genre": "khac", "speakers": {}, "lines": [], "summary_vi": "",
                                       "title_vi": meta.get("desc", "")})
        log("Không có câu thoại nào để dịch")
        return
    CH = 110
    chunks = [items[i:i + CH] for i in range(0, len(items), CH)]
    # phần 1 dịch trước để AI xác định thể loại + tên nhân vật; các phần sau dịch SONG SONG
    # (gửi cùng lúc tối đa 3 yêu cầu) và dùng lại thông tin đó cho nhất quán -> nhanh hơn video dài.
    result, model = translate_items(meta, tr["speakers"], chunks[0])
    models_used = [model]
    result.setdefault("lines", [])
    job.progress(1 / len(chunks), "dịch")
    if len(chunks) > 1:
        from concurrent.futures import ThreadPoolExecutor
        known = {"genre": result.get("genre"), "speakers": result.get("speakers")}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(translate_items, meta, tr["speakers"], ch, known) for ch in chunks[1:]]
            for k, f in enumerate(futs):
                data, model = f.result()
                models_used.append(model)
                result["lines"].extend(data.get("lines", []))
                job.progress((k + 2) / len(chunks), "dịch")

    got = {int(l["id"]): (l.get("vi") or "").strip() for l in result.get("lines", []) if "id" in l}
    missing = [it for it in items if it["id"] not in got]
    if missing:
        log(f"Thiếu {len(missing)} câu, dịch bổ sung...")
        known = {"genre": result.get("genre"), "speakers": result.get("speakers")}
        data, model = translate_items(meta, tr["speakers"], missing, known)
        models_used.append(model)
        for l in data.get("lines", []):
            got[int(l["id"])] = (l.get("vi") or "").strip()

    lines = []
    for it in items:
        vi = got.get(it["id"], "")
        lines.append({"id": it["id"], "vi": vi, "max_syll": it["max_syll"], "syll": vi_syllables(vi)})
    genre = result.get("genre") if result.get("genre") in GENRES else "khac"
    out = {"genre": genre, "title_vi": result.get("title_vi", ""), "summary_vi": result.get("summary_vi", ""),
           "speakers": result.get("speakers", {}), "lines": lines, "models": models_used}
    job.write("translation.json", out)
    over = sum(1 for l in lines if l["syll"] > l["max_syll"])
    log(f"Dịch xong: thể loại={genre}, {len(lines)} câu, {over} câu dài hơn giới hạn")


def shorten(job: Job, todo: list[dict]) -> dict:
    """todo: [{id, vi, max_syll}] -> {id: câu mới ngắn hơn}."""
    import json
    tr = job.read("translation.json")
    prompt = (RULES + "\n\nCác câu tiếng Việt dưới đây quá dài so với thời gian nói. Hãy viết lại mỗi câu "
              "ĐƠN GIẢN HƠN, nói gọn hơn, giữ ý chính và giữ nguyên xưng hô, sao cho số âm tiết <= max_syll.\n"
              f"Thông tin nhân vật: {json.dumps(tr.get('speakers', {}), ensure_ascii=False)}\n"
              f"Các câu:\n{_compact(todo)}\n\n{SCHEMA_LINES}")
    data, _ = llm.generate(prompt, temperature=0.3)
    return {int(l["id"]): (l.get("vi") or "").strip() for l in data.get("lines", []) if "id" in l}
