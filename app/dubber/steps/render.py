"""B9. Xuất video: làm mờ chữ Trung + phụ đề Việt + âm thanh mới.

- Phụ đề Việt được vẽ thành ảnh PNG (Pillow, font Be Vietnam Pro có dấu), ghép thành một
  'dải ảnh theo thời gian' (ffconcat) rồi chồng lên video. Không cần thư viện libass.
- Vùng chữ Trung: cắt dải đó, làm mờ (gblur), dán lại - chỉ bật trong các khoảng có chữ.
- Nén bằng h264_videotoolbox: bộ nén phần cứng trong chip M1 (nhanh, ít nóng).
"""
from __future__ import annotations

import time
from datetime import date

from PIL import Image, ImageDraw, ImageFont

from .. import models, paths
from ..utils import Job, ffmpeg, has_encoder, log, slugify


def _even(x):
    x = int(round(x))
    return x - (x % 2)


def _wrap(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _layout(text, W, fs0, font_path):
    dummy = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    fs = fs0
    for _ in range(4):
        font = ImageFont.truetype(font_path, fs)
        lines = _wrap(dummy, text, font, W * 0.88)
        if len(lines) <= 2:
            break
        fs = int(fs * 0.88)
    return font, lines, fs


def _render_png(path, text, W, box_h, font_path, fs0):
    img = Image.new("RGBA", (W, box_h), (0, 0, 0, 0))
    if text:
        d = ImageDraw.Draw(img)
        font, lines, fs = _layout(text, W, fs0, font_path)
        lh = int(fs * 1.25)
        total = lh * len(lines)
        y = (box_h - total) // 2
        widths = [d.textlength(l, font=font) for l in lines]
        pad = int(fs * 0.35)
        bw = max(widths) + 2 * pad
        d.rounded_rectangle([(W - bw) / 2, y - pad * 0.6, (W + bw) / 2, y + total + pad * 0.6],
                            radius=int(fs * 0.3), fill=(0, 0, 0, 120))
        sw = max(2, fs // 14)
        for l, wl in zip(lines, widths):
            d.text(((W - wl) / 2, y), l, font=font, fill=(255, 255, 255, 255), stroke_width=sw,
                   stroke_fill=(0, 0, 0, 255))
            y += lh
    img.save(path)


def _needed_h(text, W, fs0, font_path):
    font, lines, fs = _layout(text, W, fs0, font_path)
    return int(fs * 1.25) * max(1, len(lines)) + int(fs * 0.9)


def run(job: Job) -> None:
    t0 = time.time()
    s = job.settings
    meta = job.read("meta.json")
    ocr = job.read("ocr.json", {}) or {}
    fit = job.read("fit.json")
    tl = job.read("translation.json", {}) or {}
    orig_W, orig_H = int(meta["width"]), int(meta["height"])
    if abs(int(meta.get("rotation", 0) or 0)) in (90, 270):
        orig_W, orig_H = orig_H, orig_W

    # Tự động tối ưu độ phân giải xuất: Giới hạn tối đa 1080p (1920x1080 ngang hoặc 1080x1920 dọc)
    # Giúp giảm 60-75% thời gian render trên CPU mà độ nét vẫn đạt chuẩn full HD sắc nét nhất
    max_dim = max(orig_W, orig_H)
    min_dim = min(orig_W, orig_H)
    if max_dim > 1920 or min_dim > 1080:
        scale_ratio = min(1920 / max_dim, 1080 / min_dim)
        W = _even(orig_W * scale_ratio)
        H = _even(orig_H * scale_ratio)
        need_scale = True
    else:
        W, H = orig_W, orig_H
        need_scale = False

    font_path = models.font_path()
    fs0 = int(min(W, H) * 0.052)

    # ---- vùng phụ đề
    band = ocr.get("band")
    if band:
        bx, by = _even(band[0] * W), _even(band[1] * H)
        bw, bh = _even((band[2] - band[0]) * W), _even((band[3] - band[1]) * H)
        bw, bh = max(16, min(bw, W - bx)), max(16, min(bh, H - by))
        center_y = by + bh / 2
    else:
        center_y = H * 0.80
        bh = 0

    # ---- ảnh phụ đề
    cues = []
    lines = sorted(fit["lines"], key=lambda f: f["start"])
    for i, f in enumerate(lines):
        if not f["vi"]:
            continue
        st = f["start"]
        en = st + max(f["dur"], f["orig_end"] - st) + 0.25
        if i + 1 < len(lines):
            en = min(en, lines[i + 1]["start"])
        if en - st > 0.2:
            cues.append((st, en, f["vi"]))
    box_h = max([bh] + [_needed_h(c[2], W, fs0, font_path) for c in cues] + [int(fs0 * 2)])
    box_h = _even(min(box_h, H * 0.35))
    oy = _even(max(0, min(H - box_h, center_y - box_h / 2)))
    sub_dir = job.p("subs")
    sub_dir.mkdir(exist_ok=True)
    _render_png(sub_dir / "blank.png", "", W, box_h, font_path, fs0)
    entries, t = [], 0.0
    for k, (st, en, text) in enumerate(cues):
        if st > t + 0.01:
            entries.append(("blank.png", st - t))
        name = f"s{k:04d}.png"
        _render_png(sub_dir / name, text, W, box_h, font_path, fs0)
        entries.append((name, en - st))
        t = en
    total = float(meta.get("duration", t))
    entries.append(("blank.png", max(0.5, total - t + 1)))
    with open(sub_dir / "subs.ffconcat", "w", encoding="utf-8") as fh:
        fh.write("ffconcat version 1.0\n")
        for name, d in entries:
            fh.write(f"file '{name}'\nduration {d:.3f}\n")
        fh.write("file 'blank.png'\n")
    job.progress(0.2, "vẽ phụ đề xong")

    # ---- filter: làm mờ TOÀN BỘ các vùng chữ Trung (đáy, đỉnh, giữa, watermark) + chồng phụ đề
    bands = ocr.get("bands") or []
    if not bands and band:
        bands = [{"box": band, "intervals": ocr.get("intervals", []), "is_static": False, "type": "sub"}]

    blur_filters = []
    if need_scale:
        cur_v = "vscale"
        blur_filters.append(f"[0:v]scale={W}:{H}[{cur_v}];")
    else:
        cur_v = "0:v"


    for idx, b in enumerate(bands):
        bbox = b["box"]
        ibx, iby = _even(bbox[0] * W), _even(bbox[1] * H)
        ibw, ibh = _even((bbox[2] - bbox[0]) * W), _even((bbox[3] - bbox[1]) * H)
        ibw = max(16, min(ibw, W - ibx))
        ibh = max(16, min(ibh, H - iby))

        ivs = b.get("intervals") or []
        if b.get("is_static") or not ivs:
            enable_clause = ""
        else:
            expr = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in ivs[:300])
            enable_clause = f":enable='{expr}'"

        next_v = f"vb{idx}"
        blur_filters.append(
            f"[{cur_v}]split=2[in{idx}][crop{idx}];"
            f"[crop{idx}]crop={ibw}:{ibh}:{ibx}:{iby},boxblur=10[bl{idx}];"
            f"[in{idx}][bl{idx}]overlay={ibx}:{iby}{enable_clause}[{next_v}];"
        )
        cur_v = next_v

    if blur_filters:
        blur_chain = "".join(blur_filters)
    else:
        blur_chain = f"[{cur_v}]null[vb];"
        cur_v = "vb"

    fc = blur_chain + f"[1:v]format=rgba[sb];[{cur_v}][sb]overlay=0:{oy}:format=auto,format=yuv420p[vout]"

    out_dir = paths.OUTPUT / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    title = tl.get("title_vi") or meta.get("desc", "")
    base = f"{meta.get('id', job.dir.name)}_{slugify(title, 60)}"
    out = out_dir / f"{base}.mp4"
    if paths.IS_MAC and has_encoder("h264_videotoolbox"):
        br = s.get("video_bitrate", "6M")
        venc = ["-c:v", "h264_videotoolbox", "-b:v", br, "-maxrate", "10M", "-bufsize", "12M",
                "-profile:v", "high", "-allow_sw", "1"]
    else:
        preset = s.get("video_preset", "ultrafast")
        crf = str(s.get("video_crf", "22"))
        venc = ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-threads", "0"]
    mode = s.get("workflow_mode", "dub")
    is_sub_only = (mode == "sub_only")
    if is_sub_only:
        log("Xuất video chế độ Chỉ VietSub: Giữ nguyên 100% âm thanh gốc")
        try:
            ffmpeg(["-i", job.p("source.mp4"), "-f", "concat", "-safe", "0", "-i", sub_dir / "subs.ffconcat",
                    "-filter_complex", fc, "-map", "[vout]", "-map", "0:a?",
                    *venc, "-c:a", "copy", "-movflags", "+faststart", "-shortest", out])
        except Exception:
            ffmpeg(["-i", job.p("source.mp4"), "-f", "concat", "-safe", "0", "-i", sub_dir / "subs.ffconcat",
                    "-filter_complex", fc, "-map", "[vout]", "-map", "0:a?",
                    *venc, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out])
    else:
        ffmpeg(["-i", job.p("source.mp4"), "-f", "concat", "-safe", "0", "-i", sub_dir / "subs.ffconcat",
                "-i", job.p("mix.wav"), "-filter_complex", fc, "-map", "[vout]", "-map", "2:a",
                *venc, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out])

    # ---- phụ đề .srt + thông tin để đăng bài
    def ts(x):
        h, r = divmod(x, 3600)
        m, sec = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int((sec % 1) * 1000):03d}"
    with open(out.with_suffix(".srt"), "w", encoding="utf-8") as fh:
        for k, (st, en, text) in enumerate(cues, 1):
            fh.write(f"{k}\n{ts(st)} --> {ts(en)}\n{text}\n\n")
    with open(out.with_suffix(".txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Tiêu đề: {tl.get('title_vi', '')}\nTóm tắt: {tl.get('summary_vi', '')}\n"
                 f"Thể loại: {tl.get('genre', '')}\nNguồn: {meta.get('url', '')}\n"
                 f"Tác giả gốc: {meta.get('author', '')}\n")
    job.write("render.json", {"output": str(out), "seconds": round(time.time() - t0, 1), "box_h": box_h,
                              "oy": oy, "band_px": [bx, by, bw, bh] if band else None, "cues": len(cues)})
    log(f"Xuất video xong {time.time() - t0:.1f}s: {out.name}")
