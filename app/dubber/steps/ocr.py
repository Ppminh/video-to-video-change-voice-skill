"""B4. Dò phụ đề chữ Trung in cứng trên hình (bản tối ưu, nhanh hơn ~5-10 lần).

Cách hoạt động (2 lượt):
1. Lượt dò (nhanh): chỉ chụp ~40 khung hình đúng lúc đang có người nói (lấy từ bước nghe ở B3),
   đọc chữ cả khung -> tìm ra "vùng phụ đề" (vị trí có chữ Hán lặp lại nhiều nhất).
   Không thấy phụ đề -> bỏ qua lượt 2 luôn (video không có hardsub).
2. Lượt đọc: chỉ cắt dải hình chứa phụ đề (nhỏ hơn cả khung ~8 lần) 2 lần/giây.
   Khung nào gần giống khung vừa đọc (chữ chưa đổi) -> dùng lại kết quả, không đọc lại.
Mẹo tăng tốc khác: RapidOCR mặc định phóng to ảnh nhỏ lên 736px trước khi dò chữ (rất chậm);
ở đây đặt giới hạn theo cạnh dài nên ảnh nhỏ được giữ nguyên kích thước.
Kết quả dùng cho: làm mờ vùng đó, đặt phụ đề Việt, và giúp AI dịch sửa lỗi nghe nhầm.
"""
from __future__ import annotations

import subprocess
import time
from collections import Counter

import numpy as np

from .. import paths
from ..utils import Job, cjk_count, ffmpeg_bin, ffprobe_bin, log, threads

PROBES = 40          # số khung hình tối đa ở lượt dò
MIN_PROBES = 24      # ít keyframe quá thì chụp thêm cho đủ
DIFF_SAME = 3.0      # chênh lệch điểm ảnh trung bình (0-255) dưới mức này = chữ chưa đổi


def _hw() -> list:
    return ["-hwaccel", "videotoolbox"] if paths.IS_MAC else []


def _engine(side: int):
    from rapidocr_onnxruntime import RapidOCR
    try:
        return RapidOCR(det_limit_side_len=side, det_limit_type="max", intra_op_num_threads=threads())
    except Exception:
        return RapidOCR()


def _grab(src, t: float, w: int, h: int):
    """Chụp 1 khung hình tại giây t (tua nhanh tới khung khóa gần nhất rồi giải mã vài khung)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", *_hw(), "-ss", f"{t:.3f}", "-i", str(src),
           "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    if len(out) < w * h * 3:
        return None
    return np.frombuffer(out[:w * h * 3], np.uint8).reshape(h, w, 3)


def _frames(src, vf: str, fps: float, w: int, h: int, pre=(), post=()):
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", *_hw(), *pre, "-i", str(src),
           *post, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        p.stdout.close()
        p.wait()


def _detect(engine, img, oy: float = 0.0, sy: float = 1.0):
    """Đọc chữ, trả về [(x1, y1, x2, y2, chữ)] theo tọa độ 0..1 của KHUNG GỐC."""
    h, w = img.shape[:2]
    res, _ = engine(img, use_cls=False)
    out = []
    for box, text, score in res or []:
        if cjk_count(text) < 2 or float(score) < 0.6:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, x2 = min(xs) / w, max(xs) / w
        y1, y2 = oy + sy * min(ys) / h, oy + sy * max(ys) / h
        if not (0.012 < (y2 - y1) < 0.12) or (x2 - x1) < 0.05:
            continue
        out.append((x1, y1, x2, y2, text))
    return out


def _merge(times, gap):
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def _keyframes(src, w: int, h: int, max_n: int):
    """Khung khóa (keyframe) giải mã cực nhanh vì không cần giải mã các khung ở giữa."""
    r = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "packet=pts_time,flags", "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    times = []
    for line in r.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                times.append(float(parts[0]))
            except ValueError:
                pass
    times.sort()
    vf = f"scale={w}:{h}"
    cmd_extra = ["-skip_frame", "nokey"]
    frames = []
    for i, (_, img) in enumerate(_frames(src, vf, 1.0, w, h, pre=cmd_extra, post=["-fps_mode", "passthrough"])):
        frames.append(img)
    if not frames:
        return []
    if len(times) != len(frames):
        times = list(np.linspace(0, max(times or [0]), len(frames)))
    pairs = list(zip(times, frames))
    if len(pairs) > max_n:
        idx = np.linspace(0, len(pairs) - 1, max_n).round().astype(int)
        pairs = [pairs[i] for i in idx]
    return pairs


def _probe_times(job: Job, dur: float, have: list[float], need: int) -> list[float]:
    """Bổ sung thời điểm dò: giữa các câu thoại (lúc phụ đề chắc chắn đang hiện), tránh trùng keyframe."""
    tr = job.read("transcript.json", {}) or {}
    mids = [(l["start"] + l["end"]) / 2 for l in tr.get("lines", [])]
    mids = [t for t in mids if 0 <= t < dur - 0.1 and all(abs(t - k) > 1.0 for k in have)]
    if not mids:
        mids = [t for t in np.linspace(dur * 0.05, dur * 0.95, need) if all(abs(t - k) > 1.0 for k in have)]
    if len(mids) > need:
        idx = np.linspace(0, len(mids) - 1, need).round().astype(int)
        mids = [mids[i] for i in idx]
    return [round(float(t), 2) for t in mids]


def _find_all_bands(dets, n_frames: int, min_frames: int = 3, fps: float = 2.0):
    """Gom cụm và tìm TOÀN BỘ các vùng có chữ Trung Quốc (đáy, đỉnh, giữa, watermark)."""
    if not dets:
        return []
    groups = {"top": [], "mid": [], "sub": []}
    for d in dets:
        yc = (d[2] + d[4]) / 2
        if yc < 0.35:
            groups["top"].append(d)
        elif yc <= 0.65:
            groups["mid"].append(d)
        else:
            groups["sub"].append(d)

    bands = []
    for gname, gdets in groups.items():
        if not gdets:
            continue
        frames = {d[0] for d in gdets}
        if len(frames) < min_frames:
            continue

        centers = np.array([(d[2] + d[4]) / 2 for d in gdets])
        hist, edges = np.histogram(centers, bins=25, range=(0, 1))
        peaks = [k for k in range(len(hist)) if hist[k] >= max(min_frames, int(0.2 * len(frames)))]
        if not peaks:
            peaks = [int(np.argmax(hist))]

        for p_idx in set(peaks):
            mode = (edges[p_idx] + edges[p_idx + 1]) / 2
            cluster_dets = [d for d, c in zip(gdets, centers) if abs(c - mode) <= 0.08]
            c_frames = {d[0] for d in cluster_dets}
            if len(c_frames) < min_frames:
                continue

            y1 = max(0.0, float(np.percentile([d[2] for d in cluster_dets], 5)) - 0.015)
            y2 = min(1.0, float(np.percentile([d[4] for d in cluster_dets], 95)) + 0.015)
            x1 = max(0.0, float(np.percentile([d[1] for d in cluster_dets], 3)) - 0.03)
            x2 = min(1.0, float(np.percentile([d[3] for d in cluster_dets], 97)) + 0.03)

            is_static = len(c_frames) >= 0.4 * n_frames and n_frames >= 8
            iv = _merge(c_frames, gap=1.6 / fps)
            intervals = [[round(max(0, a - 0.35), 2), round(b + 0.35 + 1 / fps, 2)] for a, b in iv]

            overlap = False
            for b in bands:
                bx1, by1, bx2, by2 = b["box"]
                if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                    b["box"] = [min(bx1, x1), min(by1, y1), max(bx2, x2), max(by2, y2)]
                    b["is_static"] = b["is_static"] or is_static
                    b["dets"] += cluster_dets
                    overlap = True
                    break
            if not overlap:
                bands.append({
                    "type": gname,
                    "box": [x1, y1, x2, y2],
                    "intervals": intervals,
                    "is_static": is_static,
                    "dets": cluster_dets,
                })
    return bands


def _band_from(dets, n_frames: int, min_frames: int):
    bands = _find_all_bands(dets, n_frames, min_frames)
    sub = [b for b in bands if b["type"] == "sub"]
    if sub:
        return sub[0]["box"], sub[0]["dets"]
    if bands:
        return bands[-1]["box"], bands[-1]["dets"]
    return None, []


def run(job: Job) -> None:
    t0 = time.time()
    meta = job.read("meta.json")
    W, H = int(meta["width"]), int(meta["height"])
    dur = float(meta.get("duration", 1))
    src = job.p("source.mp4")
    fps = float(job.settings.get("ocr_fps", 2.0))
    result = {"band": None, "bands": [], "intervals": [], "captions": [], "frames": 0, "probe_frames": 0}

    # ---------- Lượt 1: dò vùng chữ trên ~40 khung (keyframe + giữa các câu thoại)
    w = 480 if W >= H else 360
    h = int(round(H * w / W / 2)) * 2
    eng = _engine(max(w, h))
    pairs = _keyframes(src, w, h, PROBES)
    extra = []
    if len(pairs) < MIN_PROBES:
        extra = _probe_times(job, dur, [t for t, _ in pairs], MIN_PROBES - len(pairs))
    times = [t for t, _ in pairs] + extra
    probe = []
    total_p = max(1, len(times))
    for k, (t, img) in enumerate(pairs):
        probe += [(t, *d) for d in _detect(eng, img)]
        if k % 5 == 0:
            job.progress(0.3 * (k + 1) / total_p, f"dò vùng chữ Trung {k + 1}/{total_p}")
    for k, t in enumerate(extra):
        img = _grab(src, t, w, h)
        if img is not None:
            probe += [(t, *d) for d in _detect(eng, img)]
        job.progress(0.3 * (len(pairs) + k + 1) / total_p, f"dò vùng chữ Trung {len(pairs) + k + 1}/{total_p}")
    result["probe_frames"] = len(times)
    probe_bands = _find_all_bands(probe, len(times), min_frames=2, fps=1.0)
    band0, _ = _band_from(probe, len(times), min_frames=2)
    t1 = time.time() - t0
    if not probe_bands and not band0:
        job.write("ocr.json", result)
        log(f"OCR: không thấy chữ Trung trên hình ({len(times)} khung dò, {t1:.1f}s) -> bỏ qua lượt đọc")
        return

    # ---------- Lượt 2: đọc dải phụ đề chính
    band_crop = band0 or (probe_bands[0]["box"] if probe_bands else [0.0, 0.7, 1.0, 0.95])
    cy1 = max(0.0, band_crop[1] - 0.05)
    cy2 = min(1.0, band_crop[3] + 0.05)
    Y1 = int(cy1 * H) // 2 * 2
    CH = max(16, int((cy2 - cy1) * H) // 2 * 2)
    cw = 640 if W >= H else 480
    ch = max(16, int(round(CH * cw / W / 2)) * 2)
    oy, sy = Y1 / H, CH / H
    eng = _engine(cw)
    vf = f"fps={fps},crop={W}:{CH}:0:{Y1},scale={cw}:{ch}"
    dets = []
    n = skipped = 0
    last_gray, last_res = None, []
    total = max(1, int(dur * fps))
    for t, img in _frames(src, vf, fps, cw, ch):
        n += 1
        gray = img.mean(axis=2, dtype=np.float32)
        if last_gray is not None and float(np.abs(gray - last_gray).mean()) < DIFF_SAME:
            res = last_res
            skipped += 1
        else:
            res = _detect(eng, img, oy, sy)
            last_gray, last_res = gray, res
        dets += [(t, *d) for d in res]
        if n % 20 == 0:
            job.progress(0.3 + 0.65 * min(1.0, n / total), f"đọc phụ đề {n}/{total} khung")

    result["frames"] = n
    # Kết hợp kết quả probe cho các vùng ngoài crop (như tiêu đề trên đỉnh, logo)
    all_dets = dets + [d for d in probe if (d[2] + d[4]) / 2 < cy1 or (d[2] + d[4]) / 2 > cy2]
    all_bands = _find_all_bands(all_dets, max(n, len(times)), min_frames=3, fps=fps)

    result["bands"] = [{"type": b["type"], "box": b["box"], "intervals": b["intervals"], "is_static": b["is_static"]}
                       for b in all_bands]

    band, band_dets = _band_from(dets, n, min_frames=3)
    if not band and all_bands:
        band = all_bands[-1]["box"]
        band_dets = all_bands[-1]["dets"]

    if band:
        frames_with = {d[0] for d in band_dets}
        result["band"] = band
        iv = _merge(frames_with, gap=1.6 / fps)
        result["intervals"] = [[round(max(0, a - 0.35), 2), round(b + 0.35 + 1 / fps, 2)] for a, b in iv]
        by_t = {}
        for d in sorted(band_dets, key=lambda d: (d[0], d[2], d[1])):
            by_t.setdefault(d[0], []).append(d[5])
        caps = []
        for t in sorted(by_t):
            txt = " ".join(by_t[t])
            if caps and caps[-1]["text"] == txt and t - caps[-1]["end"] <= 1.6 / fps:
                caps[-1]["end"] = round(t, 2)
            else:
                caps.append({"start": round(t, 2), "end": round(t, 2), "text": txt})
        result["captions"] = caps
    job.write("ocr.json", result)
    log(f"OCR: phát hiện {len(result['bands'])} vùng chữ Trung (phụ đề={result['band']}), "
        f"{len(result['captions'])} câu phụ đề, tổng {time.time() - t0:.1f}s")
