"""Kiểm tra chất lượng tự động. Video có dấu hiệu lỗi -> trạng thái 'Cần xem'."""
from __future__ import annotations

import shutil

from ..utils import Job, log, probe


def run(job: Job) -> None:
    meta = job.read("meta.json")
    fit = job.read("fit.json")
    tl = job.read("translation.json")
    rd = job.read("render.json")
    issues = []
    out = rd["output"]
    info = probe(out)
    if abs(info["duration"] - float(meta["duration"])) > 0.5:
        issues.append(f"Thời lượng lệch {info['duration']:.1f}s so với gốc {meta['duration']:.1f}s")
    if not info.get("has_audio"):
        issues.append("Video ra không có âm thanh")
    lines = fit["lines"]
    if lines:
        fast = sum(1 for f in lines if f["speed"] > 1.3)
        if fast / len(lines) > 0.15:
            issues.append(f"{fast}/{len(lines)} câu phải tăng tốc mạnh (>1.3x)")
        over = sum(1 for f in lines if f["overflow"] > 0.3)
        if over:
            issues.append(f"{over} câu bị cắt đuôi vì quá dài")
    empty = sum(1 for l in tl["lines"] if not l["vi"])
    if tl["lines"] and empty / len(tl["lines"]) > 0.2:
        issues.append(f"{empty}/{len(tl['lines'])} câu không có bản dịch")
    if not tl["lines"]:
        issues.append("Không nhận ra câu thoại nào")
    status = "review" if issues else "done"
    job.write("qc.json", {"status": status, "issues": issues, "output": out})
    log(f"QC: {status} {issues}")
    if not job.settings.get("keep_work_files", False):
        for name in ("audio.wav", "vocals.wav", "background.wav", "vocals_16k.wav", "mix.wav"):
            try:
                job.p(name).unlink(missing_ok=True)
            except Exception:
                pass
        shutil.rmtree(job.p("subs"), ignore_errors=True)
        shutil.rmtree(job.p("tts"), ignore_errors=True)
