"""Chạy thử không cần giao diện: xử lý các link trong config/test_links.txt (mỗi dòng 1 link).

Dùng: python -m dubber.cli [link hoặc đường dẫn file ...]
- Chạy 2 làn song song giống ứng dụng web.
- Nếu ứng dụng web đang mở thì chỉ thêm video vào hàng chờ và theo dõi tới khi xong.
Kết quả + thời gian từng bước ghi vào logs/test_run.txt
"""
from __future__ import annotations

import json
import sys
import time

from . import db, paths
from .steps import STEP_LABELS
from .worker import Worker, acquire_worker_lock, job_dir

FINAL = ("done", "review", "failed")


def main():
    db.init()
    links = sys.argv[1:]
    if not links:
        f = paths.CONFIG / "test_links.txt"
        links = [l.strip() for l in f.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")] if f.exists() else []
    out = paths.LOGS / "test_run.txt"
    report = [f"CHẠY THỬ {time.strftime('%Y-%m-%d %H:%M')} - {len(links)} video"]
    ids = [db.add(link) for link in links]
    for jid, link in zip(ids, links):
        print(f"##### Video #{jid}: {link}", flush=True)
    others = [j["id"] for j in db.candidates() if j["id"] not in ids]
    if others:
        print(f"Trong hàng chờ còn {len(others)} video cũ ({others}) - sẽ được xử lý cùng.", flush=True)
        ids = others + ids
    w = None
    if acquire_worker_lock():
        w = Worker()
        w.start()
    else:
        print("Ứng dụng web đang chạy -> video đã được thêm vào hàng chờ của nó, đang theo dõi...", flush=True)
    t0 = time.time()
    last = {}
    while True:
        jobs = {jid: db.get(jid) for jid in ids}
        for jid, j in jobs.items():
            if not j:
                continue
            msg = f"#{jid} {j['status']} {STEP_LABELS.get(j.get('step') or '', '')} {j.get('note') or ''}".strip()
            if last.get(jid) != msg:
                print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
                last[jid] = msg
        if all((j is None) or j["status"] in FINAL for j in jobs.values()) and not (w and w.current):
            break
        time.sleep(3)
    for jid in ids:
        job = db.get(jid)
        if not job:
            continue
        tim = json.loads(job.get("timings") or "{}")
        line = (f"#{jid} {job['status']} | tổng các bước {sum(tim.values()):.0f}s | " +
                ", ".join(f"{STEP_LABELS.get(k, k)} {v}s" for k, v in tim.items()))
        if job.get("error"):
            line += f"\n   LỖI: {job['error'][:600]}"
        if job.get("issues") and job["issues"] != "[]":
            line += f"\n   Cần xem: {job['issues']}"
        line += f"\n   File: {job.get('output', '')}\n   Thư mục làm việc: {job_dir(jid)}"
        print(line, flush=True)
        report.append(line)
    report.append(f"Tổng thời gian thực (cả {len(ids)} video, chạy song song): {time.time() - t0:.0f}s")
    print(report[-1], flush=True)
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    if w:
        w.stop_flag = True
    print("\nXONG. Kết quả: logs/test_run.txt")


if __name__ == "__main__":
    main()
