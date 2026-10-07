"""Giao diện web chạy ngay trên Mac: http://127.0.0.1:7860

Viết bằng thư viện chuẩn của Python (không cần framework) cho nhẹ RAM.
Server chạy kèm bộ điều phối (worker) trong cùng tiến trình; việc nặng chạy ở tiến trình con.
"""
from __future__ import annotations

import json
import mimetypes
import re
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import db, llm, paths
from .steps import STEP_LABELS, STEPS
from .worker import Worker, acquire_worker_lock, job_dir, reset_from

WEB = Path(__file__).parent / "web"
PORT = 7860
WORKER: Worker | None = None

EDITABLE = {"tts_mode", "workflow_mode", "threads", "paused", "keep_work_files", "video_bitrate", "duck_db", "max_speed",
            "parallel_lanes", "lead_female_voice", "lead_male_voice", "voice_variants", "tts_precision",
            "voxcpm_model", "voxcpm_timesteps", "speech_rate_sps"}
VOICE_START = STEPS.index("tts")


def split_inputs(text: str) -> list[str]:
    out = []
    for part in re.split(r"[\n|]+", text or ""):
        part = part.strip()
        if not part:
            continue
        for m in re.finditer(r"https?://\S+", part):
            out.append(m.group(0))
        if not re.search(r"https?://", part) and Path(part).expanduser().exists():
            out.append(str(Path(part).expanduser()))
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def state() -> dict:
    jobs = db.listing(300)
    for j in jobs:
        j["step_label"] = STEP_LABELS.get(j.get("step") or "", "")
        j["step_index"] = (STEPS.index(j["step"]) if j.get("step") in STEPS else
                           len(STEPS) if j["status"] in ("done", "review") else
                           VOICE_START if j["status"] == "ready" else 0)
        j["timings"] = json.loads(j.get("timings") or "{}")
        j["issues"] = json.loads(j.get("issues") or "[]")
        j["has_video"] = bool(j.get("output")) and Path(j["output"]).exists()
        j["mode"] = j.get("mode") or "dub"
    du = shutil.disk_usage(str(paths.ROOT))
    s = paths.load_settings()
    return {
        "jobs": jobs, "counts": db.counts(), "steps": [STEP_LABELS[x] for x in STEPS],
        "settings": {k: s.get(k) for k in EDITABLE},
        "disk_free_gb": round(du.free / 1e9, 1), "api_today": llm.usage_today(),
        "has_key": bool(paths.gemini_key()), "worker_alive": bool(WORKER and WORKER.is_alive()),
        "current": WORKER.current if WORKER else None, "output_dir": str(paths.OUTPUT),
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def _file(self, path: Path, ctype: str | None = None):
        size = path.stat().st_size
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = int(b) if b else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._file(WEB / "index.html", "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self._json(state())
        m = re.match(r"^/video/(\d+)$", u.path)
        if m:
            j = db.get(int(m.group(1)))
            if j and j.get("output") and Path(j["output"]).exists():
                return self._file(Path(j["output"]), "video/mp4")
            return self._json({"error": "không có video"}, 404)
        m = re.match(r"^/api/log/(\d+)$", u.path)
        if m:
            d = job_dir(int(m.group(1)))
            parts = []
            for step in STEPS:
                f = d / f"log_{step}.txt"
                if f.exists():
                    txt = f.read_text(encoding="utf-8", errors="replace")
                    parts.append(f"===== {STEP_LABELS[step]} =====\n" + txt[-6000:])
            return self._json({"log": "\n".join(parts) or "(chưa có log)"})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        if u.path == "/api/add":
            items = split_inputs(b.get("text", ""))
            mode = b.get("workflow_mode") or paths.load_settings().get("workflow_mode", "dub")
            added = 0
            for it in items:
                if not db.exists_active(it):
                    db.add(it, mode=mode)
                    added += 1
            return self._json({"added": added, "found": len(items)})
        if u.path == "/api/retry":
            jid = int(b["id"])
            reset_from(jid, b.get("from_step"))   # không có from_step: chạy tiếp từ bước lỗi
            db.update(jid, status="queued", attempts=0, error="", not_before=0, note="")
            return self._json({"ok": True})
        if u.path == "/api/remove":
            jid = int(b["id"])
            db.remove(jid)
            if WORKER and jid in WORKER.current:
                WORKER.kill(jid)
            return self._json({"ok": True})
        if u.path == "/api/settings":
            s = paths.load_settings()
            for k, v in b.items():
                if k in EDITABLE:
                    s[k] = v
            paths.save_settings(s)
            return self._json({"ok": True})
        if u.path == "/api/open":
            target = paths.OUTPUT
            if b.get("id"):
                j = db.get(int(b["id"]))
                if j and j.get("output"):
                    target = Path(j["output"])
            if paths.IS_MAC:
                args = ["open", "-R", str(target)] if target.is_file() else ["open", str(target)]
                subprocess.Popen(args)
            elif sys.platform == "win32":
                args = ["explorer", f"/select,{target}"] if target.is_file() else ["explorer", str(target)]
                subprocess.Popen(args)
            return self._json({"ok": True, "path": str(target)})
        return self._json({"error": "not found"}, 404)


def main():
    global WORKER
    db.init()
    if acquire_worker_lock():
        WORKER = Worker()
        WORKER.start()
    else:   # cửa sổ 3_CHAY_THU đang xử lý -> giao diện chỉ để xem
        print("Cửa sổ 3_CHAY_THU đang xử lý video: giao diện chỉ để xem. Muốn xử lý bằng giao diện thì "
              "đợi nó xong rồi mở lại 2_MO_UNG_DUNG.", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"Giao diện: http://127.0.0.1:{PORT}  (đóng cửa sổ Terminal này để tắt)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if WORKER:
            WORKER.stop_flag = True
            WORKER.kill_current()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
