"""Bộ điều phối: lấy video trong hàng chờ, chạy 9 bước.

Tối ưu "2 làn song song" (giống dây chuyền nhà máy):
- Làn 1 (chuẩn bị): tải -> tách nhạc (GPU) -> nghe chữ -> OCR -> dịch (mạng)
- Làn 2 (lồng tiếng): đọc giọng (CPU) -> trộn -> xuất video -> kiểm tra
Khi làn 2 đang đọc giọng video A thì làn 1 đã chuẩn bị video B -> mỗi video ra nhanh gần gấp đôi.
Làn 1 chỉ chuẩn bị trước tối đa 1 video (đỡ tốn ổ cứng), chạy ưu tiên thấp hơn và ít luồng CPU hơn
để không làm chậm làn 2. Trước bước nặng sẽ chờ đủ RAM trống.

Giữ nguyên các điểm cũ:
- Mỗi bước chạy trong tiến trình con ưu tiên thấp (nice) -> máy vẫn mượt khi bạn dùng.
- Bước nào xong sẽ có file đánh dấu .done_<bước>: mất điện/tắt máy thì chạy tiếp từ bước dở.
- Lỗi mạng (tải, dịch) -> tự thử lại sau vài phút, tối đa 3 lần.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import db, paths
from .steps import STEPS
from .utils import read_json, wait_for_memory, write_json

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
RETRYABLE = set(STEPS)                     # Tự động thử lại thông minh cho toàn bộ 9 bước
PREP = STEPS[:STEPS.index("tts")]          # download, separate, transcribe, ocr, translate
VOICE = STEPS[STEPS.index("tts"):]         # tts, mix, render, qc
NEED_MB = {"separate": 1000, "tts": 1000, "transcribe": 800, "ocr": 600}


def job_dir(job_id: int) -> Path:
    return paths.WORK / f"job_{job_id:05d}"


def first_todo(d: Path) -> str | None:
    return next((s for s in STEPS if not (d / f".done_{s}").exists()), None)


def reset_from(jid: int, step: str | None) -> list[str]:
    """Chuẩn bị chạy lại 1 video từ một bước. Tự thêm các bước cần thiết nếu file trung gian đã bị dọn
    (sau khi xong, hệ thống xóa file âm thanh lớn để tiết kiệm ổ cứng). Trả về các bước sẽ chạy lại."""
    d = job_dir(jid)
    for f in d.glob("error_*.txt"):
        f.unlink(missing_ok=True)
    if step not in STEPS:
        return []
    if not (d / "source.mp4").exists():
        step = "download"
    if step in ("render", "qc") and not (d / "mix.wav").exists():
        step = "mix"
    if step == "mix" and not (d / "tts").exists():
        step = "tts"
    redo = STEPS[STEPS.index(step):]
    wavs = ("vocals.wav", "background.wav", "vocals_16k.wav")
    if STEPS.index(step) > STEPS.index("separate") and not all((d / n).exists() for n in wavs):
        redo = ["download", "separate"] + [x for x in redo if x not in ("download", "separate")]
    for st in redo:
        (d / f".done_{st}").unlink(missing_ok=True)
    return redo


def scan_input_folder() -> None:
    """File video bỏ vào thư mục DAU_VAO sẽ tự được thêm vào hàng chờ."""
    taken = paths.INPUT_DIR / "_da_nhan"
    for f in sorted(paths.INPUT_DIR.glob("*")):
        if f.is_file() and f.suffix.lower() in VIDEO_EXT:
            try:
                if time.time() - f.stat().st_mtime < 5:   # file đang được chép
                    continue
                taken.mkdir(exist_ok=True)
                dest = taken / f.name
                shutil.move(str(f), dest)
                db.add(str(dest))
            except Exception:
                pass


_LOCK_FH = None


def acquire_worker_lock() -> bool:
    """Chỉ cho 1 bộ điều phối chạy cùng lúc (tránh 2 cửa sổ cùng xử lý 1 video)."""
    global _LOCK_FH
    lock_file = paths.WORK / "worker.lock"
    try:
        import fcntl
        _LOCK_FH = open(lock_file, "w")
        fcntl.flock(_LOCK_FH, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except ImportError:
        try:
            import msvcrt
            _LOCK_FH = open(lock_file, "w")
            msvcrt.locking(_LOCK_FH.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (OSError, IOError):
            return False
    except OSError:
        return False


class Worker:
    def __init__(self):
        self.stop_flag = False
        self.lock = threading.Lock()
        self.claimed: dict[int, str] = {}       # mã video -> làn đang xử lý
        self.procs: dict[str, subprocess.Popen] = {}
        self.threads: list[threading.Thread] = []
        self.wake = {"prep": threading.Event(), "voice": threading.Event()}   # đánh thức làn kia ngay

    def _idle(self, lane: str, sec: float = 3) -> None:
        self.wake[lane].wait(sec)
        self.wake[lane].clear()

    # ------------------------------------------------------------ vòng chạy
    def start(self):
        db.reset_running()                      # video đang dở lúc tắt máy -> cho vào hàng chờ lại
        for lane in ("prep", "voice"):
            t = threading.Thread(target=self._lane, args=(lane,), daemon=True, name=f"lane-{lane}")
            t.start()
            self.threads.append(t)

    def is_alive(self) -> bool:
        return any(t.is_alive() for t in self.threads)

    @property
    def current(self):
        return sorted(self.claimed)

    def _lane(self, lane: str):
        while not self.stop_flag:
            job = None
            try:
                s = paths.load_settings()
                if s.get("paused"):
                    self._idle(lane)
                    continue
                parallel = bool(s.get("parallel_lanes", False))
                if lane == "voice" and not parallel:
                    self._idle(lane)
                    continue
                if lane == "prep":
                    scan_input_folder()
                job = self._pick(lane, parallel, int(s.get("prep_ahead", 1)))
                if not job:
                    self._idle(lane)
                    continue
                if lane == "voice":
                    self.wake["prep"].set()              # 1 video vừa vào làn 2 -> làn 1 được chuẩn bị tiếp
                steps = STEPS if not parallel else (PREP if lane == "prep" else VOICE)
                self.process(job, steps, lane)
                self.wake["voice" if lane == "prep" else "prep"].set()
            except Exception as e:  # không để làn chết
                print(f"Làn {lane} lỗi:", e, flush=True)
                time.sleep(5)
            finally:
                if job:
                    with self.lock:
                        self.claimed.pop(job["id"], None)

    def _pick(self, lane: str, parallel: bool, ahead: int):
        with self.lock:
            if not parallel and self.claimed:
                # Chế độ tuần tự nghiêm ngặt (concurrency = 1): Đang có video chạy -> không nhận thêm video nào
                return None
            cands = [j for j in db.candidates() if j["id"] not in self.claimed]
            if lane == "prep" and parallel:
                waiting = sum(1 for j in cands if j["status"] == "ready")
                if waiting >= max(1, ahead):
                    return None
            for j in cands:
                nxt = first_todo(job_dir(j["id"]))
                in_voice = nxt is None or nxt in VOICE
                if parallel and lane == "prep" and in_voice:
                    continue
                if lane == "voice" and not in_voice:
                    continue
                self.claimed[j["id"]] = lane
                return j
        return None

    # ------------------------------------------------------------ chạy các bước
    def process(self, job: dict, steps=None, lane: str = "all") -> None:
        steps = list(steps or STEPS)
        jid = job["id"]
        d = job_dir(jid)
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "job.json").exists():
            mode = job.get("mode") or paths.load_settings().get("workflow_mode", "dub")
            write_json(d / "job.json", {"id": jid, "input": job["input"], "mode": mode})
        db.update(jid, status="running", error="", started=job.get("started") or time.time())
        timings = json.loads((db.get(jid) or job).get("timings") or "{}")
        background = lane == "prep"
        for step in steps:
            if (d / f".done_{step}").exists():
                continue
            if paths.load_settings().get("paused"):
                db.update(jid, status="queued", note="tạm dừng")
                return
            db.update(jid, step=step, frac=0, note="chờ RAM trống" if step in NEED_MB else "")
            wait_for_memory(NEED_MB.get(step, 500), timeout=30)
            db.update(jid, note="")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(paths.APP_DIR)
            env["PYTHONUNBUFFERED"] = "1"
            if background and paths.load_settings().get("parallel_lanes", False):
                env["DUBBER_THREADS"] = "2"           # nhường CPU cho làn đọc giọng
            cmd = [sys.executable, "-m", "dubber.run_step", step, str(d)]
            if paths.IS_MAC or shutil.which("nice"):
                cmd = ["nice", "-n", "10" if background else "5", *cmd]
            t0 = time.time()
            with open(d / f"log_{step}.txt", "a", encoding="utf-8") as lf:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(paths.APP_DIR))
                self.procs[lane] = proc
                while proc.poll() is None:
                    time.sleep(1.5)
                    pr = read_json(d / "progress.json", None)
                    if pr:
                        db.update(jid, frac=pr.get("frac", 0), note=pr.get("note", ""))
                rc = proc.returncode
            self.procs.pop(lane, None)
            timings = json.loads((db.get(jid) or {}).get("timings") or "{}") | timings
            timings[step] = round(time.time() - t0, 1)
            db.update(jid, timings=timings)
            (d / "progress.json").unlink(missing_ok=True)
            if step == "download":
                meta = read_json(d / "meta.json", {}) or {}
                db.update(jid, title=(meta.get("desc") or "")[:120])
            if step == "translate":
                tl = read_json(d / "translation.json", {}) or {}
                if tl.get("title_vi"):
                    db.update(jid, title=tl["title_vi"][:120])
            if rc != 0:
                if not db.get(jid):                   # video đã bị xóa khỏi danh sách
                    return
                err = (d / f"error_{step}.txt")
                msg = err.read_text(encoding="utf-8") if err.exists() else f"Bước {step} lỗi (mã {rc})"
                attempts = ((db.get(jid) or job).get("attempts") or 0) + 1
                if step in RETRYABLE and attempts <= 3:
                    backoff_sec = [5, 15, 30][min(attempts - 1, 2)]
                    db.update(jid, status="queued", attempts=attempts, error=msg[:1000],
                              not_before=time.time() + backoff_sec, note=f"sẽ thử lại sau {backoff_sec}s (lần thử lại {attempts}/3)")
                else:
                    db.update(jid, status="failed", attempts=attempts, error=msg[:1000], finished=time.time())
                return
        if first_todo(d) is not None:                  # xong làn chuẩn bị, chờ làn lồng tiếng
            db.update(jid, status="ready", step="", frac=0, note="chờ lồng tiếng")
            return
        qc = read_json(d / "qc.json", {}) or {}
        db.update(jid, status=qc.get("status", "done"), step="", frac=1, note="", output=qc.get("output", ""),
                  issues=qc.get("issues", []), finished=time.time())

    def kill(self, jid: int) -> None:
        lane = self.claimed.get(jid)
        p = self.procs.get(lane) if lane else None
        if p and p.poll() is None:
            p.terminate()

    def kill_current(self):
        for p in list(self.procs.values()):
            if p.poll() is None:
                p.terminate()
