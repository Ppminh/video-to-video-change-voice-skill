"""Tiện ích dùng chung: ffmpeg, đọc/ghi audio, JSON, đối tượng Job."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import paths


# ---------------------------------------------------------------- logging
def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------- json
def read_json(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p, data) -> None:
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------- ffmpeg
def ffmpeg_bin() -> str:
    for c in (os.environ.get("FFMPEG"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", shutil.which("ffmpeg")):
        if c and Path(c).exists():
            return c
    raise RuntimeError("Không tìm thấy ffmpeg")


def ffprobe_bin() -> str:
    ff = Path(ffmpeg_bin())
    cand = ff.with_name("ffprobe")
    return str(cand) if cand.exists() else (shutil.which("ffprobe") or "ffprobe")


def run(cmd: list, check: bool = True, quiet: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        tail = (r.stderr or "")[-2500:]
        raise RuntimeError(f"Lệnh lỗi ({r.returncode}): {' '.join(map(str, cmd[:6]))} ...\n{tail}")
    return r


def ffmpeg(args: list, check: bool = True) -> subprocess.CompletedProcess:
    return run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=check)


def probe(path) -> dict:
    r = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(r.stdout)
    info = {"duration": float(data.get("format", {}).get("duration", 0) or 0)}
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and "width" not in info:
            info["width"] = int(s.get("width", 0))
            info["height"] = int(s.get("height", 0))
            fr = s.get("avg_frame_rate") or s.get("r_frame_rate") or "30/1"
            try:
                a, b = fr.split("/")
                info["fps"] = float(a) / float(b) if float(b) else 30.0
            except Exception:
                info["fps"] = 30.0
            rot = 0
            for sd in s.get("side_data_list", []) or []:
                if "rotation" in sd:
                    rot = int(sd["rotation"])
            if "tags" in s and "rotate" in s["tags"]:
                rot = int(s["tags"]["rotate"])
            info["rotation"] = rot
        if s.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def has_encoder(name: str) -> bool:
    try:
        r = run([ffmpeg_bin(), "-hide_banner", "-encoders"], check=False)
        return name in r.stdout
    except Exception:
        return False


# ---------------------------------------------------------------- audio
def load_audio(path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Trả về (samples, sr). samples: (n,) nếu mono, (n, ch) nếu nhiều kênh. float32."""
    import soundfile as sf
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        data = data.mean(axis=1, keepdims=True)
    if sr and sr != file_sr:
        import soxr
        data = soxr.resample(data, file_sr, sr)
        file_sr = sr
    if mono:
        data = data[:, 0]
    return np.ascontiguousarray(data, dtype=np.float32), file_sr


def save_audio(path, data: np.ndarray, sr: int) -> None:
    import soundfile as sf
    data = np.asarray(data, dtype=np.float32)
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")


def to_channels_last(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a[:, None]
    if a.shape[0] <= 2 and a.shape[1] > a.shape[0]:
        return a.T
    return a


# ---------------------------------------------------------------- text
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def cjk_count(s: str) -> int:
    return len(CJK_RE.findall(s or ""))


def vi_syllables(s: str) -> int:
    return len([w for w in re.split(r"\s+", (s or "").strip()) if re.search(r"\w", w)])


def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"[#@]\S+", "", s or "")
    s = re.sub(r"[\\/:*?\"<>|\n\r\t]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:maxlen].strip() or "video"


def fmt_ts(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:04.1f}"


# ---------------------------------------------------------------- memory
def mem_available_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e6
    except Exception:
        return 99999.0


def wait_for_memory(min_mb: float = 900, timeout: float = 120) -> None:
    t0 = time.time()
    while mem_available_mb() < min_mb and time.time() - t0 < timeout:
        time.sleep(5)


# ---------------------------------------------------------------- job
class Job:
    """Một video đang xử lý. Mọi file trung gian nằm trong thư mục riêng của job."""

    def __init__(self, job_dir):
        self.dir = Path(job_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.info = read_json(self.dir / "job.json", {})

    def p(self, name: str) -> Path:
        return self.dir / name

    def read(self, name: str, default=None):
        return read_json(self.p(name), default)

    def write(self, name: str, data) -> None:
        write_json(self.p(name), data)

    def progress(self, frac: float, note: str = "") -> None:
        write_json(self.p("progress.json"), {"frac": round(float(frac), 3), "note": note, "t": time.time()})

    @property
    def settings(self) -> dict:
        if hasattr(self, "_settings_override") and self._settings_override is not None:
            return self._settings_override
        s = paths.load_settings()
        info = self.read("job.json", {}) or {}
        if info.get("mode"):
            s["workflow_mode"] = info["mode"]
        return s

    @settings.setter
    def settings(self, val: dict) -> None:
        self._settings_override = val


def threads() -> int:
    try:
        return int(os.environ.get("DUBBER_THREADS") or paths.load_settings().get("threads", 4))
    except Exception:
        return 4


def python_exe() -> str:
    return sys.executable


# ---------------------------------------------------------------- pitch & energy
def analyze_audio_pitch(samples: np.ndarray, sr: int) -> dict:
    from .pitch import analyze_segment_pitch_energy
    return analyze_segment_pitch_energy(samples, sr)


def classify_pitch_tone(f0: float, rms: float, avg_rms: float = 0.08) -> str:
    from .pitch import classify_voice_tone
    return classify_voice_tone(f0, rms, avg_rms)
