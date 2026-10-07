"""Kiểm tra từng bộ phận sau khi cài. Ghi kết quả vào logs/selftest.txt.

Mỗi phần chạy trong tiến trình riêng để đo RAM tối đa của riêng phần đó.
Dùng: python -m dubber.selftest            (chạy tất cả)
      python -m dubber.selftest <phần>      (chạy 1 phần)
"""
from __future__ import annotations

import json
import os
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time
from pathlib import Path

from . import models, paths

PARTS = ["ffmpeg", "gemini", "asr", "ocr", "separate", "tts"]
EXTRA = ["voices", "tts8", "voxcpm", "pitch"]   # chỉ chạy khi gọi --parts=...
OUT = paths.LOGS / "selftest.txt"


def peak_mb() -> float:
    if resource is not None:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3   # mac: byte, linux: KB
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def part_ffmpeg():
    from .utils import ffmpeg, has_encoder, ffmpeg_bin
    tmp = paths.WORK / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.new("RGBA", (320, 80), (0, 0, 0, 0)).save(tmp / "a.png")
    (tmp / "s.ffconcat").write_text("ffconcat version 1.0\nfile 'a.png'\nduration 1\nfile 'a.png'\n")
    enc = ["-c:v", "h264_videotoolbox", "-b:v", "2M"] if has_encoder("h264_videotoolbox") else ["-c:v", "libx264"]
    ffmpeg(["-f", "lavfi", "-i", "testsrc=size=320x568:rate=30:duration=2", "-f", "lavfi", "-i", "sine=duration=2",
            "-f", "concat", "-safe", "0", "-i", tmp / "s.ffconcat",
            "-filter_complex", "[0:v]split=2[a][b];[b]crop=320:60:0:400,gblur=sigma=18[c];[a][c]overlay=0:400:enable='between(t,0.5,1.5)'[v];[2:v]format=rgba[s];[v][s]overlay=0:420:format=auto,format=yuv420p[o]",
            "-map", "[o]", "-map", "1:a", *enc, "-c:a", "aac", "-shortest", tmp / "t.mp4"])
    return {"ffmpeg": ffmpeg_bin(), "videotoolbox": has_encoder("h264_videotoolbox"),
            "test_mp4_bytes": (tmp / "t.mp4").stat().st_size}


def part_gemini():
    from . import llm
    data, model = llm.generate('Dịch sang tiếng Việt giọng miền Nam, trả về JSON {"vi": "..."}: 你吃饭了吗？',
                               max_tokens=200)
    return {"model": model, "reply": data}


def part_asr():
    import numpy as np
    from .steps import transcribe as T
    from .utils import load_audio
    wav = models.ensure("test_wav")
    samples, _ = load_audio(wav, sr=16000, mono=True)
    t0 = time.time()
    segs = T._vad_segments(samples)
    t1 = time.time()
    turns = T._diarize(samples)
    t2 = time.time()
    rec = T._recognizer()
    streams = []
    for st, en in segs:
        s = rec.create_stream()
        s.accept_waveform(16000, samples[int(st * 16000):int(en * 16000)])
        streams.append(s)
    rec.decode_streams(streams)
    t3 = time.time()
    texts = [T._clean(s.result.text) for s in streams]
    dur = len(samples) / 16000
    return {"audio_s": round(dur, 1), "vad_segments": len(segs), "vad_s": round(t1 - t0, 2),
            "speakers": len({t[2] for t in turns}), "diar_s": round(t2 - t1, 2),
            "asr_s": round(t3 - t2, 2), "text": " | ".join(texts)[:300],
            "pitch_hz": round(T._pitch(samples[: 16000 * 10]), 1)}


def part_ocr():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from rapidocr_onnxruntime import RapidOCR
    fonts = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    fp = next((f for f in fonts if Path(f).exists()), None)
    img = Image.new("RGB", (360, 640), (40, 60, 90))
    if fp:
        d = ImageDraw.Draw(img)
        d.text((40, 520), "你好，今天天气很好", font=ImageFont.truetype(fp, 26), fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))
    eng = RapidOCR()
    arr = np.array(img)[:, :, ::-1].copy()
    t0 = time.time()
    res, _ = eng(arr, use_cls=False)
    dt = time.time() - t0
    return {"font": fp, "result": [r[1] for r in (res or [])], "ms_per_frame": round(dt * 1000)}


def part_separate():
    from .utils import ffmpeg, load_audio, save_audio
    tmp = paths.WORK / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    wav = models.ensure("test_wav")
    ffmpeg(["-i", wav, "-ac", "2", "-ar", "44100", "-t", "75", tmp / "audio.wav"])   # >30s để thử cắt khúc
    from .utils import Job
    job = Job(tmp)
    from .steps import separate
    res = {}
    for name, fn in (("demucs", separate._demucs), ("spleeter", separate._spleeter)):
        t0 = time.time()
        try:
            fn(job)
            res[name + "_s_for_57s"] = round(time.time() - t0, 1)
        except Exception as e:
            res[name + "_error"] = str(e)[:200]
    return res


def part_tts():
    import numpy as np
    from .ttsload import load as load_tts
    from .steps.tts import trim_silence
    from .utils import save_audio
    t0 = time.time()
    tts = load_tts("turbo", int(paths.load_settings().get("threads", 4)))
    load_s = time.time() - t0
    sr = tts.sample_rate
    presets = getattr(tts, "_preset_voices", {}) or {}
    south = [(n, v.get("gender"), v.get("description")) for n, v in presets.items() if "· Nam ·" in (v.get("description") or "")]
    out = paths.LOGS / "giong_mau"
    out.mkdir(exist_ok=True)
    text = "Chào cả nhà, hôm nay mình kể cho mọi người nghe một câu chuyện cực kỳ thú vị nha."
    rtfs = {}
    for name, g, desc in south:
        t1 = time.time()
        a = trim_silence(np.asarray(tts.infer(text, voice=name), dtype=np.float32), sr)
        dt = time.time() - t1
        save_audio(out / f"{'nam' if g == 'male' else 'nu'}_{name}.wav", a, sr)
        rtfs[name] = round(dt / max(0.1, len(a) / sr), 2)
    return {"load_s": round(load_s, 1), "south_voices": [f"{n} ({g})" for n, g, _ in south], "rtf": rtfs,
            "samples_dir": str(out)}


def part_voices():
    """Tạo mẫu nghe thử các biến thể giọng (logs/giong_mau/bien_the_*.wav)."""
    import numpy as np
    from .ttsload import load as load_tts
    from .steps.tts import VARIANTS, apply_variant, trim_silence
    from .utils import save_audio
    tts = load_tts("turbo", int(paths.load_settings().get("threads", 4)))
    sr = tts.sample_rate
    out = paths.LOGS / "giong_mau"
    out.mkdir(exist_ok=True)
    text = "Anh nói thật đi, chuyện hôm qua rốt cuộc là sao? Em đợi anh cả buổi tối đó."
    made = []
    for base in ("Thục Đoan", "Adam"):
        a = trim_silence(np.asarray(tts.infer(text, voice=base), dtype=np.float32), sr)
        for v in VARIANTS:
            if base == "Adam" and v == "bé":
                continue
            t0 = time.time()
            y = apply_variant(a, sr, v)
            name = f"bien_the_{base}_{v or 'goc'}.wav".replace(" ", "_")
            save_audio(out / name, y, sr)
            made.append(f"{name} ({time.time() - t0:.2f}s)")
    return {"samples": made, "dir": str(out)}


def part_tts8():
    """So sánh tốc độ giọng đọc fp32 và int8 (int8 nhanh hơn, cần nghe thử xem có méo không)."""
    import numpy as np
    from .ttsload import load as load_tts
    from .steps.tts import trim_silence
    from .utils import save_audio
    out = paths.LOGS / "giong_mau"
    out.mkdir(exist_ok=True)
    text = "Chào cả nhà, hôm nay mình kể cho mọi người nghe một câu chuyện cực kỳ thú vị nha."
    res = {}
    for prec in ("fp32", "int8"):
        t0 = time.time()
        tts = load_tts("turbo", int(paths.load_settings().get("threads", 4)), prec)
        load_s = time.time() - t0
        sr = tts.sample_rate
        tot_t = tot_a = 0.0
        for voice in ("Thục Đoan", "Adam"):
            t1 = time.time()
            a = trim_silence(np.asarray(tts.infer(text, voice=voice), dtype=np.float32), sr)
            tot_t += time.time() - t1
            tot_a += len(a) / sr
            save_audio(out / f"{prec}_{voice.replace(' ', '_')}.wav", a, sr)
        res[prec] = {"load_s": round(load_s, 1), "rtf": round(tot_t / max(0.1, tot_a), 3)}
    return res


def part_voxcpm():
    """Kiểm tra mô hình OpenBMB VoxCPM2 đọc tiếng Việt."""
    from .ttsload import load as load_tts
    from .utils import save_audio
    import numpy as np
    t0 = time.time()
    tts = load_tts("voxcpm", voxcpm_model="openbmb/VoxCPM2", voxcpm_timesteps=8)
    load_s = time.time() - t0
    sr = tts.sample_rate
    text = "Chào bạn, đây là giọng đọc tiếng Việt từ mô hình VoxCPM."
    out = paths.LOGS / "giong_mau"
    out.mkdir(exist_ok=True)
    t1 = time.time()
    ref_file = out / "nu_Thục Đoan.wav"
    ref = str(ref_file) if ref_file.exists() else None
    wav = tts.infer(text, ref_wav=ref)
    gen_s = time.time() - t1
    out_file = out / "voxcpm_tieng_viet.wav"
    save_audio(out_file, wav, sr)
    dur = len(wav) / sr
    return {
        "model": "openbmb/VoxCPM2",
        "load_s": round(load_s, 1),
        "gen_s": round(gen_s, 1),
        "audio_dur_s": round(dur, 2),
        "rtf": round(gen_s / max(0.1, dur), 2),
        "output": str(out_file)
    }


def part_pitch():
    """Kiểm tra module F0 Pitch Tracking & ánh xạ SSML."""
    from .pitch import analyze_segment_pitch_energy, classify_voice_tone, map_pitch_to_prosody
    import numpy as np
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.2 * np.sin(2 * np.pi * 105 * t)).astype(np.float32)
    t0 = time.time()
    res = analyze_segment_pitch_energy(sig, sr)
    dt = time.time() - t0
    tone = classify_voice_tone(res["f0"], res["rms"])
    prosody = map_pitch_to_prosody(res, voice_name="Nam Minh")
    return {"f0_hz": res["f0"], "tone": tone, "time_s": round(dt, 4), "prosody": prosody}


def run_part(name: str) -> dict:
    t0 = time.time()
    fn = globals()[f"part_{name}"]
    res = fn()
    res["_seconds"] = round(time.time() - t0, 1)
    res["_peak_ram_mb"] = round(peak_mb())
    return res


def main():
    parts = PARTS
    if len(sys.argv) > 1 and sys.argv[1].startswith("--parts="):
        parts = [x for x in sys.argv[1][8:].split(",") if x in PARTS + EXTRA]
    elif len(sys.argv) > 1:
        name = sys.argv[1]
        try:
            res = {"ok": True, **run_part(name)}
        except Exception as e:
            import traceback
            res = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        print("@@RESULT@@" + json.dumps(res, ensure_ascii=False))
        return
    lines = [f"KIỂM TRA HỆ THỐNG - {time.strftime('%Y-%m-%d %H:%M')}"]
    env = dict(os.environ, PYTHONPATH=str(paths.APP_DIR), PYTHONIOENCODING="utf-8")
    for name in parts:
        print(f"\n--- Kiểm tra: {name}", flush=True)
        p = subprocess.run([sys.executable, "-m", "dubber.selftest", name], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, cwd=str(paths.APP_DIR))
        res = None
        for ln in p.stdout.splitlines():
            if ln.startswith("@@RESULT@@"):
                res = json.loads(ln[10:])
        if res is None:
            res = {"ok": False, "error": (p.stderr or p.stdout)[-1500:]}
        status = "OK " if res.get("ok") else "LỖI"
        line = f"[{status}] {name}: " + json.dumps({k: v for k, v in res.items() if k not in ('ok', 'trace')},
                                                   ensure_ascii=False)
        print(line, flush=True)
        if not res.get("ok"):
            print(res.get("trace", ""), flush=True)
        lines.append(line)
    out = OUT if parts == PARTS else paths.LOGS / "selftest_part.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
