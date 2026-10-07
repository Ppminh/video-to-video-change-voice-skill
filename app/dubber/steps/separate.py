"""B2. Tách giọng nói khỏi nhạc nền.

Cách hoạt động: model AI nhìn "bản đồ tần số" của âm thanh theo thời gian và học được
phần nào là giọng người. Kết quả là 2 file:
  - vocals.wav     : chỉ có giọng (dùng để nghe tiếng Trung, phân biệt người nói)
  - background.wav : nhạc + tiếng động (làm nền cho giọng Việt)

Ưu tiên demucs-mlx (chạy trên GPU của chip Apple, rất nhanh).
Dự phòng: UVR MDX-Net qua sherpa-onnx (chạy CPU, không cần PyTorch).
"""
from __future__ import annotations

import time

import numpy as np

from .. import models
from ..utils import Job, load_audio, log, save_audio, threads, to_channels_last


def _demucs(job: Job):
    """Tách bằng demucs-mlx trên GPU chip Apple, cắt thành khúc 30 giây để GPU M1 không bị quá thời gian."""
    import os
    os.environ.setdefault("DEMUCS_MLX_COMPILE_FORWARD", "0")
    from demucs_mlx import Separator
    sep = Separator(model="htdemucs", batch_size=int(job.settings.get("demucs_batch", 1)), compile=False)
    sr = int(sep.samplerate)
    audio, _ = load_audio(job.p("audio.wav"), sr=sr)          # (n, 2)
    n = len(audio)
    chunk, ov = 30 * sr, 1 * sr
    vocals = np.zeros_like(audio)
    rest = np.zeros_like(audio)
    weight = np.zeros((n, 1), dtype=np.float32)
    pos = 0
    while pos < n:
        a, b = max(0, pos - ov), min(n, pos + chunk)
        seg = np.ascontiguousarray(audio[a:b].T)                 # (2, len)
        _, stems = sep.separate_tensor(seg)
        v = to_channels_last(np.asarray(stems["vocals"], dtype=np.float32))[: b - a]
        r = None
        for name, x in stems.items():
            if name != "vocals":
                x = to_channels_last(np.asarray(x, dtype=np.float32))[: b - a]
                r = x if r is None else r + x
        w = np.ones((b - a, 1), dtype=np.float32)
        if a > 0:                                                # chồng mép mượt
            w[:ov, 0] = np.linspace(0, 1, ov, dtype=np.float32)
        if b < n:
            w[-ov:, 0] = np.minimum(w[-ov:, 0], np.linspace(1, 0, ov, dtype=np.float32))
        vocals[a:b] += v * w
        rest[a:b] += r * w
        weight[a:b] += w
        pos = b
        job.progress(min(0.95, pos / n), "tách nhạc")
    weight[weight == 0] = 1
    return vocals / weight, rest / weight, sr


def _sherpa(job: Job, model_cfg):
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSourceSeparationConfig(model=model_cfg)
    if not cfg.validate():
        raise RuntimeError("Cấu hình tách nhạc không hợp lệ")
    sp = sherpa_onnx.OfflineSourceSeparation(cfg)
    audio, sr = load_audio(job.p("audio.wav"))           # (n, 2)
    samples = np.ascontiguousarray(audio.T)              # (2, n)
    out = sp.process(sample_rate=sr, samples=samples)
    vocals = np.transpose(np.asarray(out.stems[0].data, dtype=np.float32))
    rest = np.transpose(np.asarray(out.stems[1].data, dtype=np.float32))
    return vocals, rest, int(out.sample_rate)


def _spleeter(job: Job):
    """Spleeter: rất nhanh trên CPU, chất lượng thấp hơn demucs một chút."""
    import sherpa_onnx
    models.ensure("spleeter")
    d = models.path("spleeter").parent
    return _sherpa(job, sherpa_onnx.OfflineSourceSeparationModelConfig(
        spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
            vocals=str(d / "vocals.fp16.onnx"), accompaniment=str(d / "accompaniment.fp16.onnx")),
        num_threads=threads(), debug=False, provider="cpu"))


def _uvr(job: Job):
    import sherpa_onnx
    model = models.ensure("uvr")
    cfg = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(model)),
            num_threads=threads(), debug=False, provider="cpu"))
    if not cfg.validate():
        raise RuntimeError("Cấu hình UVR không hợp lệ")
    sp = sherpa_onnx.OfflineSourceSeparation(cfg)
    audio, sr = load_audio(job.p("audio.wav"))           # (n, 2)
    samples = np.ascontiguousarray(audio.T)              # (2, n)
    out = sp.process(sample_rate=sr, samples=samples)
    vocals = np.transpose(np.asarray(out.stems[0].data, dtype=np.float32))
    rest = np.transpose(np.asarray(out.stems[1].data, dtype=np.float32))
    return vocals, rest, int(out.sample_rate)


def run(job: Job) -> None:
    t0 = time.time()
    mode = job.settings.get("workflow_mode", "dub")
    if mode == "sub_only":
        log("Chế độ Chỉ VietSub: Bỏ qua tách nhạc Demucs, dùng âm thanh gốc để nhận diện lời thoại")
        audio, sr = load_audio(job.p("audio.wav"))
        save_audio(job.p("vocals.wav"), audio, sr)
        save_audio(job.p("background.wav"), audio, sr)
        v16, _ = load_audio(job.p("vocals.wav"), sr=16000, mono=True)
        save_audio(job.p("vocals_16k.wav"), v16, 16000)
        job.write("separate.json", {"backend": "direct_pass", "seconds": round(time.time() - t0, 1), "errors": []})
        return

    pref = job.settings.get("separator", "auto")
    result, used, errors = None, "", []
    order = {"auto": ["demucs", "spleeter", "uvr"], "demucs": ["demucs", "spleeter", "uvr"],
             "spleeter": ["spleeter", "demucs", "uvr"], "uvr": ["uvr", "demucs", "spleeter"]}.get(pref, ["demucs", "spleeter", "uvr"])
    fns = {"demucs": _demucs, "spleeter": _spleeter, "uvr": _uvr}
    for name in order:
        try:
            result = fns[name](job)
            used = name
            break
        except Exception as e:  # thử cách khác
            errors.append(f"{name}: {e}")
            log(f"Tách bằng {name} lỗi: {e}")
    if result is None:
        # không tách được: dùng toàn bộ âm thanh làm 'giọng', nền im lặng (vẫn chạy được tiếp)
        log("Không tách được nhạc nền - dùng âm thanh gốc làm giọng, nền im lặng")
        audio, sr = load_audio(job.p("audio.wav"))
        result = (audio, np.zeros_like(audio), sr)
        used = "none"
    vocals, rest, sr = result
    n = min(len(vocals), len(rest))
    save_audio(job.p("vocals.wav"), vocals[:n], sr)
    save_audio(job.p("background.wav"), rest[:n], sr)
    # bản 16kHz mono cho nhận dạng giọng nói
    v16, _ = load_audio(job.p("vocals.wav"), sr=16000, mono=True)
    save_audio(job.p("vocals_16k.wav"), v16, 16000)
    job.write("separate.json", {"backend": used, "seconds": round(time.time() - t0, 1), "errors": errors})
    log(f"Tách xong bằng {used} trong {time.time() - t0:.1f}s")
