"""B8. Trộn giọng Việt với nhạc nền gốc.

- Nền = file nhạc + tiếng động đã tách ở B2.
- Giữ lại tiếng cười/khóc/thở dài biểu cảm gốc từ 'vocals.wav', loại bỏ các đoạn Speech gây rồ xè.
- Ducking: hạ nhạc nền vài dB khi đang có lời để nghe rõ giọng.
- Chuẩn hóa độ to về -14 LUFS (chuẩn phát sóng các nền tảng video).
- Nâng cấp Studio True Peak Limiter (lookahead 5ms + soft-knee mượt mà, trần <= -1.0 dBFS)
  triệt tiêu 100% hiện tượng inter-sample clipping và rách âm ở các đoạn cao trào / tiếng hét.
"""
from __future__ import annotations

import warnings
import numpy as np

from ..utils import Job, load_audio, log, save_audio

SR = 44100


def _place(track, clip, start_s):
    a = int(round(start_s * SR))
    if a >= len(track):
        return
    b = min(len(track), a + len(clip))
    track[a:b] += clip[:b - a]


def _fade(x, n):
    n = min(n, len(x) // 2)
    if n > 1:
        r = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


HOP = SR // 100   # 10ms


def _frames_mean(x: np.ndarray) -> np.ndarray:
    nf = int(np.ceil(len(x) / HOP))
    return np.pad(x, (0, nf * HOP - len(x))).reshape(nf, HOP).mean(axis=1)


def _smooth(f: np.ndarray, frames: int) -> np.ndarray:
    frames = max(1, int(frames))
    return np.convolve(f, np.ones(frames) / frames, mode="same")


def _to_samples(f: np.ndarray, n: int) -> np.ndarray:
    """Nội suy tuyến tính từ 100 điểm/giây về từng mẫu âm thanh."""
    g = np.append(f, f[-1]).astype(np.float32)
    ramp = np.arange(HOP, dtype=np.float32) / HOP
    out = (g[:-1, None] + (g[1:] - g[:-1])[:, None] * ramp[None, :]).reshape(-1)
    return out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)), mode="edge")


def studio_true_peak_limiter(audio: np.ndarray, sr: int = SR,
                            ceiling_db: float = -1.02, knee_db: float = 1.5,
                            lookahead_ms: float = 5.0, release_ms: float = 40.0) -> np.ndarray:
    """Bộ giới hạn đỉnh chuẩn phòng thu (Studio True Peak Limiter với soft-knee và lookahead):
    - Thay thế hoàn toàn np.tanh thô sơ, triệt tiêu 100% hiện tượng inter-sample clipping và xé tiếng AAC.
    - Giới hạn trần tối đa tại <= -1.0 dBFS (linear ~0.89125).
    - Đường bao nén mềm (soft-knee) mượt mà và lookahead 5ms giúp bảo toàn độ trong trẻo và dynamic range.
    """
    import scipy.signal
    import scipy.ndimage

    if audio is None or len(audio) == 0:
        return audio
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    orig_shape = audio.shape
    if audio.ndim == 1:
        x = audio[:, None]
    else:
        x = audio.copy()

    n_samples, n_channels = x.shape
    ceiling_linear = 10.0 ** (ceiling_db / 20.0)

    # Dự phòng headroom discrete (1.2 dB) để bù trừ inter-sample peaks giữa các mẫu 1x
    # giúp True Peak sau oversampling không bị vọt trần và không làm sụt giảm độ to toàn bài
    d_ceil_db = ceiling_db - 1.2
    d_ceil_linear = 10.0 ** (d_ceil_db / 20.0)
    knee_linear = 10.0 ** ((d_ceil_db - knee_db) / 20.0)

    # Đường dẫn nhanh: nếu toàn bộ tín hiệu đã nằm dưới ngưỡng knee, không cần nén
    max_init = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
    if max_init <= knee_linear:
        return audio

    # Khung tính toán 1ms (1000 điểm/giây) cho tốc độ siêu nhanh và độ phân giải đỉnh tối ưu
    hop = sr // 1000
    nf = int(np.ceil(n_samples / hop))
    pad_len = nf * hop - n_samples
    padded_x = np.pad(x, ((0, pad_len), (0, 0)))

    # 1. Đỉnh biên độ trên từng khung 1ms (tối ưu hóa bộ nhớ và tốc độ)
    block_peaks = np.max(np.abs(padded_x.reshape(nf, hop, -1)), axis=(1, 2))

    # 2. Tính hệ số suy giảm gain tĩnh theo đường cong soft-knee hướng tới d_ceil_linear
    target_gain = np.ones(nf, dtype=np.float32)
    in_knee = (block_peaks > knee_linear) & (block_peaks <= d_ceil_linear)
    over_ceil = block_peaks > d_ceil_linear

    if np.any(in_knee):
        p_k = block_peaks[in_knee]
        p_db = 20.0 * np.log10(np.maximum(p_k, 1e-9))
        gr_db = ((p_db - (d_ceil_db - knee_db)) ** 2) / (4.0 * knee_db)
        target_gain[in_knee] = 10.0 ** (-gr_db / 20.0)

    if np.any(over_ceil):
        p_c = block_peaks[over_ceil]
        p_db = 20.0 * np.log10(np.maximum(p_c, 1e-9))
        gr_db = p_db - d_ceil_db
        target_gain[over_ceil] = 10.0 ** (-gr_db / 20.0)

    # Ràng buộc trần cứng trên gain tĩnh
    over_limit = block_peaks * target_gain > d_ceil_linear
    if np.any(over_limit):
        target_gain[over_limit] = d_ceil_linear / block_peaks[over_limit]

    # 3. Lookahead: dịch cửa sổ về phía trước để giảm gain trước khi đỉnh âm thanh xuất hiện
    look_blocks = max(2, int(lookahead_ms))
    K = look_blocks
    S = 2 * K + 1
    look_gain = scipy.ndimage.minimum_filter1d(target_gain, size=S, mode="nearest", origin=-K)

    # 4. Bộ lọc nhả mềm (smooth release filter)
    alpha_rel = 1.0 - np.exp(-1.0 / release_ms)
    gain_env = look_gain.copy()
    for i in range(1, nf):
        if gain_env[i] > gain_env[i - 1]:
            gain_env[i] = min(target_gain[i], gain_env[i - 1] + alpha_rel * (1.0 - gain_env[i - 1]))

    # 5. Làm mượt đoạn vào nén (attack smoothing) bằng cửa sổ Hann với đệm biên edge
    attack_w = max(2, look_blocks // 2)
    hann = np.hanning(attack_w * 2 + 1)
    hann /= hann.sum()
    padded_env = np.pad(gain_env, attack_w, mode="edge")
    smooth_gain_env = scipy.signal.convolve(padded_env, hann, mode="valid")
    smooth_gain_env = np.minimum(smooth_gain_env, look_gain)

    # 6. Nội suy về tần số lấy mẫu gốc
    g_padded = np.append(smooth_gain_env, smooth_gain_env[-1])
    ramp = np.arange(hop, dtype=np.float32) / hop
    full_gain = (g_padded[:-1, None] + (g_padded[1:] - g_padded[:-1])[:, None] * ramp[None, :]).reshape(-1)[:n_samples]

    out = x * full_gain[:, None]

    # 7. Đảm bảo giới hạn trần discrete peak
    d_peak = float(np.max(np.abs(out)))
    if d_peak > d_ceil_linear:
        out = np.clip(out, -d_ceil_linear, d_ceil_linear)

    # 8. Đo và kiểm soát True Peak thực tế (ITU-R BS.1770-4 4x oversampling)
    ch_max = np.max(np.abs(out), axis=0)
    best_ch = int(np.argmax(ch_max))
    up = scipy.signal.resample_poly(out[:, best_ch], 4, 1)
    tp_peak = float(np.max(np.abs(up)))
    if tp_peak > ceiling_linear:
        out *= (ceiling_linear / tp_peak)

    if orig_shape != out.shape:
        out = out.reshape(orig_shape)
    return out


def run(job: Job) -> None:
    s = job.settings
    mode = s.get("workflow_mode", "dub")
    if mode == "sub_only":
        log("Chế độ Chỉ VietSub: Bỏ qua trộn âm thanh, giữ nguyên 100% âm thanh gốc")
        if job.p("audio.wav").exists():
            import shutil
            shutil.copyfile(job.p("audio.wav"), job.p("mix.wav"))
        job.write("mix.json", {"mode": "sub_only", "kept_original_audio": True})
        return

    import pyloudnorm as pyln

    tr = job.read("transcript.json")
    fit = job.read("fit.json")
    bg, sr_bg = load_audio(job.p("background.wav"), sr=SR)
    voc, _ = load_audio(job.p("vocals.wav"), sr=SR)
    meta = job.read("meta.json")
    n = int(round(float(meta.get("duration", tr["duration"])) * SR))
    if bg.ndim == 1:
        bg = bg[:, None]
    bg = np.pad(bg, ((0, max(0, n - len(bg))), (0, 0)))[:n]
    if bg.shape[1] == 1:
        bg = np.repeat(bg, 2, axis=1)
    voc = np.pad(voc, ((0, max(0, n - len(voc))), (0, 0)))[:n]
    if voc.shape[1] == 1:
        voc = np.repeat(voc, 2, axis=1)

    # R2: Chuẩn hóa biên độ đỉnh an toàn (-1.5 dBFS headroom) cho từng clip thoại riêng lẻ
    target_clip_peak = 10.0 ** (-1.5 / 20.0)  # ~0.8414
    voice = np.zeros(n, dtype=np.float32)
    for f in fit["lines"]:
        clip, _ = load_audio(job.p("tts") / f["file"], sr=SR, mono=True)
        maxlen = int((f["window"] + 0.3) * SR)
        if len(clip) > maxlen:   # quá dài -> cắt và làm mờ đuôi
            clip = clip[:maxlen]
        pk = float(np.max(np.abs(clip))) if len(clip) > 0 else 0.0
        if pk > target_clip_peak:
            clip = clip * (target_clip_peak / pk)
        _place(voice, _fade(clip.copy(), int(0.008 * SR)), f["start"])

    # R3: Lọc sạch tiếng xè / rồ từ tạp âm giọng gốc (nonspeech)
    # Chỉ giữ lại các âm thanh biểu cảm thực sự (cười, khóc, thở dài...)
    # Loại bỏ các đoạn nhận diện nhầm là giọng nói tiếng Trung (event: Speech)
    EXPRESSIVE_EVENTS = ("laughter", "laugh", "cry", "sigh", "gasp", "groan", "sob", "cười", "khóc", "thở dài")
    raw_nonspeech = tr.get("nonspeech", [])
    valid_nonspeech = []
    for seg in raw_nonspeech:
        ev = str(seg.get("event") or "").strip().lower()
        if ev and ev != "speech" and any(k in ev for k in EXPRESSIVE_EVENTS):
            valid_nonspeech.append(seg)

    nf = int(np.ceil(n / HOP))
    keep = np.zeros(nf, dtype=np.float32)
    for seg in valid_nonspeech:
        keep[int(seg["start"] * 100):int(seg["end"] * 100)] = 1.0
    if keep.any():
        k = _to_samples(_smooth(keep, 5), n)
        bg = bg + voc * k[:, None] * 0.9
    del voc

    # R1: Cân bằng tầng âm lượng (gain staging) cho nhạc nền trước khi trộn
    # Hạn chế xung đỉnh quá lớn (tiếng kiếm va chạm, nổ) để tránh kích âm quá đà
    bg_peak = float(np.max(np.abs(bg))) if len(bg) > 0 else 0.0
    bg_safe_peak = 10.0 ** (-2.0 / 20.0)  # ~0.794 (-2.0 dBFS)
    if bg_peak > bg_safe_peak:
        bg = bg * (bg_safe_peak / bg_peak)

    # Ducking: đường bao giọng Việt -> giảm nền
    env = _smooth(_frames_mean(np.abs(voice)), 25)
    active = _smooth((env > 0.01).astype(np.float32), 20)
    duck = 10 ** (float(s.get("duck_db", -5.0)) / 20)
    gain = _to_samples((1.0 - (1.0 - duck) * np.clip(active, 0, 1)).astype(np.float32), n)
    mix = bg * gain[:, None] + voice[:, None]
    del bg

    # Chuẩn hóa độ to đạt chuẩn nền tảng video (-14.0 LUFS)
    target_lufs = float(s.get("target_lufs", -14.0))
    meter = pyln.Meter(SR)
    try:
        loud = meter.integrated_loudness(mix)
        if np.isfinite(loud):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                mix = pyln.normalize.loudness(mix, loud, target_lufs)
    except Exception as e:
        log(f"Bỏ qua chuẩn hóa độ to: {e}")

    # R1: Thay thế np.tanh thô sơ bằng Studio True Peak Limiter (-1.0 dBFS với soft-knee và lookahead)
    ceiling_db = float(s.get("limiter_ceiling_db", -1.02))
    mix = studio_true_peak_limiter(mix, sr=SR, ceiling_db=ceiling_db, knee_db=1.5, lookahead_ms=5.0, release_ms=40.0)

    # Tinh chỉnh độ to nếu bộ giới hạn làm giảm âm lượng dưới ngưỡng -14.5 LUFS
    try:
        final_loud = meter.integrated_loudness(mix)
        if np.isfinite(final_loud) and final_loud < target_lufs - 0.5:
            adj_db = target_lufs - final_loud
            adj_lin = 10.0 ** (min(adj_db, 2.0) / 20.0)
            mix = studio_true_peak_limiter(mix * adj_lin, sr=SR, ceiling_db=ceiling_db)
    except Exception:
        pass

    save_audio(job.p("mix.wav"), mix.astype(np.float32), SR)
    log(f"Trộn xong: {n / SR:.1f}s, giữ {len(valid_nonspeech)}/{len(raw_nonspeech)} đoạn biểu cảm (loại bỏ tiếng rồ/xè)")
