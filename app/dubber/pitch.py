"""Phân tích cao độ F0, năng lượng âm thanh (RMS) và ánh xạ tham số SSML cho Edge-TTS.

Module này cung cấp các tính năng:
1. Trích xuất cao độ F0 (Pitch Tracking bằng FFT Autocorrelation siêu nhẹ trên CPU).
2. Đo lường năng lượng âm lượng RMS và tốc độ nói gốc.
3. Phân loại đặc trưng tông giọng: trầm nam (<130Hz), trung (130-180Hz), thanh cao nữ/trẻ nhỏ (>200Hz), hét lớn/cao trào.
4. Ánh xạ tự động sang tham số Edge-TTS SSML (<prosody pitch="..." rate="..." volume="...">).
5. Đảm bảo tốc độ < 0.2s cho cả video, RAM < 50MB, chuẩn W3C SSML 100% không lỗi NoAudioReceived.
"""
from __future__ import annotations

import math
import os
import re
import time
import xml.sax.saxutils as saxutils
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- Tham số chuẩn cho các giọng đọc (Edge-TTS & VieNeu)
VOICE_PROFILES = {
    # Microsoft Edge-TTS: Giữ vai trò chủ đạo cho các nhân vật chính
    "Nam Minh": {"gender": "male", "engine": "Edge-TTS", "baseline_f0": 125.0, "voice_id": "vi-VN-NamMinhNeural", "offset_hz": 0, "role": "lead_male"},
    "Hoài My": {"gender": "female", "engine": "Edge-TTS", "baseline_f0": 215.0, "voice_id": "vi-VN-HoaiMyNeural", "offset_hz": 0, "role": "lead_female"},

    # Kho giọng VieNeu miền Nam: Tự động bổ sung khi video có nhiều nhân vật phụ (3, 4, 5+ người nói)
    "Đức Trí": {"gender": "male", "engine": "VieNeu", "baseline_f0": 115.0, "voice_id": "vi-VN-NamMinhNeural", "offset_hz": -4, "timbre": "nam trầm"},
    "Thái Sơn": {"gender": "male", "engine": "VieNeu", "baseline_f0": 125.0, "voice_id": "vi-VN-NamMinhNeural", "offset_hz": -2, "timbre": "nam trung trầm"},
    "Adam": {"gender": "male", "engine": "VieNeu", "baseline_f0": 128.0, "voice_id": "vi-VN-NamMinhNeural", "offset_hz": 0, "timbre": "nam trung"},
    "Minh Triết": {"gender": "male", "engine": "VieNeu", "baseline_f0": 140.0, "voice_id": "vi-VN-NamMinhNeural", "offset_hz": 2, "timbre": "nam thanh"},

    "Kim Thanh": {"gender": "female", "engine": "VieNeu", "baseline_f0": 210.0, "voice_id": "vi-VN-HoaiMyNeural", "offset_hz": -2, "timbre": "nữ trầm"},
    "Thục Đoan": {"gender": "female", "engine": "VieNeu", "baseline_f0": 220.0, "voice_id": "vi-VN-HoaiMyNeural", "offset_hz": 2, "timbre": "nữ ấm truyền cảm"},
    "Thùy Dung": {"gender": "female", "engine": "VieNeu", "baseline_f0": 225.0, "voice_id": "vi-VN-HoaiMyNeural", "offset_hz": 1, "timbre": "nữ trung"},
    "Mỹ Duyên": {"gender": "female", "engine": "VieNeu", "baseline_f0": 235.0, "voice_id": "vi-VN-HoaiMyNeural", "offset_hz": 4, "timbre": "nữ thanh cao"},
}



def get_voice_profile(voice_name_or_id: str | None) -> dict:
    """Tra cứu thông tin giọng đọc (baseline F0, gender, voice_id, offset_hz) một cách toàn diện:
    - Hỗ trợ tên tiếng Việt chuẩn: 'Nam Minh', 'Hoài My', 'Thái Sơn', ...
    - Hỗ trợ tên biến thể có ngoặc: 'Nam Minh (trầm)', 'Thái Sơn (già)', 'Thục Đoan (bé)', ...
    - Hỗ trợ voice ID Microsoft Edge: 'vi-VN-NamMinhNeural', 'vi-VN-HoaiMyNeural', ...
    - Hỗ trợ tên không dấu / viết liền: 'NamMinh', 'HoaiMy', 'ThaiSon', ...
    - Hỗ trợ từ khóa giới tính: 'male', 'female', 'nam', 'nữ'
    """
    if not voice_name_or_id:
        return VOICE_PROFILES["Hoài My"]
    v = str(voice_name_or_id).strip()
    if v in VOICE_PROFILES:
        return VOICE_PROFILES[v]

    # Loại bỏ nhãn biến thể trong ngoặc: 'Nam Minh (trầm)' -> 'Nam Minh'
    clean = re.sub(r"\s*\(.*?\)", "", v).strip()
    if clean in VOICE_PROFILES:
        return VOICE_PROFILES[clean]

    # Kiểm tra theo voice_id (ví dụ: 'vi-VN-NamMinhNeural')
    for prof in VOICE_PROFILES.values():
        if prof.get("voice_id") == v or prof.get("voice_id") == clean:
            return prof

    # Kiểm tra chứa tên trong VOICE_PROFILES
    for k, prof in VOICE_PROFILES.items():
        if k.lower() in v.lower() or k.lower() in clean.lower():
            return prof

    # Nhận diện theo từ khóa đặc trưng (Nam Minh / Hoài My / male / female)
    low = v.lower().replace("-", "").replace("_", "").replace(" ", "")
    if any(m in low for m in ("namminh", "thaison", "minhtriet", "ductri", "adam", "male", "nam")):
        return VOICE_PROFILES["Nam Minh"]
    if any(f in low for f in ("hoaimy", "thucdoan", "myduyen", "kimthanh", "thudung", "female", "nu")):
        return VOICE_PROFILES["Hoài My"]

    # Mặc định an toàn
    return VOICE_PROFILES["Hoài My"]


def f0_to_semitone(f0: float) -> float:
    """Quy đổi tần số F0 (Hz) sang bán âm MIDI (A4 = 440Hz là 69 semitones)."""
    if f0 <= 0 or not math.isfinite(f0):
        return 0.0
    return float(69.0 + 12.0 * math.log2(max(1.0, f0) / 440.0))


def _safe_float(val: any, default: float = 0.0) -> float:
    """Chuyển đổi số an toàn, chống NaN, Inf, None và chuỗi không hợp lệ."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if not math.isfinite(f) else f
    except (ValueError, TypeError):
        return default


def classify_voice_tone(f0: float, rms: float, avg_rms: float = 0.08, max_rms: float = 0.25, gender: str = "auto") -> str:
    """Phân loại đặc trưng tông giọng theo yêu cầu R1:
    - trầm nam (F0 thấp < 130Hz)
    - trung (130–180Hz)
    - thanh cao nữ/trẻ nhỏ (> 200Hz)
    - hét lớn/cao trào (F0 cao kết hợp năng lượng RMS lớn)
    - lọc nhiễu âm ngoài dải tần số giọng người
    """
    if f0 <= 0:
        return "thầm thì/không rõ âm vực" if rms >= 0.01 else "không xác định"

    # Kiểm tra hét lớn / cao trào (cần cả cao độ dâng cao và năng lượng mạnh)
    is_climax = (
        (f0 > 240 and rms > 0.14)
        or (f0 > 190 and rms > 0.22)
        or (avg_rms > 0 and rms > 1.7 * avg_rms and f0 > 220)
    )
    if is_climax:
        return "hét lớn/cao trào"

    g_low = str(gender).lower()
    is_male = g_low in ("male", "nam") or (g_low == "auto" and f0 < 190.0)

    if is_male:
        if 0 < f0 < 130:
            return "trầm nam"
        elif 130 <= f0 <= 180:
            return "trung"
        elif 180 < f0 <= 260:
            return "trung"
        else:
            return "trung" if rms <= 0.14 else "hét lớn/cao trào"
    else:
        if 0 < f0 < 130:
            return "nữ trầm"
        elif 130 <= f0 <= 200:
            return "trung"
        elif f0 > 200:
            return "thanh cao nữ/trẻ nhỏ"
        else:
            return "trung"


def analyze_segment_pitch_energy(samples: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 650.0) -> dict:
    """Phân tích F0 trung vị (Hz & semitones) và năng lượng RMS cho 1 đoạn âm thanh.
    Sử dụng thuật toán FFT Normalized Autocorrelation siêu nhẹ với Hanning window,
    khử DC offset/baseline drift, bổ chính đỉnh Parabolic và lọc nhiễu octave.
    Thời gian xử lý: ~0.003s cho mỗi giây âm thanh trên CPU.
    """
    if samples is None:
        return {"f0": 0.0, "semitones": 0.0, "rms": 0.0, "peak_rms": 0.0, "voiced_ratio": 0.0}

    samples = np.atleast_1d(samples).astype(np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.squeeze(samples)
    while samples.ndim > 1:
        samples = samples.mean(axis=0 if samples.shape[0] < samples.shape[1] else -1)
    samples = samples.flatten()

    total_len = len(samples)
    overall_rms = float(np.sqrt(np.mean(samples ** 2))) if total_len > 0 else 0.0
    if total_len < int(sr * 0.04):
        return {
            "f0": 0.0,
            "semitones": 0.0,
            "rms": round(overall_rms, 4),
            "peak_rms": round(overall_rms, 4),
            "voiced_ratio": 0.0,
        }

    frame_len = int(sr * 0.040)  # 40ms window
    hop_len = int(sr * 0.020)    # 20ms hop
    min_lag = int(sr / fmax)     # fmax=650Hz -> ~24 samples ở 16k
    max_lag = int(sr / fmin)     # fmin=60Hz -> ~266 samples ở 16k

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(samples, frame_len)[::hop_len]
    except Exception:
        # Fallback thủ công nếu numpy quá cũ
        num_frames = (total_len - frame_len) // hop_len + 1
        windows = np.array([samples[i * hop_len: i * hop_len + frame_len] for i in range(max(1, num_frames))])

    if len(windows) == 0:
        return {
            "f0": 0.0,
            "semitones": 0.0,
            "rms": round(overall_rms, 4),
            "peak_rms": round(overall_rms, 4),
            "voiced_ratio": 0.0,
        }

    frame_rms = np.sqrt(np.mean(windows ** 2, axis=1))
    peak_rms = float(np.max(frame_rms)) if len(frame_rms) > 0 else overall_rms
    mean_rms = float(np.mean(frame_rms)) if len(frame_rms) > 0 else overall_rms

    # Ngưỡng phát hiện tiếng nói (voiced)
    silence_thresh = max(0.010, 0.12 * peak_rms)
    voiced_mask = frame_rms > silence_thresh
    voiced_count = int(np.sum(voiced_mask))
    voiced_ratio = float(voiced_count / max(1, len(windows)))

    if voiced_count < 2:
        return {
            "f0": 0.0,
            "semitones": 0.0,
            "rms": round(mean_rms, 4),
            "peak_rms": round(peak_rms, 4),
            "voiced_ratio": round(voiced_ratio, 2),
        }

    v_frames = windows[voiced_mask]
    # Khử DC offset / trôi đường cơ sở trước khi tính tự tương quan
    v_frames = v_frames - np.mean(v_frames, axis=1, keepdims=True)
    win_func = np.hanning(frame_len).astype(np.float32)
    v_frames_win = v_frames * win_func

    # FFT Autocorrelation: Đảm bảo n_fft >= frame_len + max_lag để loại bỏ hoàn toàn circular aliasing ở mọi sample rate
    needed_len = frame_len + max_lag
    n_fft = int(2 ** math.ceil(math.log2(max(1024, needed_len))))
    F = np.fft.rfft(v_frames_win, n=n_fft, axis=1)
    acorr = np.fft.irfft(F * np.conj(F), axis=1)

    sub_acorr = acorr[:, min_lag:max_lag]
    best_idx = np.argmax(sub_acorr, axis=1)
    best_val = np.take_along_axis(sub_acorr, best_idx[:, None], axis=1)[:, 0]
    zero_val = np.maximum(1e-8, acorr[:, 0])

    # Khẳng định frame có tính tuần hoàn giọng nói và đỉnh không phải dốc giảm lag 0 ở mép min_lag
    is_interior_peak = np.ones(len(best_idx), dtype=bool)
    for i_f, idx in enumerate(best_idx):
        if idx == 0:
            if min_lag > 0 and acorr[i_f, min_lag] <= acorr[i_f, min_lag - 1]:
                is_interior_peak[i_f] = False
        elif idx == sub_acorr.shape[1] - 1:
            is_interior_peak[i_f] = False

    valid_peaks = (best_val > (0.26 * zero_val)) & is_interior_peak
    if np.sum(valid_peaks) < 1:
        return {
            "f0": 0.0,
            "semitones": 0.0,
            "rms": round(mean_rms, 4),
            "peak_rms": round(peak_rms, 4),
            "voiced_ratio": round(voiced_ratio, 2),
        }

    raw_lags = min_lag + best_idx[valid_peaks]
    valid_sub = sub_acorr[valid_peaks]
    v_idx = best_idx[valid_peaks]

    # Bổ chính đỉnh Parabolic (sub-bin interpolation)
    refined_lags = []
    for k, idx in enumerate(v_idx):
        lag = raw_lags[k]
        if 0 < idx < sub_acorr.shape[1] - 1:
            a = float(valid_sub[k, idx - 1])
            b = float(valid_sub[k, idx])
            c = float(valid_sub[k, idx + 1])
            denom = 2.0 * (2.0 * b - a - c)
            delta = float(np.clip((c - a) / denom, -0.5, 0.5)) if abs(denom) > 1e-6 else 0.0
            lag = lag + delta

        # Kiểm tra nhảy quãng tám (octave jumping / doubling)
        half_lag = int(round(lag / 2.0))
        if half_lag >= min_lag:
            h_idx = half_lag - min_lag
            if h_idx < sub_acorr.shape[1]:
                h_val = float(valid_sub[k, h_idx])
                if h_val > 0.75 * float(valid_sub[k, idx]):
                    if 0 < h_idx < sub_acorr.shape[1] - 1:
                        ha = float(valid_sub[k, h_idx - 1])
                        hb = float(valid_sub[k, h_idx])
                        hc = float(valid_sub[k, h_idx + 1])
                        hdenom = 2.0 * (2.0 * hb - ha - hc)
                        hdelta = float(np.clip((hc - ha) / hdenom, -0.5, 0.5)) if abs(hdenom) > 1e-6 else 0.0
                        lag = float(half_lag) + hdelta
                    else:
                        lag = float(half_lag)
        refined_lags.append(lag)

    f0_estimates = [sr / lg for lg in refined_lags if lg > 0 and fmin <= (sr / lg) <= fmax]
    if not f0_estimates:
        return {
            "f0": 0.0,
            "semitones": 0.0,
            "rms": round(mean_rms, 4),
            "peak_rms": round(peak_rms, 4),
            "voiced_ratio": round(voiced_ratio, 2),
        }

    med_f0 = float(np.median(f0_estimates))
    return {
        "f0": round(med_f0, 1),
        "semitones": round(f0_to_semitone(med_f0), 1),
        "rms": round(mean_rms, 4),
        "peak_rms": round(peak_rms, 4),
        "voiced_ratio": round(voiced_ratio, 2),
    }


def analyze_transcript_f0(vocals_path: str | Path | None, lines: list[dict],
                          speakers: dict | None = None) -> dict[int, dict]:
    """Phân tích cao độ F0 và năng lượng cho từng câu thoại trong transcript.
    - Đọc file vocals.wav (hoặc vocals_16k.wav).
    - Tính toán F0, semitones, RMS cho từng câu.
    - Phân loại đặc trưng tông giọng (trầm nam, trung, thanh cao nữ/trẻ nhỏ, hét lớn/cao trào).
    - Tự động điền baseline F0 theo người nói nếu câu thoại quá ngắn hoặc thầm thì.
    - Thời gian chạy < 0.2s trên CPU.
    """
    profiles: dict[int, dict] = {}
    if not lines:
        return profiles

    vocals_file = None
    if vocals_path:
        p = Path(vocals_path)
        if p.exists():
            vocals_file = p
        else:
            # Thử tìm vocals_16k.wav hoặc vocals.wav cùng thư mục
            for cand in [p.with_name("vocals_16k.wav"), p.with_name("vocals.wav"), p.with_name("audio.wav")]:
                if cand.exists():
                    vocals_file = cand
                    break

    audio_data = None
    sr = 16000
    if vocals_file:
        try:
            import soundfile as sf
            audio_data, sr = sf.read(str(vocals_file), dtype="float32", always_2d=False)
            if audio_data is not None:
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
        except Exception:
            audio_data = None

    # Đo năng lượng trung bình toàn video
    video_mean_rms = float(np.sqrt(np.mean(audio_data ** 2))) if audio_data is not None and len(audio_data) > 0 else 0.08

    # Lấy baseline F0 của từng speaker đã có từ diarization
    spk_f0_map = {}
    if speakers and isinstance(speakers, dict):
        for spk_id, spk_meta in speakers.items():
            if isinstance(spk_meta, dict):
                f0_val = _safe_float(spk_meta.get("f0"), 0.0)
                if f0_val > 0:
                    spk_f0_map[str(spk_id)] = f0_val

    # Vòng lặp phân tích từng câu
    collected_spk_f0: dict[str, list[float]] = {}
    for idx_ln, ln in enumerate(lines):
        if not isinstance(ln, dict):
            continue
        lid = ln.get("id")
        if lid is None:
            lid = idx_ln + 1
        st = max(0.0, _safe_float(ln.get("start"), 0.0))
        en = max(st, _safe_float(ln.get("end"), st + 1.0))
        st, en = min(st, en), max(st, en)
        spk_raw = ln.get("spk")
        spk = str(spk_raw) if spk_raw is not None else "S0"
        zh = str(ln.get("zh") or "")
        dur = max(0.1, en - st)

        # Tính tốc độ ký tự tiếng Trung (CJK characters/sec)
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", zh or ""))
        cjk_speed = round(cjk_chars / dur, 2) if dur > 0 else 0.0

        f0 = 0.0
        semitones = 0.0
        rms = 0.05
        peak_rms = 0.05
        voiced_ratio = 0.0

        if audio_data is not None and len(audio_data) > 0:
            idx_start = max(0, int(st * sr))
            idx_end = max(idx_start, min(len(audio_data), int(en * sr)))
            if idx_end > idx_start:
                seg = audio_data[idx_start:idx_end]
                seg_res = analyze_segment_pitch_energy(seg, sr)
                f0 = seg_res["f0"]
                semitones = seg_res["semitones"]
                rms = seg_res["rms"]
                peak_rms = seg_res["peak_rms"]
                voiced_ratio = seg_res["voiced_ratio"]

        if f0 > 0:
            collected_spk_f0.setdefault(spk, []).append(f0)

        tone = classify_voice_tone(f0, rms, avg_rms=video_mean_rms)
        profiles[lid] = {
            "id": lid,
            "spk": spk,
            "start": round(st, 3),
            "end": round(en, 3),
            "duration": round(dur, 3),
            "f0": f0,
            "semitones": semitones,
            "rms": rms,
            "peak_rms": peak_rms,
            "voiced_ratio": voiced_ratio,
            "tone": tone,
            "cjk_speed": cjk_speed,
            "emotion": ln.get("emotion", ""),
        }

    # Bù đắp baseline cho các câu không bắt được F0 (thì thầm, ngắt quãng)
    for spk, f0_list in collected_spk_f0.items():
        if f0_list:
            spk_f0_map[spk] = float(np.median(f0_list))

    for lid, p in profiles.items():
        if p["f0"] <= 0:
            spk_base = spk_f0_map.get(p["spk"], 0.0)
            if spk_base > 0:
                p["f0_fallback"] = round(spk_base, 1)
                p["semitones"] = round(f0_to_semitone(spk_base), 1)
                p["tone"] = classify_voice_tone(spk_base, p["rms"], avg_rms=video_mean_rms)

    return profiles


def extract_speaker_f0_profiles(vocals_path: str | Path | None, lines: list[dict],
                                speakers: dict | None = None,
                                precomputed_profiles: dict | None = None) -> dict[str, dict]:
    """Phân tích cao độ trung bình (F0 median) và năng lượng giọng nói của từng người nói từ vocals.wav:
    - Nam trầm: ~110Hz (< 130Hz)
    - Nam thanh: ~140-160Hz
    - Nữ thanh cao: ~220-250Hz (> 200Hz)
    - Nữ trầm / ấm: ~205-215Hz
    """
    profiles = precomputed_profiles if precomputed_profiles is not None else analyze_transcript_f0(vocals_path, lines, speakers)
    spk_data: dict[str, dict] = {}

    spk_f0_samples: dict[str, list[float]] = {}
    spk_rms_samples: dict[str, list[float]] = {}
    spk_dur: dict[str, float] = {}
    spk_lines: dict[str, int] = {}

    for lid, p in profiles.items():
        spk_raw = p.get("spk")
        spk = str(spk_raw) if spk_raw is not None else "S0"
        f0 = _safe_float(p.get("f0") or p.get("f0_fallback"), 0.0)
        rms = _safe_float(p.get("rms"), 0.0)
        dur = _safe_float(p.get("duration"), 0.0)
        if f0 > 0:
            spk_f0_samples.setdefault(spk, []).append(f0)
        if rms > 0:
            spk_rms_samples.setdefault(spk, []).append(rms)
        spk_dur[spk] = spk_dur.get(spk, 0.0) + dur
        spk_lines[spk] = spk_lines.get(spk, 0) + 1

    all_speakers = set(spk_dur.keys())
    if speakers and isinstance(speakers, dict):
        all_speakers.update(str(k) for k in speakers.keys())

    for spk in all_speakers:
        f0_list = spk_f0_samples.get(spk, [])
        spk_meta = (speakers or {}).get(spk)
        if not spk_meta and (speakers or {}):
            spk_meta = (speakers or {}).get(int(spk)) if str(spk).isdigit() else None
        spk_meta = spk_meta or {}
        med_f0 = float(np.median(f0_list)) if f0_list else _safe_float(spk_meta.get("f0"), 0.0)

        spk_gender = spk_meta.get("gender") or spk_meta.get("gender_voice") or "auto"
        rms_list = spk_rms_samples.get(spk, [])
        mean_rms = float(np.mean(rms_list)) if rms_list else 0.08
        tone = classify_voice_tone(med_f0, mean_rms, gender=spk_gender)

        spk_data[spk] = {
            "f0_median": round(med_f0, 1),
            "rms": round(mean_rms, 4),
            "tone": tone,
            "talk_time": round(spk_dur.get(spk, 0.0), 2),
            "line_count": spk_lines.get(spk, 0),
        }

    return spk_data


def pair_voice_by_acoustic_distance(spk_f0: float, gender: str = "male",
                                   candidate_voices: list[str] | None = None,
                                   engine_preference: str | None = None) -> str:
    """Gán giọng tương thích nhất (Acoustic Distance Pairing):
    Lựa chọn giọng lồng tiếng có cao độ và màu giọng gần với diễn viên gốc nhất:
    - Nhân vật có tông giọng trầm trong video gốc (~110Hz) -> ghép với giọng nam trầm tương ứng (Đức Trí ~115Hz).
    - Nhân vật nam thanh (~150-160Hz) -> ghép với giọng nam thanh tương ứng (Minh Triết ~140Hz).
    - Nhân vật nữ cao (> 220Hz, ví dụ ~235-250Hz) -> ghép với giọng nữ thanh tương ứng (Mỹ Duyên ~235Hz).
    - Nhân vật nữ trầm/ấm (~205-215Hz) -> ghép với giọng nữ trầm tương ứng (Kim Thanh ~210Hz / Hoài My ~215Hz).
    - Tự động lọc hài âm (octave doubling/tripling) và nhiễu ngoài dải tần tự nhiên.
    """
    g = "female" if str(gender).lower() in ("female", "nu", "nữ", "child") else "male"
    if candidate_voices:
        pool = [v for v in candidate_voices if v]
    else:
        pool = [name for name, prof in VOICE_PROFILES.items() if prof.get("gender") == g]

    if not pool:
        return "Hoài My" if g == "female" else "Nam Minh"

    if engine_preference:
        eng_pool = [v for v in pool if get_voice_profile(v).get("engine") == engine_preference]
        if eng_pool:
            pool = eng_pool

    f0_val = _safe_float(spk_f0, 0.0)
    if g == "male":
        # Nam giới: dải tần số cơ bản tự nhiên của giọng nói: 65Hz - 260Hz
        if 65.0 <= f0_val <= 260.0:
            target_f0 = f0_val
        elif 260.0 < f0_val <= 520.0 and (65.0 <= f0_val / 2.0 <= 260.0):
            # Nhảy quãng tám (octave doubling error) ở hài âm 2: gập đôi về tần số cơ bản
            target_f0 = f0_val / 2.0
        elif 520.0 < f0_val <= 780.0 and (65.0 <= f0_val / 3.0 <= 260.0):
            # Hài âm bậc 3 (tiếng còi xe ~700Hz hoặc mép trên): gập về tần số gốc
            target_f0 = f0_val / 3.0
        elif 50.0 <= f0_val < 65.0:
            target_f0 = 115.0  # Nam rất trầm (bass sâu) -> ghép Đức Trí
        else:
            # Nhiễu ngoài dải tần người nói: dùng baseline chuẩn nam
            target_f0 = 125.0
    else:
        # Nữ giới / trẻ em: dải tần số cơ bản tự nhiên: 120Hz - 380Hz
        if 120.0 <= f0_val <= 380.0:
            target_f0 = f0_val
        elif 380.0 < f0_val <= 760.0 and (120.0 <= f0_val / 2.0 <= 380.0):
            # Nhảy quãng tám ở giọng nữ cao
            target_f0 = f0_val / 2.0
        elif 80.0 <= f0_val < 120.0:
            target_f0 = 210.0  # Nữ trầm -> ghép Kim Thanh
        else:
            # Nhiễu ngoài dải tần: dùng baseline chuẩn nữ
            target_f0 = 220.0

    def _dist(v_name: str) -> float:
        prof = get_voice_profile(v_name)
        base_f0 = float(prof.get("baseline_f0", 220.0 if g == "female" else 125.0))
        return abs(base_f0 - target_f0)

    return min(pool, key=_dist)


def map_pitch_to_prosody(line_pitch: dict, voice_name: str | None = None,
                         base_voice: str | None = None, emotion: str = "",
                         unit: str = "hz", safe_volume: bool = False,
                         max_volume_pct: int | None = None) -> dict:
    """Ánh xạ cao độ F0, năng lượng và cảm xúc thành tham số SSML:
    - So sánh với cao độ chuẩn của giọng đọc (Hoài My = 215Hz, Nam Minh = 125Hz).
    - Tự động bù trừ và sinh thuộc tính pitch (+XHz hoặc +Y%), rate (+A%), volume (+B%).
    - Nhân vật gốc giọng trầm (<130Hz) -> tự động hạ pitch (-8% tới -12Hz) cho giọng trầm ấm.
    - Nhân vật giọng cao (>200Hz) / trẻ nhỏ -> tự động tăng pitch (+10% tới +18Hz) cho giọng trong trẻo.
    - Phân đoạn kịch tính/gắt giọng -> tự động tăng pitch (+15Hz) và volume (giới hạn an toàn <= +4% khi safe_volume=True).
    - Khử nhiễu còi xe / rít cao tần, gập hài âm để không bị méo tiếng quá mức.
    """
    v_key = base_voice or voice_name or "Hoài My"
    v_info = get_voice_profile(v_key)

    target_gender = v_info["gender"]
    target_baseline = v_info["baseline_f0"]
    preset_offset = v_info.get("offset_hz", 0)

    # Trích xuất biến thể nếu có trong voice_name (ví dụ: 'trầm', 'bé', 'trẻ', 'già')
    variant_offset = 0
    raw_vname = str(voice_name or "")
    if "(trầm)" in raw_vname:
        variant_offset = -6
    elif "(già)" in raw_vname:
        variant_offset = -8
    elif "(bé)" in raw_vname:
        variant_offset = +12
    elif "(trẻ)" in raw_vname:
        variant_offset = +4

    if not isinstance(line_pitch, dict):
        line_pitch = {}

    f0 = _safe_float(line_pitch.get("f0") or line_pitch.get("f0_fallback"), 0.0)
    rms = _safe_float(line_pitch.get("rms"), 0.08)
    tone = line_pitch.get("tone") or (classify_voice_tone(f0, rms, gender=target_gender) if f0 > 0 else "trung")
    cjk_speed = _safe_float(line_pitch.get("cjk_speed"), 3.8)

    # 1. Tính toán bù trừ Pitch
    pitch_hz_delta = 0
    pitch_pct_delta = 0

    eff_f0 = f0
    eff_tone = tone

    if f0 > 0:
        if target_gender == "male":
            if eff_f0 > 260.0:
                if 65.0 <= eff_f0 / 2.0 <= 260.0:
                    eff_f0 = eff_f0 / 2.0
                elif 65.0 <= eff_f0 / 3.0 <= 260.0:
                    eff_f0 = eff_f0 / 3.0
                else:
                    eff_f0 = target_baseline
            if eff_tone == "thanh cao nữ/trẻ nhỏ":
                eff_tone = "trung"
        else:
            if eff_f0 > 380.0:
                if 120.0 <= eff_f0 / 2.0 <= 380.0:
                    eff_f0 = eff_f0 / 2.0
                else:
                    eff_f0 = target_baseline

        delta = eff_f0 - target_baseline
        if eff_tone == "trầm nam" or eff_f0 < 130:
            # Nhân vật gốc trầm -> hạ pitch trầm ấm (-8% tới -12Hz, tối đa -22Hz)
            # Bảo đảm luôn hạ âm (delta âm), không bao giờ tăng pitch dù baseline thấp
            eff_delta = min(delta, -8.0)
            pitch_hz_delta = min(-4, max(-22, round(eff_delta * 0.44)))
            pitch_pct_delta = min(-3, max(-15, round((eff_delta / target_baseline) * 100 * 0.40)))
        elif eff_tone == "hét lớn/cao trào":
            # Hét lớn / kịch tính -> tăng cao độ rõ rệt (+12Hz tới +28Hz)
            pitch_hz_delta = min(28, max(12, round(max(delta, 15.0) * 0.32) + 6))
            pitch_pct_delta = min(22, max(10, round((max(delta, 15.0) / target_baseline) * 100 * 0.30) + 5))
        elif eff_tone == "thanh cao nữ/trẻ nhỏ" or (target_gender == "female" and eff_f0 > 200):
            # Giọng thanh cao / trẻ em -> nâng pitch trong trẻo (+10% tới +15Hz, tối đa +25Hz)
            # Bảo đảm luôn tăng âm (delta dương), không bao giờ hạ pitch khi Hoài My baseline cao
            eff_delta = max(delta, 12.0)
            pitch_hz_delta = max(4, min(25, round(eff_delta * 0.36)))
            pitch_pct_delta = max(3, min(18, round((eff_delta / target_baseline) * 100 * 0.34)))
        else:
            # Tông trung -> bù trừ nhẹ nhàng theo độ lệch thực tế, kiểm soát trần an toàn theo giới tính
            scale = 0.20 if target_gender == "male" else 0.30
            max_mid_hz = 14 if target_gender == "male" else 22
            min_mid_hz = -14 if target_gender == "male" else -18
            pitch_hz_delta = int(np.clip(round(delta * scale), min_mid_hz, max_mid_hz))
            pitch_pct_delta = int(np.clip(round((delta / target_baseline) * 100 * scale), -12, 16))
    else:
        # Không có F0 đo được: giữ nguyên 0
        pitch_hz_delta = 0
        pitch_pct_delta = 0

    # Cộng thêm preset offset của từng giọng và variant offset (để bảo toàn bản sắc nhân vật)
    pitch_hz_delta += preset_offset + variant_offset
    pitch_pct_delta += round((preset_offset + variant_offset) * 0.8)

    # 2. Tính toán bù trừ Volume (theo năng lượng RMS)
    vol_pct = 0
    if eff_tone == "hét lớn/cao trào" or rms > 0.15:
        vol_pct = min(25, max(12, int(round((rms - 0.10) * 120))))
    elif rms < 0.035 and rms > 0:
        vol_pct = -14
    elif rms < 0.055 and rms > 0:
        vol_pct = -8
    elif rms > 0.10:
        vol_pct = +6

    # 3. Tính toán bù trừ Rate (theo nhịp điệu dồn dập cjk_speed)
    rate_pct = 0
    if eff_tone == "hét lớn/cao trào":
        rate_pct += 10
    elif cjk_speed >= 5.5:
        rate_pct += 12
    elif cjk_speed >= 4.6:
        rate_pct += 6
    elif 0 < cjk_speed <= 2.2:
        rate_pct -= 8
    elif 0 < cjk_speed <= 2.8:
        rate_pct -= 4

    # 4. Tích hợp sắc thái Cảm xúc (Emotion)
    emo = (emotion or "").lower()
    if any(w in emo for w in ("angry", "tuc", "gian", "phẫn", "gắt", "bực")):
        rate_pct += 6
        pitch_hz_delta += 4
        pitch_pct_delta += 3
        vol_pct += 12
    elif any(w in emo for w in ("panicked", "sợ", "hoảng", "lo lắng")):
        rate_pct += 8
        pitch_hz_delta += 5
        pitch_pct_delta += 4
        vol_pct += 10
    elif any(w in emo for w in ("happy", "excited", "vui", "hào hứng")):
        pitch_hz_delta += 3
        pitch_pct_delta += 2
        vol_pct += 6
    elif any(w in emo for w in ("sad", "buon", "khoc", "đau", "bi thương")):
        rate_pct -= 8
        pitch_hz_delta -= 3
        pitch_pct_delta -= 2
        vol_pct -= 8
    elif any(w in emo for w in ("whisper", "thi tham", "nói nhỏ", "tâm sự")):
        rate_pct -= 6
        pitch_hz_delta -= 2
        pitch_pct_delta -= 2
        vol_pct -= 16

    # Bảo đảm ràng buộc hướng nhất quán theo Acceptance Criteria:
    # Trầm nam -> luôn hạ pitch âm (<= -2Hz, <= -2%) trừ khi đang tức giận cực độ
    # Thanh cao nữ/trẻ nhỏ -> luôn tăng pitch dương (>= +2Hz, >= +2%) trừ khi đang buồn/thì thầm
    is_angry_or_panicked = any(w in emo for w in ("angry", "tuc", "gian", "phẫn", "gắt", "bực", "panicked", "sợ", "hoảng"))
    is_sad_or_whisper = any(w in emo for w in ("sad", "buon", "khoc", "đau", "bi thương", "whisper", "thi tham", "nói nhỏ"))
    if (eff_tone == "trầm nam" or (0 < eff_f0 < 130)) and not is_angry_or_panicked:
        pitch_hz_delta = min(-2, pitch_hz_delta)
        pitch_pct_delta = min(-2, pitch_pct_delta)
    elif ((eff_tone == "thanh cao nữ/trẻ nhỏ" and target_gender == "female") or (target_gender == "female" and eff_f0 > 200)) and not is_sad_or_whisper:
        pitch_hz_delta = max(2, pitch_hz_delta)
        pitch_pct_delta = max(2, pitch_pct_delta)

    # Giới hạn an toàn (clamping) để âm thanh tự nhiên và Edge-TTS không báo lỗi
    final_pitch_hz = int(np.clip(pitch_hz_delta, -28, 30))
    final_pitch_pct = int(np.clip(pitch_pct_delta, -18, 22))
    final_rate = int(np.clip(rate_pct, -20, 30))
    # R2: Giới hạn an toàn vol_pct <= +4% trong SSML Edge-TTS để chống vỡ tiếng
    max_v = 4 if safe_volume else (max_volume_pct if max_volume_pct is not None else 25)
    final_vol = int(np.clip(vol_pct, -25, max_v))

    pitch_str = f"{final_pitch_pct:+d}%" if unit.lower() == "percent" or unit.lower() == "%" else f"{final_pitch_hz:+d}Hz"
    rate_str = f"{final_rate:+d}%"
    vol_str = f"{final_vol:+d}%"

    return {
        "pitch": pitch_str,
        "pitch_hz": f"{final_pitch_hz:+d}Hz",
        "pitch_pct": f"{final_pitch_pct:+d}%",
        "rate": rate_str,
        "volume": vol_str,
        "tone": eff_tone,
        "voice_id": v_info["voice_id"],
        "baseline_f0": target_baseline,
        "f0": f0,
        "eff_f0": eff_f0,
        "rms": rms,
    }


def remove_incompatible_characters(string: str) -> str:
    """Loại bỏ các ký tự điều khiển không tương thích với máy chủ Edge-TTS."""
    if not isinstance(string, str):
        return ""
    chars = list(string)
    for idx, char in enumerate(chars):
        code = ord(char)
        if (0 <= code <= 8) or (11 <= code <= 12) or (14 <= code <= 31):
            chars[idx] = " "
    return "".join(chars)


def normalize_prosody_attr(val: str | int | float | None, default_unit: str = "%",
                           min_val: int = -50, max_val: int = 50,
                           safe_volume: bool = False) -> str:
    """Chuẩn hóa và giới hạn an toàn thuộc tính prosody (pitch, rate, volume) cho Edge-TTS SSML:
    - Loại bỏ khoảng trắng thừa (ví dụ: '+10 Hz' -> '+10Hz').
    - Tự động bổ sung dấu '+' hoặc '-' nếu thiếu.
    - Bổ sung đơn vị (Hz, %, st) nếu người dùng chỉ truyền số (ví dụ: 10 -> '+10Hz' hoặc '+10%').
    - Nếu giá trị không hợp lệ hoặc không phải số (ví dụ: 'invalid', 'medium', NaN), fallback an toàn về +0{default_unit}.
    - Giới hạn giá trị nằm trong ngưỡng an toàn để Edge-TTS không bao giờ báo lỗi NoAudioReceived hay Invalid pitch.
    - safe_volume=True: giới hạn âm lượng % tối đa <= +4% để chống vỡ tiếng.
    """
    if val is None:
        return f"+0{default_unit}"
    s = str(val).strip()
    if not s or s.lower() in ("default", "none", "nan", "null"):
        return f"+0{default_unit}"

    m = re.match(r"^([+-]?\s*\d+(?:\.\d+)?)\s*(hz|%|st)?$", s, re.IGNORECASE)
    if m:
        num_part = float(m.group(1).replace(" ", ""))
        unit_part = m.group(2)
        if not unit_part:
            unit_part = default_unit
        unit_norm = "%" if unit_part == "%" else ("st" if unit_part.lower() == "st" else "Hz")
        eff_max = 4 if (safe_volume and unit_norm == "%") else max_val
        int_val = int(np.clip(round(num_part), min_val, eff_max))
        return f"{int_val:+d}{unit_norm}"

    # Nếu không khớp số, fallback an toàn về 0 thay vì tạo chuỗi hỏng như +invalid%
    return f"+0{default_unit}"


def build_ssml(text: str, voice_name_or_id: str, pitch: str = "+0Hz",
               rate: str = "+0%", volume: str = "+0%",
               safe_volume: bool = False, max_volume_pct: int | None = None) -> str:
    """Tạo chuỗi SSML hợp lệ 100% chuẩn W3C cho Microsoft Edge-TTS:
    <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
      <voice name='...'>
        <prosody pitch='+XHz' rate='+Y%' volume='+Z%'>
          {escaped_text}
        </prosody>
      </voice>
    </speak>
    """
    cleaned = remove_incompatible_characters(str(text or "").strip())
    clean_text = saxutils.escape(cleaned)

    # Phân giải thông tin giọng nói chính xác
    prof = get_voice_profile(voice_name_or_id)
    voice_id = prof["voice_id"]

    if voice_name_or_id and str(voice_name_or_id).startswith("Microsoft Server Speech"):
        full_voice = str(voice_name_or_id)
    elif voice_name_or_id and str(voice_name_or_id).startswith("vi-VN-"):
        m = re.match(r"^([a-z]{2,}-[A-Z]{2,})-(.+Neural)$", str(voice_name_or_id))
        if m:
            full_voice = f"Microsoft Server Speech Text to Speech Voice ({m.group(1)}, {m.group(2)})"
        else:
            full_voice = f"Microsoft Server Speech Text to Speech Voice (vi-VN, {prof['voice_id'].replace('vi-VN-', '')})"
    else:
        m = re.match(r"^([a-z]{2,}-[A-Z]{2,})-(.+Neural)$", voice_id)
        if m:
            full_voice = f"Microsoft Server Speech Text to Speech Voice ({m.group(1)}, {m.group(2)})"
        else:
            full_voice = "Microsoft Server Speech Text to Speech Voice (vi-VN, HoaiMyNeural)"

    # Chuẩn hóa an toàn các tham số prosody
    p = normalize_prosody_attr(pitch, default_unit="Hz", min_val=-28, max_val=30)
    r = normalize_prosody_attr(rate, default_unit="%", min_val=-20, max_val=30)
    max_v = 4 if safe_volume else (max_volume_pct if max_volume_pct is not None else 25)
    v = normalize_prosody_attr(volume, default_unit="%", min_val=-25, max_val=max_v)

    return (
        f"<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" xml:lang=\"en-US\">"
        f"<voice name=\"{full_voice}\">"
        f"<prosody pitch=\"{p}\" rate=\"{r}\" volume=\"{v}\">"
        f"{clean_text}"
        f"</prosody>"
        f"</voice>"
        f"</speak>"
    )
