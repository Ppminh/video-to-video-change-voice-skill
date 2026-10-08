"""Thuật toán Spectral Match EQ & Phân tích Âm sắc (Timbre Matching).

Mục đích:
- Phân tích màu âm (dải tần số) của giọng nói gốc trong video (vocals.wav).
- Uốn nắn phổ tần số của giọng đọc tiếng Việt (TTS) bằng bộ lọc pha tuyến tính (FIR Filter)
  sao cho giọng lồng tiếng có cùng độ ấm (bass), độ trong (mids) và không gian (treble) với diễn viên gốc.
- Đảm bảo True Peak <= -1.0 dBFS, chống vỡ tiếng, xử lý hoàn toàn trên CPU với bộ nhớ < 50MB.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.signal
import soundfile as sf


def load_audio_mono(path: str | Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Tải file âm thanh, chuyển về mono và chuẩn hóa -1.0 đến 1.0."""
    data, sr = sf.read(str(path), dtype="float32")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if target_sr is not None and sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        if num_samples > 0:
            data = scipy.signal.resample(data, num_samples).astype(np.float32)
        else:
            data = np.zeros(0, dtype=np.float32)
        sr = target_sr
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, sr


def smooth_spectrum(gains_db: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Làm mượt dải tần số theo cửa sổ trượt để tránh các đỉnh nhọn gây méo âm."""
    if len(gains_db) < window_size:
        return gains_db
    window = np.hanning(window_size)
    window /= window.sum()
    smoothed = np.convolve(gains_db, window, mode="same")
    # Giữ nguyên 2 đầu mút
    half = window_size // 2
    smoothed[:half] = gains_db[:half]
    smoothed[-half:] = gains_db[-half:]
    return smoothed


def compute_spectral_match_filter(
    ref_audio: np.ndarray,
    tgt_audio: np.ndarray,
    sr: int,
    max_gain_db: float = 6.0,
    n_fft: int = 2048,
    num_taps: int = 513,
) -> np.ndarray:
    """
    Tính toán hệ số bộ lọc FIR (Finite Impulse Response) pha tuyến tính
    để điều chỉnh phổ tần của tgt_audio cho khớp với ref_audio.
    Hỗ trợ thích ứng kích thước FFT linh hoạt cho cả câu thoại ngắn.
    """
    eps = 1e-8
    ref_audio = np.nan_to_num(np.atleast_1d(ref_audio), nan=0.0, posinf=0.0, neginf=0.0)
    tgt_audio = np.nan_to_num(np.atleast_1d(tgt_audio), nan=0.0, posinf=0.0, neginf=0.0)
    ref_voice = ref_audio[np.abs(ref_audio) > 1e-4]
    tgt_voice = tgt_audio[np.abs(tgt_audio) > 1e-4]

    min_voice_len = min(len(ref_voice), len(tgt_voice))
    if min_voice_len < 256:
        # Nếu audio quá ngắn (<256 mẫu có tiếng), trả về bộ lọc đồng nhất
        taps = np.zeros(num_taps, dtype=np.float32)
        taps[num_taps // 2] = 1.0
        return taps

    # Thích ứng n_fft theo độ dài audio có tiếng
    n_fft_eff = min(n_fft, 1 << (min_voice_len.bit_length() - 1))
    n_fft_eff = max(256, n_fft_eff)
    num_taps_eff = min(num_taps, n_fft_eff // 2 + 1)
    if num_taps_eff % 2 == 0:
        num_taps_eff += 1

    try:
        # 1. Tính mật độ phổ công suất (Welch PSD)
        freqs, psd_ref = scipy.signal.welch(ref_voice, fs=sr, nperseg=n_fft_eff)
        _, psd_tgt = scipy.signal.welch(tgt_voice, fs=sr, nperseg=n_fft_eff)
        psd_ref = np.nan_to_num(psd_ref, nan=0.0, posinf=0.0, neginf=0.0)
        psd_tgt = np.nan_to_num(psd_tgt, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Tính tỷ lệ năng lượng phổ tần
        gain_linear = np.sqrt((psd_ref + eps) / (psd_tgt + eps))
        gain_db = 20.0 * np.log10(np.maximum(gain_linear, 1e-6))
        gain_db = np.nan_to_num(gain_db, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Trừ median gain để không thay đổi âm lượng tổng thể mà chỉ chỉnh màu âm (timbre tilt)
        gain_db = gain_db - np.median(gain_db)

        # 4. Làm mượt phổ tần để giữ âm sắc mượt mà, tự nhiên
        gain_db_smooth = smooth_spectrum(gain_db, window_size=max(5, min(21, len(gain_db) // 4 * 2 + 1)))

        # 5. Giới hạn độ khuếch đại / cắt giảm tối đa (mặc định +-6dB)
        gain_db_clamped = np.clip(gain_db_smooth, -abs(max_gain_db), abs(max_gain_db))

        # Chuyển lại sang linear gain
        gains_final = 10.0 ** (gain_db_clamped / 20.0)

        # 6. Thiết kế bộ lọc FIR pha tuyến tính bằng firwin2 với kiểm soát đơn điệu nghiêm ngặt
        freqs_norm = np.asarray(freqs, dtype=np.float64) / (sr / 2.0)
        freqs_norm = np.clip(freqs_norm, 0.0, 1.0)
        freqs_norm = np.maximum.accumulate(freqs_norm)
        for i in range(1, len(freqs_norm)):
            if freqs_norm[i] <= freqs_norm[i - 1]:
                freqs_norm[i] = freqs_norm[i - 1] + 1e-6
        if freqs_norm[-1] > 1.0 or freqs_norm[0] < 0.0:
            span = freqs_norm[-1] - freqs_norm[0]
            if span > 0:
                freqs_norm = (freqs_norm - freqs_norm[0]) / span
        freqs_norm[0] = 0.0
        freqs_norm[-1] = 1.0

        taps = scipy.signal.firwin2(num_taps_eff, freqs_norm, gains_final)
        taps = np.nan_to_num(taps, nan=0.0, posinf=0.0, neginf=0.0)
        return taps.astype(np.float32)
    except Exception:
        taps = np.zeros(num_taps, dtype=np.float32)
        taps[num_taps // 2] = 1.0
        return taps


def compute_true_peak(audio: np.ndarray) -> tuple[float, float]:
    """Tính True Peak thực tế theo chuẩn ITU-R BS.1770-4 (oversampling 4x).
    Trả về (tp_linear, tp_dbfs).
    """
    if audio is None or len(audio) == 0:
        return 0.0, -100.0
    arr = np.atleast_1d(audio).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1 if arr.shape[-1] <= 2 else 0)
    arr = arr.flatten()
    if len(arr) == 0:
        return 0.0, -100.0

    try:
        up = scipy.signal.resample_poly(arr, 4, 1, axis=0)
        up = np.nan_to_num(up, nan=0.0, posinf=0.0, neginf=0.0)
        tp_linear = float(np.max(np.abs(up)))
    except Exception:
        tp_linear = float(np.max(np.abs(arr)))
    if not math.isfinite(tp_linear) or math.isnan(tp_linear):
        tp_linear = 0.0
    tp_db = float(20.0 * np.log10(max(tp_linear, 1e-9)))
    return tp_linear, tp_db


def apply_match_eq(
    ref_audio_or_path: str | Path | np.ndarray,
    tgt_audio_or_path: str | Path | np.ndarray,
    out_path: Optional[str | Path] = None,
    sr: Optional[int] = None,
    ref_sr: Optional[int] = None,
    max_gain_db: float = 6.0,
) -> Tuple[np.ndarray, int]:
    """
    Áp dụng Match EQ: Uốn nắn giọng tgt theo phổ tần của ref.
    Tự động đồng bộ tần số lấy mẫu ref_sr với tgt_sr (sr) để tránh sai lệch phổ.
    Khống chế True Peak <= -1.0 dBFS để không bao giờ bị vỡ tiếng.
    """
    if isinstance(ref_audio_or_path, (str, Path)):
        ref_audio, actual_ref_sr = load_audio_mono(ref_audio_or_path, target_sr=sr)
    else:
        ref_audio = np.atleast_1d(ref_audio_or_path).astype(np.float32)
        ref_audio = np.nan_to_num(ref_audio, nan=0.0, posinf=0.0, neginf=0.0)
        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=-1 if ref_audio.shape[-1] <= 2 else 0)
        ref_audio = ref_audio.flatten()
        actual_ref_sr = ref_sr or sr or 22050

    if isinstance(tgt_audio_or_path, (str, Path)):
        tgt_audio, tgt_sr = load_audio_mono(tgt_audio_or_path, target_sr=sr)
    else:
        tgt_audio = np.atleast_1d(tgt_audio_or_path).astype(np.float32)
        tgt_audio = np.nan_to_num(tgt_audio, nan=0.0, posinf=0.0, neginf=0.0)
        if tgt_audio.ndim > 1:
            tgt_audio = np.mean(tgt_audio, axis=-1 if tgt_audio.shape[-1] <= 2 else 0)
        tgt_audio = tgt_audio.flatten()
        tgt_sr = sr or actual_ref_sr

    ref_audio = np.nan_to_num(ref_audio, nan=0.0, posinf=0.0, neginf=0.0)
    tgt_audio = np.nan_to_num(tgt_audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Đồng bộ tần số lấy mẫu của ref_audio theo tgt_sr
    if actual_ref_sr != tgt_sr and len(ref_audio) > 0:
        if actual_ref_sr == 16000 and tgt_sr == 24000:
            ref_audio = scipy.signal.resample_poly(ref_audio, 3, 2).astype(np.float32)
        elif actual_ref_sr == 48000 and tgt_sr == 24000:
            ref_audio = scipy.signal.resample_poly(ref_audio, 1, 2).astype(np.float32)
        else:
            num_samples = int(len(ref_audio) * tgt_sr / actual_ref_sr)
            if num_samples > 0:
                ref_audio = scipy.signal.resample(ref_audio, num_samples).astype(np.float32)
            else:
                ref_audio = np.zeros(0, dtype=np.float32)
        actual_ref_sr = tgt_sr

    ref_audio = np.nan_to_num(ref_audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Tính toán bộ lọc FIR Match EQ
    taps = compute_spectral_match_filter(ref_audio, tgt_audio, sr=tgt_sr, max_gain_db=max_gain_db)

    # Lọc tín hiệu
    if len(tgt_audio) > 0:
        filtered = scipy.signal.convolve(tgt_audio, taps, mode="same")
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        filtered = tgt_audio

    # Giữ True Peak an toàn <= -1.0 dBFS (~ 0.8910 linear) theo ITU-R BS.1770-4
    tp_linear, _ = compute_true_peak(filtered)
    max_safe_peak = 0.8910
    if tp_linear > max_safe_peak:
        filtered = filtered * (max_safe_peak / (tp_linear + 1e-8))

    d_peak = float(np.max(np.abs(filtered))) if len(filtered) > 0 else 0.0
    if d_peak > max_safe_peak:
        filtered = filtered * (max_safe_peak / (d_peak + 1e-8))

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), filtered, tgt_sr, subtype="PCM_16")

    return filtered, tgt_sr


def estimate_f0_median(audio: np.ndarray, sr: int) -> float:
    """
    Ước lượng tần số cơ bản trung bình (F0 median) bằng thuật toán tự tương quan (Autocorrelation).
    Trả về F0 (Hz), ví dụ: Nam trầm ~110-140Hz, Nam cao ~150-180Hz, Nữ ~200-260Hz.
    """
    voice = audio[np.abs(audio) > 1e-3]
    if len(voice) < sr // 10:
        return 160.0  # Mặc định trung bình

    frame_size = int(sr * 0.04)  # 40ms khung phân tích
    hop_size = int(sr * 0.02)    # 20ms bước nhảy
    min_lag = int(sr / 400.0)    # Tối đa 400Hz (nữ/trẻ em)
    max_lag = int(sr / 65.0)     # Tối thiểu 65Hz (nam rất trầm)

    f0_list = []
    for i in range(0, len(voice) - frame_size, hop_size):
        frame = voice[i:i + frame_size] * np.hanning(frame_size)
        corr = scipy.signal.correlate(frame, frame, mode="full")[frame_size - 1:]
        lag_section = corr[min_lag:max_lag]
        if len(lag_section) == 0:
            continue
        best_lag = min_lag + np.argmax(lag_section)
        # Kiểm tra tính tuần hoàn tối thiểu
        if corr[best_lag] > 0.3 * corr[0]:
            f0 = sr / float(best_lag)
            f0_list.append(f0)

    if not f0_list:
        return 160.0
    return float(np.median(f0_list))
