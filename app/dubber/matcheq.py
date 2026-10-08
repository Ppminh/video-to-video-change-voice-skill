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
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if target_sr is not None and sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples).astype(np.float32)
        sr = target_sr
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
    """
    # Loại bỏ khoảng lặng tuyệt đối
    eps = 1e-8
    ref_voice = ref_audio[np.abs(ref_audio) > 1e-4]
    tgt_voice = tgt_audio[np.abs(tgt_audio) > 1e-4]

    if len(ref_voice) < n_fft or len(tgt_voice) < n_fft:
        # Nếu audio quá ngắn, trả về bộ lọc đồng nhất (identity filter)
        taps = np.zeros(num_taps, dtype=np.float32)
        taps[num_taps // 2] = 1.0
        return taps

    # 1. Tính mật độ phổ công suất (Welch PSD)
    freqs, psd_ref = scipy.signal.welch(ref_voice, fs=sr, nperseg=n_fft)
    _, psd_tgt = scipy.signal.welch(tgt_voice, fs=sr, nperseg=n_fft)

    # 2. Tính tỷ lệ năng lượng phổ tần
    gain_linear = np.sqrt((psd_ref + eps) / (psd_tgt + eps))
    gain_db = 20.0 * np.log10(gain_linear)

    # 3. Trừ median gain để không thay đổi âm lượng tổng thể mà chỉ chỉnh màu âm (timbre tilt)
    gain_db = gain_db - np.median(gain_db)

    # 4. Làm mượt phổ tần để giữ âm sắc mượt mà, tự nhiên
    gain_db_smooth = smooth_spectrum(gain_db, window_size=21)

    # 5. Giới hạn độ khuếch đại / cắt giảm tối đa (mặc định +-6dB)
    gain_db_clamped = np.clip(gain_db_smooth, -abs(max_gain_db), abs(max_gain_db))

    # Chuyển lại sang linear gain
    gains_final = 10.0 ** (gain_db_clamped / 20.0)

    # 6. Thiết kế bộ lọc FIR pha tuyến tính bằng firwin2
    freqs_norm = freqs / (sr / 2.0)
    freqs_norm[0] = 0.0
    freqs_norm[-1] = 1.0

    # Đảm bảo tần số tăng nghiêm ngặt
    for i in range(1, len(freqs_norm)):
        if freqs_norm[i] <= freqs_norm[i - 1]:
            freqs_norm[i] = freqs_norm[i - 1] + 1e-5

    taps = scipy.signal.firwin2(num_taps, freqs_norm, gains_final)
    return taps.astype(np.float32)


def apply_match_eq(
    ref_audio_or_path: str | Path | np.ndarray,
    tgt_audio_or_path: str | Path | np.ndarray,
    out_path: Optional[str | Path] = None,
    sr: Optional[int] = None,
    max_gain_db: float = 6.0,
) -> Tuple[np.ndarray, int]:
    """
    Áp dụng Match EQ: Uốn nắn giọng tgt theo phổ tần của ref.
    Khống chế True Peak <= -1.0 dBFS để không bị vỡ tiếng.
    """
    if isinstance(ref_audio_or_path, (str, Path)):
        ref_audio, ref_sr = load_audio_mono(ref_audio_or_path, target_sr=sr)
    else:
        ref_audio = ref_audio_or_path
        ref_sr = sr or 22050

    if isinstance(tgt_audio_or_path, (str, Path)):
        tgt_audio, tgt_sr = load_audio_mono(tgt_audio_or_path, target_sr=ref_sr)
    else:
        tgt_audio = tgt_audio_or_path
        tgt_sr = ref_sr

    # Tính toán bộ lọc FIR Match EQ
    taps = compute_spectral_match_filter(ref_audio, tgt_audio, sr=tgt_sr, max_gain_db=max_gain_db)

    # Lọc tín hiệu
    filtered = scipy.signal.convolve(tgt_audio, taps, mode="same")

    # Giữ True Peak an toàn <= -1.0 dBFS (~ 0.891 linear)
    peak = np.max(np.abs(filtered))
    max_safe_peak = 0.891
    if peak > max_safe_peak:
        filtered = filtered * (max_safe_peak / (peak + 1e-8))

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
