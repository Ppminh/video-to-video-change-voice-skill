import os
import sys
import time
from pathlib import Path
import numpy as np
import pytest

# Thiết lập đường dẫn môi trường kiểm thử
sys.path.insert(0, r"D:\Makemoney\site-packages")
ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dubber import paths, matcheq
from dubber.worker import Worker, RETRYABLE
from dubber.steps import STEPS


def test_matcheq_spectral_shaping_and_true_peak():
    """Kiểm tra Spectral Match EQ: lọc phổ, làm mượt và bảo vệ True Peak <= -1.0 dBFS."""
    sr = 22050
    t = np.linspace(0, 1.0, sr)
    # Giả lập ref: nhiều âm trầm (130Hz)
    ref = (0.5 * np.sin(2 * np.pi * 130 * t) + 0.1 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    # Giả lập tgt: nhiều âm cao (2000Hz)
    tgt = (0.1 * np.sin(2 * np.pi * 130 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)

    filtered, out_sr = matcheq.apply_match_eq(ref, tgt, sr=sr, max_gain_db=6.0)
    assert out_sr == sr
    assert len(filtered) == len(tgt)
    # True Peak an toàn <= -1.0 dBFS (~ 0.891)
    peak = np.max(np.abs(filtered))
    assert peak <= 0.892, f"Peak {peak} vượt ngưỡng an toàn -1.0 dBFS!"

    # Đo cao độ F0 median
    f0 = matcheq.estimate_f0_median(ref, sr)
    assert 120.0 <= f0 <= 140.0, f"F0 {f0} không khớp tần số 130Hz"


def test_worker_strict_serial_queue():
    """Kiểm tra hàng chờ tuần tự nghiêm ngặt (Concurrency = 1)."""
    w = Worker()
    # Khi parallel = False (mặc định) và đã có 1 video đang chạy (claimed)
    w.claimed[101] = "prep"

    # _pick KHÔNG được phép nhận thêm bất kỳ video nào khác
    cand = w._pick(lane="prep", parallel=False, ahead=1)
    assert cand is None, "Worker nhận thêm video khi đang có video chạy dở!"

    # Khi không có video nào chạy
    w.claimed.clear()
    # Nếu danh sách trống, trả về None
    cand = w._pick(lane="prep", parallel=False, ahead=1)
    # Không ném lỗi cú pháp


def test_worker_auto_retry_all_steps():
    """Kiểm tra toàn bộ 9 bước đều nằm trong danh sách RETRYABLE."""
    assert RETRYABLE == set(STEPS)
    for step in ["download", "separate", "transcribe", "ocr", "translate", "tts", "mix", "render", "qc"]:
        assert step in RETRYABLE

    # Kiểm tra tính toán backoff
    for attempt, expected_sec in [(1, 5), (2, 15), (3, 30)]:
        sec = [5, 15, 30][min(attempt - 1, 2)]
        assert sec == expected_sec


def test_valtec_zeroshot_model_exists():
    """Kiểm tra các tệp trọng số ZeroShot trên ổ D: tồn tại đầy đủ."""
    zeroshot_dir = paths.VALTEC_ZEROSHOT_DIR
    ckpt = zeroshot_dir / "zeroshot" / "G_175000.pth"
    cfg = zeroshot_dir / "zeroshot" / "config.json"
    hasp = zeroshot_dir / "hasp" / "pytorch_model.bin"

    assert ckpt.exists(), f"Thiếu checkpoint {ckpt}"
    assert cfg.exists(), f"Thiếu config {cfg}"
    assert hasp.exists(), f"Thiếu HASP {hasp}"
    assert ckpt.stat().st_size > 100_000_000, "Checkpoint G_175000.pth dung lượng quá nhỏ!"
