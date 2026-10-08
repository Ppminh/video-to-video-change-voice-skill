"""Đường dẫn và cấu hình chung.

- ROOT: thư mục dự án (chứa các file .command, logs, THANH_PHAM...)
- APPHOME: ~/.douyin_dubber (venv + model, đặt ở đây vì đường dẫn không có dấu cách)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
APPHOME = Path(os.environ.get("DUBBER_HOME", Path.home() / ".douyin_dubber"))
MODELS = APPHOME / "models"
WORK = Path(os.environ.get("DUBBER_WORK", ROOT / "_work"))
OUTPUT = ROOT / "THANH_PHAM"
INPUT_DIR = ROOT / "DAU_VAO"
LOGS = ROOT / "logs"
CONFIG = ROOT / "config"
DB_PATH = WORK / "jobs.db"
SETTINGS_PATH = CONFIG / "settings.json"
ENV_PATH = CONFIG / ".env"
HF_CACHE = Path("D:/Makemoney/.cache/huggingface") if Path("D:/Makemoney").exists() else Path.home() / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
VALTEC_MODEL_DIR = Path("D:/Makemoney/models/valtec_tts") if Path("D:/Makemoney/models/valtec_tts").exists() else MODELS / "valtec_tts"
VALTEC_ZEROSHOT_DIR = Path("D:/Makemoney/models/valtec_zeroshot") if Path("D:/Makemoney/models/valtec_zeroshot").exists() else MODELS / "valtec_zeroshot"

IS_MAC = sys.platform == "darwin"

for _d in (WORK, OUTPUT, INPUT_DIR, LOGS, CONFIG):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Giọng miền Nam có sẵn trong VieNeu v3 Turbo (đọc từ voices_v3_turbo.json của thư viện)
SOUTH_MALE = ["Thái Sơn", "Minh Triết", "Đức Trí", "Adam"]
SOUTH_FEMALE = ["Thục Đoan", "Mỹ Duyên", "Kim Thanh", "Thùy Dung"]

DEFAULT_SETTINGS = {
    # workflow_mode: 'dub' (lồng tiếng + vietsub), 'sub_only' (chỉ vietsub, giữ nguyên 100% âm thanh gốc)
    "workflow_mode": "dub",
    # edgetts = Microsoft Edge-TTS (truyền cảm, siêu tốc, đa cảm xúc), turbo = VieNeu Turbo, voxcpm = VoxCPM2, nano = VieNeu Nano
    "tts_mode": "edgetts",
    "voxcpm_model": "openbmb/VoxCPM2",
    "voxcpm_timesteps": 10,
    # số luồng CPU cho model
    "threads": 4,
    # tách nhạc: auto = demucs-mlx nếu có, nếu không thì UVR (sherpa-onnx)
    "separator": "auto",
    # thứ tự model dịch (model đầu hết lượt -> chuyển model sau)
    "translate_models": [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemini-3.5-flash",
    ],
    # tốc độ đọc mặc định (âm tiết/giây) khi chưa đo được của từng giọng
    "speech_rate_sps": 4.2,
    "max_speed": 1.15,        # tăng tốc tới mức này vẫn nghe tự nhiên
    "hard_max_speed": 1.45,  # mức tối đa sau khi đã rút gọn câu
    "borrow_gap_s": 1.5,     # được mượn khoảng lặng phía sau tối đa
    "duck_db": -5.0,         # hạ nhạc nền khi có lời
    "target_lufs": -14.0,
    "ocr_fps": 2.0,
    "video_bitrate": "6M",
    "keep_work_files": False,
    "paused": False,
    # Chạy tuần tự nghiêm ngặt (concurrency = 1): 1 video làm xong mới đến video tiếp theo, không xử lý 2 video cùng lúc
    "parallel_lanes": False,
    # Đồng bộ âm sắc giọng nói: 'match_eq' (Spectral Match EQ), 'zeroshot' (sao chép giọng mẫu), 'off' (tắt)
    "voice_match_mode": "match_eq",
    # Tự động cân bằng cao độ F0 trung bình khớp với nhân vật gốc
    "pitch_match": True,
    # giọng chính: nữ chính luôn dùng giọng này (để trống = theo thể loại)
    "lead_female_voice": "Thục Đoan",
    "lead_male_voice": "",
    # tạo thêm biến thể giọng (trẻ/trầm/già/bé) để mỗi nhân vật một giọng riêng
    "voice_variants": True,
    # fp32 = chất lượng tối đa; int8 = nhanh hơn (nghe thử bằng 4_CHAN_DOAN trước khi đổi)
    "tts_precision": "fp32",
    # giọng theo thể loại (thứ tự ưu tiên). Người nói nhiều nhất nhận giọng đầu danh sách.
    "genre_voices": {
        "default": {"male": ["Adam", "Thái Sơn", "Đức Trí", "Minh Triết"],
                    "female": ["Thục Đoan", "Mỹ Duyên", "Kim Thanh", "Thùy Dung"]},
        "co_trang": {"male": ["Đức Trí", "Thái Sơn", "Adam", "Minh Triết"],
                     "female": ["Mỹ Duyên", "Kim Thanh", "Thục Đoan", "Thùy Dung"]},
        "ban_hang": {"male": ["Minh Triết", "Adam", "Thái Sơn", "Đức Trí"],
                     "female": ["Thùy Dung", "Thục Đoan", "Mỹ Duyên", "Kim Thanh"]},
        "kien_thuc": {"male": ["Minh Triết", "Thái Sơn", "Adam", "Đức Trí"],
                      "female": ["Thùy Dung", "Thục Đoan", "Mỹ Duyên", "Kim Thanh"]},
        "review_phim": {"male": ["Thái Sơn", "Đức Trí", "Adam", "Minh Triết"],
                        "female": ["Thục Đoan", "Mỹ Duyên", "Kim Thanh", "Thùy Dung"]},
        "hai": {"male": ["Adam", "Thái Sơn", "Đức Trí", "Minh Triết"],
                "female": ["Thục Đoan", "Kim Thanh", "Mỹ Duyên", "Thùy Dung"]},
    },
    # VieNeu Nano chỉ có 2 giọng miền Nam
    "nano_voices": {"male": ["Adam", "Đức Trí", "Hữu Quân", "Minh Quân"],
                    "female": ["Ái Hân", "Mỹ Duyên", "Trúc Ly", "Xuân Tiên"]},
    # Valtec TTS (v-tts) gồm 5 giọng Bắc và Nam
    "valtec_voices": {"male": ["Valtec SM", "Valtec NM1", "Valtec NM2"],
                      "female": ["Valtec SF", "Valtec NF"]},
}


def load_settings() -> dict:
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    if SETTINGS_PATH.exists():
        try:
            user = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(s.get(k), dict):
                    s[k].update(v)
                else:
                    s[k] = v
        except Exception:
            pass
    return s


def save_settings(s: dict) -> None:
    """Chỉ lưu những gì KHÁC mặc định -> khi cập nhật code, mặc định mới vẫn được áp dụng."""
    diff = {k: v for k, v in s.items() if DEFAULT_SETTINGS.get(k) != v}
    SETTINGS_PATH.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")


def load_env() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or load_env().get("GEMINI_API_KEY", "")
