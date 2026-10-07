"""Nạp model giọng đọc VieNeu an toàn trên Mac.

Vấn đề: HuggingFace lưu model dạng 'liên kết' (symlink) trỏ tới các file nằm ở thư mục khác.
Bản onnxruntime mới coi file dữ liệu ngoài thư mục model là không an toàn và từ chối nạp.
Cách sửa: chuyển file thật vào đúng chỗ của liên kết (không tốn thêm ổ cứng), sau đó chạy
chế độ offline để lần sau không cần hỏi lại máy chủ HuggingFace.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

REPOS = ["pnnbao-ump/VieNeu-TTS-v3-Turbo", "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
         "pnnbao-ump/VieNeu-TTS-v3-Nano"]


def _cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        p = Path(os.environ["HF_HUB_CACHE"])
        p.mkdir(parents=True, exist_ok=True)
        return p
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def materialize() -> int:
    """Thay symlink trong snapshot bằng file thật. Trả về số file đã chuyển."""
    n = 0
    cache = _cache_dir()
    for repo in REPOS:
        snaps = cache / ("models--" + repo.replace("/", "--")) / "snapshots"
        if not snaps.exists():
            continue
        for p in snaps.rglob("*"):
            if p.is_symlink():
                target = p.resolve()
                if target.exists() and target.is_file():
                    p.unlink()
                    shutil.move(str(target), str(p))
                    n += 1
    return n


def cached(mode: str) -> bool:
    repo = REPOS[2] if mode == "nano" else REPOS[0]
    snaps = _cache_dir() / ("models--" + repo.replace("/", "--")) / "snapshots"
    return snaps.exists() and any(snaps.rglob("*.onnx"))


def _make(m: str, threads: int, precision: str):
    from vieneu import Vieneu
    kw = {"mode": m, "threads": threads}
    if m == "v3turbo" and precision and precision != "fp32":
        kw["precision"] = precision      # int8: nhanh hơn ~1,5 lần, cần nghe thử chất lượng
    return Vieneu(**kw)


class VoxCPMWrapper:
    """Wrapper cho OpenBMB VoxCPM2 hỗ trợ tiếng Việt và sao chép giọng (voice cloning)."""
    def __init__(self, model_id: str = "openbmb/VoxCPM2", timesteps: int = 10, device: str | None = None):
        import numpy as np
        from voxcpm import VoxCPM
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.timesteps = timesteps
        self.model_id = model_id
        # optimize=False để tránh lỗi torch.compile trên CPU/Windows
        # load_denoiser=False để tiết kiệm RAM
        self.model = VoxCPM.from_pretrained(
            hf_model_id=model_id,
            device=device,
            cache_dir=str(_cache_dir()),
            optimize=False,
            load_denoiser=False
        )
        self.sample_rate = int(getattr(self.model.tts_model, "sample_rate", 24000) or 24000)
        self._preset_voices = {
            "Thái Sơn": {"gender": "male"},
            "Minh Triết": {"gender": "male"},
            "Đức Trí": {"gender": "male"},
            "Adam": {"gender": "male"},
            "Thục Đoan": {"gender": "female"},
            "Mỹ Duyên": {"gender": "female"},
            "Kim Thanh": {"gender": "female"},
            "Thùy Dung": {"gender": "female"},
        }

    def infer(self, text: str, voice: str | None = None, ref_wav: str | None = None, cfg_value: float = 2.0):
        import numpy as np
        kwargs = {
            "text": text,
            "inference_timesteps": self.timesteps,
            "cfg_value": cfg_value,
            "normalize": False,
        }
        if ref_wav and Path(ref_wav).exists():
            kwargs["reference_wav_path"] = str(ref_wav)
        wav = self.model.generate(**kwargs)
        return np.asarray(wav, dtype=np.float32)

def _patch_edge_tts_smart_ssml():
    """Bổ sung bộ lọc SSML thông minh cho edge_tts.communicate mà không phá vỡ Communicate gốc.
    - Khi nhận chuỗi đã là SSML (<speak...>): giữ nguyên nội dung SSML để gửi đến máy chủ Microsoft.
    - Khi nhận chuỗi thông thường (Communicate với rate, pitch, volume): gọi mkssml gốc để bọc SSML hợp lệ.
    Loại bỏ hoàn toàn lỗi NoAudioReceived do monkey-patching làm mất thẻ SSML ở các lần gọi kế tiếp.
    """
    try:
        import edge_tts.communicate as comm_mod
        if not getattr(comm_mod, "_ssml_smart_patched", False):
            orig_mkssml = comm_mod.mkssml

            def _smart_mkssml(tc, escaped_text):
                text_str = escaped_text.decode("utf-8") if isinstance(escaped_text, bytes) else str(escaped_text)
                if text_str.lstrip().startswith("<speak"):
                    return text_str
                return orig_mkssml(tc, escaped_text)

            comm_mod.mkssml = _smart_mkssml
            comm_mod._ssml_smart_patched = True
    except Exception:
        pass


class EdgeTTSWrapper:
    """Wrapper cho Microsoft Edge-TTS: siêu tốc, tự nhiên, hỗ trợ đa giọng và cảm xúc."""
    def __init__(self):
        import numpy as np
        _patch_edge_tts_smart_ssml()
        self.sample_rate = 24000
        self._preset_voices = {
            "Hoài My": {"gender": "female", "voice_id": "vi-VN-HoaiMyNeural", "pitch": "+0Hz"},
            "Nam Minh": {"gender": "male", "voice_id": "vi-VN-NamMinhNeural", "pitch": "+0Hz"},
            "Thục Đoan": {"gender": "female", "voice_id": "vi-VN-HoaiMyNeural", "pitch": "+2Hz"},
            "Mỹ Duyên": {"gender": "female", "voice_id": "vi-VN-HoaiMyNeural", "pitch": "+4Hz"},
            "Kim Thanh": {"gender": "female", "voice_id": "vi-VN-HoaiMyNeural", "pitch": "-2Hz"},
            "Thùy Dung": {"gender": "female", "voice_id": "vi-VN-HoaiMyNeural", "pitch": "+1Hz"},
            "Thái Sơn": {"gender": "male", "voice_id": "vi-VN-NamMinhNeural", "pitch": "-2Hz"},
            "Minh Triết": {"gender": "male", "voice_id": "vi-VN-NamMinhNeural", "pitch": "+2Hz"},
            "Đức Trí": {"gender": "male", "voice_id": "vi-VN-NamMinhNeural", "pitch": "-4Hz"},
            "Adam": {"gender": "male", "voice_id": "vi-VN-NamMinhNeural", "pitch": "+0Hz"},
        }

    def infer(self, text: str, voice: str | None = None, emotion: str = "",
              pitch: str | None = None, rate: str | None = None, volume: str | None = None,
              pitch_profile: dict | None = None, ssml: str | None = None, **kwargs):
        import asyncio
        import io
        import time
        import edge_tts
        import soundfile as sf
        import numpy as np

        _patch_edge_tts_smart_ssml()

        clean_txt = (text or "").strip()
        if not clean_txt or not any(c.isalnum() for c in clean_txt):
            return np.zeros(int(self.sample_rate * 0.4), dtype=np.float32)

        info = self._preset_voices.get(voice)
        if not info and voice:
            import re
            clean = re.sub(r"\s*\(.*?\)", "", str(voice)).strip()
            info = self._preset_voices.get(clean)
        if not info and voice:
            for name, cand in self._preset_voices.items():
                if cand.get("voice_id") == voice or name.lower() in str(voice).lower():
                    info = cand
                    break
        if not info and voice:
            v_low = str(voice).lower()
            if any(m in v_low for m in ("male", "nam")):
                info = self._preset_voices.get("Nam Minh") or {"gender": "male", "voice_id": "vi-VN-NamMinhNeural"}
            elif any(f in v_low for f in ("female", "nu", "nữ")):
                info = self._preset_voices.get("Hoài My") or {"gender": "female", "voice_id": "vi-VN-HoaiMyNeural"}
        if not info:
            info = {}

        voice_id = info.get("voice_id") or ("vi-VN-NamMinhNeural" if info.get("gender") == "male" else "vi-VN-HoaiMyNeural")

        # Khởi tạo giá trị mặc định từ preset giọng và cảm xúc
        eff_pitch = info.get("pitch", "+0Hz")
        eff_rate = "+0%"
        eff_volume = "+0%"

        # Nếu có pitch_profile từ F0 pitch tracking, tính toán bù trừ theo giọng gốc (chỉ tính khi chưa có tham số trực tiếp)
        if pitch_profile and (pitch is None or rate is None or volume is None):
            try:
                from .pitch import map_pitch_to_prosody
                p_mapped = map_pitch_to_prosody(pitch_profile, voice_name=voice, base_voice=voice, emotion=emotion, safe_volume=True)
                if pitch is None:
                    eff_pitch = p_mapped.get("pitch", eff_pitch)
                if rate is None:
                    eff_rate = p_mapped.get("rate", eff_rate)
                if volume is None:
                    eff_volume = p_mapped.get("volume", eff_volume)
            except Exception:
                pass

        # Ghi đè nếu có tham số truyền trực tiếp
        if pitch is not None:
            eff_pitch = pitch
        if rate is not None:
            eff_rate = rate
        if volume is not None:
            eff_volume = volume

        # Nếu không có pitch_profile và không truyền pitch/rate/volume trực tiếp, dùng quy tắc emotion cũ
        # R2: Giới hạn volume <= +4% để tránh vỡ biên độ giọng đọc
        if not pitch_profile and pitch is None and rate is None and volume is None:
            emo = (emotion or "").lower()
            if any(w in emo for w in ("angry", "tuc", "gian", "phẫn", "gắt", "bực")):
                eff_rate = "+14%"
                eff_pitch = "+5Hz"
                eff_volume = "+4%"
            elif any(w in emo for w in ("panicked", "sợ", "hoảng", "lo lắng")):
                eff_rate = "+18%"
                eff_pitch = "+7Hz"
                eff_volume = "+4%"
            elif any(w in emo for w in ("happy", "excited", "vui", "hào hứng", "cười")):
                eff_rate = "+8%"
                eff_pitch = "+6Hz"
                eff_volume = "+4%"
            elif any(w in emo for w in ("sad", "buon", "khoc", "đau", "bi thương")):
                eff_rate = "-12%"
                eff_pitch = "-5Hz"
                eff_volume = "-10%"
            elif any(w in emo for w in ("whisper", "thi tham", "nói nhỏ", "tâm sự")):
                eff_rate = "-8%"
                eff_pitch = "-3Hz"
                eff_volume = "-25%"
            elif any(w in emo for w in ("sarcastic", "mỉa mai", "khinh bỉ")):
                eff_rate = "-4%"
                eff_pitch = "+2Hz"
                eff_volume = "+4%"
            elif any(w in emo for w in ("dramatic", "hùng hồn", "trang nghiêm", "cổ trang")):
                eff_rate = "-5%"
                eff_pitch = "-2Hz"
                eff_volume = "+4%"

        # Chuẩn hóa an toàn các tham số prosody, giới hạn an toàn volume <= +4%
        try:
            from .pitch import normalize_prosody_attr
            eff_pitch = normalize_prosody_attr(eff_pitch, default_unit="Hz", min_val=-28, max_val=30)
            eff_rate = normalize_prosody_attr(eff_rate, default_unit="%", min_val=-20, max_val=30)
            eff_volume = normalize_prosody_attr(eff_volume, default_unit="%", min_val=-25, max_val=4)
        except Exception:
            pass

        async def _gen():
            if not str(eff_pitch).endswith("Hz") or ssml:
                from .pitch import build_ssml
                full_ssml = ssml or build_ssml(clean_txt, voice_id, pitch=str(eff_pitch), rate=str(eff_rate), volume=str(eff_volume), safe_volume=True)
                comm = edge_tts.Communicate(clean_txt, voice_id)
                comm.texts = [full_ssml]
            else:
                comm = edge_tts.Communicate(clean_txt, voice_id, rate=str(eff_rate), pitch=str(eff_pitch), volume=str(eff_volume))
            buf = io.BytesIO()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            data, _ = sf.read(buf)
            await asyncio.sleep(0.01)
            return data

        import concurrent.futures

        def _run_gen():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(lambda: asyncio.run(asyncio.wait_for(_gen(), timeout=12.0))).result()
            else:
                return asyncio.run(asyncio.wait_for(_gen(), timeout=12.0))

        for attempt in range(4):
            try:
                arr = _run_gen()
                if arr is not None and len(arr) > 0:
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)
                    return np.asarray(arr, dtype=np.float32)
            except Exception:
                if attempt < 3:
                    time.sleep(0.35 * (attempt + 1))
                    continue

        # Nếu cả 4 lần đều lỗi mạng: tự động fallback sang VieNeu dự phòng hoặc khoảng lặng êm
        try:
            if not hasattr(self, "_fallback_vieneu"):
                from vieneu import Vieneu
                self._fallback_vieneu = Vieneu(mode="v3turbo", threads=2)
            v_name = "Thục Đoan" if info.get("gender") == "female" else "Thái Sơn"
            arr = self._fallback_vieneu.infer(clean_txt, voice=v_name)
            if arr is not None and len(arr) > 0:
                return np.asarray(arr, dtype=np.float32)
        except Exception:
            pass

        return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)

    def list_preset_voices(self):
        return [(name, name) for name in self._preset_voices]


def load(mode: str = "edgetts", threads: int = 4, precision: str = "fp32", **kwargs):
    if mode == "edgetts":
        return EdgeTTSWrapper()
    if mode == "voxcpm":
        model_id = kwargs.get("voxcpm_model", "openbmb/VoxCPM2")
        timesteps = int(kwargs.get("voxcpm_timesteps", 10))
        return VoxCPMWrapper(model_id=model_id, timesteps=timesteps)
    m = "v3nano" if mode == "nano" else "v3turbo"
    if cached(mode):
        materialize()
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return _make(m, threads, precision)
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        return _make(m, threads, precision)
    except Exception as e:
        if "External data path" not in str(e) and "escapes model directory" not in str(e):
            raise
    # lần đầu: model vừa tải xong nhưng còn là symlink -> chuyển thành file thật rồi nạp lại
    materialize()
    return _make(m, threads, precision)
