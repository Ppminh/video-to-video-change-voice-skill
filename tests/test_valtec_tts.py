import pytest
import numpy as np
from pathlib import Path
from app.dubber import paths, ttsload


def test_valtec_load_and_synthesize():
    """Kiểm tra nạp và tổng hợp giọng nói qua ValtecTTSWrapper."""
    tts = ttsload.load(mode="valtec")
    assert isinstance(tts, ttsload.ValtecTTSWrapper)
    assert tts.sample_rate == 24000

    # Kiểm tra danh sách giọng đọc
    voices = tts.list_preset_voices()
    voice_names = [name for name, _ in voices]
    assert "Valtec SF" in voice_names
    assert "Valtec SM" in voice_names
    assert "Valtec NF" in voice_names
    assert "Valtec NM1" in voice_names
    assert "Valtec NM2" in voice_names

    # Kiểm tra suy luận giọng đọc tiếng Việt
    text = "Xin chào, đây là bài kiểm tra Valtec TTS tự động."
    audio = tts.infer(text, voice="Valtec SF", speed=1.0)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0

    # Thời lượng audio phải hợp lý (> 1 giây)
    dur = len(audio) / tts.sample_rate
    assert dur > 1.0


def test_valtec_voice_fallback_and_speed():
    """Kiểm tra tự động gán giọng và điều chỉnh tốc độ."""
    tts = ttsload.load(mode="valtec")

    # Giọng nữ bất kỳ fallback về SF
    audio_f = tts.infer("Chào bạn", voice="Thục Đoan", speed=1.2)
    assert isinstance(audio_f, np.ndarray)
    assert len(audio_f) > 0

    # Giọng nam bất kỳ fallback về SM
    audio_m = tts.infer("Chào bạn", voice="Thái Sơn", speed=1.0)
    assert isinstance(audio_m, np.ndarray)
    assert len(audio_m) > 0
