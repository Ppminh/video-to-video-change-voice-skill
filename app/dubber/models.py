"""Danh sách model và hàm tải về (từ GitHub releases của sherpa-onnx).

Model VieNeu-TTS do thư viện vieneu tự tải từ HuggingFace ở lần chạy đầu.
Model demucs-mlx được chuyển đổi một lần bởi file cài đặt.
"""
from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

from .paths import MODELS

GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

SENSEVOICE_DIR = MODELS / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"
SPEC = {
    "sensevoice": {
        "url": f"{GH}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2",
        "check": SENSEVOICE_DIR / "model.int8.onnx",
        "tar": True,
    },
    "vad": {
        "url": f"{GH}/asr-models/silero_vad.onnx",
        "check": MODELS / "silero_vad.onnx",
    },
    "segmentation": {
        "url": f"{GH}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "check": MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx",
        "tar": True,
    },
    "embedding": {
        "url": f"{GH}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "check": MODELS / "campplus_zh.onnx",
    },
    "spleeter": {
        "url": f"{GH}/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2",
        "check": MODELS / "sherpa-onnx-spleeter-2stems-fp16" / "vocals.fp16.onnx",
        "tar": True,
    },
    "uvr": {
        "url": f"{GH}/source-separation-models/UVR-MDX-NET-Voc_FT.onnx",
        "check": MODELS / "UVR-MDX-NET-Voc_FT.onnx",
        "optional": True,
    },
    "test_wav": {
        "url": f"{GH}/speaker-segmentation-models/0-four-speakers-zh.wav",
        "check": MODELS / "test_four_speakers_zh.wav",
        "optional": True,
    },
    "font": {
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/bevietnampro/BeVietnamPro-Bold.ttf",
        "check": MODELS / "BeVietnamPro-Bold.ttf",
    },
}


def path(name: str) -> Path:
    return SPEC[name]["check"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    import time
    for attempt in range(5):
        try:
            import requests
            headers = {"User-Agent": "dubber/0.1"}
            cur_size = tmp.stat().st_size if tmp.exists() else 0
            if cur_size > 0:
                headers["Range"] = f"bytes={cur_size}-"
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416 or (cur_size > 0 and r.status_code == 200):
                    mode = "wb"
                elif cur_size > 0 and r.status_code == 206:
                    mode = "ab"
                else:
                    mode = "wb"
                r.raise_for_status()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            break
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(3)
    if dest.exists():
        dest.unlink()
    tmp.rename(dest)


def ensure(name: str, verbose: bool = True) -> Path:
    spec = SPEC[name]
    check: Path = spec["check"]
    if check.exists():
        return check
    MODELS.mkdir(parents=True, exist_ok=True)
    url = spec["url"]
    if verbose:
        print(f"  - Tải {name}: {url.rsplit('/', 1)[-1]}", flush=True)
    if spec.get("tar"):
        archive = MODELS / url.rsplit("/", 1)[-1]
        _download(url, archive)
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(MODELS)
        archive.unlink(missing_ok=True)
        # bỏ bớt file không dùng để tiết kiệm ổ cứng
        if name == "sensevoice":
            for f in SENSEVOICE_DIR.glob("*"):
                if f.name not in ("model.int8.onnx", "tokens.txt"):
                    if f.is_dir():
                        shutil.rmtree(f, ignore_errors=True)
                    else:
                        f.unlink(missing_ok=True)
    else:
        _download(url, check)
    if not check.exists():
        raise RuntimeError(f"Tải {name} xong nhưng không thấy {check}")
    return check


def font_path() -> str:
    p = path("font")
    if p.exists():
        return str(p)
    for c in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/Library/Fonts/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(c).exists():
            return c
    raise RuntimeError("Không tìm thấy font tiếng Việt")


def ensure_all(include_optional: bool = True) -> None:
    for name, spec in SPEC.items():
        if spec.get("optional") and not include_optional:
            continue
        try:
            ensure(name)
        except Exception as e:  # noqa
            if spec.get("optional"):
                print(f"  ! Bỏ qua {name}: {e}")
            else:
                raise


if __name__ == "__main__":
    ensure_all()
    print("OK")
