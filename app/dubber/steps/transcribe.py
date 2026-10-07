"""B3. Nghe tiếng Trung, ghi mốc thời gian, phân biệt người nói, đoán nam/nữ.

Cách hoạt động:
1. VAD (bộ dò tiếng nói Silero) cắt file giọng thành từng đoạn có người nói.
2. Diarization: mỗi đoạn được biến thành 'vân tay giọng' (vector CAM++), các vân tay
   giống nhau gom thành 1 người -> biết đoạn nào của ai, không cần biết trước mấy người.
3. ASR SenseVoice đọc từng đoạn -> chữ Trung (đọc cả đoạn trong 1 lượt nên rất nhanh).
4. Giới tính: đo cao độ giọng trung bình (Praat). Nam thường 85-180 Hz, nữ 165-255 Hz.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .. import models
from ..utils import Job, cjk_count, load_audio, log, threads

SR = 16000


def _vad_segments(samples: np.ndarray) -> list[tuple[float, float]]:
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(models.ensure("vad"))
    cfg.silero_vad.threshold = 0.35
    cfg.silero_vad.min_silence_duration = 0.3
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = 9
    cfg.sample_rate = SR
    try:
        cfg.num_threads = 1
    except Exception:
        pass
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=len(samples) / SR + 30)
    ws = cfg.silero_vad.window_size
    segs = []

    def drain():
        while not vad.empty():
            f = vad.front
            st = f.start / SR
            segs.append((st, st + len(f.samples) / SR))
            vad.pop()

    for i in range(0, len(samples) - ws + 1, ws):
        vad.accept_waveform(samples[i:i + ws])
        drain()
    vad.flush()
    drain()
    return segs


def _diarize(samples: np.ndarray) -> list[tuple[float, float, int]]:
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(models.ensure("segmentation"))),
            num_threads=threads()),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models.ensure("embedding")), num_threads=threads()),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.85),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise RuntimeError("Cấu hình phân biệt người nói không hợp lệ")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    res = sd.process(samples).sort_by_start_time()
    return [(r.start, r.end, int(r.speaker)) for r in res]


def _recognizer():
    import sherpa_onnx
    d = models.SENSEVOICE_DIR
    models.ensure("sensevoice")
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
        num_threads=threads(), language="zh", use_itn=True)


def _split_by_speaker(st, en, turns):
    """Cắt 1 đoạn VAD theo người nói. Trả về [(st, en, spk)]."""
    parts = []
    for a, b, s in turns:
        lo, hi = max(st, a), min(en, b)
        if hi - lo > 0.05:
            parts.append([lo, hi, s])
    if not parts:
        return [(st, en, -1)]
    # gộp các phần liền nhau cùng người
    merged = []
    for p in sorted(parts):
        if merged and merged[-1][2] == p[2] and p[0] - merged[-1][1] < 0.4:
            merged[-1][1] = max(merged[-1][1], p[1])
        else:
            merged.append(p)
    # phần quá ngắn (<0.8s) nhập vào phần dài nhất
    main = max(merged, key=lambda p: p[1] - p[0])
    keep = [p for p in merged if p[1] - p[0] >= 0.8 or p is main]
    if len(keep) == 1:
        return [(st, en, keep[0][2])]
    keep.sort()
    out = []
    for i, p in enumerate(keep):
        a = st if i == 0 else p[0]
        b = en if i == len(keep) - 1 else keep[i + 1][0]
        out.append((a, b, p[2]))
    return out


def _clean(text: str) -> str:
    t = re.sub(r"<\|[^|]*\|>", "", text or "")
    t = re.sub(r"\s+(?=[一-鿿])|(?<=[一-鿿])\s+", "", t)
    return t.strip()


def _pitch(samples: np.ndarray) -> float:
    try:
        from ..pitch import analyze_segment_pitch_energy
        res = analyze_segment_pitch_energy(samples, SR)
        if res.get("f0", 0.0) > 0:
            return float(res["f0"])
    except Exception:
        pass
    try:
        import parselmouth
        snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=SR)
        p = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
        f0 = p.selected_array["frequency"]
        f0 = f0[f0 > 0]
        return float(np.median(f0)) if len(f0) > 20 else 0.0
    except Exception:
        return 0.0


def run(job: Job) -> None:
    t0 = time.time()
    samples, _ = load_audio(job.p("vocals_16k.wav"), sr=SR, mono=True)
    dur = len(samples) / SR
    job.progress(0.05, "VAD")
    segs = _vad_segments(samples)
    log(f"VAD: {len(segs)} đoạn có tiếng nói")
    job.progress(0.2, "phân biệt người nói")
    try:
        turns = _diarize(samples)
    except Exception as e:
        log(f"Diarization lỗi ({e}) -> coi như 1 người")
        turns = [(0.0, dur, 0)]
    nspk = len({t[2] for t in turns})
    log(f"Diarization: {nspk} người nói, {len(turns)} lượt")
    job.progress(0.45, "nhận dạng tiếng Trung")

    pieces = []
    for st, en in segs:
        pieces.extend(_split_by_speaker(st, en, turns))
    rec = _recognizer()
    streams = []
    for st, en, _ in pieces:
        s = rec.create_stream()
        a, b = int(st * SR), int(en * SR)
        s.accept_waveform(SR, samples[a:b])
        streams.append(s)
    for i in range(0, len(streams), 16):
        rec.decode_streams(streams[i:i + 16])
        job.progress(0.45 + 0.45 * min(1, (i + 16) / max(1, len(streams))), "nhận dạng tiếng Trung")

    lines, nonspeech = [], []
    for (st, en, spk), s in zip(pieces, streams):
        r = s.result
        text = _clean(r.text)
        event = (getattr(r, "event", "") or "").strip("<|>")
        emotion = (getattr(r, "emotion", "") or "").strip("<|>")
        if cjk_count(text) == 0 and not re.search(r"[A-Za-z0-9]", text):
            nonspeech.append({"start": round(st, 3), "end": round(en, 3), "event": event})
            continue
        lines.append({"start": round(st, 3), "end": round(en, 3), "spk": f"S{spk}" if spk >= 0 else "S?",
                      "zh": text, "emotion": emotion, "event": event})

    # gán người nói cho đoạn không xác định = người nói gần nhất
    for i, ln in enumerate(lines):
        if ln["spk"] == "S?":
            near = [l for l in lines if l["spk"] != "S?"]
            if near:
                ln["spk"] = min(near, key=lambda l: abs(l["start"] - ln["start"]))["spk"]
            else:
                ln["spk"] = "S0"
    lines.sort(key=lambda l: l["start"])
    for i, ln in enumerate(lines, 1):
        ln["id"] = i

    # thông tin từng người nói: thời lượng nói, cao độ, giới tính đoán theo giọng
    speakers = {}
    for spk in sorted({l["spk"] for l in lines}):
        mine = [l for l in lines if l["spk"] == spk]
        talk = sum(l["end"] - l["start"] for l in mine)
        chunks, tot = [], 0.0
        for l in sorted(mine, key=lambda l: l["end"] - l["start"], reverse=True):
            a, b = int(l["start"] * SR), int(l["end"] * SR)
            chunks.append(samples[a:b])
            tot += l["end"] - l["start"]
            if tot > 30:
                break
        f0 = _pitch(np.concatenate(chunks)) if chunks else 0.0
        gender = "female" if f0 >= 170 else ("male" if f0 > 0 else "unknown")
        speakers[spk] = {"talk_time": round(talk, 1), "lines": len(mine), "f0": round(f0, 1),
                         "gender_voice": gender}

    # gộp "người nói" quá ít (<3 giây) vào người cùng giới có cao độ gần nhất (thường do chia cụm nhầm)
    big = {k: v for k, v in speakers.items() if v["talk_time"] >= 3.0}
    if big:
        remap = {}
        for k, v in speakers.items():
            if k in big:
                continue
            same = [b for b, bv in big.items() if bv["gender_voice"] == v["gender_voice"]] or list(big)
            remap[k] = min(same, key=lambda b: abs(big[b]["f0"] - v["f0"]))
        for ln in lines:
            ln["spk"] = remap.get(ln["spk"], ln["spk"])
        for k, tgt in remap.items():
            big[tgt]["talk_time"] = round(big[tgt]["talk_time"] + speakers[k]["talk_time"], 1)
            big[tgt]["lines"] += speakers[k]["lines"]
        speakers = big
    job.write("transcript.json", {"duration": round(dur, 3), "lines": lines, "nonspeech": nonspeech,
                                  "speakers": speakers})
    log(f"Nhận dạng xong: {len(lines)} câu, {len(speakers)} người, {time.time() - t0:.1f}s")
    for spk, v in speakers.items():
        log(f"  {spk}: {v['talk_time']}s, f0={v['f0']}Hz -> {v['gender_voice']}")
