"""B6 + B7. Chọn giọng cho từng nhân vật, đọc giọng Việt, khớp thời gian.

Chọn giọng (phân vai):
- 8 giọng miền Nam gốc x 5 "biến thể" (bình thường / trẻ / trầm / già / bé) = hơn 30 giọng khác nhau.
  Biến thể được tạo bằng kỹ thuật PSOLA của Praat ("Change gender"): đổi cao độ giọng (nốt nhạc)
  và độ dày giọng (formant) mà KHÔNG đổi tốc độ nói -> nghe như một người khác.
- AI dịch cho biết mỗi nhân vật: nam/nữ, tuổi (trẻ con/trẻ/trung niên/già). Người nói nhiều nhất
  được chọn giọng đầu tiên; nữ chính mặc định giọng "Thục Đoan" (đổi trong settings).
- Ưu tiên mỗi nhân vật một giọng gốc khác nhau; hết giọng gốc mới dùng biến thể.
Khớp thời gian từng câu ('ô thời gian' = từ lúc câu gốc bắt đầu tới câu kế tiếp, được mượn
thêm tối đa 1,5 giây im lặng):
  - Trước khi đọc: câu nào chắc chắn quá dài (ước lượng theo tốc độ đọc đã đo) -> nhờ AI rút gọn
    luôn, đỡ phải đọc 2 lần.
  - vừa ô -> đặt vào; dài hơn tới 1,2 lần -> tăng tốc giữ nguyên cao độ (ffmpeg atempo)
  - dài hơn nữa -> nhờ AI viết câu gọn hơn rồi đọc lại; vẫn dài thì tăng tốc tối đa 1,45 lần
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from .. import paths
from ..textnorm import normalize_vi
from ..utils import Job, ffmpeg, log, read_json, save_audio, threads, vi_syllables, write_json

RATES_FILE = paths.APPHOME / "voice_rates.json"

# biến thể: (nửa cung đổi cao độ, tỉ lệ formant). +12 nửa cung = cao gấp đôi.
VARIANTS = {
    "": (0.0, 1.0),
    "trẻ": (2.0, 1.04),
    "trầm": (-2.0, 0.97),
    "già": (-3.0, 0.94),
    "bé": (4.0, 1.12),
}
AGE_PREF = {
    "child": ["bé"],
    "young": ["", "trẻ", "trầm"],
    "adult": ["", "trầm", "trẻ"],
    "old": ["già", "trầm", ""],
}


def trim_silence(a: np.ndarray, sr: int) -> np.ndarray:
    if len(a) == 0:
        return a
    hop = int(sr * 0.01)
    n = len(a) // hop
    if n < 3:
        return a
    e = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    thr = max(1e-4, 0.03 * float(e.max()))
    idx = np.where(e > thr)[0]
    if len(idx) == 0:
        return a
    s = max(0, (idx[0] - 4) * hop)
    t = min(len(a), (idx[-1] + 6) * hop)
    return a[s:t]


def apply_variant(a: np.ndarray, sr: int, variant: str) -> np.ndarray:
    """Đổi cao độ + formant bằng Praat PSOLA, giữ nguyên thời lượng."""
    import re
    if variant in VARIANTS:
        st, fr = VARIANTS[variant]
    elif variant and re.match(r"^v\d+$", str(variant).strip()):
        k = int(re.search(r"\d+", str(variant)).group(0))
        st = float(((k * 3) % 9) - 4)
        fr = float(1.0 + (((k * 2) % 7) - 3) * 0.025)
    else:
        st, fr = (0.0, 1.0)
    if (st == 0 and fr == 1.0) or len(a) < sr * 0.2 or np.max(np.abs(a)) < 1e-4:
        return a
    try:
        import parselmouth
        from parselmouth.praat import call
        snd = parselmouth.Sound(a.astype(np.float64), sampling_frequency=sr)
        f0 = call(snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=600), "Get quantile", 0, 0, 0.5, "Hertz")
        if not f0 or not np.isfinite(f0):
            return a
        out = call(snd, "Change gender", 60, 600, fr, f0 * 2 ** (st / 12), 1.0, 1.0)
        y = np.asarray(out.values[0], dtype=np.float32)
        peak = float(np.abs(y).max() or 1.0)
        return y * min(1.0, 0.95 / peak) if peak > 0.95 else y
    except Exception as e:
        log(f"Không tạo được biến thể giọng '{variant}': {e}")
        return a


def label(base: str, variant: str) -> str:
    return base if not variant else f"{base} ({variant})"


def _pools(s: dict, tts_mode: str, genre: str, available: dict) -> dict:
    if tts_mode == "nano":
        pools = s.get("nano_voices", {})
    elif tts_mode in ("valtec", "valtec_zeroshot"):
        pools = s.get("valtec_voices") or {"male": ["Valtec SM", "Valtec NM1", "Valtec NM2"],
                                            "female": ["Valtec SF", "Valtec NF"]}
    else:
        gv = s["genre_voices"]
        pools = gv.get(genre) or gv["default"]
    out = {}
    for g in ("male", "female"):
        if available:
            lst = [v for v in pools.get(g, []) if v in available] or [v for v, gg in available.items() if gg == g]
        else:
            lst = list(pools.get(g, []))
        lead = (s.get(f"lead_{g}_voice") or "").strip()
        if lead and (not available or lead in available):
            lst = [lead] + [v for v in lst if v != lead]
        out[g] = lst or (["Thái Sơn"] if g == "male" else ["Thục Đoan"])
    return out


# Danh sách phân loại chuẩn theo engine
EDGE_TTS_VOICES = {
    "female": ["Hoài My"],
    "male": ["Nam Minh"],
}

VIENEU_SOUTH_VOICES = {
    "female": ["Thục Đoan", "Mỹ Duyên", "Kim Thanh", "Thùy Dung"],
    "male": ["Thái Sơn", "Minh Triết", "Đức Trí", "Adam"],
}


def cast(job: Job, tts_mode: str, available: dict, speaker_f0_profiles: dict | None = None) -> dict:
    """Cơ chế phân vai lai ghép thông minh (Hybrid Edge-TTS + VieNeu Casting):
    - R1: Edge-TTS làm chủ đạo cho nhân vật chính (Hoài My, Nam Minh) để đảm bảo truyền cảm, tự nhiên.
    - R1: Tự động mở rộng sang VieNeu khi thiếu giọng (3, 4, 5+ người nói) với các giọng miền Nam.
    - R1: Không trùng lặp giọng giữa các nhân vật.
    - R2: Gán giọng tương thích nhất (Acoustic Distance Pairing) theo cao độ F0 trung vị từ vocals.wav:
      nam trầm (~110Hz) -> Đức Trí, nam thanh (~150Hz) -> Minh Triết, nữ cao (~235Hz) -> Mỹ Duyên...
    - Ghi nhận rõ nguồn gốc engine ('Edge-TTS' / 'VieNeu') cho từng speaker ID.
    """
    tr = job.read("transcript.json", {}) or {}
    tl = job.read("translation.json", {}) or {}
    s = job.settings
    use_var = bool(s.get("voice_variants", True))
    llm_spk = tl.get("speakers", {}) or {}
    spk_f0_map = speaker_f0_profiles or {}

    # Tập hợp và sắp xếp danh sách người nói, đảm bảo không bỏ sót speaker nào
    spk_stats: dict[str, dict] = {str(k): v for k, v in dict(tr.get("speakers", {})).items()}
    if tr.get("lines"):
        spk_counts = Counter(str(l.get("spk")) for l in tr["lines"] if l.get("spk") is not None)
        for spk, cnt in spk_counts.items():
            if spk not in spk_stats:
                spk_stats[spk] = {"talk_time": float(cnt * 2.0), "lines": cnt}
    if llm_spk:
        for spk_k, spk_v in llm_spk.items():
            spk_str = str(spk_k)
            if spk_str not in spk_stats:
                spk_stats[spk_str] = {"talk_time": 1.0, "lines": 1}

    order = sorted(spk_stats.items(), key=lambda kv: kv[1].get("talk_time", 0.0), reverse=True)

    used_bases: set = set()
    used_pairs: set = set()
    used_voices: set = set()
    casting = {}
    by_name = {}

    clean_mode = str(tts_mode or "").lower().replace("-", "").replace("_", "").strip()
    is_hybrid = clean_mode in ("edgetts", "hybrid", "turbohybrid", "edge")

    # Xác định nhân vật chính cho từng giới tính: Ưu tiên nhãn vai chính trong metadata, sau đó theo talk_time
    main_spk_by_gender = {}
    def _get_spk_li(spk_k):
        return llm_spk.get(spk_k) or (llm_spk.get(int(spk_k)) if str(spk_k).isdigit() else None) or {}

    # Bước 1: Ưu tiên người có vai trò chính (chính, lead, main) trong translation.json
    for spk, info in order:
        li = _get_spk_li(spk)
        g = li.get("gender")
        if g not in ("male", "female"):
            g = info.get("gender_voice")
        if g not in ("male", "female"):
            g = "male"
        age = li.get("age") if li.get("age") in AGE_PREF else "adult"
        voice_g = "female" if age == "child" else g

        role_str = str(li.get("role") or "").lower()
        name_str = str(li.get("name_vi") or "").lower()
        is_explicit_lead = any(kw in role_str for kw in ("chính", "lead", "main")) or any(kw in name_str for kw in ("chính", "lead", "main"))
        if is_explicit_lead and voice_g not in main_spk_by_gender:
            main_spk_by_gender[voice_g] = spk

    # Bước 2: Với các giới tính chưa có vai chính chỉ định, gán cho người nói nhiều nhất
    for spk, info in order:
        li = _get_spk_li(spk)
        g = li.get("gender")
        if g not in ("male", "female"):
            g = info.get("gender_voice")
        if g not in ("male", "female"):
            g = "male"
        age = li.get("age") if li.get("age") in AGE_PREF else "adult"
        voice_g = "female" if age == "child" else g
        if voice_g not in main_spk_by_gender:
            main_spk_by_gender[voice_g] = spk

    from ..pitch import get_voice_profile, pair_voice_by_acoustic_distance

    for spk, info in order:
        li = _get_spk_li(spk)
        nm = (li.get("name_vi") or "").strip().lower()
        if nm and nm in by_name:
            casting[spk] = dict(by_name[nm])
            continue

        g = li.get("gender")
        if g not in ("male", "female"):
            g = info.get("gender_voice")
        if g not in ("male", "female"):
            g = "male"
        age = li.get("age") if li.get("age") in AGE_PREF else "adult"
        voice_g = "female" if age == "child" else g

        # Lấy đặc trưng F0 đo được từ vocals.wav
        spk_prof = spk_f0_map.get(str(spk)) or spk_f0_map.get(spk) or {}
        spk_f0 = float(spk_prof.get("f0_median") or info.get("f0") or 0.0)
        spk_tone = spk_prof.get("tone") or info.get("tone") or ""

        is_main = (main_spk_by_gender.get(voice_g) == spk)
        role_label = ("Nữ chính" if is_main else "Nữ phụ") if voice_g == "female" else ("Nam chính" if is_main else "Nam phụ")

        pick_base = None
        pick_variant = ""
        engine = "Edge-TTS"

        if is_hybrid:
            # R1: Edge-TTS làm chủ đạo cho nhân vật chính
            lead_cfg = (s.get(f"lead_{voice_g}_voice") or "").strip()
            edge_lead = lead_cfg if lead_cfg in EDGE_TTS_VOICES.get(voice_g, []) else EDGE_TTS_VOICES.get(voice_g, ["Hoài My" if voice_g == "female" else "Nam Minh"])[0]

            if is_main and edge_lead not in used_bases:
                pick_base = edge_lead
                engine = "Edge-TTS"
                pick_variant = "bé" if age == "child" else ""
            else:
                # R1: Tự động mở rộng sang kho giọng VieNeu miền Nam cho nhân vật phụ (3, 4, 5+ người nói)
                engine = "VieNeu"
                vieneu_cands = [v for v in VIENEU_SOUTH_VOICES.get(voice_g, []) if v not in used_bases]

                if vieneu_cands:
                    # R2: Acoustic Distance Pairing - Gán giọng VieNeu có F0 gần với diễn viên gốc nhất
                    pick_base = pair_voice_by_acoustic_distance(spk_f0, gender=voice_g, candidate_voices=vieneu_cands)
                    pick_variant = "bé" if age == "child" else ""
                else:
                    # Khi đã dùng hết giọng gốc VieNeu: Sử dụng biến thể Praat để không trùng lặp giọng
                    all_vn = VIENEU_SOUTH_VOICES.get(voice_g, ["Thục Đoan" if voice_g == "female" else "Thái Sơn"])
                    prefs = AGE_PREF.get(age, ["trẻ", "trầm", "già", "bé"]) if use_var else [""]
                    cand_pairs = [(b, v) for v in prefs for b in all_vn if (b, v) not in used_pairs]
                    if not cand_pairs:
                        cand_pairs = [(b, v) for v in VARIANTS for b in all_vn if (b, v) not in used_pairs]
                    if cand_pairs:
                        pick_base, pick_variant = min(cand_pairs, key=lambda c: (
                            abs(get_voice_profile(c[0]).get("baseline_f0", 125.0) - (spk_f0 if spk_f0 > 0 else 125.0)),
                            prefs.index(c[1]) if c[1] in prefs else 99
                        ))
                    else:
                        pick_base = all_vn[len(casting) % len(all_vn)]
                        pick_variant = f"v{len(casting) + 1}"
        else:
            # Fallback cho chế độ khác (valtec, nano...)
            pools = _pools(s, tts_mode, tl.get("genre", "khac"), available)
            pool = pools.get(voice_g) or (["Thái Sơn"] if voice_g == "male" else ["Thục Đoan"])
            prefs = AGE_PREF.get(age, [""]) if use_var else [""]
            allv = prefs + ([v for v in VARIANTS if v not in prefs and v != "bé"] if use_var else [])
            cands = [(b, v) for v in allv for b in pool if (b, v) not in used_pairs]
            if cands:
                pick_base, pick_variant = min(cands, key=lambda c: (pool.index(c[0]), allv.index(c[1])))
            else:
                pick_base = pool[len(casting) % len(pool)]
                pick_variant = prefs[0] if prefs else ""
            engine = "Valtec" if "valtec" in tts_mode else ("VieNeu" if "nano" in tts_mode or "turbo" in tts_mode else "Edge-TTS")

        used_bases.add(pick_base)
        used_pairs.add((pick_base, pick_variant))
        v_label = label(pick_base, pick_variant)
        used_voices.add(v_label)

        casting[spk] = {
            "voice": v_label,
            "base": pick_base,
            "variant": pick_variant,
            "engine": engine,
            "gender": g,
            "age": age,
            "role": role_label,
            "name": li.get("name_vi", ""),
            "f0": round(spk_f0, 1) if spk_f0 > 0 else None,
            "tone": spk_tone,
        }
        if nm:
            by_name[nm] = casting[spk]

    return casting


def atempo(src, dst, speed: float) -> None:
    ffmpeg(["-i", src, "-filter:a", f"atempo={speed:.4f}", dst])


def run(job: Job) -> None:
    t0 = time.time()
    s = job.settings
    mode = s.get("workflow_mode", "dub")
    if mode == "sub_only":
        log("Chế độ Chỉ VietSub: Tự động khớp mốc thời gian phụ đề tiếng Việt (bỏ qua đọc giọng AI)")
        tr = job.read("transcript.json", {}) or {}
        tl = job.read("translation.json", {}) or {}
        vi_map = {it["id"]: it["vi"] for it in tl.get("lines", [])}
        fit_lines = []
        for ln in tr.get("lines", []):
            st = float(ln["start"])
            en = float(ln["end"])
            dur = max(0.2, en - st)
            vi_text = vi_map.get(ln["id"]) or ln.get("zh", "")
            fit_lines.append({
                "id": ln["id"],
                "spk": ln.get("spk", 0),
                "voice": "goc",
                "start": st,
                "orig_end": en,
                "window": round(dur, 3),
                "dur_raw": round(dur, 3),
                "speed": 1.0,
                "dur": round(dur, 3),
                "overflow": 0.0,
                "file": "",
                "vi": vi_text,
            })
        out_dir = job.p("tts")
        out_dir.mkdir(parents=True, exist_ok=True)
        job.write("fit.json", {"lines": fit_lines, "stats": {"mode": "sub_only", "count": len(fit_lines)}})
        log(f"Khớp phụ đề xong {len(fit_lines)} câu cho chế độ Chỉ VietSub")
        return

    from . import translate as tmod
    mode = s.get("tts_mode", "edgetts")
    clean_mode = str(mode or "").lower().replace("-", "").replace("_", "").strip()
    is_hybrid_mode = clean_mode in ("edgetts", "hybrid", "turbohybrid", "edge")
    tr = job.read("transcript.json")
    tl = job.read("translation.json")
    lines = tr["lines"]
    vi = {l["id"]: l["vi"] for l in tl["lines"]}
    emos = {l["id"]: l.get("emotion", "") for l in tl.get("lines", [])}
    dur_total = tr["duration"]
    max_speed = float(s.get("max_speed", 1.2))
    hard_max = float(s.get("hard_max_speed", 1.45))
    borrow = float(s.get("borrow_gap_s", 1.5))
    default_rate = float(s.get("speech_rate_sps", 4.8))

    try:
        from .scene import get_scene_cuts
        cuts = get_scene_cuts(job)
    except Exception:
        cuts = []

    job.progress(0.02, "nạp model giọng đọc")
    from ..ttsload import load as load_tts
    tts = load_tts(mode, threads(), s.get("tts_precision", "fp32"),
                   voxcpm_model=s.get("voxcpm_model", "openbmb/VoxCPM2"),
                   voxcpm_timesteps=s.get("voxcpm_timesteps", 10))
    sr = int(tts.sample_rate)
    presets = getattr(tts, "_preset_voices", {}) or {}
    available = {name: (v.get("gender") or "") for name, v in presets.items()}
    if not available:
        available = {vid: "" for _, vid in tts.list_preset_voices()}

    out_dir = job.p("tts")
    out_dir.mkdir(exist_ok=True)
    todo = [l for l in lines if vi.get(l["id"])]
    idx = {l["id"]: i for i, l in enumerate(lines)}
    known_rates = read_json(RATES_FILE, {}) or {}

    # 1. Phân tích F0 và năng lượng từ giọng gốc (Pitch & Energy Extraction)
    t_pitch_start = time.time()
    vocals_f = job.p("vocals_16k.wav") if job.p("vocals_16k.wav").exists() else job.p("vocals.wav")
    from ..pitch import analyze_transcript_f0, extract_speaker_f0_profiles, map_pitch_to_prosody
    pitch_profiles = analyze_transcript_f0(vocals_f, lines, tr.get("speakers", {}))
    speaker_f0_profiles = extract_speaker_f0_profiles(vocals_f, lines, tr.get("speakers", {}), precomputed_profiles=pitch_profiles)
    t_pitch_dur = time.time() - t_pitch_start
    tone_counts = Counter(p.get("tone", "") for p in pitch_profiles.values())
    tone_summary = ", ".join(f"{k}: {v}" for k, v in tone_counts.items() if k)
    log(f"Phân tích cao độ F0 xong trong {t_pitch_dur:.2f}s ({len(pitch_profiles)} câu: {tone_summary})")
    try:
        job.write("pitch_analysis.json", pitch_profiles)
    except Exception:
        pass

    # 2. Phân vai lai ghép đa giọng (Hybrid Edge-TTS + VieNeu Casting)
    casting = cast(job, mode, available, speaker_f0_profiles=speaker_f0_profiles)
    log("=" * 80)
    log("BẢNG PHÂN VAI LAI GHÉP (HYBRID MULTI-ENGINE VOICE CASTING):")
    log(f"{'Speaker':<8} | {'Vai trò':<10} | {'Nhân vật':<14} | {'Giọng đọc':<18} | {'Engine':<10} | {'Giới tính':<8} | {'F0 / Tông gốc':<16}")
    log("-" * 80)
    for spk_id, c in casting.items():
        v_name = c['voice']
        eng = c.get('engine', 'Edge-TTS')
        role = c.get('role', 'Vai phụ')
        c_name = c.get('name') or spk_id
        g_name = "Nữ" if c['gender'] == "female" else "Nam"
        f0_str = f"{c['f0']}Hz ({c['tone']})" if c.get('f0') else (c.get('tone') or "N/A")
        log(f"{spk_id:<8} | {role:<10} | {c_name:<14} | {v_name:<18} | {eng:<10} | {g_name:<8} | {f0_str:<16}")
    log("=" * 80)
    summary_parts = [f"{spk}->{c['voice']} [{c.get('engine', 'Edge-TTS')}]" for spk, c in casting.items()]
    log("Phân vai tóm tắt: " + ", ".join(summary_parts))
    log(f"Nạp model & phân vai xong sau {time.time() - t0:.1f}s")

    # Đọc vocals để phục vụ Spectral Match EQ và Zero-shot voice cloning
    orig_vocals = None
    orig_vocals_sr = 16000
    try:
        import soundfile as sf
        if vocals_f.exists():
            orig_vocals, orig_vocals_sr = sf.read(str(vocals_f), dtype="float32")
            if orig_vocals is not None:
                orig_vocals = np.nan_to_num(orig_vocals, nan=0.0, posinf=0.0, neginf=0.0)
                if orig_vocals.ndim > 1:
                    orig_vocals = np.mean(orig_vocals, axis=1)
    except Exception as e:
        log(f"Không thể đọc vocals để Match EQ: {e}")

    # Trích xuất mẫu giọng chuẩn 3-8s của từng nhân vật phục vụ Voice Cloning
    speaker_ref_wavs = {}
    if orig_vocals is not None:
        spk_snippets_dir = out_dir / "spk_refs"
        spk_snippets_dir.mkdir(parents=True, exist_ok=True)
        for spk_id in tr.get("speakers", {}):
            best_line = None
            best_dur = 0.0
            for l_cand in lines:
                if l_cand.get("spk") == spk_id:
                    d_c = float(l_cand["end"]) - float(l_cand["start"])
                    if 3.0 <= d_c <= 8.0 and d_c > best_dur:
                        best_line = l_cand
                        best_dur = d_c
            if not best_line:
                for l_cand in lines:
                    if l_cand.get("spk") == spk_id:
                        d_c = float(l_cand["end"]) - float(l_cand["start"])
                        if d_c > best_dur:
                            best_line = l_cand
                            best_dur = d_c
            if best_line:
                s_idx = max(0, int(float(best_line["start"]) * orig_vocals_sr))
                e_idx = min(len(orig_vocals), int(float(best_line["end"]) * orig_vocals_sr))
                clip = orig_vocals[s_idx:e_idx]
                if len(clip) > int(orig_vocals_sr * 1.0):
                    ref_p = spk_snippets_dir / f"spk_{spk_id}.wav"
                    save_audio(ref_p, clip, orig_vocals_sr)
                    speaker_ref_wavs[spk_id] = str(ref_p)

    def cinfo(ln):
        spk = ln.get("spk")
        info = casting.get(spk) or casting.get(str(spk))
        if not info and str(spk).isdigit():
            info = casting.get(int(spk))
        return info or {"voice": "", "base": None, "variant": "", "engine": "Edge-TTS"}

    def synth(ln, text):
        c = cinfo(ln)
        emo = emos.get(ln["id"], "")
        p_info = pitch_profiles.get(ln["id"]) or pitch_profiles.get(str(ln["id"])) or {}
        prosody = map_pitch_to_prosody(p_info, voice_name=c["voice"], base_voice=c["base"], emotion=emo, safe_volume=True)
        engine = c.get("engine", "Edge-TTS")
        if clean_mode in ("voxcpm", "valtec_zeroshot"):
            ref_wav = speaker_ref_wavs.get(ln["spk"])
            if not ref_wav and clean_mode == "voxcpm":
                sample_dir = paths.LOGS / "giong_mau"
                if sample_dir.exists():
                    for pat in (f"*_{c['voice']}.wav", f"*{c['voice']}*.wav"):
                        found = list(sample_dir.glob(pat))
                        if found:
                            ref_wav = str(found[0])
                            break
            a = tts.infer(normalize_vi(text), voice=c["voice"], ref_wav=ref_wav)
        elif is_hybrid_mode:
            if engine == "VieNeu":
                a = tts.infer(normalize_vi(text), voice=c["base"] or c["voice"], engine="VieNeu",
                              emotion=emo, pitch=prosody["pitch"], rate=prosody["rate"], volume=prosody["volume"],
                              pitch_profile=p_info)
            else:
                a = tts.infer(normalize_vi(text), voice=c["base"] or c["voice"], engine="Edge-TTS",
                              emotion=emo, pitch=prosody["pitch"], rate=prosody["rate"], volume=prosody["volume"],
                              pitch_profile=p_info)
        else:
            a = tts.infer(normalize_vi(text), voice=c["base"] or c["voice"])
        a = trim_silence(np.asarray(a, dtype=np.float32), sr)
        if c.get("variant") and (engine == "VieNeu" or not is_hybrid_mode):
            a = apply_variant(a, sr, c["variant"])

        # KỸ THUẬT SPECTRAL MATCH EQ: Uốn nắn phổ dải tần số của giọng TTS theo dải tần gốc của câu thoại
        voice_match_mode = s.get("voice_match_mode", "match_eq")
        if voice_match_mode != "off" and orig_vocals is not None and len(a) >= int(sr * 0.05):
            try:
                from ..matcheq import apply_match_eq
                st_sample = max(0, int(float(ln["start"]) * orig_vocals_sr))
                en_sample = min(len(orig_vocals), int(float(ln["end"]) * orig_vocals_sr))
                ref_slice = orig_vocals[st_sample:en_sample]
                if len(ref_slice) >= int(orig_vocals_sr * 0.05):
                    a, _ = apply_match_eq(ref_slice, a, sr=sr, ref_sr=orig_vocals_sr, max_gain_db=6.0)
            except Exception:
                pass

        # R2: Chuẩn hóa biên độ đỉnh an toàn (peak headroom -1.5 dBFS) và True Peak <= -1.0 dBFS cho từng clip thoại riêng lẻ
        pk = float(np.max(np.abs(a))) if len(a) > 0 else 0.0
        target_clip_peak = 10.0 ** (-1.5 / 20.0) # ~0.8414
        if pk > target_clip_peak:
            a = a * (target_clip_peak / pk)
        try:
            from ..matcheq import compute_true_peak
            tp_lin, _ = compute_true_peak(a)
            if tp_lin > 0.8910:
                a = a * (0.8910 / (tp_lin + 1e-8))
        except Exception:
            pass
        save_audio(out_dir / f"{ln['id']:04d}.wav", a, sr)
        return len(a) / sr

    def window(i):
        ln = lines[i]
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else dur_total
        if cuts:
            valid_cuts = [c for c in cuts if ln["end"] - 0.1 <= c <= nxt + 0.05]
            if valid_cuts:
                nxt = min(nxt, max(ln["start"] + 0.5, min(valid_cuts) - 0.05))
        return max(0.5, min(nxt - ln["start"] - 0.08, (ln["end"] - ln["start"]) + borrow))

    def rate_for(c, measured):
        return (measured.get(c["voice"]) or known_rates.get(c["voice"]) or known_rates.get(c["base"] or "")
                or default_rate)

    shortened = 0

    def do_shorten(cands, note):
        nonlocal shortened
        if not cands:
            return []
        log(f"{len(cands)} câu quá dài -> rút gọn ({note})")
        try:
            new = tmod.shorten(job, cands)
        except Exception as e:
            log(f"Rút gọn lỗi ({e}) - dùng tăng tốc thay thế")
            return []
        changed = []
        for it in cands:
            nv = new.get(it["id"])
            if nv and vi_syllables(nv) < vi_syllables(it["vi"]):
                vi[it["id"]] = nv
                changed.append(it["id"])
                shortened += 1
        return changed

    # 1) rút gọn TRƯỚC khi đọc những câu chắc chắn quá dài (ước lượng bằng tốc độ đọc đã đo)
    pre = []
    for ln in todo:
        w = window(idx[ln["id"]])
        rate = rate_for(cinfo(ln), {})
        est = vi_syllables(vi[ln["id"]]) / rate
        if est / w > max_speed * 1.15:
            pre.append({"id": ln["id"], "vi": vi[ln["id"]], "max_syll": max(2, int(w * max_speed * rate * 0.95))})
    if pre:
        job.progress(0.04, "rút gọn câu dài")
        do_shorten(pre, "trước khi đọc")

    # 2) đọc tất cả
    raw = {}
    t_tts = time.time()
    for k, ln in enumerate(todo):
        raw[ln["id"]] = synth(ln, vi[ln["id"]])
        time.sleep(0.08)
        job.progress(0.05 + 0.7 * (k + 1) / max(1, len(todo)), f"đọc câu {k + 1}/{len(todo)}")
    speech_sec = sum(raw.values())
    log(f"Đọc {len(todo)} câu, {speech_sec:.1f}s giọng trong {time.time() - t_tts:.1f}s "
        f"(RTF {((time.time() - t_tts) / max(speech_sec, 0.1)):.2f})")

    # tốc độ đọc thực tế từng giọng (âm tiết/giây)
    rates = {}
    for ln in todo:
        d = raw[ln["id"]]
        if d > 0.8:
            rates.setdefault(cinfo(ln)["voice"], []).append(vi_syllables(vi[ln["id"]]) / d)
    rate_of = {v: float(np.median(r)) for v, r in rates.items()}

    # 3) câu vẫn quá dài -> nhờ AI rút gọn rồi đọc lại
    long_ones = []
    for ln in todo:
        w = window(idx[ln["id"]])
        if raw[ln["id"]] / w > max_speed:
            rate = rate_for(cinfo(ln), rate_of)
            long_ones.append({"id": ln["id"], "vi": vi[ln["id"]], "max_syll": max(2, int(w * max_speed * rate * 0.95))})
    if long_ones:
        job.progress(0.78, "rút gọn câu dài")
        for lid in do_shorten(long_ones, "sau khi đọc"):
            raw[lid] = synth(lines[idx[lid]], vi[lid])

    # 4) khớp tốc độ
    fit = []
    job.progress(0.92, "khớp thời gian")
    for ln in todo:
        i = idx[ln["id"]]
        w = window(i)
        d = raw[ln["id"]]
        speed = 1.0
        f = out_dir / f"{ln['id']:04d}.wav"
        if d > w:
            speed = min(d / w, hard_max)
            sp = out_dir / f"{ln['id']:04d}_fit.wav"
            atempo(f, sp, speed)
            f = sp
        final = d / speed
        p_info = pitch_profiles.get(ln["id"]) or pitch_profiles.get(str(ln["id"])) or {}
        prosody = map_pitch_to_prosody(p_info, voice_name=cinfo(ln)["voice"], base_voice=cinfo(ln)["base"], emotion=emos.get(ln["id"], ""), safe_volume=True)
        fit.append({"id": ln["id"], "spk": ln["spk"], "voice": cinfo(ln)["voice"],
                    "engine": cinfo(ln).get("engine", "Edge-TTS"),
                    "start": ln["start"], "orig_end": ln["end"], "window": round(w, 3),
                    "dur_raw": round(d, 3), "speed": round(speed, 3), "dur": round(final, 3),
                    "overflow": round(max(0.0, final - w), 3), "file": f.name, "vi": vi[ln["id"]],
                    "f0": p_info.get("f0") or p_info.get("f0_fallback") or 0.0, "tone": p_info.get("tone", ""),
                    "pitch": prosody.get("pitch", "+0Hz"), "rate": prosody.get("rate", "+0%"),
                    "volume": prosody.get("volume", "+0%")})

    # lưu tốc độ đọc trung bình để lần sau dịch ước lượng độ dài chính xác hơn
    try:
        allr = read_json(RATES_FILE, {}) or {}
        for v, r in rate_of.items():
            old = allr.get(v)
            allr[v] = round(r if old is None else 0.7 * old + 0.3 * r, 3)
        write_json(RATES_FILE, allr)
    except Exception:
        pass

    job.write("fit.json", {"sr": sr, "mode": mode, "casting": casting, "lines": fit, "rates": rate_of,
                           "shortened": shortened, "pre_shortened": len(pre), "seconds": round(time.time() - t0, 1),
                           "speech_seconds": round(speech_sec, 1)})
    sped = sum(1 for f in fit if f["speed"] > 1.0)
    log(f"TTS xong {time.time() - t0:.1f}s: {len(fit)} câu, rút gọn {shortened}, tăng tốc {sped}")
