import base64
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from dubber import ttsload
from dubber.steps import tts as step_tts
from dubber.utils import Job


class _MockColabHandler(BaseHTTPRequestHandler):
    last_payload = None

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"ok": True, "device": "Tesla T4", "model": "k2-fsa/OmniVoice"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/synthesize":
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            payload = json.loads(raw.decode("utf-8"))
            _MockColabHandler.last_payload = payload

            dur = float(payload.get("duration") or 1.5)
            sr = 24000
            t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
            wav = 0.25 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            resp = json.dumps({"audio_b64": b64, "sr": sr}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        self.send_response(404)
        self.end_headers()


class TestOmniVoiceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = HTTPServer(("127.0.0.1", 0), _MockColabHandler)
        cls.port = cls.httpd.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_omnivoice_wrapper_remote_colab(self):
        with tempfile.TemporaryDirectory() as td:
            ref_path = Path(td) / "actor_zh.wav"
            sr = 24000
            t = np.linspace(0, 3.2, int(sr * 3.2), endpoint=False, dtype=np.float32)
            sf.write(ref_path, 0.3 * np.sin(2 * np.pi * 180.0 * t), sr)

            fake_settings = {
                "tts_mode": "omnivoice",
                "omnivoice_url": self.base_url,
                "omnivoice_model": "k2-fsa/OmniVoice",
                "omnivoice_steps": 32,
            }
            with patch("dubber.paths.load_settings", return_value=fake_settings):
                engine = ttsload.OmniVoiceWrapper()
                out_wav = engine.infer(
                    "Xin chào các bạn nha",
                    voice="Gốc (OmniVoice Clone)",
                    ref_wav=str(ref_path),
                    ref_text="大家好啊",
                    duration=2.0,
                    speed=1.0,
                )

            self.assertEqual(engine.sample_rate, 24000)
            self.assertGreater(len(out_wav), 24000)
            p = _MockColabHandler.last_payload
            self.assertIsNotNone(p)
            self.assertEqual(p["language"], "vi")
            self.assertEqual(p["ref_text"], "大家好啊")
            self.assertEqual(p["model_id"], "k2-fsa/OmniVoice")
            self.assertAlmostEqual(float(p["duration"]), 2.0, places=2)
            self.assertTrue(len(p.get("ref_audio_b64", "")) > 100)

    def test_tts_step_omnivoice_utterance_cloning(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            sr = 24000
            t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False, dtype=np.float32)
            vocals = (0.35 * np.sin(2 * np.pi * 210.0 * t)).astype(np.float32)
            sf.write(work / "vocals.wav", vocals, sr)
            sf.write(work / "vocals_16k.wav", vocals[::3], 16000)

            tr_data = {
                "duration": 5.0,
                "speakers": {"S0": {"gender": "female", "age": "young"}},
                "lines": [
                    {
                        "id": 1,
                        "spk": "S0",
                        "start": 0.4,
                        "end": 3.4,
                        "zh": "今天天气真的非常好",
                    }
                ],
            }
            tl_data = {
                "genre": "default",
                "characters": {"S0": {"gender": "female", "age": "young", "role": "lead", "name_vi": "Tiểu Mỹ"}},
                "lines": [
                    {
                        "id": 1,
                        "vi": "Hôm nay thời tiết thiệt là đẹp quá đi",
                        "emotion": "happy",
                    }
                ],
            }
            (work / "transcript.json").write_text(json.dumps(tr_data, ensure_ascii=False), encoding="utf-8")
            (work / "translation.json").write_text(json.dumps(tl_data, ensure_ascii=False), encoding="utf-8")
            (work / "job.json").write_text(json.dumps({"id": 88888, "name": "test_omni"}, ensure_ascii=False), encoding="utf-8")

            fake_settings = {
                "workflow_mode": "dub",
                "tts_mode": "omnivoice",
                "omnivoice_url": self.base_url,
                "omnivoice_model": "k2-fsa/OmniVoice",
                "omnivoice_steps": 32,
                "voice_match_mode": "match_eq",
                "voice_variants": True,
                "lead_female_voice": "Hoài My",
                "lead_male_voice": "Nam Minh",
                "max_speed": 1.2,
                "hard_max_speed": 1.45,
                "borrow_gap_s": 1.5,
                "speech_rate_sps": 4.8,
            }
            with patch("dubber.paths.load_settings", return_value=fake_settings):
                job = Job(work)
                job.settings = fake_settings
                step_tts.run(job)

            fit = json.loads((work / "fit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(fit["lines"]), 1)
            wav_file = work / "tts" / "0001.wav"
            self.assertTrue(wav_file.exists())
            # Câu thoại dài 3.0s (>= 2.4s) nên phải cắt riêng utt_0001.wav kèm chữ Hán gốc
            self.assertTrue((work / "tts" / "spk_refs" / "utt_0001.wav").exists())
            self.assertEqual(_MockColabHandler.last_payload["ref_text"], "今天天气真的非常好")


if __name__ == "__main__":
    unittest.main()
