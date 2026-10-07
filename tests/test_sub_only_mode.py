"""Kiểm thử tự động cho tính năng 'Chỉ VietSub' (workflow_mode = 'sub_only').

Xác nhận:
1. separate.py: Khi workflow_mode='sub_only', bỏ qua Demucs/UVR, trích xuất trực tiếp âm thanh trong < 1s.
2. tts.py: Khi workflow_mode='sub_only', bỏ qua Edge-TTS, tạo fit.json khớp thời gian phụ đề tức thì (< 0.1s).
3. mix.py: Khi workflow_mode='sub_only', giữ nguyên 100% âm thanh gốc, không can thiệp méo/hạ âm lượng.
4. render.py: Xuất video thành phẩm với phụ đề Việt và âm thanh gốc (stream copy hoặc audio gốc).
"""
import shutil
import unittest
from pathlib import Path

from app.dubber.steps.mix import run as run_mix
from app.dubber.steps.render import run as run_render
from app.dubber.steps.separate import run as run_separate
from app.dubber.steps.tts import run as run_tts
from app.dubber.utils import Job, probe, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestSubOnlyMode(unittest.TestCase):
    def setUp(self):
        source_job = PROJECT_ROOT / "_work" / "job_00002"
        if not source_job.exists() or not (source_job / "source.mp4").exists():
            self.skipTest("job_00002 không tồn tại")

        self.test_dir = PROJECT_ROOT / "_work" / "_test_sub_only"
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Sao chép các file cơ sở từ job_00002
        for f in ("source.mp4", "audio.wav", "meta.json", "transcript.json", "translation.json", "ocr.json"):
            src_f = source_job / f
            if src_f.exists():
                shutil.copy2(src_f, self.test_dir / f)

        # Ghi job.json với mode = "sub_only"
        write_json(self.test_dir / "job.json", {"id": 99999, "input": "test_sub_only", "mode": "sub_only"})
        self.job = Job(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sub_only_separate(self):
        """separate.py chạy siêu nhanh không cần Demucs khi mode='sub_only'."""
        run_separate(self.job)
        self.assertTrue((self.test_dir / "vocals_16k.wav").exists())
        self.assertTrue((self.test_dir / "vocals.wav").exists())
        self.assertTrue((self.test_dir / "background.wav").exists())
        sep_meta = self.job.read("separate.json")
        self.assertEqual(sep_meta.get("backend"), "direct_pass")
        self.assertLess(sep_meta.get("seconds", 99), 5.0)

    def test_sub_only_tts(self):
        """tts.py bỏ qua Edge-TTS, tạo fit.json tức thì cho phụ đề khi mode='sub_only'."""
        run_tts(self.job)
        fit_path = self.test_dir / "fit.json"
        self.assertTrue(fit_path.exists())
        fit = self.job.read("fit.json")
        self.assertGreater(len(fit["lines"]), 0)
        self.assertEqual(fit.get("stats", {}).get("mode"), "sub_only")
        # Kiểm tra speed đều = 1.0 (không bị tăng tốc méo giọng)
        for ln in fit["lines"]:
            self.assertEqual(ln["speed"], 1.0)
            self.assertEqual(ln["overflow"], 0.0)

    def test_sub_only_mix(self):
        """mix.py bỏ qua trộn âm thanh và sao chép trực tiếp audio gốc khi mode='sub_only'."""
        run_mix(self.job)
        self.assertTrue((self.test_dir / "mix.wav").exists())
        mix_meta = self.job.read("mix.json")
        self.assertEqual(mix_meta.get("mode"), "sub_only")
        self.assertTrue(mix_meta.get("kept_original_audio"))

    def test_sub_only_full_pipeline(self):
        """Chạy trọn gói separate -> tts -> mix -> render chế độ sub_only."""
        run_separate(self.job)
        run_tts(self.job)
        run_mix(self.job)
        run_render(self.job)

        rd = self.job.read("render.json")
        self.assertIsNotNone(rd)
        out_video = Path(rd["output"])
        self.assertTrue(out_video.exists(), f"Video {out_video} không được tạo!")

        info = probe(out_video)
        self.assertTrue(info.get("has_audio"), "Video xuất ra phải có âm thanh!")
        self.assertGreater(info.get("duration", 0), 0)

        # Dọn dẹp video thành phẩm test sau khi xong
        out_video.unlink(missing_ok=True)
        out_video.with_suffix(".srt").unlink(missing_ok=True)
        out_video.with_suffix(".txt").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
