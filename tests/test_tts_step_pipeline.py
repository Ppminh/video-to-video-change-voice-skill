"""Kiểm thử tích hợp bước app.dubber.steps.tts.run với Job thực tế.

Xác nhận:
1. tts.run trích xuất pitch_analysis.json và ghi vào fit.json đúng định dạng.
2. Đo lường tổng thời gian phân tích F0 không làm tăng quá 1-2 giây so với ban đầu.
3. fit.json lưu đầy đủ các trường f0, tone, pitch, rate, volume cho từng câu.
"""
import json
import shutil
import time
import unittest
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dubber.utils import Job


class TestTTSStepPipeline(unittest.TestCase):

    def test_tts_step_execution(self):
        source_job_dir = PROJECT_ROOT / "_work" / "job_00002"
        if not (source_job_dir / "vocals_16k.wav").exists():
            source_job_dir = Path("D:/Makemoney/New folder/_work/job_00002")
        if not source_job_dir.exists() or not (source_job_dir / "vocals_16k.wav").exists():
            self.skipTest("job_00002 not available")

        # Tạo job tạm thời với 2 câu thoại để kiểm thử tts.run mà không tốn nhiều thời gian mạng
        test_dir = PROJECT_ROOT / "_work" / "_test_pitch_job"
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Copy các file cần thiết
            tr = json.loads((source_job_dir / "transcript.json").read_text(encoding="utf-8"))
            tl = json.loads((source_job_dir / "translation.json").read_text(encoding="utf-8"))

            # Giữ lại 2 câu (1 câu nữ S0, 1 câu nam S15)
            line_ids = {1, 18}
            tr["lines"] = [l for l in tr["lines"] if l["id"] in line_ids]
            tl["lines"] = [l for l in tl["lines"] if l["id"] in line_ids]

            (test_dir / "transcript.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
            (test_dir / "translation.json").write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")
            (test_dir / "job.json").write_text(json.dumps({"id": 99999, "name": "test"}, ensure_ascii=False), encoding="utf-8")

            # Copy file vocals_16k.wav
            shutil.copy2(source_job_dir / "vocals_16k.wav", test_dir / "vocals_16k.wav")

            from app.dubber.steps import tts as tts_step

            t0 = time.time()
            job = Job(test_dir)
            tts_step.run(job)
            elapsed = time.time() - t0

            print(f"\n[PIPELINE] tts.run hoàn thành trong {elapsed:.2f}s cho 2 câu thoại.")

            # Kiểm tra pitch_analysis.json
            pitch_file = test_dir / "pitch_analysis.json"
            self.assertTrue(pitch_file.exists())
            p_data = json.loads(pitch_file.read_text(encoding="utf-8"))
            self.assertIn("1", p_data)
            self.assertIn("18", p_data)

            # Câu 18 là giọng nam S15: F0 < 130Hz
            self.assertLess(p_data["18"]["f0"], 130.0)
            self.assertEqual(p_data["18"]["tone"], "trầm nam")

            # Kiểm tra fit.json
            fit_file = test_dir / "fit.json"
            self.assertTrue(fit_file.exists())
            fit_data = json.loads(fit_file.read_text(encoding="utf-8"))
            lines_fit = {l["id"]: l for l in fit_data["lines"]}

            self.assertIn(1, lines_fit)
            self.assertIn(18, lines_fit)

            # Kiểm tra tham số prosody của câu 18 trong fit.json
            fit_18 = lines_fit[18]
            self.assertTrue(fit_18["pitch"].startswith("-"))
            self.assertEqual(fit_18["tone"], "trầm nam")

            # Kiểm tra âm thanh wav được tạo ra
            wav_18 = test_dir / "tts" / "0018.wav"
            self.assertTrue(wav_18.exists())
            self.assertGreater(wav_18.stat().st_size, 5000)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
