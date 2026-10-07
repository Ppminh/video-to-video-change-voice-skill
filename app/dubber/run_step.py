"""Chạy MỘT bước cho MỘT video trong tiến trình riêng.

Vì sao chạy riêng: mỗi bước nạp một model AI. Chạy trong tiến trình con thì khi xong,
macOS thu hồi toàn bộ RAM -> máy 8GB không bị đầy RAM dần.

Dùng: python -m dubber.run_step <tên_bước> <thư_mục_job>
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback

from .utils import Job, log


def main() -> int:
    step, job_dir = sys.argv[1], sys.argv[2]
    job = Job(job_dir)
    t0 = time.time()
    log(f"=== Bước {step} bắt đầu")
    try:
        mod = importlib.import_module(f"dubber.steps.{step}")
        mod.run(job)
    except Exception as e:
        traceback.print_exc()
        (job.p(f"error_{step}.txt")).write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
        log(f"=== Bước {step} LỖI sau {time.time() - t0:.1f}s: {e}")
        return 1
    (job.p(f".done_{step}")).write_text(f"{time.time() - t0:.1f}", encoding="utf-8")
    log(f"=== Bước {step} xong sau {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
