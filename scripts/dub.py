"""Công cụ điều khiển hệ thống lồng tiếng Douyin -> tiếng Việt (dùng cho agent Antigravity).

Chạy qua scripts/dub.sh (tự tìm dự án + python):  bash scripts/dub.sh <lệnh> [...]
Lệnh:
  where                      đường dẫn dự án, venv, model, thành phẩm; có API key chưa; app có đang chạy
  status [--all] [--id N]    danh sách video: trạng thái, bước, thời gian, lỗi, file ra
  add <link|file> ...        thêm video vào hàng chờ (app web hoặc lệnh run sẽ xử lý)
  run [link|file ...]        xử lý ngay (2 làn song song) và chờ xong; không có link -> config/test_links.txt
  retry <id> [--from BƯỚC]   chạy lại video (từ một bước: download separate transcribe ocr translate tts mix render qc)
  logs <id> [--step BƯỚC] [--tail N]   xem nhật ký + lỗi của video
  settings [khóa=giá_trị ...]          xem / đổi cài đặt (config/settings.json)
  voices [id]                giọng có sẵn + biến thể; có id -> xem phân vai của video đó
  timing                     thời gian trung bình từng bước (để biết chỗ chậm)
  doctor [phần,...]          tự kiểm tra (ffmpeg,gemini,asr,ocr,separate,tts,voices,tts8)
  douyin [mã_video ...]      thử các cách tải Douyin, báo cách nào chạy
  serve                      mở giao diện web http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def _imports():
    from dubber import db, paths  # noqa: F401
    from dubber.steps import STEP_LABELS, STEPS  # noqa: F401
    return db, paths, STEPS, STEP_LABELS


def port_open(port=7860) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def cmd_where(args):
    db, paths, *_ = _imports()
    import shutil
    du = shutil.disk_usage(str(paths.ROOT))
    lock_busy = False
    try:
        import fcntl
        fh = open(paths.WORK / "worker.lock", "a")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            lock_busy = True
    except Exception:
        pass
    info = {
        "du_an": str(paths.ROOT), "python": sys.executable, "model": str(paths.MODELS),
        "thanh_pham": str(paths.OUTPUT), "thu_muc_lam_viec": str(paths.WORK), "dau_vao": str(paths.INPUT_DIR),
        "co_gemini_key": bool(paths.gemini_key()), "giao_dien_web_dang_chay": port_open(),
        "dang_co_bo_xu_ly_chay": lock_busy, "o_cung_trong_gb": round(du.free / 1e9, 1),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


def _fmt_job(j, STEP_LABELS, full=False):
    tim = json.loads(j.get("timings") or "{}")
    tot = sum(tim.values())
    line = (f"#{j['id']:<4} {j['status']:<8} {STEP_LABELS.get(j.get('step') or '', ''):<28} "
            f"{(j.get('note') or '')[:30]:<30} {tot:6.0f}s  {(j.get('title') or j.get('input') or '')[:60]}")
    if full:
        line += f"\n      input: {j.get('input')}"
        if tim:
            line += "\n      thời gian: " + ", ".join(f"{k} {v}s" for k, v in tim.items())
        if j.get("output"):
            line += f"\n      file ra: {j['output']}"
    if j.get("error") and j["status"] in ("failed", "queued"):
        line += f"\n      LỖI: {j['error'].splitlines()[0][:300] if j['error'] else ''}"
    if j.get("issues") and j["issues"] != "[]":
        line += f"\n      Cần xem: {j['issues']}"
    return line


def cmd_status(args):
    db, paths, STEPS, STEP_LABELS = _imports()
    db.init()
    if "--id" in args:
        j = db.get(int(args[args.index("--id") + 1]))
        print(_fmt_job(j, STEP_LABELS, True) if j else "Không có video này")
        return
    jobs = db.listing(1000 if "--all" in args else 30)
    print(f"Tổng: {db.counts()}")
    for j in jobs:
        print(_fmt_job(j, STEP_LABELS, "--all" in args))


def cmd_add(args):
    db, *_ = _imports()
    from dubber.server import split_inputs
    db.init()
    items = split_inputs(" | ".join(args))
    for it in items:
        if db.exists_active(it):
            print(f"Đã có trong hàng chờ: {it}")
        else:
            print(f"Đã thêm #{db.add(it)}: {it}")
    if not port_open():
        print("Lưu ý: giao diện web chưa chạy -> dùng lệnh 'serve' hoặc 'run' để bắt đầu xử lý.")


def cmd_run(args):
    sys.argv = ["dubber.cli", *args]
    from dubber import cli
    cli.main()


def cmd_retry(args):
    db, paths, STEPS, _ = _imports()
    from dubber.worker import reset_from
    db.init()
    jid = int(args[0])
    st = args[args.index("--from") + 1] if "--from" in args else None
    if st and st not in STEPS:
        sys.exit(f"Bước không hợp lệ. Chọn: {' '.join(STEPS)}")
    redo = reset_from(jid, st)
    if redo:
        print(f"Sẽ chạy lại #{jid} các bước: {', '.join(redo)}")
    db.update(jid, status="queued", attempts=0, error="", not_before=0, note="")
    print("Đã đưa vào hàng chờ." + ("" if port_open() else " (Giao diện chưa chạy: dùng 'serve' để xử lý.)"))


def cmd_logs(args):
    db, paths, STEPS, _ = _imports()
    from dubber.worker import job_dir
    jid = int(args[0])
    d = job_dir(jid)
    tail = int(args[args.index("--tail") + 1]) if "--tail" in args else 60
    steps = [args[args.index("--step") + 1]] if "--step" in args else STEPS
    for s in steps:
        f = d / f"log_{s}.txt"
        e = d / f"error_{s}.txt"
        if f.exists():
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"===== {s} ({len(lines)} dòng, xem {tail} dòng cuối) =====")
            print("\n".join(lines[-tail:]))
        if e.exists():
            print(f"!!!!! LỖI {s}: {e.read_text(encoding='utf-8')}")
    print("\nFile trong thư mục video:", ", ".join(sorted(p.name for p in d.iterdir())) if d.exists() else "(chưa có)")


def cmd_settings(args):
    _, paths, *_ = _imports()
    s = paths.load_settings()
    if not args:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return
    for kv in args:
        k, _, v = kv.partition("=")
        if k not in paths.DEFAULT_SETTINGS:
            sys.exit(f"Không có khóa '{k}'. Các khóa: {', '.join(paths.DEFAULT_SETTINGS)}")
        try:
            val = json.loads(v)
        except Exception:
            val = v
        s[k] = val
        print(f"{k} = {json.dumps(val, ensure_ascii=False)}")
    paths.save_settings(s)


def cmd_voices(args):
    _, paths, *_ = _imports()
    from dubber.steps.tts import AGE_PREF, VARIANTS
    print("Giọng gốc miền Nam (VieNeu v3 Turbo):")
    print("  Nam:", ", ".join(paths.SOUTH_MALE))
    print("  Nữ :", ", ".join(paths.SOUTH_FEMALE))
    print("Biến thể (nửa cung, formant):", {k or "gốc": v for k, v in VARIANTS.items()})
    print("Ưu tiên biến thể theo tuổi:", AGE_PREF)
    s = paths.load_settings()
    print("Giọng nữ chính:", s.get("lead_female_voice"), "| nam chính:", s.get("lead_male_voice") or "(theo thể loại)")
    if args:
        from dubber.worker import job_dir
        f = job_dir(int(args[0])) / "fit.json"
        if f.exists():
            c = json.loads(f.read_text(encoding="utf-8")).get("casting", {})
            for spk, v in c.items():
                print(f"  {spk}: {v.get('name') or '?'} ({v.get('gender')}, {v.get('age')}) -> {v.get('voice')}")
        else:
            print("Video này chưa tới bước đọc giọng.")


def cmd_timing(args):
    db, paths, STEPS, STEP_LABELS = _imports()
    db.init()
    rows = [j for j in db.listing(1000) if j["status"] in ("done", "review")]
    if not rows:
        print("Chưa có video nào xong.")
        return
    acc = {s: [] for s in STEPS}
    per_min = []
    from dubber.worker import job_dir
    for j in rows:
        tim = json.loads(j.get("timings") or "{}")
        for s, v in tim.items():
            acc.setdefault(s, []).append(v)
        meta = job_dir(j["id"]) / "meta.json"
        if meta.exists() and tim:
            dur = json.loads(meta.read_text(encoding="utf-8")).get("duration") or 0
            if dur:
                per_min.append(sum(tim.values()) / (dur / 60))
    print(f"{len(rows)} video đã xong. Thời gian trung bình mỗi bước:")
    for s in STEPS:
        if acc.get(s):
            print(f"  {STEP_LABELS[s]:<30} {sum(acc[s]) / len(acc[s]):6.1f}s")
    if per_min:
        print(f"Trung bình: {sum(per_min) / len(per_min):.0f} giây xử lý cho mỗi phút video (cộng dồn các bước, "
              f"chưa tính chạy 2 làn song song)")


def _py_module(mod, *a):
    _, paths, *_ = _imports()
    env = dict(os.environ, PYTHONPATH=str(paths.APP_DIR))
    return subprocess.call([sys.executable, "-m", mod, *a], env=env, cwd=str(paths.APP_DIR))


def cmd_doctor(args):
    parts = args[0] if args else "ffmpeg,gemini,asr,ocr,separate,tts"
    _py_module("dubber.diag")
    _py_module("dubber.selftest", f"--parts={parts}")


def cmd_douyin(args):
    _py_module("dubber.douyin_debug", *args)


def cmd_serve(args):
    if port_open():
        print("Giao diện đã chạy: http://127.0.0.1:7860")
    else:
        _, paths, *_ = _imports()
        env = dict(os.environ, PYTHONPATH=str(paths.APP_DIR))
        log = open(paths.LOGS / "app.log", "a")
        subprocess.Popen(["caffeinate", "-i", sys.executable, "-m", "dubber.server"] if sys.platform == "darwin"
                         else [sys.executable, "-m", "dubber.server"], env=env, cwd=str(paths.APP_DIR),
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(20):
            if port_open():
                break
            time.sleep(0.5)
        print("Đã mở giao diện: http://127.0.0.1:7860 (chạy nền, log ở logs/app.log)")
    if sys.platform == "darwin":
        subprocess.call(["open", "http://127.0.0.1:7860"])


CMDS = {k[4:]: v for k, v in globals().items() if k.startswith("cmd_")}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 1)
    CMDS[sys.argv[1]](sys.argv[2:])
