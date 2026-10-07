"""Kiểm tra các cách tải Douyin khi bị lỗi. Kết quả: logs/douyin_debug/report.txt

Dùng: python -m dubber.douyin_debug [mã_video ...]
Thử từng cách (open / feed / page / share), báo cách nào lấy được link video và tải thử 1MB đầu.
"""
from __future__ import annotations

import sys
import time

import requests

from . import paths
from .steps import download as dl


def test(vid: str) -> list[str]:
    out = [f"== Video {vid}"]
    sess = requests.Session()
    methods = [("open", lambda: dl.fetch_info_open(vid, sess)),
               ("feed", lambda: dl.fetch_info_feed(vid, sess)),
               ("page", lambda: dl.fetch_info_page(vid)[0]),
               ("share", lambda: dl.fetch_info(vid, sess))]
    for name, fn in methods:
        t0 = time.time()
        try:
            info = fn()
            r = sess.get(info["play_url"], headers=dl.DL_HEADERS, stream=True, timeout=30, allow_redirects=True)
            got = 0
            for chunk in r.iter_content(1 << 16):
                got += len(chunk)
                if got >= 1 << 20:
                    break
            r.close()
            out.append(f"[OK ] {name}: {time.time() - t0:.1f}s, HTTP {r.status_code}, tải thử {got // 1024}KB, "
                       f"mô tả: {info['desc'][:50]!r}")
        except Exception as e:
            out.append(f"[LỖI] {name}: {time.time() - t0:.1f}s, {str(e)[:300]}")
        print(out[-1], flush=True)
    return out


def main():
    ids = sys.argv[1:] or ["7682470240348933427", "7666397642670402297"]
    d = paths.LOGS / "douyin_debug"
    d.mkdir(exist_ok=True)
    lines = [f"KIỂM TRA TẢI DOUYIN {time.strftime('%Y-%m-%d %H:%M')}"]
    for v in ids:
        lines += test(v)
    (d / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("XONG -> logs/douyin_debug/report.txt")


if __name__ == "__main__":
    main()
