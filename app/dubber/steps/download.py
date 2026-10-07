"""B1. Tải video Douyin (bản không logo) hoặc nhận file có sẵn.

Cách hoạt động:
1. Lấy mã video (aweme_id) từ link (link rút gọn v.douyin.com sẽ được mở để lấy link thật).
2. Hỏi máy chủ Douyin thông tin video, thử lần lượt (cách nào được thì dừng):
   a) "Cửa open.douyin.com": gọi API chi tiết video, giả làm trang open.douyin.com -> không cần
      chữ ký hay cookie. Nhanh nhất (~1 giây). (Cách các dự án Diana, nonebot-parser đang dùng 9/2026)
   b) API của ứng dụng điện thoại (feed) -> trả về danh sách, lọc đúng video.
   c) Chrome chạy ẩn với hồ sơ riêng lưu lâu dài: mở douyin.com để nhận cookie khách, rồi gọi API
      NGAY TRONG trang -> mã bảo vệ của Douyin tự ký các tham số (a_bogus...). Chắc chắn nhất nhưng chậm.
   d) Cách cũ: trang chia sẻ, yt-dlp.
3. Chọn link 'play' chất lượng cao nhất (bản không logo; 'playwm' là bản có logo).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import requests

from ..utils import Job, ffmpeg, log, probe

UA_MOBILE = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 "
             "(KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1")
HEADERS = {"User-Agent": UA_MOBILE, "Referer": "https://www.douyin.com/", "Accept-Language": "zh-CN,zh;q=0.9"}


def extract_url(text: str) -> str | None:
    m = re.search(r"https?://[^\s，。,一-鿿]+", text or "")
    return m.group(0) if m else None


def _find_id(s: str) -> str | None:
    for pat in (r"/(?:video|note|slides)/(\d{8,})", r"modal_id=(\d{8,})", r"aweme_id=(\d{8,})", r"item_ids=(\d{8,})"):
        m = re.search(pat, s or "")
        if m:
            return m.group(1)
    return None


def resolve_id(url: str, sess: requests.Session) -> str:
    vid = _find_id(url)
    if vid:
        return vid
    r = sess.get(url, headers=HEADERS, allow_redirects=True, timeout=20)
    for u in [h.headers.get("Location", "") for h in r.history] + [r.url]:
        vid = _find_id(u)
        if vid:
            return vid
    vid = _find_id(r.text[:20000])
    if vid:
        return vid
    raise RuntimeError(f"Không lấy được mã video từ link: {url}")


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def fetch_info(aweme_id: str, sess: requests.Session) -> dict:
    url = f"https://www.iesdouyin.com/share/video/{aweme_id}/"
    r = sess.get(url, headers=HEADERS, timeout=20)
    m = re.search(r"window\._ROUTER_DATA\s*=\s*(\{.*?\})\s*</script>", r.text, re.S)
    if not m:
        raise RuntimeError("Trang chia sẻ Douyin không có dữ liệu video (có thể Douyin đã đổi cách hiển thị)")
    data = json.loads(m.group(1))
    item = None
    for d in _walk(data):
        if isinstance(d.get("item_list"), list) and d["item_list"]:
            item = d["item_list"][0]
            break
    if not item:
        raise RuntimeError("Không tìm thấy item_list trong dữ liệu Douyin")
    video = item.get("video") or {}
    urls = (video.get("play_addr") or {}).get("url_list") or []
    if not urls:
        raise RuntimeError("Video không có link phát (có thể là bài ảnh hoặc bị giới hạn)")
    play = urls[0].replace("playwm", "play")
    return {
        "id": aweme_id,
        "desc": item.get("desc", ""),
        "author": (item.get("author") or {}).get("nickname", ""),
        "play_url": play,
        "duration_ms": video.get("duration") or 0,
    }

UA_PC = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/146.0.0.0 Safari/537.36")
UA_APP = ("com.ss.android.ugc.aweme/300904 (Linux; U; Android 12; zh_CN; SM-G9730; Build/SQ3A.220705.004; "
          "Cronet/TTNetVersion:2e47e0ce 2022-05-19)")
DL_HEADERS = {"User-Agent": UA_PC, "Referer": "https://www.douyin.com/"}
DETAIL_Q = ("/aweme/v1/web/aweme/detail/?device_platform=webapp&aid=6383&channel=channel_pc_web&pc_client_type=1"
            "&version_code=290100&version_name=29.1.0&cookie_enabled=true&platform=PC&aweme_id=")


def _best_play_url(video: dict) -> str | None:
    """Chọn link video tốt nhất (không logo) từ dữ liệu aweme."""
    cands = []
    for br in video.get("bit_rate") or []:
        pa = br.get("play_addr") or {}
        urls = pa.get("url_list") or []
        h = int(pa.get("height") or 0)
        if urls and h <= 1920:
            cands.append((int(br.get("bit_rate") or 0), urls[0]))
    if cands:
        return max(cands)[1]
    urls = (video.get("play_addr") or {}).get("url_list") or []
    return urls[0].replace("playwm", "play") if urls else None


def _info_from_detail(aweme_id: str, det: dict) -> dict:
    video = det.get("video") or {}
    play = _best_play_url(video)
    if not play:
        uri = (video.get("play_addr") or {}).get("uri") or ""
        if uri.startswith("http"):
            play = uri
        elif uri:
            play = f"https://aweme.snssdk.com/aweme/v1/play/?video_id={uri}&ratio=1080p&line=0"
    if not play:
        raise RuntimeError("Video không có link phát (có thể là bài đăng ảnh)")
    return {"id": aweme_id, "desc": det.get("desc", ""), "author": (det.get("author") or {}).get("nickname", ""),
            "play_url": play, "duration_ms": video.get("duration") or 0}


def fetch_info_open(aweme_id: str, sess: requests.Session) -> dict:
    """Cách a: API chi tiết video, giả làm trang open.douyin.com (không cần chữ ký)."""
    r = sess.get("https://www.douyin.com/aweme/v1/web/aweme/detail/", params={"aweme_id": aweme_id, "aid": "6383"},
                 headers={"User-Agent": UA_PC, "Origin": "https://open.douyin.com", "Referer": "https://open.douyin.com/",
                          "Accept": "application/json, text/plain, */*"}, timeout=20)
    if not r.text.strip():
        raise RuntimeError(f"HTTP {r.status_code}, trả về rỗng")
    data = r.json()
    det = data.get("aweme_detail")
    if not det:
        reason = (data.get("filter_detail") or {}).get("filter_reason") or data.get("status_msg") or data.get("status_code")
        raise RuntimeError(f"không có dữ liệu video ({reason})")
    return _info_from_detail(aweme_id, det)


def fetch_info_feed(aweme_id: str, sess: requests.Session) -> dict:
    """Cách b: API của ứng dụng điện thoại."""
    last = None
    for host in ("api5-normal-c-hl.amemv.com", "aweme.snssdk.com"):
        try:
            r = sess.get(f"https://{host}/aweme/v1/feed/", params={"aweme_id": aweme_id, "aid": "1128"},
                         headers={"User-Agent": UA_APP}, timeout=20)
            for it in (r.json().get("aweme_list") or []):
                if str(it.get("aweme_id")) == str(aweme_id):
                    return _info_from_detail(aweme_id, it)
            last = "không có video này trong kết quả"
        except Exception as e:
            last = e
    raise RuntimeError(f"feed: {last}")


def fetch_info_page(aweme_id: str) -> tuple[dict, list]:
    """Cách c: Chrome ẩn, hồ sơ riêng lưu lâu dài (~/.douyin_dubber/douyin_profile, không phải Chrome của bạn).
    Gọi API từ BÊN TRONG trang douyin.com -> mã bảo vệ của Douyin tự gắn chữ ký vào yêu cầu."""
    from playwright.sync_api import sync_playwright
    from .. import paths
    profile = paths.APPHOME / "douyin_profile"
    profile.mkdir(parents=True, exist_ok=True)
    got = {}

    def on_resp(r):
        if "aweme_detail" in got or "/aweme/v1/web/aweme/detail" not in r.url:
            return
        try:
            det = (r.json() or {}).get("aweme_detail")
            if det:
                got["aweme_detail"] = det
        except Exception:
            pass

    with sync_playwright() as p:
        ctx = None
        for kw in ({"channel": "chrome"}, {}):
            try:
                ctx = p.chromium.launch_persistent_context(
                    str(profile), headless=True, args=["--disable-blink-features=AutomationControlled"],
                    user_agent=UA_PC, locale="zh-CN", viewport={"width": 1280, "height": 800}, **kw)
                break
            except Exception as e:
                log(f"Không mở được trình duyệt {kw}: {e}")
        if ctx is None:
            raise RuntimeError("Không mở được Chrome để tải Douyin")
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.on("response", on_resp)
            page.goto("https://www.douyin.com/", wait_until="domcontentloaded", timeout=60000)
            for _ in range(20):                       # chờ cookie khách (ttwid, s_v_web_id)
                names = {c["name"] for c in ctx.cookies()}
                if {"ttwid", "s_v_web_id"} <= names:
                    break
                page.wait_for_timeout(500)
            page.wait_for_timeout(1500)               # chờ mã bảo vệ của trang sẵn sàng
            for attempt in range(3):
                try:
                    txt = page.evaluate("async u => { const r = await fetch(u, {credentials: 'include'});"
                                        " return await r.text(); }", DETAIL_Q + aweme_id)
                    det = (json.loads(txt) if txt and txt.strip().startswith("{") else {}).get("aweme_detail")
                    if det:
                        got["aweme_detail"] = det
                        break
                except Exception as e:
                    log(f"Gọi API trong trang lỗi: {e}")
                page.wait_for_timeout(1500 * (attempt + 1))
            if "aweme_detail" not in got:             # dự phòng: mở trang video, đọc dữ liệu trang tự tải
                page.goto(f"https://www.douyin.com/video/{aweme_id}", wait_until="domcontentloaded", timeout=60000)
                for _ in range(50):
                    if "aweme_detail" in got:
                        break
                    page.wait_for_timeout(500)
            cookies = ctx.cookies()
        finally:
            ctx.close()
    det = got.get("aweme_detail")
    if not det:
        raise RuntimeError("Chrome mở được Douyin nhưng không lấy được dữ liệu video (có thể bị bắt xác minh)")
    return _info_from_detail(aweme_id, det), cookies


def fetch_info_browser(aweme_id: str) -> tuple[dict, list]:
    """Mở trang video bằng Google Chrome chạy ẩn (hồ sơ tạm, không dùng tài khoản của bạn).
    JavaScript của Douyin tự gọi API lấy dữ liệu -> ta đọc câu trả lời đó để lấy link video."""
    from playwright.sync_api import sync_playwright
    got = {}

    def on_resp(r):
        u = r.url
        if "aweme_detail" in got:
            return
        if "/aweme/v1/web/aweme/detail" in u or "/web/api/v2/aweme/iteminfo" in u:
            try:
                data = r.json()
                det = data.get("aweme_detail") or (data.get("item_list") or [None])[0]
                if det:
                    got["aweme_detail"] = det
            except Exception:
                pass

    with sync_playwright() as p:
        browser = None
        for kw in ({"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"], **kw)
                break
            except Exception as e:
                log(f"Không mở được trình duyệt {kw}: {e}")
        if browser is None:
            raise RuntimeError("Không mở được Chrome để tải Douyin")
        ctx = browser.new_context(user_agent=UA_PC, locale="zh-CN", viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.on("response", on_resp)
        page.goto(f"https://www.douyin.com/video/{aweme_id}", wait_until="domcontentloaded", timeout=60000)
        for _ in range(60):
            if "aweme_detail" in got:
                break
            page.wait_for_timeout(500)
        cookies = ctx.cookies()
        browser.close()
    det = got.get("aweme_detail")
    if not det:
        raise RuntimeError("Chrome mở được trang nhưng không thấy dữ liệu video (có thể Douyin bắt xác minh)")
    play = _best_play_url(det.get("video") or {})
    if not play:
        raise RuntimeError("Dữ liệu video không có link phát")
    info = {"id": aweme_id, "desc": det.get("desc", ""), "author": (det.get("author") or {}).get("nickname", ""),
            "play_url": play, "duration_ms": (det.get("video") or {}).get("duration") or 0}
    return info, cookies


def download_file(url: str, dest: Path, sess: requests.Session, headers: dict | None = None) -> None:
    with sess.get(url, headers=headers or HEADERS, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        n = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                n += len(chunk)
    if n < 100_000:
        raise RuntimeError(f"File tải về quá nhỏ ({n} byte) - có thể bị chặn")
    tmp.rename(dest)


def download_ytdlp(url: str, dest: Path) -> dict:
    import yt_dlp
    opts = {"outtmpl": str(dest.with_suffix(".%(ext)s")), "format": "best[ext=mp4]/best", "quiet": True,
            "noprogress": True, "merge_output_format": "mp4"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    for f in dest.parent.glob(dest.stem + ".*"):
        if f.suffix in (".mp4", ".mkv", ".webm") and f != dest:
            f.rename(dest)
    return {"desc": info.get("title", ""), "author": info.get("uploader", ""), "id": info.get("id", "")}


def run(job: Job) -> None:
    src = job.p("source.mp4")
    meta = job.read("meta.json", {}) or {}
    inp = job.info.get("input", "")
    if not src.exists():
        local = Path(inp)
        if inp and local.exists():
            log(f"Dùng file có sẵn: {local.name}")
            shutil.copy2(local, src)
            meta.update({"desc": local.stem, "id": local.stem, "source": "file"})
        else:
            url = extract_url(inp)
            if not url:
                raise RuntimeError(f"Không thấy link trong: {inp[:100]}")
            sess = requests.Session()
            vid = resolve_id(url, sess)
            errors = []
            methods = [
                ("open", lambda: (fetch_info_open(vid, sess), [])),
                ("feed", lambda: (fetch_info_feed(vid, sess), [])),
                ("page", lambda: fetch_info_page(vid)),
                ("share", lambda: (fetch_info(vid, sess), [])),
            ]
            for name, fn in methods:
                try:
                    info, cookies = fn()
                    for c in cookies:
                        sess.cookies.set(c["name"], c["value"], domain=c.get("domain"))
                    log(f"Douyin {vid} (cách {name}): {info['desc'][:60]}")
                    download_file(info["play_url"], src, sess, headers=DL_HEADERS)
                    meta.update(info)
                    meta["source"] = name
                    break
                except Exception as e:
                    errors.append(f"{name}: {str(e)[:200]}")
                    log(f"Cách {name} lỗi: {e}")
            if not src.exists():
                try:
                    info = download_ytdlp(f"https://www.douyin.com/video/{vid}", src)
                    meta.update(info)
                    meta["source"] = "yt-dlp"
                except Exception as e:
                    errors.append(f"yt-dlp: {str(e)[:200]}")
                    raise RuntimeError("Không tải được video. " + " | ".join(errors)[:900] +
                                       " -> Có thể tải tay rồi bỏ file vào thư mục DAU_VAO")
            meta["url"] = url
    info = probe(src)
    if not info.get("has_audio"):
        raise RuntimeError("Video không có âm thanh")
    meta.update({k: info[k] for k in ("duration", "width", "height", "fps", "rotation") if k in info})
    if abs(int(info.get("rotation") or 0)) in (90, 270):     # video quay dọc bằng cờ xoay -> đổi rộng/cao
        meta["width"], meta["height"] = info["height"], info["width"]
    job.write("meta.json", meta)
    # Âm thanh gốc 44.1kHz stereo cho bước tách nhạc
    ffmpeg(["-i", src, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", job.p("audio.wav")])
    log(f"Đã tải: {info.get('width')}x{info.get('height')}, {info['duration']:.1f}s")
