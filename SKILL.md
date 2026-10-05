---
name: video-to-video-change-voice
description: Đổi giọng nói video sang video, lồng tiếng tự động video tiếng Trung (Douyin) sang tiếng Việt giọng miền Nam trên MacBook M1 8GB, miễn phí. Dùng khi người dùng dán link Douyin hoặc gửi file video cần đổi giọng / lồng tiếng Việt; hỏi tiến độ, video lỗi, đổi giọng, chỉnh chất lượng hay tốc độ; muốn tối ưu, sửa hoặc cài lại hệ thống lồng tiếng.
---

# Lồng tiếng Douyin → tiếng Việt

Skill này giúp bạn (agent) vận hành, sửa lỗi và cải tiến một hệ thống đã chạy được trên Mac của người dùng. Hệ thống tự làm 9 bước: tải video không logo → tách giọng khỏi nhạc nền → nghe tiếng Trung và phân biệt người nói → dò phụ đề chữ Trung trên hình → AI dịch Việt hóa → đọc giọng Việt theo từng nhân vật và khớp thời gian → trộn với nhạc nền gốc → làm mờ chữ Trung, đè phụ đề Việt, xuất mp4 → kiểm tra chất lượng.

## 1. Cách làm việc với người dùng (BẮT BUỘC)
- Người dùng nói tiếng Việt và **không biết code**. Luôn trả lời bằng tiếng Việt, **ngắn gọn, dễ hiểu**.
- **Mỗi hành động đều giải thích**: đang làm gì, vì sao, và kỹ thuật đó hoạt động thế nào. Ví dụ: "Mình tách giọng bằng demucs: model AI nhìn bản đồ tần số âm thanh để biết phần nào là giọng người". Giải thích 1–2 câu, không viết dài dòng.
- Tự làm luôn, không bắt người dùng gõ lệnh. Bạn có terminal, hãy chạy lệnh rồi báo kết quả. Chỉ nhờ người dùng khi bắt buộc phải có họ, ví dụ bấm "Cho phép" khi macOS hỏi quyền.
- Câu hỏi nào sai thì tốn kém (xóa dữ liệu, đổi cách làm lớn, tốn tiền) thì hỏi trước. Việc dễ làm lại thì cứ làm rồi báo.
- Ngân sách: **miễn phí**, tối đa 20.000đ/tháng. Chỉ dùng tài khoản Google (Gemini API miễn phí, Colab). Không đề xuất dịch vụ trả phí khi người dùng chưa đồng ý.
- **Bảo mật:** API key nằm trong `config/.env`. Không bao giờ in, đọc to hay chép key vào chat, log hay tài liệu. Chỉ kiểm tra "có key hay chưa".
- Báo kết quả: file nằm ở đâu, mất bao lâu, có gì cần xem. Không kể lể từng bước.

## 2. Yêu cầu sản phẩm (đã chốt với người dùng)
- **Máy:** MacBook Pro M1 8GB. Hệ thống phải chạy trong khoảng 4GB RAM để máy không đơ. Cho phép chạy qua đêm.
- **Đầu vào:** dán danh sách link Douyin (video 3–10 phút, nhiều chủ đề), hoặc thả file vào thư mục `DAU_VAO/`.
- **Giọng:**
  - Giọng **miền Nam**, mỗi nhân vật một giọng, tự chọn theo giới tính, tuổi và thể loại. **Nữ chính = Thục Đoan.**
  - Không nhân bản giọng người thật, chỉ dùng giọng có sẵn.
- **Dịch:** Việt hóa tự nhiên; nói gọn lại cho vừa thời gian câu gốc.
- **Hình và tiếng:**
  - Giữ nhạc nền gốc.
  - Làm mờ phụ đề chữ Trung, đè phụ đề Việt lên.
  - Chỉ xuất bản gốc, không logo, không intro.
- **Vận hành:** 100% tự động qua giao diện web. Sản lượng càng nhiều càng tốt nhưng phải cân bằng với chất lượng. Tự đăng TikTok, YouTube, Facebook là giai đoạn sau.

## 3. Bản đồ hệ thống
- **Dự án:** thư mục chứa `app/dubber/`. Chạy `bash scripts/dub.sh where` để biết đường dẫn chính xác. Trên máy hiện tại là `~/Desktop/video-to-video-change-voice-from-chineses to vietnamese/`.
- **Môi trường:** `~/.douyin_dubber/venv` (Python 3.11, cài bằng uv). Model AI nằm ở `~/.douyin_dubber/models`. VieNeu tự tải vào `~/.cache/huggingface`.
- **Nhấp đúp để dùng** (nằm ở thư mục dự án):
  - `1_CAI_DAT`: cài đặt.
  - `2_MO_UNG_DUNG`: giao diện web http://127.0.0.1:7860.
  - `3_CHAY_THU`: chạy các link trong `config/test_links.txt`.
  - `4_CHAN_DOAN`: tự kiểm tra.
  - `5_DEBUG_DOUYIN`: thử các cách tải.
  - `CAP_NHAT`: cài lại thư viện.
  - `6_CAI_SKILL_ANTIGRAVITY`: cài skill này.
- **Kết quả:**
  - Video thành phẩm: `THANH_PHAM/<ngày>/<mã>_<tiêu đề>.mp4`, kèm `.srt` và `.txt`.
  - File trung gian: `_work/job_XXXXX/`.
  - Hàng chờ: `_work/jobs.db` (SQLite).
  - Nhật ký: `logs/`.
- **Code:** `app/dubber/`.
  - `steps/<bước>.py` mỗi file là một bước.
  - `worker.py` điều phối 2 làn.
  - `server.py` cùng `web/index.html` là giao diện.
  - `llm.py` gọi Gemini.
  - `ttsload.py` nạp giọng đọc.
  - `paths.py` chứa cài đặt mặc định.
  - `selftest.py`, `diag.py`, `douyin_debug.py` dùng để chẩn đoán.

Chi tiết kỹ thuật từng bước: đọc `references/PIPELINE.md` khi cần giải thích hoặc sửa một bước.

## 4. Lệnh điều khiển (chạy trong terminal)
Mọi lệnh chạy dạng `bash <thư mục skill>/scripts/dub.sh <lệnh>`. Script tự tìm dự án và Python.

| Việc | Lệnh |
|---|---|
| Xem dự án, có API key chưa, app có đang chạy không | `dub.sh where` |
| Mở giao diện web (chạy nền) | `dub.sh serve` |
| Thêm video vào hàng chờ | `dub.sh add "<link1>" "<link2>" "/đường/dẫn/file.mp4"` |
| Xử lý ngay rồi chờ tới khi xong | `dub.sh run "<link>"` (lệnh chạy lâu: để chạy nền rồi kiểm tra bằng `status`) |
| Xem tiến độ | `dub.sh status`, `dub.sh status --id 12`, `dub.sh status --all` |
| Xem lỗi / nhật ký | `dub.sh logs 12`, `dub.sh logs 12 --step tts --tail 100` |
| Chạy lại | `dub.sh retry 12` (tiếp từ bước lỗi) hoặc `dub.sh retry 12 --from tts` (làm lại từ một bước) |
| Xem hoặc đổi cài đặt | `dub.sh settings`, `dub.sh settings max_speed=1.25 lead_male_voice="Thái Sơn"` |
| Xem giọng, phân vai của 1 video | `dub.sh voices`, `dub.sh voices 12` |
| Xem bước nào chậm | `dub.sh timing` |
| Tự kiểm tra | `dub.sh doctor` hoặc `dub.sh doctor separate,tts,voices,tts8` |
| Thử các cách tải Douyin | `dub.sh douyin 7682470240348933427` |

Lưu ý:
- Chỉ **một** bộ xử lý được chạy cùng lúc (file khóa `_work/worker.lock`).
- Nếu giao diện web đang chạy thì `add` hoặc `retry` là đủ, bộ xử lý của web sẽ tự làm.
- Nếu web chưa chạy thì dùng `serve` (nên dùng) hoặc `run`.

## 5. Quy trình theo tình huống
**A. Người dùng dán link hoặc gửi file**
1. Chạy `where`. Nếu chưa có key hoặc chưa cài môi trường, xem mục 7.
2. Chạy `add` với tất cả link.
3. Nếu app chưa chạy thì `serve`.
4. Báo cho người dùng: đã thêm mấy video; ước tính khoảng 1,5–3 phút xử lý cho mỗi phút video khi chạy 2 làn; xem tiến độ ở http://127.0.0.1:7860.

**B. Hỏi tiến độ hoặc "xong chưa"**
- Chạy `status`. Video xong thì báo đường dẫn file mp4 và có thể mở bằng `open -R "<file>"`.
- Trạng thái "Cần xem" (review) thì đọc dòng "Cần xem" và giải thích ngắn gọn.

**C. Video lỗi (failed)**
1. Chạy `logs <id>` và đọc `error_<bước>.txt`.
2. Đối chiếu với `references/TROUBLESHOOTING.md`.
3. Sửa nguyên nhân: cấu hình, file, hoặc code.
4. Chạy `retry <id> --from <bước>`.
5. Kiểm tra lại bằng `status`.
6. Giải thích cho người dùng: lỗi gì, vì sao, đã sửa thế nào.

**D. Đổi giọng hoặc thấy giọng không hợp**
- Nữ chính và nam chính: `settings lead_female_voice=... lead_male_voice=...`. Giọng hợp lệ:
  - Nữ: Thục Đoan, Mỹ Duyên, Kim Thanh, Thùy Dung.
  - Nam: Adam, Thái Sơn, Đức Trí, Minh Triết.
- Danh sách ưu tiên theo thể loại: `genre_voices` (các thể loại `co_trang`, `ban_hang`, `kien_thuc`, `review_phim`, `hai`, `default`). Biến thể giọng: `voice_variants=true/false`.
- Làm lại giọng cho video đã xong: `retry <id> --from tts`. Hệ thống tự chạy lại bước tách nhạc nếu file âm thanh trung gian đã bị dọn. Bản dịch cũ được giữ nguyên, không tốn lượt AI.
- Cho người dùng nghe mẫu giọng: `doctor voices`, mẫu nằm ở `logs/giong_mau/`.

**E. Chất lượng hoặc tốc độ**
- Câu bị đọc nhanh quá: giảm `speech_rate_sps` (4.8 → 4.5) để AI dịch ngắn hơn, hoặc tăng `borrow_gap_s`.
- Nhạc nền át giọng: giảm `duck_db` (−5 → −8).
- Máy đơ: `parallel_lanes=false` hoặc `threads=3`. Muốn nhanh hơn: thử `tts_precision=int8` sau khi nghe mẫu `logs/giong_mau/int8_*.wav` so với `fp32_*.wav`.
- Xem `references/OPTIMIZATION.md`: những gì đã đo, đã tối ưu, và các ý tưởng còn lại.

**F. Sửa hoặc cải tiến code**: làm theo `references/CODE_GUIDE.md`. Tóm tắt:
1. Đọc file liên quan.
2. Sửa nhỏ, giữ đúng kiểu code: mỗi bước là một hàm `run(job)`; ghi kết quả bằng `job.write`; báo tiến độ bằng `job.progress`; ghi nhật ký tiếng Việt bằng `log`.
3. Chạy `python -m py_compile`.
4. Thử bằng `dub.sh doctor <phần>` hoặc 1 video ngắn (`dub.sh run "<link>"`).
5. Đo lại bằng `dub.sh timing`.
6. Báo trước/sau bằng con số.
- RAM: mỗi tiến trình con nên dưới khoảng 2GB. Không nạp 2 model nặng cùng lúc trong 1 bước.

## 6. Hiểu nhanh 2 làn và hàng chờ
- **Trạng thái:**
  - `queued`: đang chờ.
  - `running`: đang xử lý.
  - `ready`: đã chuẩn bị xong, chờ đọc giọng.
  - `done`: xong.
  - `review`: xong nhưng cần xem.
  - `failed`: lỗi.
- **Thử lại tự động:** download, translate và tts tự thử lại tối đa 3 lần, cách nhau 2 phút × số lần.
- **Tắt máy giữa chừng:** không sao. Mở lại app sẽ chạy tiếp từ bước dở, nhờ các file `.done_<bước>`.

## 7. Cài đặt trên máy mới hoặc khi chưa có dự án
- Nếu `dub.sh where` báo không tìm thấy dự án: chạy `bash scripts/setup_project.sh [thư mục đích]`. Script giải nén bản sao dự án đi kèm skill (`bundle/project.zip`), rồi mở trình cài đặt.
- Cài môi trường: nhấp đúp hoặc chạy `bash "<dự án>/1_CAI_DAT.command"`. Trình cài sẽ cài ffmpeg (brew), uv, Python 3.11, thư viện, model, demucs-mlx, rồi tự kiểm tra. Lần đầu mất khoảng 15–30 phút và tải vài GB.
- Lấy API key miễn phí: hướng dẫn người dùng vào https://aistudio.google.com/apikey → Create API key, rồi tự ghi vào `config/.env` dạng `GEMINI_API_KEY=...` và chạy `chmod 600 config/.env`. Không nhắc lại key trong chat.

## 8. Tài liệu kèm theo
- `references/PIPELINE.md`: kỹ thuật từng bước, file vào/ra, model, tham số.
- `references/TROUBLESHOOTING.md`: lỗi đã gặp và cách sửa.
- `references/OPTIMIZATION.md`: số đo, những tối ưu đã làm (có nguồn), ý tưởng tiếp theo.
- `references/CODE_GUIDE.md`: cấu trúc code, quy ước, cách thêm hoặc sửa một bước, cách kiểm thử.
- `examples/hoi-dap.md`: mẫu cách trả lời người dùng.
