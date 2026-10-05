# Hướng dẫn sửa code

## Cấu trúc
```
<dự án>/
├── 1_CAI_DAT.command … 6_CAI_SKILL_ANTIGRAVITY.command   # nhấp đúp để chạy (bash)
├── HUONG_DAN.txt                     # hướng dẫn cho người dùng
├── config/.env                       # GEMINI_API_KEY (chmod 600, KHÔNG đọc ra/chia sẻ)
├── config/settings.json              # chỉ chứa những gì khác mặc định (paths.save_settings)
├── config/test_links.txt             # link chạy thử (dòng bắt đầu bằng # bị bỏ qua)
├── DAU_VAO/  THANH_PHAM/  logs/  _work/
├── .agents/skills/douyin-long-tieng/ # skill này
└── app/
    ├── requirements.txt
    └── dubber/
        ├── paths.py        # đường dẫn + DEFAULT_SETTINGS (thêm cài đặt mới ở đây)
        ├── utils.py        # log, ffmpeg, probe, load/save_audio, Job, threads(), wait_for_memory
        ├── models.py       # danh sách model sherpa-onnx (tải từ GitHub releases) + font
        ├── run_step.py     # chạy 1 bước trong tiến trình con, ghi .done_/error_
        ├── worker.py       # 2 làn (prep/voice), hàng chờ, thử lại, reset_from()
        ├── db.py           # SQLite: jobs(id,input,title,status,step,frac,note,error,attempts,…,timings,issues,not_before)
        ├── server.py       # web API: /api/state /api/add /api/retry /api/remove /api/settings /api/open /video/<id>
        ├── web/index.html  # giao diện (tiếng Việt, tự làm mới mỗi 2 giây)
        ├── cli.py          # chạy thử không cần giao diện
        ├── llm.py          # Gemini: danh sách model dự phòng, tắt suy nghĩ, JSON, đếm lượt
        ├── ttsload.py      # nạp VieNeu (materialize symlink, offline, precision)
        ├── textnorm.py     # số/ký hiệu -> chữ tiếng Việt
        ├── selftest.py diag.py douyin_debug.py
        └── steps/  download separate transcribe ocr translate tts mix render qc
```

## Quy ước
- Mỗi bước là một hàm `run(job: Job)` trong `steps/<bước>.py`.
  - Đọc đầu vào bằng `job.read("x.json")` hoặc `job.p("file")`. Ghi kết quả bằng `job.write("y.json", data)`; hàm này ghi nguyên tử nên không bao giờ để lại file hỏng.
  - Báo tiến độ bằng `job.progress(0..1, "ghi chú")`. Giao diện hiện ghi chú này.
  - Lỗi thì `raise RuntimeError("thông báo tiếng Việt dễ hiểu")`. Thông báo này hiện trên giao diện.
  - Nhật ký ghi bằng `log("...")` bằng tiếng Việt. Câu tóm tắt cuối mỗi bước phải có số đo: số giây, số câu...
- Docstring đầu mỗi file giải thích **cách hoạt động** bằng tiếng Việt đơn giản. Khi đổi kỹ thuật thì cập nhật docstring luôn.
- Cài đặt mới: thêm vào `DEFAULT_SETTINGS` trong `paths.py`, kèm chú thích. Muốn chỉnh được từ giao diện thì thêm vào `EDITABLE` trong `server.py`.
- Số luồng CPU lấy bằng `utils.threads()`. Hàm này đọc biến `DUBBER_THREADS`; làn chuẩn bị đặt sẵn biến này bằng 2.
- Gọi ffmpeg qua `utils.ffmpeg([...])`. Hàm này ưu tiên `/opt/homebrew/bin/ffmpeg`. Trên Mac dùng `-hwaccel videotoolbox` khi giải mã, và `h264_videotoolbox` khi mã hóa.
- Không thêm thư viện nặng (PyTorch...) vào môi trường chính. Nếu bắt buộc phải dùng thì dùng venv tạm, như cách trình cài chuyển đổi demucs.
- Bước mới: thêm tên vào `steps/__init__.py` (`STEPS`, `STEP_LABELS`), đặt đúng làn (`PREP` là các bước trước `tts`), và thêm vào `NEED_MB` nếu bước tốn RAM.

## Cách kiểm thử sau khi sửa
1. Kiểm tra cú pháp: `~/.douyin_dubber/venv/bin/python -m py_compile app/dubber/steps/<file>.py`
2. Tự kiểm tra từng phần: `dub.sh doctor <phần>`. Các phần: ffmpeg, gemini, asr, ocr, separate, tts, voices, tts8.
3. Chạy lại 1 bước trên video có sẵn: `dub.sh retry <id> --from <bước>`, rồi `dub.sh serve` hoặc `dub.sh run`.
4. Chạy cả quy trình: `dub.sh run "<link>"`, hoặc nhấp đúp `3_CHAY_THU.command` (lấy link trong `config/test_links.txt`). Kết quả ghi ở `logs/test_run.txt`.
5. Đo lại tốc độ: `dub.sh timing`, rồi báo người dùng số trước và sau khi sửa.
6. Xem thành phẩm: mở file mp4 (`open "<file>"`), kiểm tra phụ đề có đè đúng chỗ chữ Trung, giọng có khớp miệng, nhạc nền có còn.

## Những điều KHÔNG làm
- Không xóa `THANH_PHAM/`, `config/.env`, `~/.douyin_dubber/` khi người dùng chưa đồng ý.
- Không in API key ra chat, log hay tài liệu.
- Không chạy 2 bộ xử lý cùng lúc. Kiểm tra bằng `dub.sh where` → `dang_co_bo_xu_ly_chay`.
- Không đổi sang dịch vụ trả phí hoặc giọng nhân bản khi người dùng chưa đồng ý.
- Không tải hay chạy file từ nguồn không rõ ràng.
