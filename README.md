# Video-to-Video Change Voice (Đổi giọng nói video sang video)

Skill dành cho **Google Antigravity**: Tự động đổi giọng nói và lồng tiếng video tiếng Trung (Douyin hoặc file video MP4) sang **tiếng Việt giọng miền Nam**, chạy trực tiếp trên máy Mac (tối ưu chip Apple Silicon M1/M2/M3), hoàn toàn miễn phí.

---

## 🌟 Điểm nổi bật
- **Đổi giọng video-to-video:** Nhận link Douyin hoặc kéo thả trực tiếp file video vào hàng chờ.
- **Tách âm thanh bằng AI:** Sử dụng `demucs-mlx` chạy trên Metal GPU của chip Apple để tách riêng giọng nói gốc và nhạc nền, giữ trọn vẹn nhạc nền và hiệu ứng âm thanh.
- **AI Dịch thuật thông minh:** Sử dụng Google Gemini API dịch nghĩa tự nhiên, tối ưu độ dài câu nói cho vừa khớp với thời gian phát âm của video gốc.
- **Giọng đọc miền Nam tự nhiên:** Tự động phân vai nhân vật (nam/nữ, chính/phụ, độ tuổi) với các giọng:
  - *Nữ:* Thục Đoan (nữ chính mặc định), Mỹ Duyên, Kim Thanh, Thùy Dung.
  - *Nam:* Thái Sơn, Adam, Đức Trí, Minh Triết.
- **Xử lý hình ảnh:** Tự động làm mờ phụ đề tiếng Trung gốc và đè phụ đề tiếng Việt đồng bộ thời gian.
- **Giao diện Web trực quan:** Theo dõi tiến độ thời gian thực, quản lý hàng chờ và xem lại video thành phẩm tại `http://127.0.0.1:7860`.

---

## 🚀 Cài đặt Skill vào Antigravity

### Cách 1: Clone vào thư mục Skill của Antigravity
```bash
git clone https://github.com/Ppminh/video-to-video-change-voice-skill.git ~/.gemini/config/skills/video-to-video-change-voice
```

### Cách 2: Khởi tạo dự án lồng tiếng trên máy mới
Sau khi tải skill về máy, bạn có thể tạo toàn bộ mã nguồn dự án lồng tiếng bằng lệnh:
```bash
bash scripts/setup_project.sh ~/Desktop/video-to-video-change-voice
```
Lệnh này sẽ tự động giải nén bản sao dự án sạch (đã loại bỏ thông tin cá nhân và API key) từ gói `bundle/project.zip`.

---

## 💻 Sử dụng với Antigravity
Khi pair programming cùng Antigravity, bạn chỉ cần:
- Dán link video: `"Lồng tiếng video Douyin này giúp tôi: https://..."`
- Hoặc đổi giọng file video: `"Đổi giọng video này sang tiếng Việt giúp tôi: /đường/dẫn/video.mp4"`
- Xem tiến độ: `"Tiến độ video đến đâu rồi?"`
- Đổi giọng nhân vật: `"Đổi giọng nam chính sang Thái Sơn"`

---

## ⚙️ Cấu trúc thư mục Skill
- `SKILL.md`: Định nghĩa và hướng dẫn vận hành skill cho agent.
- `scripts/`: Chứa các script điều khiển (`dub.sh`, `dub.py`, `setup_project.sh`).
- `references/`: Tài liệu kỹ thuật chi tiết (`PIPELINE.md`, `OPTIMIZATION.md`, `TROUBLESHOOTING.md`, `CODE_GUIDE.md`).
- `examples/`: Mẫu hội thoại và câu hỏi thường gặp (`hoi-dap.md`).
- `bundle/`: Bản sao mã nguồn dự án rút gọn để bootstrap trên máy mới.
