# Quy tắc cho agent trong dự án này

Đây là hệ thống lồng tiếng tự động video Douyin (tiếng Trung) sang tiếng Việt giọng miền Nam, chạy trên MacBook M1 8GB, miễn phí.
Dùng skill **douyin-long-tieng** (`.agents/skills/douyin-long-tieng/SKILL.md`) cho mọi việc liên quan.

- Trả lời bằng **tiếng Việt, ngắn gọn, dễ hiểu**. Người dùng không biết code.
- **Mỗi hành động đều giải thích**: đang làm gì, vì sao, kỹ thuật đó hoạt động thế nào (1–2 câu).
- Tự chạy lệnh, rồi báo kết quả. Chỉ nhờ người dùng khi bắt buộc, ví dụ bấm "Cho phép" khi macOS hỏi quyền.
- Không bao giờ in hay chép API key trong `config/.env`.
- Không xóa `THANH_PHAM/`, `config/.env`, `~/.douyin_dubber/` khi người dùng chưa đồng ý.
- Chỉ dùng công cụ miễn phí, tối đa 20.000đ/tháng. Chỉ dùng tài khoản Google.
- Giữ RAM dưới khoảng 4GB. Trước khi chạy xử lý, kiểm tra xem đã có bộ xử lý khác đang chạy chưa: `bash .agents/skills/douyin-long-tieng/scripts/dub.sh where`.
