# Mẫu cách trả lời (ngắn, có giải thích kỹ thuật, tự làm luôn)

## 1. Người dùng dán link
**Người dùng:** https://www.douyin.com/jingxuan?modal_id=7682470240348933427 | https://v.douyin.com/xxxx/ làm giúp tôi

**Agent:** chạy `dub.sh where` → `dub.sh add "<link1>" "<link2>"` → `dub.sh serve`

**Trả lời:**
> Đã thêm 2 video vào hàng chờ và mở giao diện ở http://127.0.0.1:7860.
> - Cách tải: hệ thống hỏi thẳng máy chủ Douyin thông tin video, rồi lấy link bản **không logo**. Bản có logo tên là "playwm", bản sạch là "play".
> - Thời gian: video 3 phút mất khoảng 5–8 phút. Chạy 2 làn song song nên video thứ hai ra nhanh hơn.
> Xong mình báo bạn đường dẫn file.

## 2. Hỏi tiến độ
**Agent:** `dub.sh status`

**Trả lời:**
> - Video #7 **xong**, mất 6 phút 10 giây. File ở `THANH_PHAM/2026-10-05/768..._Tieu de.mp4`.
> - Video #8 đang ở bước **đọc giọng Việt** (câu 23/40). Bước này chạy model VieNeu trên CPU: biến chữ thành sóng âm 48kHz, mỗi nhân vật một giọng.

## 3. Video lỗi
**Agent:** `dub.sh logs 9 --step download` → thấy `open: không có dữ liệu video` và `page: bị bắt xác minh` → xóa `~/.douyin_dubber/douyin_profile` → `dub.sh retry 9`

**Trả lời:**
> Video #9 lỗi ở bước tải: Douyin bắt trình duyệt ẩn xác minh (captcha).
> - Mình đã xóa hồ sơ trình duyệt cũ để Douyin cấp cookie khách mới, rồi chạy lại.
> - Nếu vẫn lỗi thì video này có thể là video riêng tư hoặc phải đăng nhập mới xem được. Khi đó bạn tải tay rồi thả file vào thư mục `DAU_VAO`.

## 4. Đổi giọng
**Người dùng:** giọng nam chính nghe không hợp, đổi giọng trầm hơn

**Agent:** `dub.sh voices 7` (thấy nam chính = Adam) → `dub.sh settings lead_male_voice="Thái Sơn"` → `dub.sh retry 7 --from tts`

**Trả lời:**
> Đã đổi nam chính sang **Thái Sơn**, giọng kể chuyện, trầm hơn Adam, và đang làm lại giọng cho video #7.
> - Bản dịch cũ được giữ nguyên, nên không tốn thêm lượt AI. Chỉ chạy lại bước đọc, trộn và xuất video, mất khoảng 3 phút.
> - Mẹo: muốn trầm hơn nữa thì có biến thể "trầm". Hệ thống dùng kỹ thuật PSOLA của Praat hạ giọng 2 nửa cung mà tốc độ nói vẫn giữ nguyên.

## 5. Hỏi "sao chậm vậy"
**Agent:** `dub.sh timing`

**Trả lời:**
> Bước chậm nhất là **đọc giọng** (khoảng 45% thời gian). Lý do: model chạy trên CPU, đọc 10 giây giọng mất khoảng 4–6 giây.
> - Cách nhanh hơn: chế độ **int8**. Model dùng số nhỏ gọn hơn nên nhanh hơn khoảng 1,5 lần, nhưng cần nghe thử xem giọng có bị méo không. Mình đã tạo 2 mẫu `logs/giong_mau/fp32_*.wav` và `int8_*.wav` để bạn so sánh.
> - Bạn thấy giống nhau thì mình bật int8.
