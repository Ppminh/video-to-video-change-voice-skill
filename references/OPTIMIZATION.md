# Tối ưu: số đo, những gì đã làm, ý tưởng tiếp theo

## Số đo gốc (bản đầu tiên, MacBook M1 8GB, video 2 phút 03 giây, 37 câu)
| Bước | Bản cũ | Nguyên nhân chậm |
|---|---|---|
| download | 37.6s | Mở Chrome ẩn và chờ trang tự gọi API |
| separate | 60.5s | demucs bị GPU Timeout nên phải chạy Spleeter dự phòng trên CPU |
| transcribe | 18.1s | (ổn) |
| ocr | 66.4s | Đọc toàn khung 2 lần/giây (246 khung). RapidOCR phóng ảnh lên 736px |
| translate | 194.1s | Gemma 4 31B "suy nghĩ" khoảng 190 giây |
| tts | 127.5s | Đọc 91 giây giọng (RTF 0.61), cộng 52 giây gọi AI rút gọn câu (Gemma) |
| mix | 43.7s | `np.convolve` cửa sổ 11.025 mẫu trên từng mẫu âm thanh |
| render | 30.2s | (ổn, có dùng chip mã hóa phần cứng) |
| **Tổng** | **~580s (cả lượt ~747s)** | |

## Đã tối ưu (10/2026)
1. **Tải bằng "cửa open.douyin.com"**: gọi thẳng API chi tiết, giả làm trang open.douyin.com, không cần chữ ký hay cookie. Đây là cách mà Diana PR #612 và nonebot-plugin-parser-lite PR #312 đang dùng (9/2026). Nếu không được thì chuyển sang feed của app điện thoại, rồi tới Chrome ẩn có hồ sơ lâu dài gọi `fetch()` trong trang. Bản thân trang Douyin (bdms/secsdk) sẽ tự gắn `a_bogus`, `x-secsdk-web-signature`, `uifid`. Lý do: từ 8–9/2026 Douyin có thêm cổng "ArgusSecurityPlugin", chặn các yêu cầu thiếu chữ ký.
2. **demucs cắt khúc 30 giây + `batch_size=1` + tắt compile**: tránh GPU Timeout trên M1. Selftest: 30 giây âm thanh tách trong 10.2 giây, gồm cả thời gian nạp model.
3. **OCR 2 lượt**: dò vùng phụ đề trên keyframe (giải mã rất nhanh), rồi chỉ đọc dải phụ đề, bỏ qua khung không đổi. Đặt `det_limit_type=max` để RapidOCR không phóng to ảnh. Thử trong máy ảo (2 nhân): 42 giây xuống 13 giây, kết quả vùng phụ đề và chữ y hệt. Video không có hardsub: chỉ mất vài giây.
4. **Dịch**: dùng Gemini Flash-Lite trước, `thinking_level=minimal`, JSON thuần. Phần sau của video dài thì dịch song song (3 luồng).
5. **TTS**:
   - Rút gọn câu **trước** khi đọc, ước lượng bằng tốc độ đọc đã đo, nên ít câu phải đọc 2 lần.
   - Có thêm lựa chọn `tts_precision=int8`. VieNeu ghi "nhanh khoảng 3 lần mỗi khung". README đo trên Intel: RTF 0.36 so với 0.6. Trên Intel cần VNNI để không méo; M1 là ARM nên cần nghe thử (`doctor tts8`).
6. **Trộn âm**: đường bao âm lượng tính trên khung 10ms. Máy ảo: video 3 phút trộn trong 3.6 giây.
7. **2 làn song song**: làn chuẩn bị (GPU, mạng, CPU 2 luồng, nice 10) chạy cùng lúc với làn lồng tiếng (CPU 4 luồng). Mô phỏng 3 video: 41 giây so với 63 giây khi chạy tuần tự.
8. **Biến thể giọng bằng Praat PSOLA ("Change gender")**: khoảng 0.2 giây cho mỗi câu 6 giây, cao độ ra đúng như yêu cầu (+2 nửa cung: 186→209Hz), thời lượng giữ nguyên.

## Kết quả đo sau tối ưu
Chạy `dub.sh timing` để có số mới nhất. Ghi lại ở đây sau mỗi lần đo lớn:
- (chờ đo trên Mac)

## Ý tưởng tiếp theo (chưa làm), xếp theo lợi ích ước tính
1. **TTS int8**: nếu nghe không méo thì nhanh hơn khoảng 1,5–3 lần ở bước chậm nhất. Bật bằng `settings tts_precision=int8`.
2. **Giữ model TTS trong bộ nhớ** giữa các video (một tiến trình TTS chạy lâu, nhận việc qua hàng đợi): bớt được 12–35 giây nạp model cho mỗi video. Đánh đổi: tốn khoảng 1.4GB RAM thường trực.
3. **VieNeu `infer_stream`**: không nhanh hơn trên CPU (chỉ giảm thời gian chờ âm thanh đầu tiên), bỏ qua.
4. **CoreML cho ONNX**: VieNeu khóa cứng `CPUExecutionProvider`. Mỗi khung gọi khoảng 17 lần ONNX nhỏ, nên chuyển dữ liệu qua lại tốn hơn phần lợi, khó có lợi. RapidOCR thì có thể thử CoreML, nhưng OCR đã nhanh.
5. **Ngưỡng `DIFF_SAME` của OCR**: nền phía sau phụ đề chuyển động nhiều thì khung ít được bỏ qua hơn. Có thể so sánh chỉ phần trung tâm dải (nơi có chữ) để bỏ qua được nhiều hơn.
6. **Dịch theo lô nhiều video trong 1 yêu cầu**: tiết kiệm lượt miễn phí khi chạy hàng trăm video/ngày. Hạn mức miễn phí hiện xem tại aistudio.google.com/rate-limit (trang tài liệu không còn ghi số cố định; các nguồn ngoài cho biết Flash-Lite khoảng 500–1.500 lượt/ngày).
7. **Tăng `prep_ahead` lên 2** khi ổ cứng còn nhiều chỗ trống và mạng chậm.

## Nguồn tham khảo
- Antigravity skills: https://antigravity.google/docs/skills · rules: https://antigravity.google/docs/rules
- Douyin: https://github.com/SuInk/Diana/pull/612 · https://github.com/sokoko-org/nonebot-plugin-parser-lite/pull/312 · https://github.com/Evil0ctal/Douyin_TikTok_Download_API · https://github.com/JoeanAmier/TikTokDownloader · https://github.com/Johnserf-Seed/f2 · https://github.com/yt-dlp/yt-dlp/issues/16803
- VieNeu-TTS: https://github.com/pnnbao97/VieNeu-TTS (docs/streaming.md) · https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo
- Gemini: https://ai.google.dev/gemini-api/docs/thinking · https://ai.google.dev/gemini-api/docs/rate-limits · https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api
- RapidOCR config (limit_type/limit_side_len): mã nguồn rapidocr_onnxruntime 1.4.4 `config.yaml`
- ffmpeg rubberband không có trong Homebrew ffmpeg (chỉ có ở ffmpeg-full), nên dùng Praat PSOLA thay thế: https://github.com/Homebrew/homebrew-core/blob/HEAD/Formula/f/ffmpeg.rb
