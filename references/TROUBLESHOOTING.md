# Xử lý lỗi: các lỗi đã gặp và cách sửa

Cách điều tra chung:
1. Chạy `bash scripts/dub.sh status` để xem video nào lỗi và lỗi ở bước nào.
2. Chạy `bash scripts/dub.sh logs <id> --step <bước>` để xem nhật ký và file `error_<bước>.txt`.
3. So với bảng dưới đây, sửa nguyên nhân, rồi chạy `bash scripts/dub.sh retry <id> --from <bước>`.
4. Không chắc lỗi do đâu thì chạy `bash scripts/dub.sh doctor` (tự kiểm tra từng bộ phận, kết quả ở `logs/selftest.txt`).

## Tải Douyin
| Dấu hiệu | Nguyên nhân | Cách sửa |
|---|---|---|
| `open: không có dữ liệu video (...)` | Video riêng tư, đã xóa, chỉ xem được khi đăng nhập, hoặc Douyin đã đóng cửa này | Hệ thống tự chuyển sang feed/page. Chạy `dub.sh douyin <mã>` để xem cách nào còn chạy |
| `page: ... không lấy được dữ liệu video (có thể bị bắt xác minh)` | Douyin hiện captcha cho trình duyệt ẩn | Xóa hồ sơ `~/.douyin_dubber/douyin_profile` rồi thử lại. Hoặc tải tay video rồi thả vào `DAU_VAO/` |
| `Không mở được Chrome` | Máy chưa cài Google Chrome, hoặc Playwright chưa có trình duyệt | Cài Google Chrome, hoặc chạy `~/.douyin_dubber/venv/bin/python -m playwright install chromium` |
| `yt-dlp: Fresh cookies ... are needed` | yt-dlp không còn tải được Douyin (issue #16803) | Bỏ qua, các cách khác vẫn chạy |
| `File tải về quá nhỏ` | CDN chặn, link hết hạn | Chạy lại (link được lấy mới mỗi lần) |
| `Trang chia sẻ ... không có dữ liệu` | iesdouyin đã bỏ `_ROUTER_DATA` từ khoảng 8/2026 | Bình thường, chỉ là cách dự phòng |
| File trong `~/Downloads` báo không đọc được | macOS chưa cho Terminal quyền vào thư mục Downloads | Bấm "Cho phép" khi Mac hỏi, hoặc vào Cài đặt → Quyền riêng tư → Tệp và thư mục → Terminal |

Nếu cả 4 cách đều chết (Douyin thay đổi): tìm trên GitHub cách mới (từ khóa: `aweme/v1/web/aweme/detail a_bogus 2026`, các repo Evil0ctal/Douyin_TikTok_Download_API, JoeanAmier/TikTokDownloader, Johnserf-Seed/f2). Thêm hàm `fetch_info_<tên>` vào `download.py` theo mẫu `fetch_info_open`, rồi đưa vào danh sách `methods` trong `run()`.

## Tách nhạc
| Dấu hiệu | Nguyên nhân | Cách sửa |
|---|---|---|
| `[METAL] Command buffer execution failed: GPU Timeout` | GPU M1 xử lý một khúc quá lâu | Đã có cắt khúc 30 giây. Nếu vẫn lỗi thì giảm khúc xuống 15–20 giây trong `_demucs` (`chunk = 20 * sr`) |
| `separate.json` ghi `backend: spleeter/uvr` | demucs lỗi nên đã dùng dự phòng (chất lượng kém hơn, chậm hơn) | Xem `errors` trong separate.json. Chạy `dub.sh doctor separate` |
| `No module named demucs_mlx` | Bản cài thiếu demucs-mlx (không bắt buộc) | Chạy lại `1_CAI_DAT.command` |

## Nghe / phân biệt người nói
| Dấu hiệu | Cách sửa |
|---|---|
| Một người bị tách thành quá nhiều người | Tăng ngưỡng gom cụm trong `_diarize` (hiện 0.85, có thể lên 0.9) |
| Hai người khác nhau bị gộp làm một | Giảm ngưỡng xuống 0.75–0.8 |
| Đoán sai giới tính | AI dịch đã sửa phần lớn (`speakers.gender`). Ngưỡng cao độ 170Hz nằm trong transcribe.py |

## Dịch (Gemini)
| Dấu hiệu | Nguyên nhân | Cách sửa |
|---|---|---|
| `Chưa có GEMINI_API_KEY` | Thiếu key | Tạo key miễn phí tại aistudio.google.com/apikey, ghi vào `config/.env` dạng `GEMINI_API_KEY=...` rồi `chmod 600`. **Không bao giờ in key ra màn hình hay chat** |
| `429 RESOURCE_EXHAUSTED` | Hết lượt miễn phí trong ngày/phút | Hệ thống tự chuyển model. Lượt reset lúc nửa đêm giờ Mỹ (khoảng 14–15h giờ Việt Nam). Xem lượt đã dùng ở `~/.douyin_dubber/quota.json` |
| `404 ... not found` | Model không có cho tài khoản này (ví dụ gemini-2.5-flash-lite với người dùng mới) | Bỏ model đó khỏi `translate_models` |
| `model trả về rỗng` | Gemma 26B đôi khi trả về rỗng | Tự chuyển model khác |
| Dịch rất chậm (>100 giây) | Model đang "suy nghĩ" | Kiểm tra log có `thinking` không. `llm._config` đặt `thinking_level=minimal` |
| Lỗi 400 về `thinking`/`mime` | Model không hỗ trợ tùy chọn đó | Tự bỏ tùy chọn và thử lại (`_NO_THINK_CFG`) |

Chạy `bash scripts/dub.sh doctor gemini` hoặc `python -m dubber.diag` để thử từng model, kết quả ghi vào `logs/diag.txt`.

## Giọng đọc (VieNeu)
| Dấu hiệu | Nguyên nhân | Cách sửa |
|---|---|---|
| `External data path escapes model directory` | Cache HuggingFace dùng symlink, onnxruntime bản mới từ chối | `ttsload.materialize()` đã tự sửa. Nếu vẫn lỗi: xóa `~/.cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Turbo` rồi chạy lại để tải mới |
| Giọng bị méo khi dùng `tts_precision=int8` | int8 chưa ổn trên máy này | Đặt lại `tts_precision=fp32` |
| Câu đầu tiên đọc chậm (RTF >1) | Model đang khởi động | Bình thường |
| Nhiều câu phải tăng tốc >1.3 lần (QC "Cần xem") | Bản dịch quá dài | Giảm `speech_rate_sps` (ví dụ 4.5) để AI dịch ngắn hơn, hoặc tăng `max_speed` lên 1.25 |

## Xuất video
| Dấu hiệu | Cách sửa |
|---|---|
| `Unknown encoder h264_videotoolbox` | Code tự dùng libx264 (chậm hơn) nếu không có encoder. Kiểm tra ffmpeg: `/opt/homebrew/bin/ffmpeg -encoders \| grep videotoolbox` |
| Phụ đề Việt bị lệch vị trí | Xem `band` trong ocr.json. Video có cờ xoay thì download đã đổi rộng/cao |
| Lỗi font | `models.font_path()` sẽ dùng Arial Bold nếu không có Be Vietnam Pro |

## Hệ thống / máy
| Dấu hiệu | Cách sửa |
|---|---|
| Nhấp đúp file .command báo "Permission denied" | Chạy `chmod +x *.command` trong thư mục dự án |
| Máy chậm hoặc đơ | Tắt chạy 2 làn (`parallel_lanes=false`) hoặc giảm `threads` xuống 3 |
| "Đang có một cửa sổ xử lý khác chạy" | Chỉ 1 bộ xử lý được chạy (file khóa `_work/worker.lock`). Đóng cửa sổ 3_CHAY_THU hoặc 2_MO_UNG_DUNG cũ |
| Video kẹt ở "Đang xử lý" sau khi tắt máy | Khi mở lại, video đang chạy dở tự quay về hàng chờ và chạy tiếp từ bước dở |
| Ổ cứng đầy | Xóa `_work/job_*` của các video đã xong (thành phẩm nằm ở `THANH_PHAM/`). Hỏi người dùng trước khi xóa |
