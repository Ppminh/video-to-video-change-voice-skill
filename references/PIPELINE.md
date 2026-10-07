# Dây chuyền 9 bước: cách từng bước hoạt động

Mỗi video có thư mục riêng `_work/job_XXXXX/`. Mỗi bước là một tiến trình con riêng (`python -m dubber.run_step <bước> <thư_mục>`). Bước xong thì RAM được trả lại hết, nên máy 8GB không bị đầy RAM dần. Bước xong tạo file `.done_<bước>`, bước lỗi tạo `error_<bước>.txt`. Nhật ký của mỗi bước nằm ở `log_<bước>.txt`.

Có 2 làn chạy song song, giống dây chuyền nhà máy (`worker.py`):
- **Làn 1 – chuẩn bị:** download → separate → transcribe → ocr → translate. Làn này chạy ưu tiên thấp (nice 10) và chỉ dùng 2 luồng CPU.
- **Làn 2 – lồng tiếng:** tts → mix → render → qc. Làn này chạy nice 5.
- Làn 1 chỉ chuẩn bị trước tối đa 1 video (`prep_ahead`). Video đã chuẩn bị xong mang trạng thái `ready` ("Chờ lồng tiếng").
- Trước mỗi bước nặng, hệ thống chờ cho tới khi đủ RAM trống: separate và tts cần 1500MB, transcribe 900MB, ocr 700MB.

| # | Bước | File tạo ra | Thời gian đo trên M1 (video 2 phút) |
|---|------|-------------|---------------------|
| 1 | download | source.mp4, audio.wav, meta.json | khoảng 2–40 giây |
| 2 | separate | vocals.wav, background.wav, vocals_16k.wav, separate.json | khoảng 20 giây (demucs) |
| 3 | transcribe | transcript.json | khoảng 18 giây |
| 4 | ocr | ocr.json | khoảng 10–15 giây (bản cũ 66 giây) |
| 5 | translate | translation.json | khoảng 10–30 giây (bản cũ 194 giây) |
| 6 | tts | tts/*.wav, fit.json | khoảng 60–130 giây |
| 7 | mix | mix.wav | khoảng 3 giây (bản cũ 44 giây) |
| 8 | render | THANH_PHAM/<ngày>/*.mp4 + .srt + .txt, render.json | khoảng 30 giây |
| 9 | qc | qc.json; xóa file âm thanh lớn nếu keep_work_files=false | khoảng 1 giây |

## 1. download – Tải video không logo (`steps/download.py`)
1. Lấy mã video (aweme_id) từ link. Link có thể có dạng `/video/ID`, `modal_id=ID`, hoặc là link rút gọn `v.douyin.com` (khi đó mở link để lấy link thật).
2. Hỏi thông tin video, thử lần lượt từng cách, cách nào được thì dừng:
   - **open:** gửi `GET https://www.douyin.com/aweme/v1/web/aweme/detail/?aweme_id=ID&aid=6383`, kèm header `Origin/Referer: https://open.douyin.com` và User-Agent Chrome bản mới (146). Cách này không cần chữ ký hay cookie.
   - **feed:** gọi API của app điện thoại `api5-normal-c-hl.amemv.com/aweme/v1/feed/?aweme_id=ID&aid=1128`, rồi lọc đúng video trong kết quả.
   - **page:** dùng Playwright mở Chrome thật ở chế độ ẩn, với hồ sơ riêng lưu lâu dài tại `~/.douyin_dubber/douyin_profile`. Vào douyin.com để nhận cookie khách (ttwid, s_v_web_id), rồi gọi `fetch()` API chi tiết ngay trong trang. Mã bảo vệ của Douyin (bdms/secsdk) sẽ tự gắn chữ ký `a_bogus` cùng các tham số khác. Nếu vẫn không được thì mở trang video và đọc dữ liệu mà trang tự tải.
   - **share** (trang chia sẻ cũ, đã chết từ khoảng 8/2026) và **yt-dlp** (lỗi "Fresh cookies"): chỉ giữ lại để dự phòng.
3. Chọn link trong `bit_rate[]` có chất lượng cao nhất (chiều cao ≤1920). `play` là bản không logo, `playwm` là bản có logo.
4. Dùng ffprobe đọc thông tin video. Nếu video có cờ xoay 90/270 độ thì đổi chỗ rộng và cao. Sau đó dùng ffmpeg tách âm thanh ra `audio.wav` (44.1kHz stereo).
5. Video từ file có sẵn (đường dẫn, hoặc thả vào thư mục `DAU_VAO/`) thì chỉ cần chép vào.

## 2. separate – Tách giọng khỏi nhạc nền (`steps/separate.py`)
- **demucs-mlx (htdemucs)** chạy trên GPU chip Apple qua thư viện MLX. Model nhìn "bản đồ tần số" của âm thanh theo thời gian để biết phần nào là giọng người.
- Âm thanh được cắt thành **khúc 30 giây, chồng mép 1 giây, nối mượt (crossfade)**, với `batch_size=1`, `compile=False` và `DEMUCS_MLX_COMPILE_FORWARD=0`. Nếu chạy cả bài một lần, GPU M1 sẽ bị hết thời gian chờ (lỗi "GPU Timeout").
- Nếu demucs lỗi, hệ thống chuyển sang dự phòng: Spleeter 2 stem (sherpa-onnx, CPU), sau đó UVR MDX-Net (chậm).
- Cộng các phần "không phải giọng" (drums, bass, other) lại thành `background.wav`. Ngoài ra tạo thêm `vocals_16k.wav` (mono 16kHz) để dùng cho bước nghe.

## 3. transcribe – Nghe tiếng Trung và phân biệt người nói (`steps/transcribe.py`)
- **Silero VAD:** tìm các đoạn có tiếng người (ngưỡng 0.35, im lặng tối thiểu 0.3 giây, mỗi đoạn dài tối đa 9 giây).
- **Phân biệt người nói:** dùng sherpa-onnx với model pyannote segmentation 3.0 và giọng đặc trưng 3D-Speaker CAM++ zh-cn. Gom cụm với ngưỡng 0.85; ngưỡng thấp hơn sẽ tách một người thành quá nhiều người.
- **SenseVoice-Small int8:** chuyển giọng thành chữ Hán, kèm cảm xúc và sự kiện (tiếng cười...). Chạy rất nhanh: 57 giây âm thanh mất khoảng 7 giây.
- **Giới tính:** đo cao độ trung vị (Praat/parselmouth). Từ 170Hz trở lên là nữ.
- Người nói tổng cộng dưới 3 giây được gộp vào người cùng giới có cao độ gần nhất.
- Kết quả: `lines[{id,start,end,spk,zh,emotion,event}]`, `speakers{S#:{talk_time,f0,gender_voice}}`, và `nonspeech` (những đoạn cười, khóc... để giữ lại khi trộn).

## 4. ocr – Dò phụ đề chữ Trung in cứng trên hình (`steps/ocr.py`)
Bản tối ưu chia làm 2 lượt:
1. **Lượt dò:** giải mã chỉ các **keyframe** (`-skip_frame nokey`, rất nhanh), tối đa 40 khung. Nếu ít hơn 24 keyframe thì chụp thêm ở **giữa các câu thoại**, là lúc chắc chắn phụ đề đang hiện. RapidOCR đọc toàn khung. Chữ đứng yên suốt video (logo, tên kênh) bị loại. **Vị trí có chữ Hán lặp lại nhiều nhất** chính là vùng phụ đề. Nếu không thấy vùng nào thì bỏ qua lượt 2, vì video không có hardsub.
2. **Lượt đọc:** ffmpeg cắt riêng dải phụ đề (±5% chiều cao) với 2 khung mỗi giây. Khung nào **gần giống khung vừa đọc** (chênh lệch điểm ảnh trung bình dưới 3/255) thì dùng lại kết quả cũ. Mẹo: RapidOCR mặc định phóng ảnh nhỏ lên 736px (`limit_type=min`), rất chậm. Ở đây đặt `det_limit_type=max` để giữ nguyên kích thước ảnh.
- Kết quả gồm `band` (tọa độ 0..1), `intervals` (các khoảng thời gian có chữ, dùng để làm mờ) và `captions` (chữ theo thời gian, giúp AI dịch sửa chỗ nghe nhầm).

## 5. translate – Dịch và Việt hóa (`steps/translate.py`, `llm.py`)
- Gửi **cả kịch bản** cho Gemini trong một lượt: mọi câu, ai nói, thời lượng, chữ OCR. Làm vậy để giữ mạch truyện và xưng hô nhất quán.
- Mỗi câu có `max_syll` = số âm tiết tối đa vừa khung thời gian. Công thức: thời gian có thể nói × tốc độ đọc đo được (âm tiết/giây) × 0.95. Tiếng Việt đếm âm tiết bằng cách đếm số từ cách nhau bởi dấu cách.
- Quy tắc trong đề (RULES): giọng miền Nam; xưng hô theo quan hệ, cổ trang dùng ta–ngươi; tên riêng đổi sang Hán-Việt; số viết thành chữ; tin chữ OCR khi ASR nghe sai; câu chỉ có tiếng ồn thì để trống.
- AI trả về JSON gồm: `genre`, `title_vi`, `summary_vi`, `speakers{S#:{name_vi,gender,age,role}}`, `lines[{id,vi}]`.
- Video dài được cắt thành phần 110 câu. Phần 1 dịch trước để chốt thể loại và tên nhân vật, các phần sau **dịch song song** (3 yêu cầu cùng lúc). Câu nào bị thiếu sẽ được dịch bổ sung.
- Model được thử lần lượt: gemini-3.5-flash-lite → gemini-3.1-flash-lite → gemma-4-31b-it → gemma-4-26b-a4b-it → gemini-3.5-flash.
  - Gặp 429 (hết lượt) thì chuyển model.
  - Gặp 404 thì đánh dấu model chết.
  - Model trả về rỗng cũng tính là lỗi.
  - Nếu tất cả đều lỗi: chờ 60 giây rồi thử thêm 1 vòng.
- **Tối ưu:** `thinking_level="minimal"`, tức là tắt chế độ suy nghĩ; Gemma 4 31B có suy nghĩ mất khoảng 190 giây. `response_mime_type="application/json"` để AI trả về JSON thuần. Model không nhận các tùy chọn này thì tự bỏ tùy chọn và thử lại.

## 6. tts – Đọc giọng Việt và khớp thời gian (`steps/tts.py`, `ttsload.py`)
- **VieNeu-TTS v3 Turbo** (mặc định: `tts_mode=turbo`, ONNX, CPU, 48kHz, Apache-2.0) có 8 giọng miền Nam:
  - Nam: Thái Sơn, Minh Triết, Đức Trí, Adam.
  - Nữ: Thục Đoan, Mỹ Duyên, Kim Thanh, Thùy Dung.
  - RTF khoảng 0.35–0.6 trên CPU, tốc độ nhanh và nhẹ RAM.
- **OpenBMB VoxCPM2** (tùy chọn: `tts_mode=voxcpm`, model `openbmb/VoxCPM2`):
  - Mô hình Diffusion Autoregressive 2B thông minh từ OpenBMB, hỗ trợ tiếng Việt bản xứ.
  - Khả năng **Voice Cloning** (sao chép bất kỳ giọng mẫu nào từ file audio mẫu hoặc giọng của diễn viên) và **Voice Design**.
  - Chất lượng âm thanh cao cấp 48kHz. Khuyên dùng khi có GPU hoặc khi cần sao chép giọng đặc thù.
  - `ttsload.materialize()` biến symlink trong cache HuggingFace thành file thật. Cần làm vậy vì onnxruntime bản mới từ chối file dữ liệu nằm ngoài thư mục model.
- **Biến thể giọng:** dùng Praat "Change gender" (PSOLA) để đổi cao độ và formant mà vẫn giữ nguyên tốc độ nói.
  - trẻ: +2 nửa cung, formant ×1.04
  - trầm: −2, ×0.97
  - già: −3, ×0.94
  - bé: +4, ×1.12
- **Phân vai:**
  - Xếp nhân vật theo thời lượng nói, nhiều nhất trước.
  - Lấy giới tính và tuổi do AI dịch cung cấp; nếu AI không có thì dùng cao độ. Trẻ con dùng giọng nữ biến thể "bé".
  - Nữ chính dùng `lead_female_voice` (mặc định Thục Đoan).
  - Ưu tiên giọng gốc chưa ai dùng, rồi tới biến thể hợp tuổi. Giọng gốc của vai chính được để dành cho vai chính.
  - Nhiều mã người nói nhưng AI nhận ra cùng một tên thì dùng chung một giọng.
- **Khớp thời gian:**
  - "Ô" của mỗi câu = min(thời điểm câu kế tiếp − thời điểm câu này bắt đầu − 0.08 giây, thời lượng gốc + 1.5 giây mượn khoảng lặng).
  - (a) Trước khi đọc: câu nào ước lượng dài hơn 1.2×1.15 lần ô thì nhờ AI **rút gọn trước**, khỏi phải đọc 2 lần.
  - (b) Đọc tất cả các câu, đo tốc độ đọc thật của từng giọng, lưu vào `~/.douyin_dubber/voice_rates.json` (trung bình động).
  - (c) Câu đọc ra dài hơn 1.2 lần ô thì nhờ AI "viết đơn giản hơn" rồi đọc lại.
  - (d) Câu còn dài hơn ô thì tăng tốc bằng ffmpeg `atempo` (giữ nguyên cao độ), tối đa 1.45 lần.
- `normalize_vi` (`textnorm.py`) đổi số, %, đơn vị, tiền thành chữ để model đọc đúng.

## 7. mix – Trộn âm thanh (`steps/mix.py`)
- Nền = `background.wav`, cộng lại những đoạn không phải lời thoại trong giọng gốc (cười, khóc) ×0.9.
- Đặt từng câu giọng Việt vào đúng giây của nó. Câu nào dài quá ô + 0.3 giây thì cắt bớt và làm nhỏ dần ở đuôi.
- **Ducking:** hạ nhạc nền −5dB khi đang có lời.
  - Đường bao âm lượng được tính trên khung 10ms (100 điểm/giây), rồi nội suy về từng mẫu âm thanh.
  - Nhờ vậy nhanh hơn hàng trăm lần so với làm mượt trên 44.100 điểm/giây.
- Chuẩn hóa độ to về −14 LUFS bằng pyloudnorm, là mức các nền tảng video thường dùng. Phần vượt 0.9 được nén mềm bằng hàm tanh.

## 8. render – Xuất video (`steps/render.py`)
- Phụ đề Việt được vẽ thành ảnh PNG bằng Pillow (font Be Vietnam Pro Bold, viền chữ, hộp nền mờ, tối đa 2 dòng), rồi ghép thành chuỗi ảnh theo thời gian (ffconcat). Cách này không cần libass, vì ffmpeg của Homebrew có thể thiếu thư viện này.
- Làm mờ vùng chữ Trung trong các khoảng thời gian có chữ: `crop` → `gblur=sigma=18` → `overlay ... enable='between(t,a,b)+...'`.
- Mã hóa bằng chip phần cứng `h264_videotoolbox` (6Mbps), âm thanh AAC 192k, bật `faststart`.
- Xuất thêm file `.srt` (phụ đề) và `.txt` (tiêu đề + tóm tắt) nằm cạnh file mp4.

## 9. qc – Kiểm tra chất lượng (`steps/qc.py`)
Video bị đánh dấu "Cần xem" (review) nếu gặp một trong các trường hợp:
- Thời lượng lệch quá 0.5 giây so với bản gốc.
- Video ra không có âm thanh.
- Hơn 15% số câu phải tăng tốc trên 1.3 lần.
- Có câu bị cắt đuôi quá 0.3 giây.
- Hơn 20% số câu không có bản dịch.

Sau khi kiểm tra, xóa các file wav lớn cùng thư mục tts/ và subs/, trừ khi bật `keep_work_files=true`. `source.mp4` và các file json được giữ lại, nên có thể chạy lại từ bất kỳ bước nào (`reset_from` sẽ tự thêm các bước cần thiết).
