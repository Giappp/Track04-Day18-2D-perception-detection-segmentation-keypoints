# Hướng dẫn làm bài Lab 18 — 2D Perception: Detection · Segmentation · Keypoints

## 1. Mục tiêu bài lab

Bài lab này giúp bạn nắm cách các mô hình 2D perception hoạt động trong thực tế ứng dụng an ninh giám sát:
- Đếm người trong ảnh/video
- Phân đoạn đối tượng (person, vehicle, ...)
- Phát hiện keypoints và nhận diện người ngã
- Fine-tune mô hình pose trên dữ liệu custom

Câu hỏi chính của bài lab là: "Camera ở cổng nhà máy cần biết có bao nhiêu người, ai không đội mũ bảo hộ, và ai vừa ngã?" Từ đó, bạn sẽ tự xây dựng các bước xử lý từ đầu đến cuối bằng nhiều mô hình khác nhau.

Notebook chính của bài là [lab_2d_perception_student.ipynb](lab_2d_perception_student.ipynb). Tham khảo thêm [README.md](README.md) và [rubric.md](rubric.md) để hiểu rõ yêu cầu chấm điểm và cấu trúc bài.

---

## 2. Tóm tắt cấu trúc bài làm

### Phần 0 — Setup
- Mở notebook trên Colab
- Chạy 5 ô đầu tiên
- Chọn runtime GPU T4
- Kiểm tra môi trường, tải weights và dataset
- Mục tiêu: đảm bảo notebook chạy được và sẵn sàng cho các phần tiếp theo

### Phần 1 — Object Detection
- Tìm hiểu output của Faster R-CNN và YOLO26
- Viết các hàm:
  - `box_iou`
  - `nms`
  - `batched_nms`
  - (bonus) `average_precision`
- Mục tiêu: học cách lọc box, loại bỏ box chồng chéo và đếm đối tượng đúng cách

### Phần 2 — Segmentation
- So sánh semantic segmentation và instance segmentation
- Viết các hàm:
  - `mask_iou`
  - `polygon_to_mask`
  - `mask_to_yolo_seg`
- Mục tiêu: tạo nhãn segmentation tự động từ mask SAM và chuyển thành định dạng YOLO-seg

### Phần 3 — Keypoints và Pose
- So sánh Keypoint R-CNN và YOLO26-pose
- Viết các hàm:
  - `oks`
  - `joint_angle`
- Mục tiêu: phát hiện keypoints và xác định trạng thái người ngã

### Phần 4 — Fine-tune Pose Model
- Sửa `FLIP_IDX`
- Fine-tune mô hình YOLO26n-pose trên dataset `tiger-pose`
- Phân tích lỗi trên validation set
- Mục tiêu: học cách custom hóa mô hình cho một đối tượng mới và đánh giá nhạy cảm với dữ liệu lật ảnh

---

## 3. Hướng dẫn làm từng phần

### Phần 0 — Setup (10 phút)

Cách làm:
1. Bấm "Open in Colab" ở README hoặc mở file notebook.
2. Chọn "File → Save a copy in Drive".
3. Vào "Runtime → Change runtime type → T4 GPU".
4. Chạy lần lượt 5 ô đầu của phần 0.
5. Chờ phần tải weights và dataset hoàn tất.

Bạn đã thành công khi:
- `device = cuda`
- Có thông báo `🔧 Sẵn sàng: ...`
- Xuất hiện danh sách 9 dấu `✓`
- Có hai ảnh mẫu `bus.jpg` và `zidane.jpg`

Lưu ý:
- Notebook cố định `ultralytics==8.4.171`.
- Nếu không có GPU, notebook vẫn chạy được nhưng phần fine-tune sẽ bị giảm xuống 3 epoch và không tối ưu cho nộp bài.

---

### Phần 1 — Detection (35 phút)

#### 1A — Khảo sát output của detector
Mục tiêu:
- Hiểu output format của Faster R-CNN và YOLO26
- Kiểm tra shape và cách các box được biểu diễn

Cần làm:
- Chạy các ô demo
- Quan sát `boxes`, `labels`, `scores`
- Trả lời Q1

Kết quả mong đợi:
- Faster R-CNN trả về `boxes / labels / scores`
- YOLO26n trả về tensor có shape `(1, 84, 8400)` trên ảnh 640×640 (và `(1, 84, 6300)` cho ảnh `bus.jpg`)

#### 1B — Viết NMS / IoU / batched NMS
Mục tiêu:
- Làm sạch nhiều box chồng chéo
- Chọn box tốt nhất theo IoU và confidence

Các hàm cần hoàn thành:
- `box_iou`: tính IoU giữa các box
- `nms`: loại bỏ box chồng chéo theo ngưỡng IoU
- `batched_nms`: áp dụng NMS theo class để tránh loại bỏ nhầm box của lớp khác nhau

Cách làm:
1. Đọc kỹ mô tả từng bước trong notebook.
2. Thay các dấu `...` bằng code đúng theo logic toán học.
3. Chạy từng ô kiểm tra (`check_box_iou()`, `check_nms()`, `check_batched_nms()`).
4. Nếu sai, đọc message lỗi dưới ô để biết đang sai ở bước nào.

Dấu hiệu đúng:
- IoU(A, B) đúng = `0.3333`
- Với ngưỡng `0.3`, NMS giữ `A` và `C`
- Với ngưỡng `0.5`, NMS giữ `A`, `B`, `C`
- `gate("1B")` hiện `✅ Mục 1B xong`

#### 1C — Đếm người và so sánh NMS
Mục tiêu:
- So sánh các chiến lược:
  - one-to-many + NMS
  - one-to-one
- Tính latency và đếm người đúng

Cần làm:
- Chạy các ô đã chuẩn bị trong notebook
- Kiểm tra số lượng box trước và sau NMS
- Trả lời Q2, Q3

Kết quả mong đợi:
- Số box giảm từ `51 → 5 → 5`
- NMS của bạn khớp với NMS của Ultralytics
- Camera đếm được khoảng `4 người` và `1 xe buýt`
- Có bảng latency đủ 4 cấu hình

#### 1D — Bonus: average_precision
Mục tiêu:
- Tính AP (Average Precision) theo ví dụ trên slide

Các bước:
- Hoàn thành hàm `average_precision`
- Chạy kiểm tra
- Vẽ đường PR

Kết quả mong đợi:
- `✅ average_precision đạt`
- Đường PR gần với ví dụ trên slide với AP khoảng `0.535`

---

### Phần 2 — Segmentation (30 phút)

#### 2A — Semantic vs instance segmentation
Mục tiêu:
- Hiểu vì sao semantic segmentation có thể đếm sai số người
- Hiểu vai trò của panoptic segmentation

Cần làm:
- Chạy các ô mở rộng của phần 2A
- Quan sát bản đồ semantic và mask instance
- Trả lời Q4

Kết quả mong đợi:
- Semantic map của class `person` có nhiều vùng liên thông hơn số người thật
- Một nguyên nhân gây thiếu: các phần người bị tách hoặc chỉ là một nửa thân không kết nối
- Một nguyên nhân gây thừa: background / vật thể / nhiều vùng liền kề bị gán cùng class `person`
- Panoptic segmentation giải quyết bằng cách gán từng instance và class riêng biệt, đồng thời tách `thing` và `stuff`

#### 2B — Viết mask IoU và chuyển mask -> YOLO-seg
Mục tiêu:
- So sánh hai mask bằng IoU
- Chuyển polygon thành mask và ngược lại
- Tạo nhãn YOLO-seg từ mask SAM

Các hàm cần hoàn thành:
- `mask_iou`
- `polygon_to_mask`
- `mask_to_yolo_seg`

Cách làm:
1. Đọc mô tả từng bước trong notebook
2. Tính diện tích giao và hợp theo đúng công thức
3. Chạy bộ kiểm tra từng hàm
4. Kiểm tra kết quả `IoU` và hình ảnh

Kết quả mong đợi:
- `✅ mask_iou đạt`
- `✅ polygon_to_mask đạt`
- `✅ mask_to_yolo_seg đạt`
- Ví dụ hai hình vuông cho `IoU = 0.1429`
- Bảng ghép cặp YOLO26n-seg với Mask R-CNN có khoảng 5 dòng
- Mask IoU khoảng `0.84` đến `0.94`

#### 2C — Auto-label bằng SAM
Mục tiêu:
- Tạo file nhãn tự động cho đối tượng mới
- Kiểm tra định dạng YOLO-seg và độ khớp với mask SAM

Cần làm:
- Chạy các ô auto-label
- Kiểm tra file `submission/autolabel/bus.txt`
- Trả lời Q6

Kết quả mong đợi:
- Ảnh A: mask mô tả cả người (IoU ≈ 0.99)
- Ảnh B: mảnh nhỏ (IoU ≈ 0.00)
- File `autolabel/bus.txt` có 5 dòng
- Polygon và mask khớp chặt với vật thể
- Dòng `✅ autolabel/bus.txt hợp lệ`

---

### Phần 3 — Keypoints và Pose (25 phút)

#### 3A — Keypoint R-CNN vs YOLO26-pose
Mục tiêu:
- Hiểu sự khác nhau giữa định dạng output keypoint của model
- Biết cách đọc `xy` và `conf`

Cần làm:
- Chạy 2 ô đầu
- Đổi `KP_THR = -100` và chạy lại ô thứ hai
- Trả lời Q7

Kết quả mong đợi:
- Keypoint R-CNN output shape `(2, 17, 3)`
- YOLO26n-pose output shape `(2, 17, 2)`
- Khi bỏ ngưỡng, model vẽ cả đầu gối, mắt cá dù ngoài ảnh

#### 3B — Tính OKS
Mục tiêu:
- Đánh giá độ chính xác của keypoint bằng OKS
- Hiểu ảnh hưởng của kích thước người đến điểm số

Cần làm:
- Hoàn thành hàm `oks`
- Chạy bộ kiểm tra
- Trả lời Q8

Kết quả mong đợi:
- `✅ oks đạt`
- Đồ thị có ba đường cong theo kích thước người
- Dòng ví dụ: "Lệch 8 px: mắt còn 0.53, hông còn 0.97"

#### 3C — Tính góc khớp và phát hiện ngã
Mục tiêu:
- Từ keypoints, tính góc ở khớp để nhận dạng tư thế ngã

Cần làm:
- Hoàn thành hàm `joint_angle`
- Chạy ô áp dụng lên ảnh
- Trả lời Q9

Kết quả mong đợi:
- `✅ joint_angle đạt`
- Góc khớp hiển thị đúng hướng và có thể dùng làm dấu hiệu "ngã"

---

### Phần 4 — Fine-tune Pose Model (20 phút)

#### 4A — Sửa `FLIP_IDX`
Mục tiêu:
- Khôi phục đúng quy ước giải phẫu khi lật ảnh
- Đảm bảo training và evaluation không bị sai do keypoint bị đánh nhầm vị trí

Cần làm:
- Sửa `FLIP_IDX` theo quy ước của dữ liệu `tiger-pose`
- Chạy kiểm tra

Kết quả mong đợi:
- `FLIP_IDX` đúng
- Fine-tune không bị lệch khi ảnh bị lật

#### 4B — Fine-tune YOLO26n-pose
Mục tiêu:
- Huấn luyện mô hình trên dữ liệu custom
- So sánh hiệu năng trước/sau fine-tune

Cách làm:
1. Chạy phần train của notebook
2. Sử dụng GPU T4 nếu có
3. Đợi mô hình học xong (thường 40 epoch với `imgsz=640`)
4. Quan sát bảng mAP
5. Ghi nhận lỗi phát hiện trên validation set

#### Q11 — Phân tích lỗi
Đây là câu hỏi quan trọng trong rubric. Bạn phải:
- Chỉ ra ít nhất hai kiểu lỗi quan sát thấy trên validation set
- Mỗi lỗi phải gắn với ảnh cụ thể của chính bạn
- Đề xuất cách khắc phục cho từng lỗi

Điểm quan trọng:
- Trả lời chung chung không đủ điểm
- Cần nêu rõ lỗi và cách sửa cụ thể

---

## 4. Cách làm hiệu quả khi làm notebook

### Workflow đề xuất
1. Chạy từng ô theo thứ tự, không bỏ qua các ô kiểm tra.
2. Đọc chỗ lỗi ngay khi nó xuất hiện; không để kéo dài quá 3 phút.
3. Nếu bị kẹt ở một hàm, có thể dùng `lifeline=True` để tiếp tục, nhưng hãy quay lại làm đúng sau.
4. Lưu lại output của các ô quan trọng để làm câu trả lời Q1–Q12.
5. Khi nào đã hoàn thành, chạy lại toàn bộ notebook bằng chức năng `Restart session and run all` để kiểm tra tổng quát.

### Mẹo khi code sai
- Kiểm tra shape tensor trước khi tính toán
- Luôn đảm bảo giá trị nằm trong phạm vi [0, 1] cho YOLO-seg
- Với NMS: cần sắp xếp theo confidence giảm dần, rồi giữ box theo threshold
- Với IoU: cẩn thận với phép toán `max` và `clamp(min=0)` để tránh negative area
- Với keypoints: xử lý đúng keypoint không có nhãn (`nan`/`None`/vị trí không hợp lệ)

---

## 5. Yêu cầu nộp bài

Theo [rubric.md](rubric.md), repo cần có các file sau:
- [lab_2d_perception_student.ipynb](lab_2d_perception_student.ipynb) chạy hết, giữ nguyên output
- `submission/ket_qua.json`
- `submission/autolabel/bus.txt`
- (tuỳ chọn) báo cáo bonus trong `submission/`

### Lưu ý nộp lên GitHub
- Đẩy repo lên public
- Tạo repo theo tên: `<username-của-bạn>/Track04-Day18-2D-perception-detection-segmentation-keypoints`
- Dán URL repo vào ô LMS
- Giữ repo public cho đến khi có điểm

---

## 6. Checklist cuối cùng trước khi nộp

- [ ] Notebook đã chạy xong từ đầu tới cuối
- [ ] Không còn ô nào báo lỗi
- [ ] Đã trả lời Q1–Q12
- [ ] Đã hoàn thành các hàm bắt buộc: `box_iou`, `nms`, `batched_nms`, `mask_iou`, `polygon_to_mask`, `mask_to_yolo_seg`, `oks`, `joint_angle`
- [ ] File `submission/ket_qua.json` tồn tại
- [ ] File `submission/autolabel/bus.txt` tồn tại và hợp lệ
- [ ] Repo GitHub public và URL đúng

---

## 7. Tóm tắt nhanh

Nếu cần làm nhanh, hãy nhớ 4 nguyên tắc chính:
1. Chạy từng phần theo thứ tự, không bỏ qua setup.
2. Hoàn thành từng hàm TODO đúng logic, không chỉ copy code.
3. Đọc kỹ chỗ báo lỗi để biết sai ở bước nào.
4. Luôn kiểm tra output và file nộp cuối cùng trước khi submit.

Chúc bạn làm bài tốt và đạt điểm cao!
