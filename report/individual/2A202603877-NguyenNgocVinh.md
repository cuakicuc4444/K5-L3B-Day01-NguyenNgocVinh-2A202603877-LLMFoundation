# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603877
- **Họ và tên:** Nguyễn Ngọc Vinh
- **SEED** (in ở ô đầu notebook):5132

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
| Phương pháp | F1 test | F1 SUB 40 | p95 (ms/tin) | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.867 | 0.857 | 0.07 | 0.0000 |
| **2. BoW + MLP (deep learning)** | 0.938 | 0.952 | 0.20 | 0.0000 |
| **3a. Encoder (MiniLM) + kNN** | 0.911 | 0.909 | 57.80 | 0.0000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.935 | 0.952 | 35.85 | 0.0000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.931 | 0.952 | 35.74 | 0.0000 |

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
 Dự đoán 1 (`ml_f1`)
   Đoán: F1 ~ 0.85, precision cao hơn recall.
   Ra: F1 test = 0.867, precision = 0.987, recall = 0.772.
   Vì sao lệch: Lệch nhẹ về F1 (+0.017) do tập vocabulary 2,000 từ thu thập được các từ khóa mang tính phân loại spam cực mạnh (chẳng hạn `call`, `txt`, `free`), giúp độ chính xác của lớp dự đoán spam đạt rất cao.
 Dự đoán 2 (`ml_lr`)
   Đoán: `lr=0.001` loss dạng đường thẳng, `lr=50` loss dạng đường cong.
   Ra: Với `lr=50`, đường loss vọt tăng vọt lên khoảng 2.5 ở vài epoch đầu tiên rồi mới bắt đầu giảm dần về sát 0.
   Vì sao lệch: Khi learning rate quá lớn (`lr=50`), độ dài bước nhảy gradient ở các bước đầu tiên vượt quá vùng tối ưu local làm cho trọng số và biên độ lỗi bộc phát mạnh trước khi mô hình tự điều chỉnh hướng giảm loss.
 Dự đoán 3 (`dl_relu`)
   Đoán: Mạng MLP bỏ activation ReLU sẽ tương đương Logistic Regression và làm chỉ số F1 giảm xuống.
   Ra: F1 test vẫn đạt mức 0.938 (bằng với mô hình MLP có ReLU).
   Vì sao lệch: Tập dữ liệu BoW đầu vào có tính phân tách tuyến tính khá tốt đối với cấu hình tham số lớp ẩn $2000 \to 64 \to 1$. Sự kết hợp hai phép biến đổi tuyến tính nối tiếp $W_2(W_1 X)$ đóng vai trò tương tự một phép nén không gian đặc trưng mà không làm suy giảm năng lực phân loại của loss sigmoid ở tầng cuối cùng.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Nhóm case bị tụt điểm mạnh nhất:

Mô hình tụt điểm nhiều nhất là 1. BoW + LogReg, tập trung ở nhóm case Spam dạng tin nhắn ngắn, thiếu từ khóa đặc trưng (ví dụ: các tin nhắn cố tình viết tắt mã dịch vụ hoặc chứa thông tin quảng cáo dịch vụ nhạy cảm/prescription/banned nhưng dùng ký hiệu lạ).

Case 1: "2/2 146tf150p"

Kết quả: Thật = spam, Mô hình đoán = ham.

Cơ chế sai: Tin nhắn chỉ bao gồm các chuỗi ký tự viết tắt dịch vụ tổng đài rất ngắn. Khi đi qua bộ tokenize, các chuỗi này tạo ra các token lạ không nằm trong từ điển Vocabulary (2,000 từ phổ biến nhất) hoặc có trọng số bằng 0, làm cho vector đặc trưng biến thành vector toàn số 0. Mô hình Logistic Regression gán xác suất σ(0 + b) ≈ 0 và phân loại sai thành ham.

Case 2: "Mobile Club: Choose any of the top quality items for your mobile. 7cfca1a"

Kết quả: Thật = spam, Mô hình đoán = ham.

Cơ chế sai: Tin nhắn sử dụng các từ ngữ mang nghĩa thông thường (mobile, club, choose, items) mà không dùng từ khóa bị phạt nặng như FREE hay WIN. Các từ kéo về nhãn ham áp đảo tổng trọng số w^T x + b < 0, dẫn đến mô hình bị lừa và gán nhãn ham.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Yêu cầu: p95 < 50 ms/tin cho 10 triệu SMS/ngày.

Lựa chọn phương án: Phương án 2 (BoW + MLP) hoặc Phương án 1 (BoW + LogReg).

Số liệu dẫn chứng:

Trễ hệ thống (Latency): p95 của BoW + LogReg là 0.07 ms và BoW + MLP là 0.20 ms, thấp hơn rất nhiều so với ngưỡng 50 ms. Trong khi đó, các phương án dùng Embedding (MiniLM) có p95 dao động từ 35 ms đến 57.80 ms (kNN vi phạm ngưỡng p95 < 50 ms với 57.80 ms).

Thông lượng (Throughput): BoW + LogReg đạt 49,407 tin/s; BoW + MLP đạt 49,527 tin/s. Với nhu cầu 10 triệu SMS/ngày (≈ 116 tin/s trung bình, cao điểm khoảng 1,000 - 2,000 tin/s), hai phương án BoW đáp ứng thoải mái trên duy nhất 1 CPU instance giá rẻ.

Chi phí: 0.0000 USD / 1k tin (tự host mô hình nhẹ trên hạ tầng hiện tại).

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
AI Log:

Hỏi: Yêu cầu AI hướng dẫn cách xây dựng hàm train_mlp trong PyTorch xử lý phép tính gradient và cập nhật trọng số Adam đúng chuẩn.

Nhận: AI cung cấp đoạn mã 5 bước gồm: opt.zero_grad(), tính logits, tính BCEWithLogitsLoss, loss.backward() và opt.step().

Sửa: Chỉnh lại kích thước tensor đầu ra logits = model(xb).squeeze(1) để match chính xác với vector nhãn y dạng float32.

Điều hiểu ra:

Một mô hình học máy phức tạp hay LLM đắt tiền không phải lúc nào cũng là giải pháp tối ưu cho sản phẩm thực tế; việc lựa chọn mô hình phụ thuộc chặt chẽ vào bài toán đánh đổi giữa trễ hệ thống (p95), chi phí vận hành và chỉ số chất lượng (F1).

Kỹ thuật Bag-of-Words tuy đơn giản nhưng hiệu quả bất ngờ với các bài toán có dấu hiệu nhận biết trực diện (như từ khóa rác), tuy nhiên điểm yếu cốt lõi của nó là hoàn toàn mù tịt về ngữ cảnh và cấu trúc câu so với mô hình chuỗi hay Embedding.
