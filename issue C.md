# [Task C] Phân tích, Chẩn đoán & Biện luận Báo cáo (Analyst & Reporter)
## 1. Danh sách Công việc (Task Flow)

Xem các task trong này được đánh dấu tương ứng từ C1->C8: [part2_team_structure.pdf](https://github.com/user-attachments/files/28245456/part2_team_structure.pdf)

* [ ] **Task C1 (Phase 1):** Đọc và phân tích bảng VIF, t-statistics, p-values từ output của Member B. Quyết định loại bỏ biến nào, cập nhật vào biên bản vòng lặp và chốt `final_drop_list` cho Member A.
* [ ] **Task C8 (Phase 1 - Bonus):** Lập trình lõi thuật toán phi tuyến (Kernel Ridge) vào file `advanced_methods.py`. Bàn giao hàm bọc (wrapper) để Member B cắm vào lò huấn luyện `train_models`.
* [ ] **Task C2 (Phase 3):** Gọi hàm `evaluate_gauss_markov_assumptions` và `residual_plots()` để vẽ 4 biểu đồ chẩn đoán phần dư cho mô hình vô địch.
* [ ] **Task C3 (Phase 3):** Gọi hàm `plot_coefficients()` (đưa `metadata['feature_names']` vào) để trực quan hóa tầm quan trọng của các biến sạch.
* [ ] **Task C4 (Phase 3):** Trực quan hóa Ridge Trace (sự hội tụ của hệ số $\beta$ khi tăng $\lambda$) và đường cong K-Fold CV.
* [ ] **Task C5, C6, C7 (Phase 4):** Trích xuất toàn bộ insights từ Notebook, viết phân tích biện luận Toán học - Kinh tế và biên dịch thành báo cáo PDF bằng Typst.

---

## 2. Bản đồ Luồng Quyết định & Phân tích (ASCII Flow)

Để Member C hiểu rõ vai trò "Thẩm phán" và "Người kể chuyện" của mình, luồng công việc được chia làm 2 hệ thống cốt lõi sau:

**A. SƠ ĐỒ QUYẾT ĐỊNH VÒNG LẶP CHẨN ĐOÁN (Phase 1 - Gatekeeper Flow)**
Biểu đồ này mô tả cách DA đọc số liệu và điều khiển vòng lặp của Data Engineer (DE).

```text
[Bảng Output: VIF, SE, p-value từ Member B]
                 │
                 ▼
         (Task C1) THẨM ĐỊNH TOÁN HỌC
                 │
                 ├─> [Cửa 1] VIF > 10? 
                 │      └─> Bị đa cộng tuyến nặng. Xem xét loại bỏ.
                 │
                 ├─> [Cửa 2] p-value > 0.05? 
                 │      └─> Biến không có ý nghĩa thống kê do sai số (SE) bị lạm phát.
                 │
                 └─> [Cửa 3] Kéo theo hệ quả Kinh tế?
                        └─> Ví dụ: Rooms và Bedroom2 trùng lặp. Giữ lại Rooms vì bao quát hơn.
                 │
                 ▼
 [Đưa ra Phán quyết] ──(Vòng lặp)──> Báo Member A cập nhật current_drop
                 │
          (Nếu toàn bộ VIF < 5)
                 ▼
        CHỐT final_drop_list!

```

**B. SƠ ĐỒ BIỆN LUẬN BÁO CÁO (Phase 3 & 4 - Interpretation Engine)**
Biểu đồ này mô tả cách chuyển hóa các mảng số vô tri (Numpy) thành ngôn ngữ Kinh tế Bất động sản trong báo cáo Typst.

```text
[Từ điển results_dict & Bảng Leaderboard từ Member B]
                 │
                 ▼
       (Task C2, C3, C4) TRỰC QUAN HÓA TRÊN NOTEBOOK
                 ├─> Hàm plot_coefficients() ──> [Biểu đồ Tầm quan trọng của Đặc trưng]
                 ├─> Hàm residual_plots()    ──> [4 Biểu đồ Kiểm định Gauss-Markov]
                 └─> Dữ liệu cv_scores       ──> [Biểu đồ Ridge Trace & CV Error]
                 │
                 ▼
       (Task C5, C6, C7) BIỆN LUẬN THỰC TẾ (Viết vào Typst Report)
                 │
                 ├─> Đọc Hệ số Beta ──> "Nhà cách trung tâm thêm 1km thì giá giảm bao nhiêu?"
                 ├─> Đọc Phần dư    ──> "Sai số dạng hình phễu chứng tỏ OLS dự đoán rất kém cho giới siêu giàu."
                 └─> Đọc RMSE / R2  ──> "Ridge Regression thắng OLS nhờ khả năng bóp nghẹt phương sai của các biến nhiễu."

```

---

## 3. Sứ mệnh & Tư duy cốt lõi (Bắt buộc đọc kỹ)

Code của Member A và Member B dù có chạy nhanh và sạch đến đâu, nếu không có phần phân tích của Member C, đồ án chỉ là một cỗ máy tính toán vô hồn.

* **Quyền lực đi kèm trách nhiệm tại Phase 1:** Bạn không được phép xóa cột một cách máy móc chỉ vì thấy VIF cao. Bạn phải đối chiếu với thực tế thị trường. Hai biến tương quan mạnh, biến nào dễ thu thập dữ liệu hơn ở ngoài đời? Biến nào mang lại ngữ nghĩa kinh tế rõ ràng hơn? Bạn là người giải thích lý do giữ/bỏ vào trong báo cáo.
* **Toán học phục vụ Kinh doanh:** Tại Phase 3, khi nhìn vào biểu đồ `Feature Importance`, việc nói "Biến Distance có trọng số âm lớn" là chưa đủ. Bạn phải diễn dịch thành: *"Vị trí địa lý là yếu tố chi phối mạnh nhất đến giá bất động sản Melbourne, vượt xa các yếu tố diện tích hay số phòng."*
* **Bắt bệnh Mô hình qua Gauss-Markov:** Rất hiếm khi một bộ dữ liệu thực tế thỏa mãn trọn vẹn 5 giả định Gauss-Markov. Đừng sợ hãi nếu biểu đồ phần dư (Residuals vs Fitted) có hình dạng bất thường. Nhiệm vụ của bạn là chỉ ra khuyết điểm đó một cách trung thực: *"Phương sai sai số bị thay đổi (Heteroscedasticity) cho thấy mô hình tuyến tính gặp giới hạn khi dự đoán các biệt thự giá cao đột biến."* Giảng viên đánh giá cao sự trung thực và hiểu biết này hơn là một mô hình hoàn hảo nhưng bị làm giả số liệu.
* **Giao diện nâng cao:** Tại Task C8, khi lập trình class `KernelRegression`, bạn phải thiết kế một hàm bọc (wrapper) trả về mảng `y_pred` với cấu trúc I/O giống hệt hàm `ols_fit`. Điều này đảm bảo Member B có thể nhét mô hình của bạn vào `train_models()` mà không cần sửa bất kỳ dòng code cốt lõi nào của họ.

---

## 4. Điểm đồng bộ (Sync Points - Bắt buộc Dừng)

* **Sync Point 2 (Trạm kiểm soát VIF):** Khi Member B in ra bảng VIF đầu tiên. Bạn nắm quyền chủ trì. Đọc kết quả, chỉ đạo vòng lặp loại biến, chốt hạ danh sách và yêu cầu Member A xây dựng `pipeline_best`.
* **Sync Point 3 (Hội nghị Tổng kết):** Sau khi có Bảng Phong Thần (Leaderboard). Cùng team họp lại: Chúng ta chọn mô hình nào làm gương mặt đại diện? Ridge vì RMSE thấp nhất? Hay OLS vì dễ giải thích hệ số? Khi cả team đã thống nhất cốt truyện, bạn mới bắt tay vào gõ code Typst.