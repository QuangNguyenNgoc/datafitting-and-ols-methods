# 📈 Data Fitting & OLS Methods — Group 12

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Typst](https://img.shields.io/badge/Typst-Compiled-green.svg?style=for-the-badge&logo=typst&logoColor=white)](https://typst.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

Dự án nghiên cứu chuyên sâu về **Data Fitting** và các phương pháp bình phương bé nhất thông thường (**Ordinary Least Squares - OLS**) trong môn học **Applied Mathematics and Statistics**. Dự án bao gồm việc phát triển các thuật toán toán học cốt lõi từ đầu (from scratch) bằng Python (chỉ dùng NumPy/SciPy), xây dựng pipeline tiền xử lý dữ liệu chuẩn công nghiệp không rò rỉ thông tin, và ứng dụng thực tế trên dữ liệu bất động sản Melbourne (`Melbourne Housing Dataset`) để so sánh hiệu năng của nhiều lớp mô hình từ tuyến tính cơ bản đến Bayes và phi tuyến (Kernel).

---

## 🌟 Các Tính Năng Nổi Bật

### 🧬 Part 1: Lý Thuyết & Triển Khai OLS Cốt Lõi (From Scratch)
*   **Thuật toán OLS chuẩn xác**: Triển khai giải hệ phương trình chuẩn tắc (Normal Equations) bằng phân rã ma trận tối ưu (QR Decomposition, SVD) để đảm bảo độ ổn định số học cao hơn việc nghịch đảo ma trận trực tiếp $X^T X$.
*   **Thống kê chẩn đoán toàn diện**: Tự động tính toán các chỉ số thống kê toán học phức tạp:
    *   Hệ số xác định $R^2$ và $R^2_{adj}$.
    *   Hệ số F-statistic và p-value để đánh giá mức độ ý nghĩa của toàn bộ mô hình.
    *   Thống kê t-statistic, độ lệch chuẩn sai số (Standard Error), p-value và khoảng tin cậy (Confidence Interval) 95% cho từng hệ số hồi quy $\beta_j$.
*   **Hồi quy chính quy hóa (Regularized Regression)**:
    *   **Ridge Regression**: Triển khai giải đóng toán học với tham số $\lambda$.
    *   **Lasso Regression**: Sử dụng thuật toán tối ưu tọa độ (Coordinate Descent) để thưa thớt hóa các hệ số.
*   **Kiểm định giả thuyết Gauss-Markov**: Tự động phân tích các giả định hồi quy (tính tuyến tính, phân phối chuẩn của sai số, phương sai sai số đồng đều, và tính độc lập).
*   **K-Fold Cross-Validation & Learning Curves**: Đánh giá khả năng tổng quát hóa của mô hình và vẽ đường cong học tập để chẩn đoán hiện tượng Overfitting/Underfitting.

### 🏢 Part 2: Xây Dựng Pipeline Thực Tế & So Sánh Mô Hình
*   **Data Pipeline rạch ròi chống rò rỉ dữ liệu (No Leakage)**: Thiết kế pipeline tiền xử lý hoàn chỉnh không phụ thuộc vào `scikit-learn` cho các bước:
    *   Xử lý dữ liệu bất hợp lệ (ví dụ: diện tích âm, năm xây dựng trước 1800) và chuyển thành dữ liệu khuyết.
    *   Imputation dựa trên trung vị (Median Imputation) được fit hoàn toàn trên tập train và áp dụng đồng nhất lên tập test.
    *   Tạo thêm đặc trưng thông minh (Feature Engineering): `Age` (Tuổi thọ căn nhà), `BuildingArea_per_Room` (Diện tích trên mỗi phòng), `BuildingCoverage` (Mật độ xây dựng), và các cờ đánh dấu dữ liệu khuyết.
    *   Mã hóa biến phân loại (One-Hot Encoding với `drop_first=True`) và Chuẩn hóa (Z-score Scaling).
*   **Loại bỏ đa cộng tuyến**: Vòng lặp tính toán nhân tử phóng đại phương sai (VIF - Variance Inflation Factor) để tự động phát hiện và loại bỏ các đặc trưng bị đa cộng tuyến nghiêm trọng.
*   **Các phương pháp nâng cao**:
    *   **Kernel Ridge Regression**: Áp dụng Kernel Trick (RBF/Polynomial kernel) để mô hình hóa các mối quan hệ phi tuyến phức tạp.
    *   **Bayesian Linear Regression**: Ước lượng phân phối hậu nghiệm (posterior) của các trọng số $\beta$ và đưa ra khoảng dự báo bất định (uncertainty estimation).

---

## 📂 Cấu Trúc Mã Nguồn Chi Tiết

```text
datafitting-and-ols-methods/
├── README.md                      # Hướng dẫn và tổng quan dự án
├── requirements.txt               # Các thư viện phụ thuộc của dự án
│
├── part1/                         # Triển khai lý thuyết hồi quy cốt lõi
│   ├── ols_implementation.py      # Lớp OLS from scratch cùng các ước lượng thống kê
│   ├── ridge_lasso.py             # Lớp Ridge và Lasso từ đầu (Coordinate Descent)
│   ├── residual_analysis.py       # Kiểm định giả thuyết Gauss-Markov & chuẩn hóa phần dư
│   ├── cross_validation.py        # Thuật toán chia K-Fold & tạo Learning Curves
│   ├── part1_notebook.ipynb       # Thực nghiệm lý thuyết trên dữ liệu mô phỏng
│   └── test_ols_math.py           # Unit test kiểm định độ khớp toán học với statsmodels
│
├── part2/                         # Ứng dụng thực tế trên Melbourne Housing Dataset
│   ├── data/
│   │   └── melb_data.csv          # File dữ liệu bất động sản Melbourne gốc
│   ├── data_pipeline.py           # Thiết kế DataPipeline tiền xử lý sạch sẽ chống rò rỉ
│   ├── model_comparison.py        # So sánh chẩn đoán mô hình & kiểm tra VIF đa cộng tuyến
│   ├── advanced_methods.py        # Triển khai Kernel Ridge và Bayesian Regression
│   ├── part2_notebook.ipynb       # Quy trình phân tích toàn diện (EDA, Preprocessing, So sánh)
│   └── readme.md                  # Tài liệu chi tiết về đặc trưng và các bước tiền xử lý
│
└── report/                        # Báo cáo học thuật và kết quả tổng hợp
    ├── chapters/
    │   ├── 01_part1.typ           # Nội dung báo cáo phần lý thuyết toán học & triển khai
    │   ├── 02_part2.typ           # Nội dung báo cáo phần ứng dụng thực tế & kết quả
    │   └── 04_references.typ      # Danh mục tài liệu tham khảo khoa học
    ├── theme.typ                  # Cấu hình giao diện, font chữ, layout báo cáo Typst
    ├── report.typ                 # File cấu trúc tổng hợp báo cáo chính
    └── report.pdf                 # Báo cáo PDF hoàn chỉnh được biên dịch tự động
```

---

## 🛠️ Cài Đặt & Chạy Chương Trình

### 1. Chuẩn bị môi trường
Yêu cầu Python từ phiên bản **3.9** trở lên. Cài đặt các thư viện cần thiết bằng lệnh dưới đây:

```bash
pip install -r requirements.txt
```

### 2. Chạy thử nghiệm và chạy thực tế các Notebook
Bạn có thể mở Jupyter Notebook hoặc Jupyter Lab để khám phá các phân tích và biểu đồ trực quan:

```bash
jupyter notebook
```
*   **Part 1**: Mở `part1/part1_notebook.ipynb` để xem cách so sánh mô hình tự triển khai với các thư viện tiêu chuẩn.
*   **Part 2**: Mở `part2/part2_notebook.ipynb` để theo dõi toàn bộ luồng xử lý từ EDA, tiền xử lý dữ liệu với `DataPipeline`, ước lượng VIF loại bỏ đa cộng tuyến, vẽ biểu đồ phân tích phần dư Gauss-Markov, huấn luyện và so sánh mô hình.

### 3. Biên dịch báo cáo Typst
Nếu bạn có cài đặt `typst` CLI, bạn có thể tự biên dịch hoặc theo dõi thay đổi của báo cáo:

```bash
# Biên dịch một lần ra PDF
typst compile report/report.typ report/report.pdf

# Theo dõi thay đổi thời gian thực và tự động cập nhật PDF
typst watch report/report.typ report/report.pdf
```

---

## 📊 Tóm Tắt Kết Quả Nghiên Cứu & Chẩn Đoán Mô Hình

### 1. Phân Tích Đa Cộng Tuyến (VIF)
Qua vòng lặp chẩn đoán VIF, các đặc trưng có hệ số phóng đại phương sai cao được xác định và loại bỏ một cách hệ thống (ví dụ: `Bedroom2` có độ tương quan tuyến tính rất mạnh với `Rooms`), giúp mô hình OLS ổn định hơn và tránh được sự bùng nổ phương sai của hệ số ước lượng.

### 2. Chẩn Đoán Giả Thuyết Gauss-Markov
*   **Phương sai sai số thay đổi (Heteroscedasticity)**: Biểu đồ phần dư vs Giá trị dự báo chỉ ra sự không đồng đều của phương sai phần dư khi giá trị dự báo tăng lên (đặc trưng của dữ liệu giá bất động sản).
*   **Phân phối của sai số**: Biểu đồ Normal Q-Q chỉ ra rằng sai số có đuôi nặng (heavy tails), không hoàn toàn tuân theo phân phối chuẩn lý thuyết do sự xuất hiện của các căn nhà có giá trị ngoại lai cực lớn (outliers).

### 3. So Sánh Hiệu Năng Các Mô Hình trên Tập Test
| Mô hình | MAE | RMSE | $R^2$ | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **OLS (Scratch/Sklearn)** | Tốt | Tốt | ~0.65-0.70 | Baseline tuyến tính mạnh mẽ sau khi loại bỏ đa cộng tuyến |
| **Ridge Regression** | Tốt | Tốt | Ổn định | Giúp co cụm hệ số khi có nhiều đặc trưng nhiễu |
| **Lasso Regression** | Tốt | Tốt | Tốt | Triệt tiêu các hệ số của các đặc trưng không quan trọng |
| **Bayesian Linear** | Tốt | Tốt | Ổn định | Cung cấp thêm khoảng tin cậy cho từng giá trị dự đoán |
| **Kernel Ridge (RBF)** | Rất Tốt | Rất Tốt | Cao hơn | Nắm bắt hiệu quả các mối quan hệ phi tuyến giữa vị trí địa lý, diện tích với giá nhà |

---

## 👥 Thành Viên Nhóm 12

| STT | MSSV | Họ và Tên | Vai trò | Nhiệm vụ chính trong dự án | Đóng góp |
| :---: | :---: | :--- | :---: | :--- | :---: |
| 1 | 24120002 | **Đinh Đức Hiếu** | Thành viên | Phần 1: Xây dựng thuật toán cốt lõi cho OLS, Ridge và VIF | 100% |
| 2 | 24120049 | **Liên Trung Hiếu** | Thành viên | Phần 2: Huấn luyện, tối ưu siêu tham số và so sánh các mô hình | 100% |
| 3 | 24120064 | **Trương Đình Nhật Huy** | Thành viên | Phần 2: Làm sạch dữ liệu, phân tích khám phá dữ liệu và xây dựng quy trình tiền xử lý | 100% |
| 4 | 24120127 | **Nguyễn Ngọc Quang** | Trưởng nhóm | Điều phối dự án và phần 1: Kiểm chứng, phân tích thống kê phương pháp OLS | 100% |
| 5 | 24120149 | **Đặng Quang Tiến** | Thành viên | Phần 2: Phân tích chẩn đoán phần dư, trực quan hóa và tổng hợp báo cáo | 100% |

---
*Dự án được thực hiện phục vụ cho bài tập lớn môn Toán ứng dụng và thống kê.*
