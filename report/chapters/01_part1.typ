= Lý thuyết hồi quy tuyến tính và mô phỏng thực nghiệm

== Mô hình sinh dữ liệu

Mô hình dữ liệu giả lập được xây dựng dựa trên phương trình hồi quy tuyến tính:

$ y = X beta_"true" + epsilon, quad epsilon ~ cal(N)(0, sigma^2 I) $

Trong đó $X in RR^(50 times 4)$ là ma trận thiết kế với cột đầu tiên chứa toàn số 1 đại diện cho hệ số chặn, $beta_"true" in RR^4$, và độ lệch chuẩn của nhiễu là $sigma = 1.5$. Việc mô phỏng nhằm kiểm chứng độ chính xác của thuật toán bình phương bé nhất thông thường tự cài đặt so với các thư viện chuẩn.

== Phương pháp bình phương bé nhất thông thường

Vector hệ số ước lượng $hat(beta)$ được xác định nhằm tối thiểu hóa tổng bình phương phần dư. Bằng cách giải hệ phương trình chuẩn:

$ X^T X hat(beta) = X^T y $

Khi ma trận $X^T X$ khả nghịch, nghiệm có dạng đóng:

$ hat(beta) = (X^T X)^(-1) X^T y $

Kết quả ước lượng $hat(beta)$ từ chương trình tự cài đặt khớp với các thư viện `scikit-learn` và `statsmodels` với sai số ở mức $10^(-15)$ do giới hạn phần cứng máy tính.

== Đánh giá mô hình và ma trận hình mũ

Hệ số xác định $R^2$ được tính toán theo công thức:

$ R^2 = 1 - (R S S) / (T S S) $

Trên tập dữ liệu mô phỏng, mô hình giải thích được 94.39% biến thiên của biến mục tiêu ($R^2 = 0.943864$).

Ma trận hình mũ đại diện cho phép chiếu không gian được xác định bởi:
$ H = X (X^T X)^(-1) X^T $

Kết quả kiểm tra thuộc tính của ma trận hình mũ cho thấy vết của ma trận bằng đúng số lượng tham số ước lượng ($upright("tr")(H) = 4$). Thuộc tính lũy đẳng $H^2 = H$ cũng được thỏa mãn với sai số số học ở mức $1.25 times 10^(-16)$.

== Các vấn đề phân tích nâng cao

Đối với đa cộng tuyến, các giá trị nhân tử phóng đại phương sai của dữ liệu mô phỏng đều xấp xỉ 1.0 (ví dụ đối với biến thứ nhất là $1.0798$), xác nhận các biến độc lập không có hiện tượng tương quan tuyến tính mạnh.

Về suy diễn thống kê, phương sai sai số ước lượng đạt $hat(sigma)^2 = 1.922732$. Từ đó, các chỉ số sai số chuẩn, giá trị thống kê $t$ và trị số $p$ được tính toán. Kết quả cho thấy tất cả các hệ số hồi quy đều có ý nghĩa thống kê ở mức ý nghĩa 5% ($p < 0.05$).

Tóm lại, các hàm tự cài đặt như `ols_fit`, `hat_matrix`, `model_metrics`, `coef_inference` và `vif` đều vượt qua các thử nghiệm toán học, tạo cơ sở vững chắc để áp dụng vào thực tế phân tích dữ liệu bất động sản.
