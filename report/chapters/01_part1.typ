= Lý thuyết Data Fitting và Minh họa

== Data Generation Model

Mô hình dữ liệu giả lập được tạo dựa trên phương trình hồi quy tuyến tính:

$ y = X beta_"true" + epsilon, quad epsilon ~ cal(N)(0, sigma^2 I) $

Trong đó $X in RR^(50 times 4)$ là ma trận thiết kế (với cột đầu tiên chứa toàn số 1 đại diện cho hệ số chặn intercept), $beta_"true" in RR^4$, và độ lệch chuẩn của nhiễu là $sigma = 1.5$.
Mục đích của việc giả lập này là để chứng minh rằng thuật toán OLS tự cài đặt tạo ra kết quả chính xác tuyệt đối so với các thư viện chuẩn.

== Phương pháp Ordinary Least Squares (OLS)

Nghiệm OLS $hat(beta)$ được định nghĩa là giá trị làm cực tiểu hóa tổng bình phương phần dư (RSS). Bằng cách giải hệ phương trình chuẩn:

$ X^T X hat(beta) = X^T y $

Khi $X^T X$ khả nghịch, nghiệm OLS có dạng đóng:

$ hat(beta) = (X^T X)^(-1) X^T y $

Thử nghiệm trên dữ liệu giả lập cho thấy, kết quả $hat(beta)$ tính bằng code thủ công hoàn toàn khớp với `scikit-learn` và `statsmodels` (sai số ở mức $10^(-15)$, do giới hạn làm tròn số của máy tính).

== Đánh giá Mô hình & Ma trận Hat

Hệ số xác định $R^2$ được tính bằng:

$ R^2 = 1 - (R S S) / (T S S) $

Trên tập dữ liệu giả lập, mô hình OLS giải thích được 94.39% phương sai của $y$ ($R^2 = 0.943864$).

Ma trận chiếu:
$ H = X (X^T X)^(-1) X^T $
Kiểm tra tính chất của $H$ cho thấy dấu vết $upright("tr")(H) = 4$, đúng bằng số lượng tham số $p+1$. Tính lũy đẳng $H^2 = H$ cũng được thỏa mãn với sai số $1.25 times 10^(-16)$.

== Các vấn đề nâng cao trong Data Fitting

Đa cộng tuyến:
Các giá trị VIF của dữ liệu giả lập đều xấp xỉ 1.0 (ví dụ $x_1: 1.0798$), chứng tỏ các biến độc lập không bị đa cộng tuyến.

Bảng suy diễn hệ số:
Mô hình ước lượng được phương sai sai số $hat(sigma)^2 = 1.922732$. Từ đó tính toán được các sai số chuẩn (Standard Errors), $t$-statistics và $p$-values. Tất cả các hệ số đều có ý nghĩa thống kê ($p < 0.05$).

Tóm lại, toàn bộ các hàm tự cài đặt như `ols_fit`, `hat_matrix`, `model_metrics`, `coef_inference` và `vif` đều vượt qua các bài kiểm thử toán học, sẵn sàng để ứng dụng vào dữ liệu thực tế ở Phần 2.
