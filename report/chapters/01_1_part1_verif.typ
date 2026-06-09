== Nhóm 1: Ước lượng và Đánh giá độ phù hợp

=== 1. Phân rã phương sai và Thống kê tổng thể (`model_metrics`)

Để đánh giá chất lượng của mô hình hồi quy tuyến tính OLS, ta cần phân tích các thành phần phương sai và kiểm định sự phù hợp tổng thể của mô hình. Trong phần này, các công thức toán học lõi được cài đặt hoàn toàn từ đầu (from scratch) bằng Đại số tuyến tính.

*a. Ma trận chiếu (Hat Matrix) và Tính chất* \
Giá trị dự đoán của mô hình được tính bằng biểu thức $hat(y) = X hat(beta)$. Khi thay $hat(beta) = (X^T X)^(-1) X^T y$, ta có:
$ hat(y) = X (X^T X)^(-1) X^T y = H y $
Trong đó, $H = X(X^T X)^(-1)X^T$ được gọi là Ma trận chiếu (Hat matrix) vì nó "đội mũ" (chiếu) vector thực tế $y$ xuống không gian tuyến tính sinh bởi các cột của ma trận $X$.

Định lý toán học chứng minh $H$ có 2 tính chất vô cùng quan trọng:
- *Tính đối xứng:* $ H^T = (X (X^T X)^(-1) X^T)^T = X ((X^T X)^(-1))^T X^T = X (X^T X)^(-1) X^T = H $
- *Tính lũy đẳng (Idempotent):* $ H^2 = H H = X(X^T X)^(-1)X^T X(X^T X)^(-1)X^T = X(X^T X)^(-1) (X^T X) (X^T X)^(-1) X^T = X(X^T X)^(-1)X^T = H $ 
Ta có vector phần dư $e = y - hat(y) = (I - H)y$. \
*Chứng minh tính trực giao:* Phần dư luôn trực giao với không gian sinh bởi ma trận thiết kế $X$:
$ X^T e = X^T (I - H) y = (X^T - X^T X (X^T X)^(-1) X^T) y = (X^T - X^T) y = 0 $
*Hệ quả:*
1. Tổng các phần dư bằng 0 ($sum e_i = 0$) do $X$ có chứa cột hệ số chặn (toàn số 1).
2. Phần dư trực giao với giá trị dự đoán: $hat(y)^T e = (X hat(beta))^T e = hat(beta)^T X^T e = 0$.

*b. Định lý phân rã phương sai* \
Sự biến thiên của dữ liệu (TSS) được phân rã thành hai thành phần: phần được mô hình giải thích (ESS) và phần sai số không giải thích được (RSS).
- *Tổng bình phương phần dư (RSS):* Đo lường sai số của mô hình.
  $ R S S = e^T e = sum_(i=1)^n (y_i - hat(y)_i)^2 $
  
  *Chứng minh (Phân tích ma trận $e^T e$):* \
  Khai triển vector phần dư $e = y - X hat(beta)$, ta có:
  $ e^T e &= (y - X hat(beta))^T (y - X hat(beta)) \
          &= y^T y - y^T X hat(beta) - hat(beta)^T X^T y + hat(beta)^T (X^T X) hat(beta) $
  Vì $y^T X hat(beta)$ là một số vô hướng (scalar), nên chuyển vị của nó bằng chính nó: $(y^T X hat(beta))^T = hat(beta)^T X^T y$.
  Đồng thời, áp dụng phương trình chuẩn (Normal Equations) $X^T X hat(beta) = X^T y$, phương trình được rút gọn thành:
  $ e^T e &= y^T y - 2 hat(beta)^T X^T y + hat(beta)^T (X^T y) \
          &= y^T y - hat(beta)^T X^T y $

- *Tổng bình phương toàn phần (TSS):* Đo lường sự phân tán của dữ liệu so với giá trị trung bình. Dạng vector hóa với vector trung tâm $y_c = y - overline(y) bold(1)$ (với $bold(1)$ là vector cột gồm $n$ số 1):
  $ T S S = y_c^T y_c = sum_(i=1)^n (y_i - overline(y))^2 $
  
  *Chứng minh (Phân tích ma trận $y_c^T y_c$):* \
  Khai triển tích vô hướng của vector trung tâm:
  $ y_c^T y_c &= (y - overline(y) bold(1))^T (y - overline(y) bold(1)) \
              &= y^T y - y^T (overline(y) 1) - (overline(y) bold(1))^T y + (overline(y) bold(1))^T (overline(y) bold(1)) $
  Với tính chất $1^T y = sum_(i=1)^n y_i = n overline(y)$ và $1^T 1 = n$, ta thay vào biểu thức:
  $ y_c^T y_c &= y^T y - 2 overline(y) (1^T y) + overline(y)^2 (1^T 1) \
              &= y^T y - 2 overline(y) (n overline(y)) + overline(y)^2 (n) \
              &= y^T y - 2 n overline(y)^2 + n overline(y)^2 \
              &= y^T y - n overline(y)^2 $
*c. Các chỉ số đánh giá (Metrics)* \
Từ phân rã trên, các phép đo độ phù hợp của mô hình được cài đặt bằng công thức:
- *Hệ số xác định* ($R^2$): Cho biết tỷ lệ phương sai của biến phụ thuộc được giải thích bởi mô hình.
  $ R^2 = 1 - (R S S) / (T S S) $
- *$R^2$ hiệu chỉnh* ($R^2_"adj"$): Bổ sung hình phạt khi mô hình có quá nhiều biến độc lập $k$ (bao gồm cả intercept), tránh hiện tượng Overfitting:
  $ overline(R)^2 = 1 - (("RSS" / (n - k))) / (("TSS" / (n - 1))) $

*d. Kiểm định F (Đánh giá toàn mô hình)* \
Kiểm định giả thuyết $H_0$: Tất cả các hệ số góc $beta_1 = beta_2 = ... = beta_p = 0$. Phép kiểm định này sử dụng tỷ số giữa phương sai được giải thích và phương sai phần dư:
$ F = ((T S S - R S S) / (k - 1)) / ("RSS" / (n - k)) $
Áp dụng *Phép biến đổi Paulson (1942)* nhằm xấp xỉ phân phối Fisher-F bất đối xứng thành phân phối chuẩn tắc $Z$, cho phép thuật toán đạt độ phức tạp $O(1)$.

