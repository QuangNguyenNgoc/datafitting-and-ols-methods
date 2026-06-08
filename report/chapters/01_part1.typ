= Lý thuyết hồi quy tuyến tính — Chứng minh và Mô phỏng thực nghiệm

== Mô hình hồi quy tuyến tính

Cho $n$ quan sát và $p$ biến độc lập. Mô hình hồi quy tuyến tính được viết dưới dạng ma trận:

$ y = X beta + epsilon, quad epsilon ~ cal(N)(0, sigma^2 I_n) $

trong đó $y in RR^n$ là vector mục tiêu, $X in RR^(n times (p+1))$ là ma trận thiết kế, $beta in RR^(p+1)$ là vector hệ số cần ước lượng, và $epsilon$ là vector nhiễu ngẫu nhiên với phương sai đồng đều.

== Phương pháp Bình phương Bé nhất (OLS)

=== Chứng minh công thức $hat(beta)$

Ước lượng OLS tìm $hat(beta)$ tối thiểu hóa tổng bình phương phần dư:

$ cal(L)(beta) = ||y - X beta||^2 = (y - X beta)^T (y - X beta) $

Khai triển và lấy đạo hàm theo $beta$:

$ (partial cal(L)) / (partial beta) = -2 X^T (y - X beta) = 0 $

Thu được *phương trình chuẩn*:

$ X^T X hat(beta) = X^T y $

#block(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
)[
*Định lý:* Khi $X^T X$ khả nghịch (tức các cột của $X$ độc lập tuyến tính), nghiệm duy nhất là:

$ hat(beta) = (X^T X)^(-1) X^T y $
]

Đây là ước lượng tuyến tính không chệch tốt nhất (BLUE) theo Định lý Gauss–Markov.

=== Cài đặt qua Economic SVD

Thay vì tính $(X^T X)^(-1)$ trực tiếp (kém ổn định khi $X$ gần suy biến), ta dùng phân rã giá trị kỳ dị kinh tế $X = U Sigma V^T$, trong đó $U in RR^(n times k)$, $Sigma in RR^(k times k)$, $V^T in RR^(k times (p+1))$, $k = min(n, p+1)$.

Thay vào công thức OLS:

$ hat(beta) = (V Sigma U^T U Sigma V^T)^(-1) V Sigma U^T y = V Sigma^(-1) U^T y $

Phương pháp này ổn định hơn vì chỉ chia cho các giá trị kỳ dị $sigma_i > 0$, tránh khuếch đại sai số số học.

== Ma trận Hình mũ (Hat Matrix)

=== Định nghĩa và dẫn xuất

Giá trị dự đoán $hat(y) = X hat(beta)$. Thay $hat(beta)$:

$ hat(y) = X (X^T X)^(-1) X^T y =: H y $

Ma trận $H = X (X^T X)^(-1) X^T$ được gọi là *ma trận hình mũ* (hat matrix) hay *ma trận chiếu*.

=== Tính chất và Chứng minh

**(1) Lũy đẳng ($H^2 = H$):**

$ H^2 = [X(X^T X)^(-1) X^T][X(X^T X)^(-1) X^T] = X(X^T X)^(-1) underbrace((X^T X)(X^T X)^(-1), = I) X^T = H $

**(2) Đối xứng ($H^T = H$):**

$ H^T = [X(X^T X)^(-1) X^T]^T = X [(X^T X)^(-1)]^T X^T = X (X^T X)^(-1) X^T = H $

vì $(X^T X)$ đối xứng nên $(X^T X)^(-1)$ cũng đối xứng.

**(3) Vết $upright(tr)(H) = p + 1$:**

$ upright(tr)(H) = upright(tr)(X(X^T X)^(-1) X^T) = upright(tr)((X^T X)^(-1) X^T X) = upright(tr)(I_(p+1)) = p + 1 $

sử dụng tính chất vòng của vết: $upright(tr)(A B) = upright(tr)(B A)$.

**(4) Biểu diễn qua SVD:** Với $X = U Sigma V^T$, ta có $H = U U^T$, cho phép tính đường chéo $h_(i i) = ||U_(i,·)||^2$ mà không cần lưu toàn bộ $H in RR^(n times n)$.

== Hồi quy Ridge

=== Vấn đề của OLS khi có đa cộng tuyến

Khi các cột của $X$ gần tuyến tính phụ thuộc, $X^T X$ gần kỳ dị, khiến $(X^T X)^(-1)$ có các phần tử rất lớn, dẫn đến $hat(beta)$ có phương sai cao dù không chệch.

=== Bài toán điều chuẩn Tikhonov

Ridge Regression thêm hạng phạt $lambda ||beta||^2$ vào hàm mục tiêu:

$ hat(beta)_"ridge" = arg min_beta { ||y - X beta||^2 + lambda ||beta||^2 } $

Lấy đạo hàm và đặt bằng 0:

$ -2 X^T (y - X beta) + 2 lambda beta = 0 $
$ (X^T X + lambda I) beta = X^T y $

#block(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
)[
*Nghiệm Ridge:*
$ hat(beta)_"ridge" = (X^T X + lambda I)^(-1) X^T y $
]

Ma trận $(X^T X + lambda I)$ *luôn khả nghịch* với $lambda > 0$ vì tất cả trị riêng của $X^T X + lambda I$ đều $>= lambda > 0$. Khi $lambda = 0$, thu về OLS. Khi $lambda -> infinity$, $hat(beta)_"ridge" -> 0$.

=== Đánh đổi Bias–Variance

Ước lượng Ridge có độ chệch $upright(E)[hat(beta)_"ridge"] - beta = -(X^T X + lambda I)^(-1) lambda beta != 0$, nhưng phương sai nhỏ hơn OLS:

$ upright("Var")(hat(beta)_"ridge") = sigma^2 (X^T X + lambda I)^(-1) X^T X (X^T X + lambda I)^(-1) $

Với $lambda$ phù hợp, giảm phương sai đủ bù cho độ chệch, cải thiện MSE tổng thể.

== Nhân tử Phóng đại Phương sai (VIF)

=== Định nghĩa

VIF của biến $x_j$ đo mức độ phương sai của $hat(beta)_j$ bị thổi phồng do đa cộng tuyến, so với trường hợp $x_j$ độc lập với các biến còn lại:

$ V I F_j = 1 / (1 - R_j^2) $

trong đó $R_j^2$ là hệ số xác định của mô hình hồi quy $x_j$ trên tất cả các biến độc lập còn lại (có hệ số chặn).

=== Chứng minh liên hệ với phương sai của $hat(beta)_j$

Từ ma trận hiệp phương sai của $hat(beta)$:

$ upright("Var")(hat(beta)) = sigma^2 (X^T X)^(-1) $

Phần tử thứ $j$ trên đường chéo là $upright("Var")(hat(beta)_j) = sigma^2 [(X^T X)^(-1)]_(j j)$.

Có thể chứng minh (bằng công thức nghịch đảo theo khối) rằng:

$ [(X^T X)^(-1)]_(j j) = 1 / (S_(j j) (1 - R_j^2)) $

trong đó $S_(j j) = sum_(i=1)^n (x_(i j) - bar(x)_j)^2$ là biến thiên mẫu của $x_j$. Do đó:

$ upright("Var")(hat(beta)_j) = sigma^2 / (S_(j j)) times underbrace(1/(1 - R_j^2), = V I F_j) $

VIF cho biết phương sai của $hat(beta)_j$ lớn hơn $upright("VIF")_j$ lần so với trường hợp $x_j$ hoàn toàn độc lập ($R_j^2 = 0$).

=== Quy tắc chẩn đoán

- $upright("VIF")_j approx 1$: Không có đa cộng tuyến.
- $5 < upright("VIF")_j < 10$: Đa cộng tuyến vừa phải, cần theo dõi.
- $upright("VIF")_j > 10$: Đa cộng tuyến nghiêm trọng; nên xem xét loại biến hoặc dùng Ridge.

== Kết quả kiểm chứng

Cả bốn hàm `ols_fit`, `hat_matrix`, `ridge_fit` và `vif` đều được kiểm chứng trên dữ liệu giả lập ($n = 100$, $p = 3$, $sigma = 1.5$, seed = 42):

- `ols_fit`: Sai số so với `numpy.linalg.lstsq` đạt $1.8 times 10^(-15)$ (giới hạn máy).
- `hat_matrix`: Tính lũy đẳng $max|H^2 - H| = 1.1 times 10^(-16)$; $upright(tr)(H) = p + 1$ chính xác.
- `ridge_fit`: Sai số so với nghiệm chuẩn `numpy.linalg.solve` đạt $6.7 times 10^(-16)$.
- `vif`: Kết quả khớp với `statsmodels.variance_inflation_factor` trên cả hai kịch bản độc lập và đa cộng tuyến.
