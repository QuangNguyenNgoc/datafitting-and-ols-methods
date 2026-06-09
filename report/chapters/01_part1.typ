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

*Cài đặt thực tế* dùng hàm `svd_decomp` từ `utils/decomposition.py` — thuần Python, dựa trên phân tích trị riêng (eigendecomposition) của $X^T X$, không dùng `numpy.linalg.svd`. NumPy chỉ dùng để sinh dữ liệu giả lập và kiểm chứng kết quả cuối.

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

**(4) Biểu diễn qua SVD (thuần Python):** Với $X = U Sigma V^T$ (tính bởi `svd_decomp` từ `utils/decomposition.py`), ta có $H = U_r U_r^T$ trong đó $U_r$ gồm các cột ứng với $sigma_i > "varepsilon"$. Cài đặt dùng vòng lặp Python thuần: $H[a][b] += U[a][k] times U[b][k]$ cho mỗi $k$ có $sigma_k > "varepsilon"$, không dùng `numpy.linalg.svd`.

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

Cả bốn hàm `ols_fit`, `hat_matrix`, `ridge_fit` và `vif` đều được kiểm chứng trên dữ liệu giả lập ($n = 60$, $p = 3$, seed = 42). Phân rã SVD trong `ols_fit` và `hat_matrix` dùng `svd_decomp` thuần Python từ `utils/decomposition.py` (không dùng `numpy.linalg.svd`):

- `ols_fit`: Sai số so với `numpy.linalg.lstsq` đạt $2.8 times 10^(-14)$ (sai số làm tròn máy từ SVD thuần Python).
- `hat_matrix`: Tính lũy đẳng $max|H^2 - H| = 6.9 times 10^(-17)$; đối xứng tuyệt đối; $upright(tr)(H) = p + 1$ sai số $< 10^(-15)$.
- `ridge_fit`: Sai số so với nghiệm chuẩn `numpy.linalg.solve` đạt $4.4 times 10^(-16)$.
- `vif`: VIF $approx 1$ trên dữ liệu độc lập; VIF $> 13000$ khi biến thứ ba $approx 0.99 x_1$ (đa cộng tuyến mạnh).
== Ước lượng và Đánh giá độ phù hợp

=== Phân rã phương sai và Thống kê tổng thể (`model_metrics`)

Để đánh giá chất lượng của mô hình hồi quy tuyến tính OLS, ta cần phân tích các thành phần phương sai và kiểm định sự phù hợp tổng thể của mô hình. Trong phần này, các công thức toán học lõi được cài đặt hoàn toàn từ đầu (from scratch) bằng Đại số tuyến tính.

*a. Ma trận chiếu (Hat Matrix) và Tính chất* \
Giá trị dự đoán của mô hình được tính bằng biểu thức $hat(y) = X hat(beta)$. Khi thay $hat(beta) = (X^T X)^(-1) X^T y$, ta có:
$ hat(y) = X (X^T X)^(-1) X^T y = H y $
Trong đó, $H = X(X^T X)^(-1)X^T$ được gọi là Ma trận chiếu (Hat matrix) vì nó "đội mũ" (chiếu) vector thực tế $y$ xuống không gian tuyến tính sinh bởi các cột của ma trận $X$.

Định lý toán học chứng minh $H$ có 2 tính chất vô cùng quan trọng:
- *Tính đối xứng:* $ H^T = (X (X^T X)^(-1) X^T)^T = X ((X^T X)^(-1))^T X^T = X (X^T X)^(-1) X^T = H $
- *Tính lũy đẳng (Idempotent):* $ H^2 &= H H = X(X^T X)^(-1)X^T X(X^T X)^(-1)X^T \ &= X(X^T X)^(-1) (X^T X) (X^T X)^(-1) X^T = X(X^T X)^(-1)X^T = H $
* (\*) Vector phần dư và hệ quả* \
- Ta có vector phần dư $e = y - hat(y) = (I - H)y$. \
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


== Suy diễn Thống kê và Định lý Gauss-Markov

=== Định lý Gauss-Markov và Mô phỏng Monte Carlo (`gauss_markov_simulation`)

Định lý Gauss-Markov là nền tảng lý thuyết chứng tỏ sự ưu việt của phương pháp OLS. *Định lý phát biểu rằng:* Dưới các giả định cơ bản (Kỳ vọng sai số bằng 0, Phương sai sai số không đổi và không có tự tương quan), OLS là ước lượng tuyến tính không chệch tốt nhất (*BLUE* - Best Linear Unbiased Estimator).

*a. Chứng minh tính không chệch (Unbiasedness):* \
Kỳ vọng của vector hệ số ước lượng $hat(beta)$ phải bằng chính giá trị thực $beta$.
$ E[hat(beta)] &= E[(X^T X)^(-1) X^T y] \
               &= E[(X^T X)^(-1) X^T (X beta + epsilon)] \
               &= beta + (X^T X)^(-1) X^T E[epsilon] $
Vì giả định mô hình có $E[epsilon] = 0$, ta thu được $E[hat(beta)] = beta$.

*b. Tính "Tốt nhất" (Minimum Variance) và Thực nghiệm Monte Carlo:* \
Để chứng minh OLS có phương sai nhỏ nhất trong lớp các ước lượng không chệch, đồ án đã cài đặt thuật toán mô phỏng Monte Carlo. Quá trình sinh ra hàng ngàn mẫu ngẫu nhiên và so sánh ma trận hiệp phương sai của OLS với một phép ước lượng thay thế (Alternative Estimator - được tạo bằng cách làm nhiễu trọng số). 
Kết quả thực nghiệm từ chương trình (ví dụ: $"Var"_"OLS" = 0.1532 < "Var"_"Alt" = 0.2865$) là minh chứng xác nhận bằng số học rằng đồ thị phân phối của OLS luôn hẹp và hội tụ tốt nhất.

=== 3. Suy diễn hệ số (`coef_inference`)

Sau khi tìm được $hat(beta)$ và chứng minh được nó là ước lượng tốt nhất, bước tiếp theo là đánh giá độ tin cậy của từng hệ số riêng biệt nhằm xác định xem biến độc lập tương ứng có thực sự tác động lên biến phụ thuộc hay không.

*a. Ma trận Hiệp phương sai và Sai số chuẩn (Standard Error):* \
Phương sai của sai số được ước lượng không chệch bằng: $hat(sigma)^2 = (R S S) / (n - k)$. \
Ma trận hiệp phương sai của vector hệ số được tính thông qua biến đổi đại số:
$ V a r(hat(beta)) = hat(sigma)^2 (X^T X)^(-1) $
Sai số chuẩn của từng hệ số $S E(hat(beta)_j)$ chính là căn bậc hai của các phần tử nằm trên đường chéo chính của ma trận này.

*b. Kiểm định t (t-test) và Khoảng tin cậy:* \
Kiểm định giả thuyết $H_0: beta_j = 0$. Giá trị thống kê $t$ được tính theo tỷ số giữa giá trị hệ số và sai số chuẩn của nó:
$ t_j = (hat(beta)_j) / (S E(hat(beta)_j)) $
Khoảng tin cậy 95% cho hệ số được thiết lập bởi:
$ C I = hat(beta)_j plus.minus t_(alpha/2, n-k) times S E(hat(beta)_j) $

*c. Ghi chú về giải thuật tính $p$-value ($O(1)$):* \
- Sử dụng *Phép xấp xỉ đa thức Wallace* cho hàm phân phối tích lũy (CDF) để tính $p$-value.
- Sử dụng *Khai triển Cornish-Fisher* để tìm giá trị tới hạn (Critical Value) cho khoảng tin cậy.
Việc chuẩn hóa phân phối $t$ bằng các phép biến đổi giải tích giúp thuật toán tính xác suất đạt độ phức tạp $O(1)$.

== Chẩn đoán và Tổng quát hóa Mô hình

=== Phân tích Phần dư (Residual Analysis - `residual_plots`)

Để kiểm chứng xem dữ liệu có vi phạm các giả định Gauss-Markov hay không, đồ án cài đặt thuật toán vẽ 4 biểu đồ chẩn đoán tiêu chuẩn. Quá trình này đòi hỏi phải tính toán Phần dư chuẩn hóa (Standardized Residuals) và Giá trị đòn bẩy (Leverage).

*a. Cơ sở Toán học của các chỉ số:*
- *Giá trị đòn bẩy ($h_(i i)$):* Là các phần tử trên đường chéo chính của Ma trận chiếu $H$. Nó đo lường mức độ "cực đoan" của điểm dữ liệu $X_i$ so với trung tâm dữ liệu.
- *Phần dư chuẩn hóa ($r_i$):* Để so sánh công bằng, phần dư được chia cho sai số chuẩn tại vị trí của nó:
  $ r_i = e_i / (hat(sigma) sqrt(1 - h_(i i))) $

*b. Ý nghĩa của 4 biểu đồ chẩn đoán:*
1. *Residuals vs Fitted:* Kiểm tra tính tuyến tính. Nếu các điểm phân tán ngẫu nhiên quanh trục hoành 0 mà không tạo thành đường cong, mô hình tuyến tính là phù hợp.
2. *Normal Q-Q:* Kiểm tra phân phối chuẩn. Thuật toán tự cài đặt sử dụng *Phép xấp xỉ Tukey Lambda* để tìm các phân vị lý thuyết (Theoretical Quantiles) thay vì gọi thư viện. Nếu phần dư bám sát đường chéo, giả định phân phối chuẩn được thỏa mãn.
3. *Scale-Location:* Dùng để phát hiện hiện tượng phương sai thay đổi (Heteroscedasticity). Trục tung sử dụng $sqrt(|r_i|)$. Nếu các điểm toe ra thành hình cái phễu, phương sai không đồng đều, OLS sẽ mất đi tính hiệu quả tối ưu (không còn là BLUE).
4. *Residuals vs Leverage:* Xác định các Điểm ảnh hưởng (Influential points). Những điểm nằm ở góc phải (có $h_(i i)$ lớn) và xa trục 0 (có $r_i$ lớn) sẽ làm xô lệch đường hồi quy và cần được loại bỏ.

=== Kiểm định chéo (K-Fold Cross Validation - `kfold_cv`)

Việc đánh giá mô hình bằng "RSS" trên tập huấn luyện (Train set) là không đủ, vì khi thêm càng nhiều biến (thậm chí là biến rác), "RSS" luôn có xu hướng giảm, dẫn đến hiện tượng Học vẹt (Overfitting). Đồ án tự cài đặt giải thuật K-Fold Cross Validation để đánh giá chính xác năng lực tổng quát hóa (Generalization) của mô hình.

*a. Thuật toán phân tách:* \
Tập dữ liệu kích thước $n$ được chia thành $k$ phần rời rạc (folds) có kích thước xấp xỉ nhau: $S_1, S_2, ..., S_k$.

*b. Vòng lặp huấn luyện và đánh giá:* \
Với mỗi fold $i$ từ 1 đến $k$:
- Chọn phần $S_i$ làm tập Validation (kích thước $m$). Phần còn lại $S \\ S_i$ làm tập Train.
- Tìm $hat(beta)^((i))$ bằng OLS trên tập Train.
- Tính vector dự đoán trên tập Validation: $hat(y)_"val"^((i)) = X_"val"^((i)) hat(beta)^((i))$
- Tính Lỗi bình phương trung bình ("MSE") cho fold $i$:
  $ "MSE"_i = 1 / m sum_(j=1)^m (y_("val", j)^((i)) - hat(y)_("val", j)^((i)))^2 $

*c. Điểm đánh giá tổng hợp:* \
Điểm kiểm định chéo là trung bình cộng của lỗi trên tất cả $k$ vòng lặp:
$ "CV_Score" = 1 / k sum_(i=1)^k "MSE"_i $
Thực nghiệm trong đồ án cho thấy: Mô hình đơn giản chứa các biến có ý nghĩa sẽ có $"CV_Score"$ thấp hơn (tốt hơn) so với mô hình bị nhét thêm các biến rác, chứng minh K-Fold CV đã chặn đứng thành công hiện tượng Overfitting.
