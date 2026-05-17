#import "../theme.typ": *
= Phụ lục <sec:appendix>

== Tổng hợp các trường hợp kiểm chứng

#styled-table(
  (0.4fr, 1.8fr, 2.0fr, 1.8fr),
  (
    [*#text(white)[Case]*],
    [*#text(white)[Mục đích kiểm tra]*],
    [*#text(white)[Hàm liên quan]*],
    [*#text(white)[Kết luận chính]*],
  ),
  (
    [1], [Pivot gần 0], [gaussian_eliminate, back_substitution, determinant, inverse], [Pivoting hoạt động tốt, nghiệm đúng với Numpy],
    [2], [Vô số nghiệm], [gaussian_eliminate, rank_and_basis, basis], [Rank < n, det = 0, tìm đúng biến tự do],
    [3], [Vô nghiệm], [gaussian_eliminate], [Phát hiện mâu thuẫn],
    [4], [Gần suy biến], [gaussian_eliminate, verify_solution], [Sai số cực thấp],
  ),
  [Bảng tổng hợp các trường hợp kiểm chứng thuật toán khử Gauss],
  inset: 8pt,
)

== Thiết lập benchmark

Tóm tắt cấu hình benchmark đã dùng trong @sec:benchmark nhằm giúp việc đối chiếu số liệu được rõ ràng:
-  Thực nghiệm A sử dụng ma trận chéo trội nghiêm ngặt với các kích thước
  $n in {50, 100, 200, 500, 1000}$.
- thực nghiệm B sử dụng hai lớp dữ liệu:
  + ma trận `Hilbert` với các kích thước $n in {3, 5, 8, 10, 12, 15}$,
  + ma trận `SPD` ngẫu nhiên với các kích thước $n in {5, 10, 20, 50, 100}$.
- Với mỗi bài toán, nghiệm chuẩn được chọn là
  $x_"true" = (1,1,dots,1)^T$
  và vế phải được dựng theo công thức
  $b = A x_"true"$.
- Mỗi thuật toán được chạy lặp lại nhiều lần để lấy thời gian thực thi trung bình.
- Các thước đo chính gồm: `Average Runtime`, `Relative Residual Error`, `Solution Error` và `Condition Number`.

== Bảng số liệu benchmark
#styled-table(
  (1fr, 1.2fr, 1.2fr, 1.2fr),
  (
    [*#text(white)[Kích thước $n$]*],
    [*#text(white)[Gauss (s)]*],
    [*#text(white)[LU (s)]*],
    [*#text(white)[Gauss-Seidel (s)]*],
  ),
  (
    [50],   [0.0074], [0.0079], [0.0051],
    [100],  [0.0546], [0.0536], [0.0183],
    [200],  [0.4138], [0.3867], [0.0700],
    [500],  [6.5889], [6.0864], [0.4855],
    [1000], [66.8685], [49.2168], [2.1516],
  ),
  [Thời gian thực thi trung bình của ba phương pháp theo kích thước $n$],
  inset: 7pt,
) <tbl:runtime-full>

#styled-table(
  (1fr, 1.5fr, 1.5fr, 1.5fr),
  (
    [*#text(white)[Kích thước $n$]*],
    [*#text(white)[Gauss: Relative Residual]*],
    [*#text(white)[LU: Relative Residual]*],
    [*#text(white)[Gauss-Seidel: Relative Residual]*],
  ),
  (
    [50],   [$2.57 times 10^(-16)$], [$1.80 times 10^(-16)$], [$1.47 times 10^(-11)$],
    [100],  [$3.99 times 10^(-16)$], [$2.38 times 10^(-16)$], [$2.10 times 10^(-11)$],
    [200],  [$6.29 times 10^(-16)$], [$3.61 times 10^(-16)$], [$2.24 times 10^(-11)$],
    [500],  [$8.51 times 10^(-16)$], [$4.84 times 10^(-16)$], [$4.01 times 10^(-12)$],
    [1000], [$1.22 times 10^(-15)$], [$7.14 times 10^(-16)$], [$4.12 times 10^(-12)$],
  ),
  [Sai số tương đối trên ma trận chéo trội theo kích thước $n$],
  inset: 7pt,
) <tbl:residual-full>

#v(0.6em)

#styled-table(
  (0.8fr, 1.5fr, 1fr, 1fr, 1fr),
  (
    [*#text(white)[$n$]*],
    [*#text(white)[$kappa(H_n)$]*],
    [*#text(white)[Gauss]*],
    [*#text(white)[LU]*],
    [*#text(white)[Gauss-Seidel]*],
  ),
  (
    [3],  [$5.24 dot 10^(2)$], [$8.03 dot 10^(-15)$], [$6.31 dot 10^(-16)$], [$2.94 dot 10^(-9)$],
    [5],  [$4.77 dot 10^(5)$], [$3.49 dot 10^(-13)$], [$4.08 dot 10^(-13)$], [$1.56 dot 10^(-2)$],
    [8],  [$1.53 dot 10^(10)$], [$3.59 dot 10^(-8)$], [$1.20 dot 10^(-7)$], [$1.18 dot 10^(-2)$],
    [10], [$1.60 dot 10^(13)$], [$8.24 dot 10^(-5)$], [$7.67 dot 10^(-5)$], [$1.86 dot 10^(-2)$],
    [12], [$1.76 dot 10^(16)$], [$0.24$], [$0.22$], [$1.94 dot 10^(-2)$],
    [15], [$3.68 dot 10^(17)$], [inf], [inf], [$1.48 dot 10^(-2)$],
  ),
  [Condition number và sai số nghiệm trên ma trận Hilbert],
  inset: 6pt,
) <tbl:hilbert-full>

#v(0.6em)

#styled-table(
  (0.8fr, 1.5fr, 1fr, 1fr, 1fr),
  (
    [*#text(white)[$n$]*],
    [*#text(white)[$kappa(A)$]*],
    [*#text(white)[Gauss]*],
    [*#text(white)[LU]*],
    [*#text(white)[Gauss-Seidel]*],
  ),
  (
    [5],   [8.26], [$3.88 times 10^(-16)$], [$3.88 times 10^(-16)$], [$3.19 times 10^(-11)$],
    [10],  [26.67], [$1.89 times 10^(-15)$], [$2.44 times 10^(-15)$], [$9.63 times 10^(-11)$],
    [20],  [101.65], [$3.64 times 10^(-15)$], [$3.23 times 10^(-15)$], [$1.87 times 10^(-10)$],
    [50],  [624.11], [$2.83 times 10^(-14)$], [$3.42 times 10^(-14)$], [$1.71 times 10^(-09)$],
    [100], [2494.19], [$1.33 times 10^(-13)$], [$1.29 times 10^(-13)$], [0.0505],
  ),
  [Condition number và sai số nghiệm trên ma trận SPD ngẫu nhiên],
  inset: 6pt,
) <tbl:spd-full>
