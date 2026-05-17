#import "../theme.typ": *
= Phép khử Gauss và ứng dụng <sec:gauss>
Phần này tập trung vào phép khử Gauss và các ứng dụng trực tiếp của nó trong đại số tuyến tính số. Thay vì chỉ sử dụng phép khử Gauss để giải hệ phương trình tuyến tính, nhóm khai thác cùng một ý tưởng biến đổi dòng để xây dựng các chức năng quan trọng khác như tính định thức, tìm ma trận nghịch đảo, xác định hạng và mô tả các cơ sở liên quan của ma trận.

Vì các bài toán này có chung nền tảng toán học nhưng khác nhau ở đầu ra cần thu được, nhóm tổ chức phần cài đặt thành các hàm riêng biệt nhằm giúp việc kiểm thử, giải thích và tái sử dụng thuật toán được rõ ràng hơn.

== Cài đặt thuật toán
// === Các hàm phu trợ
// Các hàm phụ trợ có tác dụng chuẩn hóa dữ liệu đầu vào và thao tác ma trận. 
// - Nguồn: `utils.py`.
// - Bao gồm:
//     - `_to_matrix` : 
//     - `_to_vector` :
//     - `_shape` :
//     - `_copy_matrix` :
//     - `identity` :
//     - `_clean_small_entries` :
//     - `_augment` :
//     - `_swap_rows` :
=== `gaussian_eliminate(A, b)`
_Mục tiêu_: giải hệ $A x = b$ bằng phép khử Gauss và phân loại loại nghiệm theo 3 trường hợp: có nghiệm duy nhất, có vô số nghiệm hoặc vô nghiệm.

_Đầu vào_: ma trận hệ số kích thước $m times n$ (kí hiệu: $A$), vector vế phải kích thước $m times 1$ (kí hiệu: $b$), ngưỡng nhận diện số gần bằng 0 (kí hiệu: $epsilon$, tùy chọn với giá trị mặc định $10^(-12)$).

_Đầu ra_: ma trận tăng cường ở dạng `REF`, một cấu trúc `solution_info` mô tả loại nghiệm và nội dung nghiệm, cùng với `swap_count` là số lần hoán đổi dòng.

_Ý tưởng thuật toán:_
Trước hết, hệ được viết dưới dạng ma trận tăng cường $[A | b]$. Thuật toán thực hiện khử Gauss có `partial pivoting`: ở mỗi cột, chọn phần tử có trị tuyệt đối lớn nhất làm `pivot` để giảm rủi ro chia cho số quá nhỏ. Sau khi khử về dạng `REF`, hệ được phân loại như sau:

- Nếu xuất hiện hàng $[0 space dots space 0 | c]$ với $c != 0$, hệ vô nghiệm;
- Nếu số cột `pivot` bằng số ẩn, hệ có nghiệm duy nhất;
- Nếu còn cột tự do, hệ có vô số nghiệm.

_Cách cài đặt:_
Trong code, ma trận tăng cường được tạo bằng `_augment`. Bước khử tiến được thực hiện trong `_forward_elimination_ref`, đồng thời theo dõi `pivot_columns`, `swap_count` và cảnh báo nếu `pivot` quá nhỏ. Từ kết quả khử, quá trình phân nhánh theo từng loại nghiệm:
- Nếu xuất hiện hàng mâu thuẫn thì trả về hệ vô nghiệm;
- Nếu hệ có nghiệm duy nhất, chương trình tách hệ tam giác trên bằng `_extract_upper_system` rồi chuyển cho `back_substitution`.
- Nếu hệ có vô số nghiệm, chương trình dùng `_rref` và `_build_general_solution_from_rref` để dựng nghiệm tổng quát.

Cách tổ chức này giúp phần cài đặt bám sát đúng ba trường hợp đã nêu trong phần ý tưởng thuật toán, đồng thời làm cho từng nhánh xử lý có thể được kiểm tra độc lập.

=== `back_substitution(U, c, eps)`
_Mục tiêu:_
Giải hệ tam giác trên $U x = c$ trong trường hợp hệ có nghiệm duy nhất.

_Đầu vào_:
Ma trận tam giác trên $U$, vector vế phải $c$, và ngưỡng $epsilon$.

_Đầu ra:_
Vector nghiệm $x$.

_Ý tưởng thuật toán:_
Thực hiện thế ngược từ hàng cuối lên hàng đầu. Ở mỗi bước, một ẩn được tính dựa trên các ẩn đã biết trước đó.

_Cách cài đặt:_
Hàm duyệt chỉ số hàng từ cuối về đầu, tính phần tổng các hạng tử đã biết rồi suy ra giá trị của ẩn hiện tại. Nếu phần tử trên đường chéo chính quá nhỏ hoặc bằng 0, hàm báo lỗi vì không thể tiếp tục thế ngược.

=== `determinant(A, eps)`
_Mục tiêu:_
Tính định thức của ma trận vuông.

_Đầu vào_:
Ma trận vuông `A` và ngưỡng `eps`.

_Đầu ra:_
Giá trị định thức của ma trận.

_Ý tưởng thuật toán:_
Đưa ma trận về dạng tam giác trên bằng phép khử Gauss. Khi đó, định thức được tính bằng tích các phần tử trên đường chéo. Nếu có đổi dòng, dấu của định thức thay đổi theo số lần hoán đổi.

_Cách cài đặt:_
Hàm tái sử dụng `_forward_elimination_ref` để nhận ma trận sau khử và `swap_count`. Sau đó tính tích các phần tử đường chéo và nhân với dấu $(-1)^s$. Nếu đầu vào không phải ma trận vuông $A$, hàm báo lỗi.

=== `inverse(A, eps)`
_Mục tiêu:_
Tính ma trận nghịch đảo $A^(-1)$.

_Đầu vào_:
Ma trận vuông `A` và ngưỡng `eps`.

_Đầu ra:_
Ma trận nghịch đảo của A.

_Ý tưởng thuật toán:_
Dùng phương pháp `Gauss-Jordan` trên ma trận ghép $[A | I]$. Nếu đưa được vế trái về ma trận đơn vị $I$, thì vế phải chính là $A^(-1)$.

_Cách cài đặt:_
Hàm ghép $A$ với ma trận đơn vị cùng kích thước. Với mỗi cột, chương trình chọn `pivot` phù hợp, đổi dòng nếu cần, chuẩn hóa dòng `pivot` rồi khử tất cả các dòng còn lại. Sau khi hoàn tất, phần bên phải của ma trận ghép được tách ra làm ma trận nghịch đảo. Nếu không chọn được `pivot` hợp lệ, hàm báo lỗi do ma trận suy biến.

=== `rank_and_basis(A, eps)`

_Mục tiêu:_
Tính hạng của ma trận và xác định các cơ sở liên quan.

_Đầu vào_:
Ma trận $A$ và ngưỡng $epsilon$.

_Đầu ra:_
Một cấu trúc dữ liệu chứa `rank, pivot_columns`, `free_columns`, `column_space_basis`, `row_space_basis` và `null_space_basis`.

Từ dạng `RREF`, số cột `pivot` cho biết hạng ma trận. Các cột `pivot` của ma trận gốc tạo thành cơ sở không gian cột; các dòng khác $0$ của `RREF` tạo thành cơ sở không gian dòng; các cột tự do được dùng để dựng cơ sở không gian nghiệm (`Nullspace`).

_Cách cài đặt:_
Hàm gọi `_rref` để lấy ma trận rút gọn và danh sách `pivot_columns`. Từ đó, chương trình suy ra `free_columns` và lần lượt dựng các basis tương ứng. Đây là bước chuyển trực tiếp từ khái niệm toán học sang cấu trúc dữ liệu có thể dùng tiếp trong code.

=== `verify_solution(A, x, b, tol)`
_Mục tiêu:_
Kiểm chứng nghiệm hoặc thông tin nghiệm do phần cài đặt chính trả về.

_Đầu vào_:
Ma trận $A$, nghiệm hoặc `solution_info`, vector $b$, và ngưỡng $tau$.

_Đầu ra_:
Một cấu trúc chứa `is_close`, `residual_norm`, `Relative_residual` và nghiệm tham chiếu từ `NumPy`.

_Ý tưởng thuật toán:_
Không tham gia vào phần tính toán lõi. Hàm này chỉ dùng để đối chiếu kết quả và đo mức độ sai số thông qua `residual`.

_Cách cài đặt:_
Nếu đầu vào là nghiệm duy nhất hoặc nghiệm riêng của hệ vô số nghiệm, hàm chuyển sang mảng `NumPy` để kiểm tra $A x approx b$. Nếu đầu vào thuộc nhánh vô nghiệm, hàm báo lỗi thay vì cố kiểm tra một nghiệm không tồn tại.


== Kiểm chứng và kết quả
Sau khi hoàn thành phần cài đặt, các case tiêu biểu được sử dụng để kiểm tra thuật toán. Mục tiêu của phần này không chỉ là xác nhận chương trình chạy đúng, mà còn làm rõ ý nghĩa toán học của từng loại kết quả và đối chiếu với hành vi mong đợi của thuật toán.

=== Pivot ban đầu gần bằng 0

Trong trường hợp này, đồ án sử dụng một ma trận $3 times 3$ có tính chất đặc biệt: phần tử đầu tiên $A_(11)$ cực nhỏ ($10^(-10)$). Nếu không có `partial pivoting`, thuật toán sẽ chia cho một số gần bằng 0, dẫn đến sai số làm tròn cực lớn.

Xét hệ phương trình tuyến tính $A_1 x = b_1$, trong đó:
$
A_1 = mat(
  10^(-10), 2.34567, -1.23456;
  3.5, -4.2, 1.125;
  2.25, 1.33333, 5.75
),
b_1 = mat(
  1.2345;
  -2.3456;
  3.4567
).
$

Phần tử ở vị trí đầu tiên của $A_1$ là $10^(-10)$, có độ lớn rất nhỏ. Vì vậy, nếu chọn trực tiếp phần tử này làm pivot thì phép chia trong quá trình khử có thể gây sai số lớn. Khi dùng `partial pivoting`, thuật toán đổi dòng để chọn pivot có trị tuyệt đối lớn hơn.

Ma trận tăng cường sau khử Gauss thu được:

$
M_1^("REF") = mat(
  3.5000, -4.2000, 1.1250, -2.3456;
  0.0000, 4.0333, 5.0268, 4.9646;
  0.0000, 0.0000, -4.1580, -1.6528
).
$

Các cột pivot là:

$
P_1 = [0, 1, 2], space s_1 = 2,
$

trong đó $s_1$ là số lần hoán đổi dòng. Vì cả ba cột ẩn đều là cột pivot, hệ có nghiệm duy nhất. Nghiệm trả về bởi `gaussian_eliminate` là:

$
x_1 = mat(
  0.0847;
  0.7355;
  0.3975
).
$

Khi tách hệ tam giác trên từ $M_1^("REF")$ và áp dụng thế ngược, ta cũng thu được:

$
x_1^("back") = mat(
  0.0847;
  0.7355;
  0.3975
).
$

Định thức và ma trận nghịch đảo của $A_1$ là:

$
det(A_1) = -58.6970.
$

$
A_1^(-1) = mat(
  0.4370, 0.2578, 0.0434;
  0.2997, -0.0473, 0.0736;
  -0.2405, -0.0899, 0.1399
).
$

Kết quả kiểm chứng nghiệm:

$
||A_1 x_1 - b_1||_2 = 0.0000, space
frac(||A_1 x_1 - b_1||_2, ||b_1||_2) = 0.0000.
$

Nghiệm tham chiếu từ `NumPy` là:

$
x_1^("numpy") = mat(
  0.0847;
  0.7355;
  0.3975
).
$

- _Kết quả thực nghiệm_: Như vậy, nghiệm từ thuật toán, nghiệm từ thế ngược và nghiệm tham chiếu trùng nhau ở bốn chữ số thập phân. Kết quả này cho thấy việc đổi dòng đã xử lý đúng trường hợp pivot ban đầu gần bằng 0.

=== Hệ có vô số nghiệm

Khả năng phân loại hệ được kiểm chứng bằng một ma trận phụ thuộc tuyến tính kích thước $3 times 4$. Ở trường hợp này, mục tiêu không phải là một nghiệm duy nhất mà là cấu trúc của không gian nghiệm.

Xét hệ $A_2 x = b_2$ với $A_2$ là ma trận $3 times 4$:

$
A_2 = mat(
  1.2, 2.4, -0.6, 3.0;
  2.4, 4.8, -1.2, 6.0;
  0.6, 1.2, -0.3, 1.5
),
b_2 = mat(
  3.6;
  7.2;
  1.8
).
$

Các dòng của $A_2$ phụ thuộc tuyến tính. Cụ thể, dòng thứ hai bằng $2$ lần dòng thứ nhất và dòng thứ ba bằng $0.5$ lần dòng thứ nhất. Sau khi khử Gauss, ma trận tăng cường có dạng:

$
M_2^("REF") = mat(
  2.4000, 4.8000, -1.2000, 6.0000, 7.2000;
  0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  0.0000, 0.0000, 0.0000, 0.0000, 0.0000
).
$

Do chỉ có một cột pivot nên:

$
"rank"(A_2) = 1, space P_2 = [0], space F_2 = [1, 2, 3].
$

Trong đó $P_2$ là tập cột pivot và $F_2$ là tập cột tự do. Vì số ẩn là $4$ nhưng hạng của ma trận chỉ bằng $1$, hệ có vô số nghiệm.

Nghiệm tổng quát do chương trình dựng từ dạng rút gọn theo dòng là:

$
x = mat(
  3.0000;
  0.0000;
  0.0000;
  0.0000
)
+ t_1 mat(
  -2.0000;
  1.0000;
  0.0000;
  0.0000
)
+ t_2 mat(
  0.5000;
  0.0000;
  1.0000;
  0.0000
)
+ t_3 mat(
  -2.5000;
  0.0000;
  0.0000;
  1.0000
), space t_1, t_2, t_3 in RR.
$

Từ đó, một nghiệm riêng là:

$
x_p = mat(
  3.0000;
  0.0000;
  0.0000;
  0.0000
).
$

Cơ sở của không gian nghiệm thuần nhất $N(A_2)$ gồm ba vector:

$
N(A_2) = "span"(
mat(-2.0000; 1.0000; 0.0000; 0.0000),
mat(0.5000; 0.0000; 1.0000; 0.0000),
mat(-2.5000; 0.0000; 0.0000; 1.0000)
).
$

Kết quả `rank_and_basis` cũng cho:

$
"Col"(A_2) = "span"(mat(1.2000; 2.4000; 0.6000)).
$

$
"Row"(A_2) = "span"(mat(1.0000, 2.0000, -0.5000, 2.5000)).
$

Kết quả kiểm chứng nghiệm riêng:

$
||A_2 x_p - b_2||_2 = 0.0000, space
frac(||A_2 x_p - b_2||_2, ||b_2||_2) = 0.0000.
$

Nghiệm tham chiếu từ `NumPy` là một nghiệm cụ thể khác của cùng hệ:

$
x_2^(("numpy")) = mat(
  0.2609;
  0.5217;
  -0.1304;
  0.6522
).
$

- _Kết quả thực nghiệm_: Sự khác nhau giữa $x_p$ và $x_2^(("numpy"))$ không mâu thuẫn, vì vậy hệ có vô số nghiệm.

=== Hệ vô nghiệm

Để kiểm tra nhánh xử lý mâu thuẫn, một hệ phương trình không tương thích được đưa vào thực nghiệm.

Xét hệ $A_3 x = b_3$:

$
A_3 = mat(
  1, 1, 1;
  2, 2, 2;
  1, -1, 0
),
b_3 = mat(
  3;
  7;
  1
).
$

Sau khi khử Gauss, ma trận tăng cường thu được:

$
M_3^("REF") = mat(
  2.0000, 2.0000, 2.0000, 7.0000;
  0.0000, -2.0000, -1.0000, -2.5000;
  0.0000, 0.0000, 0.0000, -0.5000
).
$

Dòng cuối của ma trận trên tương ứng với phương trình:

$
0 x_1 + 0 x_2 + 0 x_3 = -0.5000.
$

Đây là mệnh đề sai, vì vế trái luôn bằng $0$ nhưng vế phải khác $0$. Do đó:

$
S_3 = emptyset.
$

Chương trình trả về loại nghiệm `inconsistent`, với các cột pivot:

$
P_3 = [0, 1], space s_3 = 2.
$

- _Kết quả thực nghiệm_: Sau bước khử tiến, thuật toán phát hiện hàng có dạng $[0 space dots space 0 | c]$ với $c != 0$. Chương trình dừng lại ngay lập tức và trả về `type`: `inconsistent`. Điều này bảo vệ hệ thống khỏi việc cố gắng tính toán trên một tập nghiệm rỗng. Trường hợp này không thực hiện thế ngược, không tính nghịch đảo và không kiểm chứng residual của nghiệm, vì tập nghiệm rỗng.

=== Hệ gần suy biến và cảnh báo `ill-conditioned`

Trường hợp cuối cùng sử dụng ma trận có các dòng gần như song song (ví dụ: các hệ số lệch nhau $10^(-10)$).

Xét hệ $A_4 x = b_4$:

$
A_4 = mat(
  1.0, 1.0;
  1.0, 1.0 + 10^(-10)
),
b_4 = mat(
  2.0;
  2.0 + 10^(-10)
).
$

Hai dòng của $A_4$ gần như trùng nhau, chỉ khác nhau ở lượng rất nhỏ $10^(-10)$. Sau bước khử, nếu chưa làm tròn, dòng thứ hai có pivot cỡ $10^(-10)$:

$
M_4^("REF") approx mat(
  1.0000, 1.0000, 2.0000;
  0.0000, 10^(-10), 10^(-10)
).
$

Khi in ra với bốn chữ số thập phân, dòng thứ hai được hiển thị như sau:

$
mat(0.0000, 0.0000, 0.0000).
$

Tuy nhiên, pivot thứ hai không bằng $0$ tuyệt đối mà chỉ rất nhỏ. Vì vậy, chương trình vẫn phân loại hệ là có nghiệm duy nhất nhưng đồng thời phát sinh cảnh báo pivot gần bằng $0$.

Nghiệm trả về là:

$
x_4 = mat(
  1.0000;
  1.0000
), space P_4 = [0, 1], space s_4 = 0.
$

Kết quả kiểm chứng:

$
||A_4 x_4 - b_4||_2 = 0.0000, space
frac(||A_4 x_4 - b_4||_2, ||b_4||_2) = 0.0000.
$

Nghiệm tham chiếu từ `NumPy` là:

$
x_4^(("numpy")) = mat(
  1.0000;
  1.0000
).
$

- _Kết quả thực nghiệm_: Thuật toán vẫn giải ra nghiệm nhưng đồng thời đưa ra cảnh báo `pivot` `is near zero`; `the system may be ill-conditioned`. Cảnh báo này cực kỳ quan trọng trong tính toán khoa học, giúp người dùng nhận thức được độ tin cậy của kết quả khi làm việc với ma trận nhạy cảm về số học.
