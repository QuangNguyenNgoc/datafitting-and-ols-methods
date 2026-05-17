#import "../theme.typ": *
= Phân rã ma trận và trực quan hóa với Manim <sec:decomposition>

== Chéo hóa ma trận (Diagonalization)

Thay vì thiết lập chéo hóa thành một mảng thuật toán phức tạp, phần cài đặt được thiết kế gọn gàng dựa trên ma trận demo. Cụ thể, đầu vào hàm sẽ tiếp nhận một nhóm ma trận kiểm thử đại diện và thực hiện chéo hóa nguyên bản từ đầu, bao gồm các công đoạn thuật toán số học chi tiết sau đối với nhóm ma trận nhỏ:

- _Thiết lập đa thức đặc trưng (Thuật toán `Faddeev-LeVerrier`)_: Hàm trích xuất chuỗi hệ số của đa thức $c_n, c_(n-1), ...$ bằng kỹ thuật biến đổi trên vết (Trace) đệ quy của ma trận, né tránh được độ phức tạp bù trừ định thức cồng kềnh.
- _Dò tìm Trị riêng (Phương pháp lặp `Durand-Kerner`)_: Thuật toán lặp đồng thời `Durand-Kerner` được triển khai để dò tìm trên không gian nghiệm phức, giúp giải chính xác toàn bộ rễ của đa thức (tức là Trị riêng).
- _Dựng không gian Vector riêng_: Tái sử dụng thuật toán khử Gauss tìm hạng từ @sec:gauss, không gian nghiệm `Nullspace` của hệ ($A - lambda I$) được thiết lập tính toán. Các vector tìm được sẽ tiếp tục qua rây lọc _Trực chuẩn hóa `Gram-Schmidt`_ nhằm bảo toàn tuyệt đối không gian nền cơ sở.

Từ các nguyên liệu trên, chương trình sẽ kiểm tra tính chéo hóa khả thi của ma trận (có đủ $n$ vector cơ sở độc lập tuyến tính) nhằm hoàn thành bài toán tổng kết định dạng biểu diễn:
$ A = P D P^(-1) $

== Phân rã ma trận

_Phân rã kỳ dị (Singular Value Decomposition - `SVD`)_ được chọn làm phương pháp cốt lõi để nghiên cứu.

_Lý do lựa chọn:_ Khác với một số cách phân rã bị giới hạn bởi tính đối xứng hay dạng vuông của ma trận, `SVD` linh hoạt xử lý mọi kích cỡ ma trận $m times n$. Phân rã này phân tích mọi cấu trúc dữ liệu thành các phép biến đổi không gian trực quan (rotate-scale-rotate), tạo nền tảng vững chắc cho nhiều ứng dụng khoa học lớn như nén dữ liệu.

_Thành phần cấu trúc và ý nghĩa:_
Phân rã biến đổi ma trận gốc đầu vào $A$ thành hệ đầu ra 3 thành phần:
$ A = U Sigma V^T $
- _Đầu vào_: Ma trận $A$ bất kỳ ($m times n$).
- _Đầu ra_: Ma trận chéo $Sigma$ chứa tập hợp _trị kỳ dị (singular values)_ đại diện cho mức độ co giãn. Bộ hệ cơ sở trực chuẩn $U$ (Left singular vectors) mô tả không gian đích và $V$ (Right singular vectors) đóng vai trò căn phối dữ liệu nguồn. Việc thu nhận phân tích giúp giải thích chính xác hệ thống tọa độ đang phóng to và dịch chuyển theo góc uốn nào.

Cột thư viện ma trận $U$ được đối xứng qua biểu thức $u_i = (A dot v_i) / sigma_i$. Đặc biệt, tính ổn định luôn được đảm bảo: mỗi khi $A$ thất bại thiết lập trực giao do lỗi mất hạng (`Nullspace`), thuật toán thực hiện chèn trực giao hóa `Gram-Schmidt` kèm hệ cơ sở chuẩn để thu được toàn bộ các cột, đảm bảo cấu trúc $U$ giữ nguyên vẹn.

== Thư viện trực quan hóa Manim

`Manim` là một thư viện Python được thiết kế để tạo ra các hình ảnh động toán học chất lượng cao. Trong đồ án này, `Manim` được sử dụng để trực quan hóa các khái niệm trừu tượng của đại số tuyến tính, giúp dễ dàng nắm bắt các phép biến đổi ma trận, sự thay đổi cơ sở và các tính chất của trị riêng, vector riêng.

Nội dung video xoay quanh giới thiệu lý thuyết, mô phỏng thuật toán cho ma trận A cụ thể:
- Mô phỏng quá trình chéo hóa ma trận $A=P D P^(-1)$
- Trực quan quá trình phân rã SVD. Cụ thể là trục quan phép biến đổi hình học rotate-scale-rotate trên hình tròn đơn vị.

*Link video:* #link("https://youtu.be/sWpLmDTU0R0")[https://youtu.be/sWpLmDTU0R0]

#figure(
  image("../images/manim_svd_snapshot.png", width: 100%),
  caption: [Minh họa hệ vector thay đổi cơ sở và dãn không gian theo các trị kỳ dị (SVD)]
) <fig:manim-svd>

#figure(
  image("../images/manim_eigen_snapshot.png", width: 100%),
  caption: [Tiến trình phát hiện và tính toán các đặc trưng giá trị riêng dựa trên sự bảo toàn phương hướng]
) <fig:manim-eigen>
