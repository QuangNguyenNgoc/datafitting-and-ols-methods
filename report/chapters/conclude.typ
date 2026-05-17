#import "../theme.typ": *
= Kết luận <sec:conclude>

== Tóm tắt kết quả đạt được
Trong đồ án này, các mục tiêu đề ra về mặt lý thuyết lẫn thực hành đối với các kỹ thuật cốt lõi của đại số tuyến tính số đã được hoàn thành:
- _@sec:gauss Phép khử Gauss:_ Xây dựng và cài đặt thành công phép khử Gauss với chiến lược chọn phần tử chốt (`Partial Pivoting`) một cách độc lập, ứng dụng vào việc giải hệ phương trình tuyến tính, tính định thức, tìm ma trận nghịch đảo, hạng và xuất cơ sở của các không gian ma trận liên quan.
- _@sec:decomposition Phân rã ma trận:_ Cài đặt thành công kỹ thuật chéo hóa ma trận và phân rã kỳ dị (`SVD`). Sử dụng thư viện `Manim` để xây dựng video minh họa giúp kết nối lý thuyết đại số với hoạt cảnh trực quan.
- _@sec:benchmark Benchmark:_ Thực hiện đánh giá thực nghiệm hoàn chỉnh trên các lớp ma trận ngẫu nhiên (chéo trội, `SPD`, `Hilbert`). Phân tích chi phí tính toán thực tế qua đồ thị log-log và kiểm chứng sự đánh đổi giữa giải thuật trực tiếp (Gauss, `LU`) hay lặp (`Gauss-Seidel`) trước sự biến thiên của số điều kiện ($kappa$).

== Khó khăn gặp phải
Trong quá trình thực hiện đồ án, đã xuất hiện một số thách thức nhất định:
- _Cài đặt thuật toán thuần tuý:_ Cài đặt các phép toán trên cấu trúc danh sách lồng nhau của `Python` mà không sử dụng `NumPy` buộc người thực hiện phải đối mặt trực tiếp với các vấn đề về xử lý tốc độ tính toán và cẩn thận quản lý từng phần tử.
- _Xử lý rủi ro số học:_ Đối với các lớp ma trận cực kỳ kém ổn định như ma trận `Hilbert`, số điều kiện lớn khiến sai số làm tròn lan truyền đáng kể.
- _Khó khăn về `Manim`:_ Việc phải xây dựng các hoạt ảnh toán học phức tạp đòi hỏi phải nắm chắc cả kiến thức lập trình đồ hoạ, tốn nhiều thời gian nghiên cứu tài liệu kỹ thuật.

== Bài học rút ra
Thấy rõ ranh giới lớn giữa _chứng minh hội tụ lý thuyết_ và _thực thi an toàn trên máy tính_. Một thuật toán tốt phải cân bằng được giữa chi phí tính toán lý thuyết và khả năng vận hành ổn định trước hiện tượng nhiễu của dấu phẩy động.

Hệ thống quản lý `GitHub` đã được tận dụng tốt cùng với việc tuân thủ các nguyên tắc làm việc chung để tránh xung đột mã nguồn. Có thêm kinh nghiệm trong việt viết mã sạch, chuẩn bị tài liệu mô tả, báo cáo.