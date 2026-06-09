= Kết luận

== Tóm tắt kết quả nghiên cứu
- Nghiên cứu đã hoàn thành toàn bộ mục tiêu đề ra từ việc tự xây dựng nền tảng toán học cho ước lượng OLS, chẩn đoán đa cộng tuyến VIF, hồi quy Ridge ở Phần 1 đến việc ứng dụng vào bộ dữ liệu thực tế bất động sản Melbourne ở Phần 2.
- Mô hình hồi quy tuyến tính chọn lọc (OLS_selected) sau khi loại bỏ đa cộng tuyến bằng trạm kiểm soát VIF đã đạt sự cân bằng tối ưu giữa hiệu năng dự báo và tính diễn giải kinh tế (căn bậc hai sai số bình phương trung bình trên tập kiểm thử đạt 419984 và hệ số xác định đạt 0.5750).

== Bài học kinh nghiệm và ý nghĩa thực tiễn
- Ý nghĩa của tiền xử lý và khám phá dữ liệu: Phân tích EDA đóng vai trò then chốt giúp nhận diện các điểm dị biệt (ngoại lai cực đại của diện tích đất, diện tích xây dựng và năm xây dựng phi lý) và đề xuất các giải pháp làm sạch dữ liệu hiệu quả, tạo tiền đề vững chắc cho mô hình hồi quy hoạt động ổn định.
- Giá trị của chẩn đoán mô hình: Trực quan hóa phần dư đã giúp phát hiện sự vi phạm giả định về phương sai sai số đồng nhất và giả định phân phối chuẩn, giúp nhà phân tích hiểu rõ giới hạn của mô hình tuyến tính ở phân khúc bất động sản siêu sang.
- Cầu nối giữa toán học và kinh doanh: Các ước lượng hệ số của mô hình không chỉ mang ý nghĩa thống kê thuần túy mà đã được chuyển hóa thành các đề xuất hành động thực tiễn cho doanh nghiệp bất động sản trong việc định vị sản phẩm theo khoảng cách, tối ưu hóa không gian thiết kế, quản trị rủi ro định giá phân khúc cao cấp và phân bổ nguồn lực theo khu vực địa lý.

== Hướng phát triển tương lai
- Áp dụng các phép biến đổi phi tuyến: Thực hiện biến đổi logarit cho biến mục tiêu giá nhà ($ln("Price")$) để khắc phục hiện tượng phương sai thay đổi và đưa phân phối phần dư về gần phân phối chuẩn.
- Bổ sung các đặc trưng chất lượng: Thu thập thêm các đặc trưng phi cấu trúc hoặc đặc trưng chất lượng (như khoảng cách đến trường học, bệnh viện, chỉ số an ninh và chất lượng môi trường sống) để cải thiện độ giải thích của mô hình vượt qua giới hạn $R^2 = 57.5%$.

