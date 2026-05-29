= Ứng dụng Data Fitting vào Dữ liệu thực tế

== Thông tin Bộ dữ liệu

Bộ dữ liệu được nhóm sử dụng là Melbourne Housing (`melb_data.csv`) thu thập thông tin về thị trường bất động sản tại Melbourne, Úc với mục tiêu dự đoán giá nhà (`Price`).
- Kích thước gốc: 13.580 quan trắc (dòng) và 21 đặc trưng (cột).
- Tính chất: Là bộ dữ liệu thực tế, chứa nhiều dữ liệu bị khuyết và các biến phân loại phức tạp. Dữ liệu này thỏa mãn hoàn toàn các tiêu chí chọn lựa khắt khe của Đồ án 2 (Kích thước lớn, biến mục tiêu liên tục, có missing values).

== Tiền xử lý Dữ liệu

Để mô hình có thể học được chính xác, dữ liệu thô đã được xử lý tự động thông qua một `DataPipeline` chặt chẽ:
- Xử lý dữ liệu không hợp lệ & Khuyết thiếu: Các giá trị phi logic (ví dụ `BuildingArea <= 0` hay `YearBuilt < 1800`) được ép về dạng missing. Sau đó, nhóm áp dụng phương pháp K-Nearest Neighbors (KNNImputer) để điền khuyết cho các biến định lượng. Việc điền khuyết (fit) chỉ diễn ra trên tập huấn luyện nhằm tránh rò rỉ dữ liệu.
- Trích xuất đặc trưng: Tạo thêm các biến có ý nghĩa cao như Tuổi của căn nhà (`Age = 2026 - YearBuilt`) hay Tỷ lệ xây dựng (`BuildingCoverage = BuildingArea / Landsize`).
- Mã hóa và Chuẩn hóa: Áp dụng One-hot encoding cho các biến phân loại (`Type`, `Regionname`). Tất cả các đặc trưng định lượng cuối cùng đều được đưa về cùng một thang đo bằng Z-score Standardization.
- Lược bỏ biến: Các biến có tính định danh quá cao (như `Address`, `SellerG`) hoặc gây trùng lặp thông tin (như `Postcode`, `Bedroom2`) đều bị loại bỏ nhằm giúp ma trận thiết kế gọn gàng, hạn chế hiện tượng đa cộng tuyến.

== So sánh Mô hình

Dựa trên kết quả đo lường từ tập kiểm thử, mô hình OLS_selected (OLS với các biến đã qua chọn lọc) đạt hiệu năng tốt nhất với RMSE xấp xỉ 419,817 và $R^2 approx 0.575$. Mặc dù mô hình Ridge Regression (với siêu tham số $lambda = 1000$) có cơ chế phạt giúp giảm thiểu over-fitting, nhưng trong trường hợp này, khi hiện tượng đa cộng tuyến đã được nhóm chủ động xử lý bằng cách loại bỏ biến (chẳng hạn `Bedroom2`, `YearBuilt`), mô hình OLS cơ bản vẫn duy trì được độ chính xác và tính dễ diễn giải mà không bị lép vế trước Ridge.

== Tầm quan trọng của các Đặc trưng

#figure(
  image("../images/feature_importance.png", width: 90%),
  caption: [Biểu đồ Tầm quan trọng của các Đặc trưng (OLS_selected)]
)

Nhìn vào biểu đồ trọng số (Coefficients), ta có thể rút ra những hiểu biết sâu sắc (insights) về thị trường bất động sản Melbourne:
- Vị trí địa lý là yếu tố then chốt: Biến `Distance` (khoảng cách tới trung tâm thành phố) có trọng số âm lớn nhất. Điều này hoàn toàn phù hợp với thực tế: nhà càng xa trung tâm thì giá càng giảm mạnh.
- Không gian sống quyết định giá trị gia tăng: Các biến liên quan đến quy mô như `Rooms` (số phòng) và `BuildingArea` (diện tích xây dựng) có trọng số dương rất cao, kéo giá trị của bất động sản lên đáng kể.
- Các yếu tố này kết hợp lại cho thấy người mua nhà tại Melbourne sẵn sàng trả giá rất cao cho sự tiện lợi (gần trung tâm) và không gian rộng rãi.

== Tác động của Regularization

#figure(
  image("../images/cv_error.png", width: 70%),
  caption: [Đường cong Lỗi K-Fold CV theo siêu tham số $lambda$]
)

Thông qua phương pháp K-Fold Cross-Validation ($k=5$), mô hình Ridge đã khảo sát và tìm được siêu tham số tối ưu $lambda = 1000$. Biểu đồ cho thấy khi $lambda$ tăng, lỗi RMSE trên tập validation ban đầu có xu hướng đi ngang rồi tăng vọt nếu $lambda$ quá lớn (bóp nghẹt quá mức các hệ số). Dù vậy, hiệu năng của Ridge vẫn không vượt qua được OLS, chứng tỏ ma trận dữ liệu sau khi được chọn lọc biến đã đủ vững chãi (well-conditioned).

== Kiểm định giả định Gauss-Markov & Bắt bệnh mô hình

#figure(
  image("../images/residual_diagnostics.png", width: 100%),
  caption: [4 Biểu đồ chẩn đoán phần dư cho mô hình OLS_selected]
)

Dựa vào các biểu đồ chẩn đoán, ta có thể đánh giá một cách trung thực về giới hạn của mô hình OLS:
- Hiện tượng phương sai thay đổi: Biểu đồ _Residuals vs Fitted_ có dạng hình phễu mở rộng dần sang phải. Điều này phản ánh sai số dự báo của mô hình rất thấp đối với các căn nhà giá rẻ, nhưng lại sai số cực lớn đối với các bất động sản đắt tiền. Nguyên nhân là do hành vi định giá của phân khúc "siêu giàu" chịu tác động bởi các yếu tố cảm xúc, kiến trúc độc bản hoặc phong thủy – những biến ẩn không có trong file dữ liệu.
- Phân phối của phần dư: Biểu đồ _Normal Q-Q_ cho thấy phần dư bám sát đường chéo ở đoạn giữa nhưng lệch mạnh ở hai đuôi (tails). Điều này vi phạm giả định chuẩn (GM5), cho thấy dữ liệu có chứa nhiều điểm ngoại lai (outliers) hoặc phần đuôi dày (heavy tails).
- Điểm ảnh hưởng (Leverage): Biểu đồ _Residuals vs Leverage_ cho thấy một vài điểm dữ liệu có đòn bẩy cao, có thể là những căn biệt thự có diện tích đất cực kỳ lớn làm xoay trục đường hồi quy.

Bài học rút ra: Để khắc phục điểm yếu này trong tương lai, một giải pháp khả thi là thực hiện biến đổi Logarithm cho biến mục tiêu `Price` để kéo các giá trị siêu lớn về quy mô tuyến tính, giúp mô hình OLS giảm bớt hiện tượng phương sai thay đổi.
