= Ứng dụng Data Fitting vào Dữ liệu thực tế

== Thông tin Bộ dữ liệu

Bộ dữ liệu được nhóm sử dụng là Melbourne Housing (`melb_data.csv`) thu thập thông tin về thị trường bất động sản tại Melbourne, Úc với mục tiêu dự đoán giá nhà (`Price`).
- Kích thước gốc: 13.580 quan trắc (dòng) và 21 đặc trưng (cột).
- Tính chất: Là bộ dữ liệu thực tế, chứa nhiều dữ liệu bị khuyết và các biến phân loại phức tạp. Dữ liệu này thỏa mãn hoàn toàn các tiêu chí chọn lựa khắt khe của Đồ án 2 (Kích thước lớn, biến mục tiêu liên tục, có missing values).

== Tiền xử lý Dữ liệu và Xử lý Đa cộng tuyến

Để mô hình hồi quy tuyến tính OLS hoạt động vững chãi và đạt hiệu quả tối ưu, toàn bộ quá trình biến đổi dữ liệu thô thành ma trận thiết kế $X$ đã được tự động hóa thông qua lớp kiến trúc `DataPipeline` hướng đối tượng. Nhóm đã thực hiện tiền xử lý và loại bỏ đa cộng tuyến qua một quy trình hai giai đoạn chặt chẽ như sau:

=== Giai đoạn 1: Làm sạch dữ liệu, Kỹ nghệ đặc trưng và Chuẩn hóa (Phase 1)

Ở giai đoạn này, luồng xử lý tập trung vào việc làm sạch nhiễu cơ bản, khắc phục khuyết thiếu và kỹ nghệ đặc trưng:
- *Xử lý giá trị không hợp lệ (Invalid Values):* Bộ dữ liệu thực tế chứa nhiều dữ liệu phi logic do sai sót nhập liệu. Nhóm đã áp dụng các bộ lọc miền giá trị nghiêm ngặt:
  - Các giá trị âm không hợp lý ở các cột đếm và khoảng cách như `Rooms`, `Bathroom`, `Car`, `Distance`, `Propertycount` được chuyển thành giá trị khuyết (`NaN`).
  - Biến số phòng `Rooms` nếu bằng $0$ (vô lý đối với nhà ở) cũng được chuyển thành khuyết.
  - Các bất động sản có diện tích xây dựng `BuildingArea <= 0` hoặc năm xây dựng `YearBuilt < 1800` (trước thời kỳ định cư hiện đại của Melbourne) được quy về khuyết.
  - Để lưu giữ tín hiệu khuyết thiếu của các thuộc tính quan trọng này, pipeline tự động tạo ra các biến chỉ báo nhị phân: `BuildingArea_missing`, `YearBuilt_missing` và `Landsize_zero_or_missing` trước khi thực hiện nội suy.
- *Điền khuyết (Imputation):* Thay vì sử dụng các phương pháp phức tạp gây rò rỉ thông tin, nhóm áp dụng phương pháp điền khuyết bằng trung vị (`median`) của từng cột định lượng. Việc tính toán trung vị này được thực hiện nghiêm ngặt *chỉ trên tập huấn luyện* (`X_train`) và được lưu giữ để áp dụng đồng nhất lên tập kiểm thử (`X_test`), ngăn ngừa triệt để hiện tượng rò rỉ dữ liệu (data leakage).
- *Kỹ nghệ đặc trưng (Feature Engineering):* Nhóm xây dựng ba đặc trưng phái sinh mang giá trị kinh tế cao:
  - *Tuổi của căn nhà (`Age`):* Được tính bằng hiệu số giữa năm giao dịch và năm xây dựng (`SaleYear - YearBuilt`), trong đó năm giao dịch được trích xuất trực tiếp từ trường `Date` (cột gốc sau đó được loại bỏ để tránh đa cộng tuyến trực tiếp).
  - *Diện tích xây dựng trung bình mỗi phòng (`BuildingArea_per_Room`):* Được tính bằng `BuildingArea / Rooms` để nắm bắt mật độ không gian sinh hoạt.
  - *Mật độ xây dựng (`BuildingCoverage`):* Được tính bằng `BuildingArea / Landsize` để phản ánh tỷ lệ đất được sử dụng cho việc xây cất.
- *Mã hóa biến phân loại và Chuẩn hóa (Encoding & Scaling):* Áp dụng One-hot Encoding cho hai biến phân loại chính là `Type` (loại nhà) và `Regionname` (khu vực địa lý) với tùy chọn `drop_first=True` để tránh bẫy đa cộng tuyến hoàn hảo (dummy variable trap). Cuối cùng, toàn bộ dữ liệu định lượng và các biến chỉ báo nhị phân được đưa về chung một thang đo bằng phương pháp chuẩn hóa Z-score ($z = (x - mu)/sigma$), trong đó trung bình $mu$ và độ lệch chuẩn $sigma$ được cố định từ tập huấn luyện.
- *Lọc biến định danh cơ bản:* Loại bỏ các thuộc tính định danh có cardinality quá cao hoặc không mang thông tin dự báo trực tiếp như `Address`, `SellerG`, `Suburb`, `Postcode`, `Method`, và `CouncilArea`.

=== Giai đoạn 2: Chẩn đoán và Xử lý Đa cộng tuyến bằng VIF (Phase 2)

Sau Giai đoạn 1, ma trận thiết kế $X$ vẫn có khả năng tiềm ẩn hiện tượng đa cộng tuyến (multicollinearity). Hiện tượng này làm cho ma trận nghịch đảo $X^T X$ tiến sát trạng thái suy biến, khiến phương sai của các ước lượng hệ số $hat(beta)$ bị thổi phồng, làm mất tính ổn định và khả năng diễn giải kinh tế của mô hình.

Để chẩn đoán và loại bỏ đa cộng tuyến, nhóm đã thiết lập một trạm kiểm soát toán học sử dụng Nhân tử Phóng đại Phương sai (VIF - Variance Inflation Factor). Quy trình ra quyết định tuân thủ chặt chẽ 3 tiêu chí:
1. *Toán học (VIF > 10):* Xác định các biến bị tương quan tuyến tính mạnh với các biến độc lập khác.
2. *Thống kê (p-value > 0.05):* Xem xét ý nghĩa thống kê của biến đó; đa cộng tuyến thường thổi phồng sai số chuẩn và đẩy p-value lên cao.
3. *Ý nghĩa kinh tế tế thực tế:* Đối chiếu giữa các cặp biến tương quan mạnh để quyết định giữ biến gốc/biến bao quát hơn và loại bỏ biến phái sinh/thứ cấp.

Quy trình chẩn đoán được thực hiện qua các vòng lặp (với danh sách loại bỏ sơ khởi `current_drop` và chốt chặn `final_drop_list`):

- *Vòng 0 (Chưa loại biến - `current_drop = []`):* 
  Kết quả tính VIF cho thấy hai hiện tượng đa cộng tuyến nghiêm trọng:
  - Cặp biến `BuildingArea` (VIF = 36.23) và `BuildingArea_per_Room` (VIF = 35.92) bị trùng lặp thông tin trầm trọng.
  - Cặp biến `Rooms` (VIF = 10.50) và `Bedroom2` (VIF = 8.87) đẩy VIF của biến số phòng vượt ngưỡng an toàn.
- *Vòng 1 (Loại bỏ `Bedroom2`):*
  Khi loại bỏ `Bedroom2` (một đặc trưng thứ cấp thường chứa nhiều sai số nhập liệu hơn so với biến gốc `Rooms`), chỉ số VIF của `Rooms` lập tức giảm mạnh từ $10.50$ xuống mức an toàn $3.26$. Tuy nhiên, cặp biến diện tích vẫn duy trì VIF ở mức báo động (~36).
- *Vòng 2 (Loại bỏ `Bedroom2` và `BuildingArea_per_Room`):*
  Nhóm quyết định giữ lại biến diện tích gốc `BuildingArea` (do phản ánh trực tiếp quy mô vật lý của tài sản) và loại bỏ biến tỷ lệ phái sinh `BuildingArea_per_Room`. Sau vòng lặp này, toàn bộ các đặc trưng còn lại đều có chỉ số VIF dưới ngưỡng an toàn 5.0 (với khu vực `Western Metropolitan` đạt tối đa 6.65, các biến khác đều nhỏ hơn 5.0).
  
Danh sách loại bỏ cuối cùng được xác lập là `final_drop_list = ["Bedroom2", "BuildingArea_per_Room"]`. 

Sau khi loại bỏ các đặc trưng này, nhóm thực hiện kiểm tra kiểm định toán học ma trận thiết kế:
- Kiểm tra hạng (rank check): Hạng của ma trận thiết kế bằng đúng số lượng cột ($"rank"(X) = 23$), xác nhận ma trận đạt đầy đủ hạng cột (full column rank).
- Chỉ số điều kiện (condition number): Đạt mức $approx 3.79$, nhỏ hơn rất nhiều so với ngưỡng cảnh báo $10^(10)$. Điều này bảo đảm ma trận nghịch đảo $X^T X$ hoàn toàn khả nghịch và không bị suy biến về mặt số học.


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
