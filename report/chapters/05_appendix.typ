= Phụ lục

== Bảng phân tích nhân tử phóng đại phương sai

Dưới đây là kết quả đo lường chỉ số nhân tử phóng đại phương sai từ quy trình tiền xử lý tại thời điểm đánh giá và loại bỏ các biến số tương quan mạnh, làm cơ sở toán học để loại bỏ các cột `BuildingArea_per_Room` và `Bedroom2`.

#table(
  columns: (auto, auto, auto),
  align: (left, center, center),
  [*Tên đặc trưng*], [*Chỉ số cột*], [*Điểm VIF*],
  [`Rooms`], [0], [10.50],
  [`Distance`], [1], [2.16],
  [`Bedroom2`], [2], [8.88],
  [`Bathroom`], [3], [1.80],
  [`Car`], [4], [1.27],
  [`Landsize`], [5], [1.13],
  [`BuildingArea`], [6], [36.23],
  [`Lattitude`], [7], [2.64],
  [`Longtitude`], [8], [4.02],
  [`Propertycount`], [9], [1.16],
  [`BuildingArea_missing`], [10], [2.50],
  [`YearBuilt_missing`], [11], [2.50],
  [`Landsize_zero_or_missing`], [12], [1.70],
  [`Age`], [13], [1.34],
  [`BuildingArea_per_Room`], [14], [35.93],
  [`BuildingCoverage`], [15], [1.01],
  [`Type_t`], [16], [1.22],
  [`Type_u`], [17], [2.29],
  [`Regionname_Eastern Victoria`], [18], [1.24],
  [`Regionname_Northern Metropolitan`], [19], [4.84],
  [`Regionname_Northern Victoria`], [20], [1.29],
  [`Regionname_South-Eastern Metropolitan`], [21], [2.01],
  [`Regionname_Southern Metropolitan`], [22], [4.34],
  [`Regionname_Western Metropolitan`], [23], [6.65],
  [`Regionname_Western Victoria`], [24], [1.39],
)

Việc phát hiện và loại bỏ các biến có nhân tử phóng đại phương sai vượt quá ngưỡng 5.0 (đặc biệt là các cặp biến tỷ lệ nghịch hoặc đồng dạng như diện tích xây dựng và diện tích mỗi phòng) đóng vai trò quyết định trong việc bảo toàn hạng cột và tránh hiện tượng suy biến của ma trận $X^T X$.

== Chứng minh toán học cho mô hình hồi quy Ridge

Nghiệm đóng của phương pháp hồi quy tuyến tính thông thường đã được xây dựng từ trước. Để mở rộng khả năng kiểm soát phương sai sai số, phần này trình bày chứng minh toán học cho phương pháp hồi quy Ridge để giải quyết triệt để tính bất định của ma trận hồi quy khi xảy ra hiện tượng đa cộng tuyến.

Hàm mục tiêu của hồi quy Ridge bổ sung thêm thành phần phạt bậc hai theo chuẩn $L_2$ với tham số phạt $lambda$:
$ L(beta) = (y - X beta)^T (y - X beta) + lambda beta^T beta $

Lấy đạo hàm bậc nhất theo vector hệ số $beta$ và cho bằng vector 0:
$ (partial L(beta)) / (partial beta) = -2 X^T (y - X beta) + 2 lambda beta = 0 $

Thu gọn và loại bỏ hằng số 2:
$ X^T (y - X beta) = lambda beta $

Khai triển ma trận:
$ X^T y - X^T X beta = lambda beta $

Chuyển các số hạng chứa hệ số $beta$ sang một vế:
$ X^T X beta + lambda beta = X^T y $

Sử dụng ma trận đơn vị $I$ để đưa hệ số $beta$ làm nhân tử chung:
$ (X^T X + lambda I) beta = X^T y $

Nhân nghịch đảo hai vế, nghiệm đóng của hồi quy Ridge thu được là:
$ hat(beta)_"Ridge" = (X^T X + lambda I)^(-1) X^T y $

Ý nghĩa toán học: Ma trận xuyên suốt $(X^T X)$ có thể rơi vào trạng thái gần suy biến với định thức xấp xỉ 0 khi xuất hiện đa cộng tuyến. Việc cộng thêm ma trận đường chéo $lambda I$ giúp dịch chuyển tất cả các trị riêng của ma trận tổng lên một khoảng tối thiểu bằng $lambda > 0$, đảm bảo ma trận thu được luôn khả nghịch số học. Điều này giải thích khả năng kiểm soát hiện tượng quá khớp và đa cộng tuyến của phương pháp hồi quy Ridge.
