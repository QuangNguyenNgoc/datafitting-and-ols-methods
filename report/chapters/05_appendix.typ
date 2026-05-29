= Phụ lục

== Bảng kiểm định Đa cộng tuyến (VIF)

Dưới đây là bảng kết quả đo lường chỉ số VIF từ `DataPipeline` tại Trạm kiểm soát số 2 (Sync Point 2), cung cấp bằng chứng toán học để đưa ra quyết định loại bỏ các cột `BuildingArea_per_Room` và `Bedroom2`.

#table(
  columns: (auto, auto, auto),
  align: (left, center, center),
  [*Tên Đặc trưng*], [*Chỉ số Cột*], [*Điểm VIF*],
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

Việc phát hiện và loại bỏ các biến có VIF > 5.0 (đặc biệt là các cặp tỷ lệ thuận như Tổng diện tích và Diện tích mỗi phòng) đã đóng vai trò sống còn trong việc bảo vệ tính khả nghịch của ma trận $X^T X$.

== Chứng minh Toán học Mô hình Ridge Regression

Ở Phần 1, nhóm đã trình bày nghiệm đóng của Ordinary Least Squares (OLS). Để mở rộng, phần này trình bày chứng minh toán học của phương pháp Ridge (phương pháp được sử dụng ở Phần 2) nhằm giải quyết triệt để tính bất định của ma trận $X^T X$ khi có đa cộng tuyến.

Hàm mục tiêu của Ridge Regression bổ sung thêm thành phần phạt L2 ($lambda$):
$ L(beta) = (y - X beta)^T (y - X beta) + lambda beta^T beta $

Tiến hành lấy đạo hàm bậc 1 theo vector $beta$ và cho bằng vector 0:
$ (partial L(beta)) / (partial beta) = -2 X^T (y - X beta) + 2 lambda beta = 0 $

Triển khai và triệt tiêu hằng số 2:
$ X^T (y - X beta) = lambda beta $

Phân phối ma trận $X^T$:
$ X^T y - X^T X beta = lambda beta $

Chuyển vế chứa $beta$ sang một bên:
$ X^T X beta + lambda beta = X^T y $

Rút $beta$ làm nhân tử chung (với $I$ là ma trận đơn vị cùng cấp):
$ (X^T X + lambda I) beta = X^T y $

Nhân nghịch đảo hai vế, ta thu được nghiệm đóng (closed-form solution) của Ridge:
$ hat(beta)_"ridge" = (X^T X + lambda I)^(-1) X^T y $

*Ý nghĩa toán học:* Ma trận $(X^T X)$ có thể bị suy biến (định thức xấp xỉ 0) nếu có đa cộng tuyến. Việc cộng thêm ma trận đường chéo $lambda I$ giúp ép tất cả các trị riêng (eigenvalues) của ma trận này lớn hơn 0, đảm bảo nó luôn tồn tại ma trận nghịch đảo. Điều này giải thích tại sao Ridge Regression có khả năng xử lý hiện tượng Over-fitting và đa cộng tuyến cực kỳ mạnh mẽ.
