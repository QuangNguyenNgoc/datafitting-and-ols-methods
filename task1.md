_Bản báo cáo nội dung bộ dữ liệu sẽ chọn, đảm bảo bộ dữ liệu thỏa mãn yêu cầu và có thuyết phục lí do chọn bộ dữ liệu này._

## Nguồn dữ liệu

**Nguồn:** [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

Nguồn gốc học thuật của bộ dữ liệu đến từ bài báo/dữ liệu mở **Hotel Booking Demand Datasets** của Nuno António, Ana de Almeida và Luis Nunes. Bộ dữ liệu mô tả booking của **một resort hotel** và **một city hotel**, với dữ liệu đặt phòng trong giai đoạn từ **01/07/2015 đến 31/08/2017**. Các thông tin định danh cá nhân và thông tin nhận diện khách sạn đã được loại bỏ.

- **Vị trí file trong repo:** Đã upload file `.csv` vào đường dẫn `./part2/data/` trong nhánh này chưa? _(Chưa — cần upload file `hotel_bookings.csv` vào `./part2/data/`)_

---

## Mô tả

### 1. **Dữ liệu này trình bày điều gì?**

Bộ dữ liệu **Hotel Booking Demand** ghi lại thông tin đặt phòng của khách hàng tại hai loại khách sạn: **City Hotel** và **Resort Hotel**. Mỗi dòng dữ liệu tương ứng với một lượt đặt phòng, bao gồm các thông tin như thời gian đặt trước, tháng nhận phòng, số đêm lưu trú, số người lớn/trẻ em, loại phòng, kênh phân phối, yêu cầu đặc biệt, số chỗ đậu xe và giá phòng trung bình mỗi ngày.

Trong đồ án này, nhóm sẽ dùng bộ dữ liệu để xây dựng bài toán hồi quy: **dự đoán giá phòng trung bình mỗi ngày**, tức cột `adr` — **Average Daily Rate**.

---

### 2. **Tại sao lại chọn bộ dữ liệu này?**

Nhóm chọn bộ dữ liệu này vì đây là một bài toán thực tế trong lĩnh vực **khách sạn, du lịch và quản trị doanh thu**. Việc dự đoán giá phòng trung bình có ý nghĩa rõ ràng: khách sạn có thể ước lượng doanh thu, phân tích yếu tố ảnh hưởng đến giá phòng và hiểu hành vi đặt phòng của khách hàng.

Dataset này cũng phù hợp với yêu cầu đồ án vì:

- Đây là **dữ liệu thực**, không phải dữ liệu toy hoặc synthetic.
- Có số lượng dòng lớn, đủ để chia train/test và cross-validation.
- Có nhiều biến đặc trưng dạng số và dạng phân loại.
- Có missing values rõ ràng ở một số cột như `agent`, `company`, `country`, `children`.
- Biến mục tiêu `adr` là biến số liên tục, phù hợp với bài toán **regression**.
- Dataset này thường được dùng cho bài toán classification dự đoán hủy phòng, nhưng nhóm chọn hướng **dự đoán giá phòng `adr`**, nên giảm khả năng bị trùng hướng làm với nhóm khác.

---

### 3. **Các thông số của bộ dữ liệu**

| Tiêu chí | Thông số cụ thể của bộ data này | Yêu cầu Đồ án |
| :--- | :--- | :--- |
| **Số lượng dòng ($n$)** | 119,390 dòng | $n \ge 200$ |
| **Số lượng đặc trưng ($p$)** | Khoảng 31–32 cột ban đầu, sau khi bỏ biến mục tiêu vẫn còn trên 3 đặc trưng | $p \ge 3$ |
| **Tỷ lệ Missing Values** | Có cột thiếu rất cao: `company` khoảng 94%, `agent` khoảng 13.7%; ngoài ra có missing ở `country`, `children` | $\ge 5\%$ |

> Ghi chú: Nếu tính trên toàn bộ bảng thì tỷ lệ missing tổng thể có thể thấp hơn 5%, nhưng yêu cầu quan trọng là dữ liệu gốc có **ít nhất một số cột missing đáng kể để xử lý**. Với dataset này, `company` và `agent` vượt xa mức 5%, nên đủ ý nghĩa cho phần xử lý missing values.

---

### 4. **Phân tích bài toán hồi quy**

**Biến mục tiêu ($y$):** Cột `adr` — Average Daily Rate.

> Trả lời: `adr` là giá phòng trung bình mỗi ngày của lượt đặt phòng. Đây là giá trị số liên tục, ví dụ `75.0`, `98.5`, `120.3`, nên bài toán này là **hồi quy**, không phải phân loại.

**Các biến đặc trưng ($X$):** Liệt kê ít nhất 3 cột sẽ dùng để dự đoán $y$.

> Trả lời: Nhóm dự kiến sử dụng các cột sau để dự đoán `adr`:

```text
hotel
lead_time
arrival_date_month
stays_in_weekend_nights
stays_in_week_nights
adults
children
babies
meal
market_segment
distribution_channel
reserved_room_type
customer_type
required_car_parking_spaces
total_of_special_requests
```

Các biến này hợp lý vì giá phòng có thể bị ảnh hưởng bởi loại khách sạn, thời điểm nhận phòng, thời gian lưu trú, số lượng khách, loại phòng, kênh đặt phòng và nhu cầu đặc biệt của khách hàng.

---

## 5. Phân tích đánh giá khả thi

### Vấn đề Missing Values

**Các giá trị bị thiếu nằm ở cột nào? Xảy ra do ngẫu nhiên hay có quy luật? Dự kiến xử lý thế nào?**

> Trả lời: Missing values chủ yếu nằm ở các cột `company`, `agent`, `country` và `children`. Trong đó, `company` bị thiếu rất nhiều vì không phải lượt đặt phòng nào cũng đến từ khách doanh nghiệp. Cột `agent` bị thiếu khi booking không thông qua đại lý. Vì vậy, missing ở `company` và `agent` nhiều khả năng **không hoàn toàn ngẫu nhiên**, mà phản ánh đặc điểm thật của quy trình đặt phòng.

Dự kiến xử lý:

- Với `company`: do thiếu quá nhiều, nhóm có thể **không dùng cột này** trong mô hình chính, hoặc chuyển thành biến nhị phân `has_company`.
- Với `agent`: có thể chuyển thành biến `has_agent`, biểu diễn booking có/không thông qua đại lý.
- Với `country`: có thể điền bằng mode hoặc gộp missing thành nhóm `Unknown`.
- Với `children`: số lượng thiếu rất ít, có thể điền bằng median hoặc 0 tùy phân tích EDA.

---

### Kiểm tra tính tương thích với 3 mô hình bắt buộc

#### 1. **OLS cơ bản**

Dữ liệu có đủ nhiều cột để ném tất cả vào huấn luyện ban đầu không?

> Trả lời: Có. Dataset có nhiều đặc trưng dạng số và dạng phân loại. Sau khi xử lý missing values, one-hot encoding các biến categorical và chuẩn hóa các biến số, nhóm có thể đưa toàn bộ tập đặc trưng vào mô hình **OLS cơ bản** để tạo baseline ban đầu.

Các biến có thể dùng cho baseline:

```text
hotel
lead_time
arrival_date_month
stays_in_weekend_nights
stays_in_week_nights
adults
children
babies
meal
market_segment
distribution_channel
reserved_room_type
customer_type
required_car_parking_spaces
total_of_special_requests
```

---

#### 2. **OLS chọn biến**

Nhìn lướt qua, có những cột nào khả năng cao bị trùng lặp ý nghĩa để loại bỏ bớt không?

> Trả lời: Có. Một số nhóm biến có khả năng bị trùng lặp ý nghĩa hoặc gây đa cộng tuyến:

- `stays_in_weekend_nights` và `stays_in_week_nights`: đều mô tả thời gian lưu trú.
- Có thể tạo thêm `total_stays = stays_in_weekend_nights + stays_in_week_nights`, khi đó cần cân nhắc bỏ hai cột gốc hoặc không dùng `total_stays` cùng lúc.
- `reserved_room_type` và `assigned_room_type`: đều liên quan đến loại phòng. Tuy nhiên `assigned_room_type` có thể phát sinh sau quá trình xử lý booking nên có nguy cơ gây leakage.
- `market_segment` và `distribution_channel`: đều liên quan đến kênh đặt phòng, có thể tương quan mạnh.
- `is_canceled`, `reservation_status`, `reservation_status_date`: không nên dùng vì không phù hợp để dự đoán giá tại thời điểm đặt phòng và có thể gây rò rỉ thông tin.

Nhóm sẽ dùng **VIF** và p-value để loại bớt các biến dư thừa trong mô hình OLS chọn biến.

---

#### 3. **Ridge/Lasso**

Dữ liệu có đặc điểm gì để biện minh cho việc dùng mô hình nắn chỉnh không?

> Trả lời: Có. Dataset có nhiều biến categorical như `hotel`, `meal`, `market_segment`, `distribution_channel`, `reserved_room_type`, `customer_type`. Sau khi one-hot encoding, số lượng cột sẽ tăng lên đáng kể. Ngoài ra, nhiều biến có thể tương quan với nhau, ví dụ thời gian lưu trú, loại khách sạn, loại phòng và kênh đặt phòng.

Vì vậy, Ridge/Lasso phù hợp vì:

- **Ridge** giúp giảm ảnh hưởng của đa cộng tuyến.
- **Lasso** có thể đưa hệ số của các biến ít quan trọng về 0, hỗ trợ chọn biến.
- Regularization giúp mô hình ổn định hơn khi dữ liệu có nhiều biến sau encoding.

---

### Xác định thước đo đánh giá

Bài toán thực tế này nên ưu tiên chỉ số nào để đánh giá sự thành công?

> Trả lời: Nên ưu tiên **RMSE** và dùng thêm **MAE**, **R²** để bổ sung.

Lý do:

- `adr` là giá phòng trung bình mỗi ngày, nên dự đoán sai quá lớn có thể ảnh hưởng trực tiếp đến phân tích doanh thu.
- **RMSE** phạt mạnh các lỗi dự đoán lớn, phù hợp khi muốn hạn chế các trường hợp dự đoán giá phòng lệch quá xa.
- **MAE** dễ diễn giải vì cho biết trung bình mô hình dự đoán sai bao nhiêu đơn vị tiền.
- **R²** giúp đánh giá mô hình giải thích được bao nhiêu phần biến thiên của giá phòng.

---

### Ý tưởng cho phân tích Kernel/Bayesian

> Trả lời: Có thể mở rộng bằng **Kernel Ridge Regression** hoặc **Bayesian Linear Regression**.

**Kernel Ridge Regression** phù hợp vì mối quan hệ giữa giá phòng và các yếu tố như tháng nhận phòng, số đêm lưu trú, loại phòng, loại khách sạn có thể không hoàn toàn tuyến tính. Ví dụ, giá phòng vào mùa du lịch có thể tăng mạnh hơn bình thường, hoặc số ngày lưu trú quá dài có thể đi kèm chính sách giá khác.

**Bayesian Linear Regression** cũng phù hợp vì bài toán dự đoán giá phòng có tính bất định cao. Thay vì chỉ dự đoán một giá trị `adr`, Bayesian có thể đưa ra khoảng tin cậy cho dự đoán, ví dụ: “giá phòng trung bình dự kiến nằm trong khoảng 95–120 với độ tin cậy cao”.

---

## Cam kết

- [x] Đã hiểu rõ lý thuyết cốt lõi: data fitting, OLS, 3 mô hình bắt buộc.
- [x] Dữ liệu thực tế, nguồn tin cậy.
  - Có bị trùng nhóm khác không?: _(Khả năng thấp hơn House Prices/Ames Housing; dataset phổ biến nhưng hướng dự đoán `adr` ít trùng hơn hướng classification `is_canceled`)_
- [x] Thỏa mãn yêu cầu bài toán hồi quy, tuyệt đối **không phải phân lớp**.
- [x] Đáp ứng được nhu cầu cho **đánh giá khả thi**.
  - Mở rộng có đáp ứng cho mô hình Kernel/Bayesian: _(Có)_

---

## Chia sẻ thêm thông tin

Bộ dữ liệu này khá phù hợp để làm phần **Residual Analysis**. Giá phòng khách sạn thường có outlier do mùa cao điểm, loại phòng đặc biệt hoặc booking bất thường. Khi vẽ residual plot, nếu thấy phần dư phân tán mạnh ở các mức giá cao, nhóm có thể giải thích rằng mô hình tuyến tính chưa nắm hết được các yếu tố phi tuyến như mùa du lịch, loại khách hàng, chính sách giá hoặc kênh đặt phòng.

Ngoài ra, nhóm cần lưu ý không dùng các biến có nguy cơ rò rỉ thông tin như `reservation_status`, `reservation_status_date`, `assigned_room_type` hoặc `is_canceled` nếu mục tiêu là dự đoán `adr` tại thời điểm đặt phòng. Điều này giúp bài toán thực tế và sạch hơn về mặt kỹ thuật.
