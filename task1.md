_Bản báo cáo nội dung bộ dữ liệu sẽ chọn, đảm bảo bộ dữ liệu thỏa mãn yêu cầu và có thuyết phục lí do chọn bộ dữ liệu này._
## Nguồn dữ liệu
**Nguồn:** [Boston Airbnb Open Data - Kaggle](https://www.kaggle.com/datasets/airbnb/boston)
- **Vị trí file trong repo:** Đã upload file `.csv` vào đường dẫn `./part2/data/` trong nhánh này chưa? _(Có)_

## Mô tả
1. Dữ liệu này trình bày điều gì?
Dữ liệu cung cấp thông tin chi tiết về các phòng/nhà cho thuê trên nền tảng Airbnb tại thành phố Boston, bao gồm các thông tin về đặc điểm của nơi lưu trú, vị trí, đánh giá của người dùng, và mức giá cho thuê.

2. Tại sao lại chọn bộ dữ liệu này?
Bộ dữ liệu này rất phổ biến và "lý tưởng" cho việc thực hành các phương pháp học máy cơ bản, đặc biệt là bài toán hồi quy (Regression). Tính thực tế cao (thị trường lưu trú), đa dạng kiểu dữ liệu (số, phân loại, văn bản). Đặc biệt, có nhiều vấn đề cần xử lý (missing values, tiền xử lý cột text) --> có thể học thêm kha khá

3. Các thông số của bộ dữ liệu?

| Tiêu chí | Thông số cụ thể của bộ data này | Yêu cầu Đồ án |
| :--- | :--- | :--- |
| **Số lượng dòng ($n$)** | 3,585 dòng | $n \ge 200$ |
| **Số lượng đặc trưng ($p$)** | 95 cột | $p \ge 3$ |
| **Tỷ lệ Missing Values** | ~20-25% ở các cột đánh giá (review), có cột >90% (`square_feet`) | $\ge 5\%$ |

4. Phân tích bài toán hồi quy
**Biến mục tiêu ($y$):** Cột nào là biến cần dự đoán? (VD: Cột `Giá_tiền`, `Ping_ms`). Đây có chắc chắn là giá trị số liên tục không? 
> Trả lời: Biến mục tiêu là cột `price` (giá thuê) là một biến số liên tục hoàn toàn phù hợp để dự đoán bằng bài toán hồi quy.

**Các biến đặc trưng ($X$):** Liệt kê ít nhất 3 cột sẽ dùng để dự đoán $y$.
> Trả lời: `accommodates` (số người chứa được), `bedrooms` (số phòng ngủ), `bathrooms` (số phòng tắm), `number_of_reviews` (số lượng đánh giá), `review_scores_rating` (điểm số đánh giá tổng quát).

 5. Phân tích đánh giá khả thi (Góc nhìn kỹ thuật)
- **Vấn đề Missing Values:** Các giá trị bị thiếu (missing values) nằm ở cột nào? Xảy ra do ngẫu nhiên hay có quy luật? Dự kiến xử lý thế nào?
> Trả lời: Missing values tập trung nhiều ở các cột về đánh giá (`review_scores_rating`, `review_scores_accuracy`...) do có nhiều phòng chưa từng được thuê hoặc khách không để lại đánh giá. Một số cột khác thiếu trầm trọng như `square_feet`, `monthly_price`. Hướng xử lý: Bỏ đi các cột có tỷ lệ thiếu quá lớn (>50%), điền giá trị trung bình/trung vị cho các cột thiếu ít, điền 0 hoặc đánh dấu là một phân loại riêng biệt cho các phòng chưa có review.

- **Kiểm tra tính tương thích với 3 mô hình bắt buộc:**
  1. OLS cơ bản: Dữ liệu có đủ nhiều cột (đặc trưng) để ném tất cả vào huấn luyện ban đầu không?
  > Trả lời: Rất đủ. Với 95 cột ban đầu, ta có thể lọc ra khoảng 30-40 cột dạng số và categorical (sau khi One-hot encoding) để trực tiếp đưa vào OLS cơ bản.

  2. OLS chọn biến: Nhìn lướt qua, có những cột nào khả năng cao bị trùng lặp ý nghĩa (ví dụ: cột 'Năm sinh' và cột 'Tuổi') để loại bỏ bớt không?
  > Trả lời: Có rất nhiều sự trùng lặp. Ví dụ: `beds`, `bedrooms`, và `accommodates` có tương quan thuận rất mạnh. Các cột về khu vực như `neighbourhood`, `zipcode`, `latitude`, `longitude` cũng mang ý nghĩa khá giống nhau. Việc chọn lọc biến sẽ giúp loại bỏ bớt các biến này để giảm hiện tượng đa cộng tuyến.

  3. Ridge/Lasso: Dữ liệu có đặc điểm gì (nhiều nhiễu, nhiều cột) để biện minh cho việc dùng mô hình nắn chỉnh (Regularization) không?
  > Trả lời: Do dữ liệu có rất nhiều cột đặc trưng và hiện tượng đa cộng tuyến giữa các biến là rõ ràng (như số phòng ngủ và sức chứa), đây là trường hợp thích hợp để dùng Ridge (giảm trọng số các biến có tương quan) và Lasso (ép trọng số của các biến không quan trọng về 0, giúp tự động chọn biến).

- **Xác định thước đo (đánh giá):** Bài toán thực tế này nên ưu tiên chỉ số nào để đánh giá sự thành công? (Ví dụ: Dự đoán giá nhà sai lệch lớn sẽ trả giá đắt, nên cần RMSE nhỏ. Nếu chỉ cần dự đoán đúng xu hướng chung, R-squared là đủ).
> Trả lời: Ưu tiên dùng RMSE (Root Mean Squared Error) và MAE (Mean Absolute Error) vì chúng cho biết trung bình mô hình dự đoán lệch bao nhiêu USD so với giá thực tế. R-squared ($R^2$) cũng sẽ được dùng kết hợp để xem các biến đầu vào giải thích được bao nhiêu % sự biến động của giá nhà.

- (Tùy chọn) Nếu có ý tưởng cho phân tích Kernel/Bayesian, mô tả rõ về mô hình này đối với bộ dữ liệu, điểm đặc trưng so với 3 mô hình trên
> Trả lời: Giá nhà thường không tuyến tính hoàn toàn với các đặc điểm (VD: ở các khu đông đúc thì giá tăng vọt phi tuyến). Sử dụng Kernel Ridge Regression có thể giúp mô hình nắm bắt các mối quan hệ phức tạp và phi tuyến này tốt hơn so với OLS hay Ridge thông thường.

## Cam kết
- [x] Đã hiểu rõ lý thuyết cốt lõi (data fitting, OLS, 3 mô hình bắt buộc).
- [x] Dữ liệu thực tế, Nguồn tin cậy.
	- [x] Có bị trùng nhóm khác không?: _(Không / Chưa rõ)_
- [x] Thỏa mãn yêu cầu bài toán hồi quy (Regression), tuyệt đối KHÔNG phải Phân lớp (Classification).
- [x] Đáp ứng được nhu cầu cho **"đánh giá khả thi"**
	- [x] (Mở rộng) có đáp ứng cho mô hình Kernel/Bayesian: _(Có)_

## Chia sẻ thêm thông tin
Mức độ "nhiễu" và phức tạp của nó vừa đủ để thực hành trọn vẹn quy trình làm sạch dữ liệu (như xử lý định dạng tiền tệ trong cột price), xử lý missing values, và quan trọng nhất là tính thực tiễn cao giúp việc giải thích kết quả của mô hình (ví dụ: thêm 1 phòng tắm thì giá tăng bao nhiêu) trở nên rất sinh động.
