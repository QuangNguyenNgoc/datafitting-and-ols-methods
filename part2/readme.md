# Ghi chú

## Tổng quan

- Dataset gốc: `melb_data.csv`, 13,580 dòng, 21 cột.
- Target: `Price`, giữ nguyên dạng gốc.
- Dữ liệu được chia thô thành `X_train_raw` 70% và `X_test_raw` 30% trước mọi bước xử lý.
- Output cuối: `X_train`, `X_test`, `y_train`, `y_test`, `metadata`.

## Cột sử dụng

- Numeric gốc: `Rooms`, `Distance`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `Lattitude`, `Longtitude`, `Propertycount`.
- Categorical được one-hot encode: `Type`, `Regionname`.
- Feature tạo thêm:
  - `Age = SaleYear - YearBuilt`
  - `BuildingArea_per_Room = BuildingArea / Rooms`
  - `BuildingCoverage = BuildingArea / Landsize`
  - `BuildingArea_missing`
  - `YearBuilt_missing`
  - `Landsize_zero_or_missing`

## Cột bị drop

- `Address`: gần như định danh từng căn nhà, cardinality rất cao.
- `SellerG`: cardinality cao, không cần cho baseline OLS/Ridge gọn.
- `Suburb`: cardinality cao; giữ `Regionname`, `Lattitude`, `Longtitude` làm tín hiệu vị trí chính.
- `Date`: chỉ dùng để lấy `SaleYear` khi tạo `Age`, sau đó drop cột gốc.
- `Postcode`: tín hiệu vị trí bị trùng với `Regionname`, latitude và longitude.
- `Method`, `CouncilArea`: chưa đưa vào baseline để giữ encoding gọn; `CouncilArea` có thể hữu ích cho performance nhưng làm tăng số chiều.
- `YearBuilt`: đã chuyển thành `Age`, drop để tránh phụ thuộc tuyến tính với intercept.
- `Bedroom2`: drop qua cấu hình `DataPipeline(drop_columns=["Bedroom2"])` để thể hiện cơ chế drop cột theo cấu hình.

Drop được thực hiện trong pipeline sau bước repair invalid value, median imputation và feature engineering.

## Quá trình xử lí data

- Invalid values:
  - `BuildingArea <= 0` được xem là missing.
  - `YearBuilt < 1800` được xem là missing.
  - `Landsize <= 0` được xem là missing trước khi tính `BuildingCoverage`.
  - Giá trị âm ở các cột đếm/khoảng cách chính được chuyển thành missing.
  - `Rooms == 0` được chuyển thành missing.
- Missing values:
  - Các cột numeric được điền bằng median từng cột.
  - Median chỉ fit trên `X_train_raw` và được reuse khi transform `X_test_raw`.
  - NaN phát sinh sau feature engineering được fill bằng median của feature cuối đã fit từ train.
  - Không dùng `scikit-learn` trong preprocessing pipeline.
- Encoding:
  - One-hot encode `Type`, `Regionname` với `drop_first=True`.
  - Test data reindex theo `self.encoded_columns` đã fit từ train data.
  - Logic nằm trong `DataPipeline._engineer_and_encode()`.
- Scaling:
  - Z-score toàn bộ feature numeric cuối cùng bằng mean/std từ train data.
  - Test data dùng lại `self.scalers` đã fit từ train data.
  - Dummy và missing flag cũng được scale để giữ pipeline baseline gọn; nếu cần diễn giải trực tiếp hệ số dummy thì có thể tách binary columns ở phiên bản sau.
- Validation:
  - Pipeline kiểm tra schema đầu vào và báo lỗi nếu thiếu cột bắt buộc.
  - Sau one-hot/drop, nếu còn cột non-numeric thì báo lỗi.
  - Notebook kiểm tra feature contract, rank/condition number của design matrix, target validity và alignment giữa `X` và `y`.
