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
  - `Age = 2026 - YearBuilt`
  - `OtherRooms = Rooms - Bathroom`
  - `BuildingArea_per_Room = BuildingArea / Rooms`
  - `BuildingCoverage = BuildingArea / Landsize`
  - `BuildingArea_missing`
  - `YearBuilt_missing`
  - `Landsize_zero_or_missing`

## Cột bị drop

- `Address`: gần như định danh từng căn nhà, cardinality rất cao.
- `SellerG`: cardinality cao, không cần cho baseline OLS/Ridge gọn.
- `Suburb`: cardinality cao; giữ `Regionname`, `Lattitude`, `Longtitude` làm tín hiệu vị trí chính.
- `Date`: không dùng trong pipeline mới vì `Age` được tính theo mốc 2026.
- `Postcode`: tín hiệu vị trí bị trùng với `Regionname`, latitude và longitude.
- `Method`, `CouncilArea`: bị drop để giữ đúng phạm vi one-hot của task A4 là `Type` và `Regionname`.
- `Bedroom2`: drop qua cấu hình `DataPipeline(drop_columns=["Bedroom2"])` để thể hiện cơ chế drop cột theo cấu hình.

Drop được thực hiện trong pipeline sau bước repair invalid value, KNN imputation và feature engineering.

## Quá trình xử lí data

- Invalid values:
  - `BuildingArea <= 0` được xem là missing.
  - `YearBuilt < 1800` được xem là missing.
  - `Landsize <= 0` được xem là missing trước khi tính `BuildingCoverage`.
- Missing values:
  - Các cột numeric được xử lý bằng `KNNImputer`.
  - `KNNImputer` chỉ fit trên `X_train_raw` và được reuse khi transform `X_test_raw`.
- Encoding:
  - One-hot encode `Type`, `Regionname`.
  - Test data reindex theo `self.encoded_columns` đã fit từ train data.
  - Logic nằm trong `DataPipeline._engineer_and_encode()`.
- Scaling:
  - Z-score toàn bộ feature numeric cuối cùng bằng mean/std từ train data.
  - Test data dùng lại `self.scalers` đã fit từ train data.
- Validation:
  - Pipeline kiểm tra schema đầu vào và báo lỗi nếu thiếu cột bắt buộc.
  - Sau one-hot/drop, nếu còn cột non-numeric thì báo lỗi.
  - Notebook kiểm tra feature contract, target validity và alignment giữa `X` và `y`.
