# Ghi chú

## Tổng quan

- Dataset gốc: `melb_data.csv`, 13,580 dòng, 21 cột.
- Target: `Price`, giữ nguyên dạng gốc, không log-transform.
- EDA thực hiện trên train split để hạn chế data snooping.
- Output cuối: `X_train`, `X_test`, `y_train`, `y_test`, `metadata`.

## Cột sử dụng

- Numeric gốc: `Rooms`, `Distance`, `Bedroom2`, `Bathroom`, `Car`, `Landsize`, `BuildingArea`, `YearBuilt`, `Lattitude`, `Longtitude`, `Propertycount`.
- Categorical: `Type`, `Method`, `Regionname`, `CouncilArea`.
- Feature tạo thêm:
  - `PropertyAge = sale_year - YearBuilt`
  - `OtherRooms = Rooms - Bathroom`
  - `BuildingArea_per_Room = BuildingArea / Rooms`
  - `BuildingCoverage = BuildingArea / Landsize`
  - `BuildingArea_missing`
  - `YearBuilt_missing`
  - `Landsize_zero_or_missing`

## Cột bị drop

- `Address`: gần như định danh từng căn nhà, cardinality rất cao.
- `SellerG`: nhiều giá trị khác nhau, không cần cho baseline OLS/Ridge gọn.
- `Suburb`: cardinality cao; giữ `Regionname`, `CouncilArea`, `Lattitude`, `Longtitude` làm tín hiệu vị trí chính.
- `Date`: chỉ dùng để lấy năm bán khi tạo `PropertyAge`, sau đó drop cột gốc.
- `Postcode`: tín hiệu vị trí bị trùng với `Regionname`, `CouncilArea`, latitude và longitude.

Drop được thực hiện trong pipeline sau các bước repair invalid value, imputation và feature extraction, để các cột như `Date` vẫn có thể phục vụ tạo feature trước khi bị loại.

## Quá trình xử lí data

- Invalid values:
  - `BuildingArea <= 0` được xem là missing.
  - `YearBuilt < 1800` được xem là missing.
  - `Landsize <= 0` được xem là missing trước khi tính `BuildingCoverage`.
  - `Landsize_zero_or_missing` được thêm để giữ lại tín hiệu `Landsize` không dùng được cho ratio.
- Missing values:
  - `BuildingArea`, `YearBuilt`, `Car` dùng median theo `Type`, fit trên train data.
  - Nếu một nhóm `Type` không có median, dùng global median của train data.
  - Missing `CouncilArea` được fill bằng `Unknown`.
- Encoding:
  - One-hot encode `Type`, `Method`, `Regionname`, `CouncilArea`.
  - Test data reindex theo encoded columns đã fit từ train data.
- Scaling:
  - Standardize toàn bộ feature cuối cùng bằng mean/std của train data.
  - Test data dùng lại fitted state từ train data.
- Validation:
  - Pipeline kiểm tra schema đầu vào và báo lỗi nếu thiếu cột bắt buộc.
  - Sau one-hot/drop, nếu còn cột non-numeric thì báo lỗi thay vì ép thành missing value.
  - Notebook kiểm tra feature contract, target validity và alignment giữa `X` và `y`.

## Nhận xét đưa ra sau khi EDA

- `Price`: lệch phải mạnh và có outlier lớn.
- `Rooms`, `Bedroom2`, `Bathroom`: cùng mô tả quy mô căn nhà; `Bedroom2` gần với `Rooms`, nên thêm `OtherRooms` để giữ tín hiệu layout đơn giản.
- `Distance`: phần lớn nhà nằm gần trung tâm hơn, khoảng cách xa xuất hiện ít hơn.
- `Landsize`: có nhiều giá trị 0 và outlier rất lớn; giá trị không dương không phù hợp khi tính tỷ lệ coverage.
- `BuildingArea`: thiếu nhiều và có outlier lớn; dùng median theo `Type` hợp lý hơn mean.
- `YearBuilt`: thiếu nhiều và có một giá trị bất hợp lý rất cũ; được sửa trước khi tạo `PropertyAge`.
- `Type`: phân tách rõ house, townhouse, unit; median price khác nhau đáng kể.
- `Method`: cardinality thấp, encode trực tiếp được.
- `Regionname`: giữ tín hiệu vùng lớn, cardinality thấp.
- `CouncilArea`: có missing nhưng vẫn hữu ích cho vị trí; missing được giữ thành nhóm `Unknown`.
- `Lattitude`, `Longtitude`: giữ tín hiệu không gian trực tiếp; geo scatter chỉ lọc top 1% giá để dễ nhìn, không drop khỏi pipeline.
