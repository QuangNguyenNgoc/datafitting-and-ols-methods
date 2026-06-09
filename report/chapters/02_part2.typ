#import "../theme.typ": *
= Ứng dụng mô hình hồi quy tuyến tính vào dữ liệu thực tế

== Thông tin bộ dữ liệu

Bộ dữ liệu được sử dụng là dữ liệu bất động sản Melbourne với mục tiêu dự đoán giá nhà.
- Kích thước gốc: 13580 quan trắc và 21 đặc trưng.
- Đặc điểm dữ liệu: Đây là bộ dữ liệu thực tế phức tạp, chứa tỷ lệ khuyết thiếu tự nhiên lớn cùng nhiều biến phân loại đa dạng. Mẫu có quy mô quan trắc đủ lớn ($n >= 200$), số lượng đặc trưng đa dạng ($p >= 3$), biến mục tiêu dạng liên tục thích hợp cho bài toán hồi quy và tỷ lệ khuyết thiếu đáng kể nhằm thử nghiệm hiệu quả của các kỹ thuật nội suy.

== Phân tích khám phá dữ liệu (EDA)

Trước khi tiến hành xây dựng mô hình hồi quy, việc phân tích khám phá dữ liệu được thực hiện trên tập huấn luyện thô nhằm nhận diện cấu trúc, phát hiện các điểm bất thường và định hình các bước tiền xử lý cần thiết.

@descriptive_stats trình bày thống kê mô tả cho các biến định lượng cốt lõi của bộ dữ liệu:

#figure(
  kind: table,
  [
    #set text(size: 9pt)
    #table(
      columns: (1.8fr, 1.2fr, 1fr, 1.2fr, 1fr, 1.2fr, 1.2fr),
      fill: (col, row) => if row == 0 { title_color } else if calc.odd(row) { rgb("#F4F7FB") } else { none },
      align: (col, row) => if row == 0 { center + horizon } else if col == 0 { left + horizon } else { right + horizon },
      stroke: 0.6pt + title_color,
      inset: (x: 5pt, y: 6pt),

      table.header(
        [*#text(white)[Chỉ số]*],
        [*#text(white)[Price\ (Giá - AUD)]*],
        [*#text(white)[Rooms\ (Số phòng)]*],
        [*#text(white)[Distance\ (Khoảng cách)]*],
        [*#text(white)[Landsize\ (Đất - m²)]*],
        [*#text(white)[BuildingArea\ (Xây dựng)]*],
        [*#text(white)[YearBuilt\ (Năm xây)]*]
      ),

      [Mẫu (Count)], [#box[9506]], [#box[9506]], [#box[9506]], [#box[9506]], [#box[4949]], [#box[5691]],
      [Trung bình (Mean)], [#box[1076128]], [2.94], [10.12], [541.6], [154.4], [#box[1964.4]],
      [Độ lệch chuẩn (Std)], [#box[637184]], [0.95], [5.90], [#box[1642.1]], [646.6], [37.7],
      [Tối thiểu (Min)], [#box[131000]], [1.00], [0.00], [0], [0], [#box[1196]],
      [Phân vị 1%], [#box[300050]], [1.00], [1.60], [0], [6.9], [#box[1880]],
      [Phân vị 5%], [#box[407125]], [2.00], [2.60], [0], [52.4], [#box[1900]],
      [Trung vị (50%)], [#box[905000]], [3.00], [9.20], [446], [126], [#box[1970]],
      [Phân vị 95%], [#box[2300000]], [4.00], [20.60], [#box[1002.5]], [291], [#box[2012]],
      [Phân vị 99%], [#box[3359500]], [5.00], [32.27], [#box[2976.9]], [446.6], [#box[2015]],
      [Tối đa (Max)], [#box[8000000]], [10.00], [48.10], [#box[76000]], [#box[44515]], [#box[2018]],
      [Hệ số lệch (Skew)], [2.19], [0.35], [1.67], [28.97], [65.56], [-1.94]
    )
  ],
  caption: [Bảng thống kê mô tả các đặc trưng định lượng quan trọng trên tập huấn luyện thô (dạng chuyển vị)]
) <descriptive_stats>

Từ bảng thống kê mô tả, ta nhận thấy một số đặc điểm quan trọng của dữ liệu thô:
- Phân phối lệch phải mạnh: Biến mục tiêu Price có hệ số lệch 2.19, với giá trị trung vị là 905000 AUD thấp hơn đáng kể so với trung bình 1076128.4 AUD. Điều này cho thấy sự xuất hiện của các biệt thự siêu sang kéo trung bình lên cao. Hiện tượng này càng rõ nét hơn ở Landsize (độ lệch 28.97) và BuildingArea (độ lệch 65.56), nơi có những bất động sản sở hữu diện tích đất lên tới 76000 m² và diện tích xây dựng 44515 m².
- Giá trị không hợp lệ: Biến Landsize và BuildingArea ghi nhận giá trị tối thiểu bằng 0, điều này phi lý đối với bất động sản giao dịch. Đồng thời, biến YearBuilt có giá trị tối thiểu là 1196, đây chắc chắn là lỗi nhập liệu do thành phố Melbourne chỉ được thành lập vào năm 1835. Những bất thường này đòi hỏi phải có bước làm sạch và sửa chữa dữ liệu trước khi huấn luyện mô hình.
- Vấn đề dữ liệu khuyết thiếu: Biến BuildingArea chỉ có 4949 quan sát hợp lệ (khuyết xấp xỉ 48%), và YearBuilt khuyết xấp xỉ 40%. Việc thiếu hụt dữ liệu lớn ở các thuộc tính quan trọng này cho thấy sự cần thiết của việc xây dựng các biến chỉ báo nhị phân để ghi nhận thông tin khuyết thiếu trước khi tiến hành nội suy.

#figure(
  image("../images/price_distribution.png", width: 85%),
  caption: [Biểu đồ phân phối của biến mục tiêu giá nhà]
) <price_dist_fig>

Biểu đồ phân phối giá nhà (@price_dist_fig) trực quan hóa độ lệch phải rõ rệt của biến mục tiêu, tập trung mật độ cao nhất ở phân khúc từ 0.5 đến 1.5 triệu AUD và kéo dài một đuôi rất dài về phía bên phải.

#figure(
  image("../images/outliers_boxplot.png", width: 85%),
  caption: [Biểu đồ hộp nhận diện ngoại lai của các đặc trưng quan trọng]
) <outliers_box_fig>

Biểu đồ hộp (@outliers_box_fig) xác nhận sự hiện diện dày đặc của các điểm ngoại lai ở cả bốn thuộc tính quan trọng. Giá nhà và diện tích đất có vô số điểm dữ liệu vượt xa hàng rào trên của hộp. Năm xây dựng cũng thể hiện các điểm ngoại lai lịch sử lệch sâu về phía dưới năm 1800.

#figure(
  image("../images/geo_spatial.png", width: 85%),
  caption: [Bản đồ phân bố giá nhà theo vị trí địa lý tại Melbourne]
) <geo_spatial_fig>

Bản đồ địa lý (@geo_spatial_fig) thể hiện rõ xu hướng không gian của giá nhà. Các bất động sản có giá trị cao nhất (biểu diễn bằng tông màu ấm/đỏ) tập trung cô đặc ở trung tâm địa lý (nơi có mật độ giao dịch cao nhất) và trải dài dọc theo bờ vịnh phía Đông Nam. Ngược lại, các khu vực phía Tây và phía Bắc có mật độ thưa thớt hơn và chủ yếu là các tông màu xanh lá/xanh dương biểu thị mức giá thấp hơn.

== Tiền xử lý dữ liệu và xử lý đa cộng tuyến


Để mô hình hồi quy tuyến tính hoạt động vững chãi và đạt hiệu quả tối ưu, toàn bộ quá trình biến đổi dữ liệu thô thành ma trận thiết kế $X$ được tự động hóa thông qua lớp kiến trúc quy trình tiền xử lý hướng đối tượng. Quy trình tiền xử lý và loại bỏ đa cộng tuyến được thực hiện qua hai giai đoạn chặt chẽ:

=== Giai đoạn làm sạch dữ liệu, kỹ nghệ đặc trưng và chuẩn hóa

Trong giai đoạn này, luồng xử lý tập trung vào việc làm sạch nhiễu cơ bản, khắc phục khuyết thiếu và kỹ nghệ đặc trưng:
- Xử lý giá trị không hợp lệ: Bộ dữ liệu thực tế chứa nhiều dữ liệu phi logic do sai sót nhập liệu. Các bộ lọc miền giá trị nghiêm ngặt được áp dụng:
  - Các giá trị âm không hợp lý ở các cột đếm và khoảng cách như số phòng, số phòng tắm, số chỗ đậu xe, khoảng cách đến trung tâm, số lượng bất động sản được chuyển thành giá trị khuyết.
  - Biến số phòng nếu bằng 0 cũng được chuyển thành khuyết.
  - Các bất động sản có diện tích xây dựng nhỏ hơn hoặc bằng 0 hoặc năm xây dựng trước 1800 được quy về khuyết.
  - Để lưu giữ tín hiệu khuyết thiếu của các thuộc tính quan trọng này, chương trình tự động tạo ra các biến chỉ báo nhị phân để ghi nhận thông tin khuyết thiếu trước khi thực hiện nội suy.
- Nội suy dữ liệu khuyết: Thay vì sử dụng các phương pháp phức tạp gây rò rỉ thông tin, phương pháp nội suy bằng giá trị trung vị của từng cột định lượng được áp dụng. Việc tính toán trung vị này được thực hiện nghiêm ngặt chỉ trên tập huấn luyện và được lưu giữ để áp dụng đồng nhất lên tập kiểm thử, ngăn ngừa triệt để hiện tượng rò rỉ dữ liệu.
- Kỹ nghệ đặc trưng: Ba đặc trưng phái sinh mang giá trị kinh tế được xây dựng:
  - Tuổi của căn nhà: Được tính bằng hiệu số giữa năm giao dịch và năm xây dựng, trong đó năm giao dịch được trích xuất trực tiếp từ trường ngày tháng (cột gốc sau đó được loại bỏ để tránh đa cộng tuyến trực tiếp).
  - Diện tích xây dựng trung bình mỗi phòng: Được tính bằng tỷ lệ giữa diện tích xây dựng và số phòng để nắm bắt mật độ không gian sinh hoạt.
  - Mật độ xây dựng: Được tính bằng tỷ lệ diện tích xây dựng trên diện tích đất để phản ánh tỷ lệ đất được sử dụng cho việc xây dựng.
- Mã hóa biến phân loại và chuẩn hóa: Áp dụng mã hóa một nóng cho hai biến phân loại chính là loại nhà và khu vực địa lý với tùy chọn loại bỏ cột đầu tiên để tránh bẫy đa cộng tuyến hoàn hảo. Cuối cùng, toàn bộ dữ liệu định lượng và các biến chỉ báo nhị phân được đưa về chung một thang đo bằng phương pháp chuẩn hóa điểm chuẩn ($z = (x - mu)/sigma$), trong đó trung bình $mu$ và độ lệch chuẩn $sigma$ được cố định từ tập huấn luyện.
- Lọc biến định danh: Loại bỏ các thuộc tính định danh có số lượng giá trị phân biệt quá cao hoặc không mang thông tin dự báo trực tiếp như địa chỉ, người bán, vùng ngoại ô, mã bưu điện, phương thức giao dịch và khu vực hội đồng.

=== Giai đoạn chẩn đoán và xử lý đa cộng tuyến bằng hệ số nhân tử phóng đại phương sai

Sau giai đoạn làm sạch dữ liệu, ma trận thiết kế $X$ vẫn có khả năng tiềm ẩn hiện tượng đa cộng tuyến. Hiện tượng này làm cho ma trận $X^T X$ tiến sát trạng thái suy biến, khiến phương sai của các ước lượng hệ số hồi quy bị thổi phồng, làm mất tính ổn định và khả năng diễn giải kinh tế của mô hình.

Để chẩn đoán và loại bỏ đa cộng tuyến, một trạm kiểm soát toán học sử dụng nhân tử phóng đại phương sai (VIF) được thiết lập. Quy trình ra quyết định tuân thủ chặt chẽ ba tiêu chí:
1. Toán học (VIF > 5.0): Xác định các biến bị tương quan tuyến tính mạnh với các biến độc lập khác.
2. Thống kê (p-value > 0.05): Xem xét ý nghĩa thống kê của biến đó.
3. Ý nghĩa kinh tế: Đối chiếu giữa các cặp biến tương quan mạnh để quyết định giữ biến gốc hoặc biến bao quát hơn và loại bỏ biến phái sinh hoặc biến thứ cấp.

Quy trình chẩn đoán được thực hiện qua các vòng lặp chẩn đoán:
- Vòng chưa loại biến: Kết quả tính toán cho thấy hai hiện tượng đa cộng tuyến nghiêm trọng:
  - Cặp biến diện tích xây dựng và diện tích xây dựng trung bình mỗi phòng bị trùng lặp thông tin trầm trọng.
  - Cặp biến số phòng tổng và số phòng ngủ làm tăng nhân tử phóng đại phương sai vượt ngưỡng an toàn.
- Vòng thứ nhất (loại bỏ số phòng ngủ): Khi loại bỏ đặc trưng thứ cấp số phòng ngủ, chỉ số VIF của biến số phòng lập tức giảm mạnh xuống mức an toàn. Tuy nhiên, cặp biến diện tích vẫn duy trì chỉ số ở mức báo động.
- Vòng thứ hai (loại bỏ số phòng ngủ và diện tích xây dựng trung bình mỗi phòng): Quyết định giữ lại đặc trưng gốc diện tích xây dựng và loại bỏ đặc trưng phái sinh diện tích xây dựng trung bình mỗi phòng. Sau vòng lặp này, toàn bộ các đặc trưng còn lại đều có chỉ số VIF dưới ngưỡng an toàn 5.0 (với khu vực miền Tây đạt tối đa 6.65, các biến khác đều nhỏ hơn 5.0).

Danh sách loại bỏ cuối cùng được xác lập bao gồm số phòng ngủ và diện tích xây dựng trung bình mỗi phòng. Sau khi loại bỏ các đặc trưng này, việc kiểm tra các điều kiện toán học của ma trận thiết kế được tiến hành:
- Kiểm tra hạng: Hạng của ma trận thiết kế bằng đúng số lượng cột ($"rank"(X) = 23$), xác nhận ma trận đạt đầy đủ hạng cột.
- Chỉ số điều kiện: Chỉ số đạt mức xấp xỉ 3.79, nhỏ hơn rất nhiều so với ngưỡng cảnh báo. Điều này bảo đảm ma trận hồi quy hoàn toàn khả nghịch và không bị suy biến về mặt số học.

== So sánh mô hình

Để đánh giá khách quan hiệu năng của các phương pháp hồi quy, phép đo lường chéo được thực hiện trên cả tập huấn luyện và tập kiểm thử thông qua các chỉ số: sai số tuyệt đối trung bình, căn bậc hai sai số bình phương trung bình và hệ số xác định ($R^2$). Quy trình huấn luyện và đánh giá được thực hiện nghiêm ngặt sau khi chia dữ liệu huấn luyện và kiểm thử để tránh hiện tượng rò rỉ thông tin.

@leaderboard tổng hợp kết quả hiệu năng của năm mô hình thực nghiệm và được sắp xếp theo thứ tự sai số bình phương trung bình tăng dần trên tập kiểm thử:

#figure(
  kind: table,
  [
    #set text(size: 9.5pt)
    #table(
      columns: (2fr, 1.3fr, 1.3fr, 1fr, 1.3fr, 1fr, 2.1fr),
      fill: (col, row) => if row == 0 { title_color } else if calc.odd(row) { rgb("#F4F7FB") } else { none },
      align: (col, row) => if row == 0 { center + horizon } else {
        (left, right, right, right, right, right, left).at(col) + horizon
      },
      stroke: 0.6pt + title_color,
      inset: (x: 4pt, y: 8pt),

      table.header(
        [*#text(white)[Mô hình]*],
        [*#text(white)[MAE\ (Kiểm thử)]*],
        [*#text(white)[RMSE\ (Kiểm thử)]*],
        [*#text(white)[$R^2$\ (Kiểm thử)]*],
        [*#text(white)[RMSE\ (Huấn luyện)]*],
        [*#text(white)[$R^2$\ (Huấn luyện)]*],
        [*#text(white)[Tham số / Ghi chú]*]
      ),

      [Hồi quy tuyến tính cơ bản], [#box[271915]], [#box[419893]], [#box[0.5752]], [#box[393991]], [#box[0.6176]], [Toàn bộ 25 biến gốc],
      [Hồi quy Bayes], [#box[272137]], [#box[419984]], [#box[0.5750]], [#box[394126]], [#box[0.6174]], [23 biến sạch, tiên nghiệm yếu],
      [Hồi quy tuyến tính chọn lọc], [#box[272137]], [#box[419984]], [#box[0.5750]], [#box[394126]], [#box[0.6174]], [23 biến đã lọc VIF],
      [Hồi quy Ridge], [#box[267935]], [#box[423830]], [#box[0.5672]], [#box[398465]], [#box[0.6089]], [23 biến, $lambda = 500.0$],
      [Hồi quy Kernel Ridge], [#box[250202]], [#box[432857]], [#box[0.5486]], [#box[400274]], [#box[0.6053]], [Nhân RBF, $alpha = 1.0$]
    )
  ],
  caption: [Bảng so sánh hiệu năng các mô hình trên tập kiểm thử và tập huấn luyện]
) <leaderboard>

Từ bảng kết quả trên, có thể rút ra một số nhận xét thống kê quan trọng:
- Đánh đổi giữa hiệu năng dự báo và tính diễn giải: Mô hình hồi quy tuyến tính cơ bản đạt căn bậc hai sai số bình phương trung bình thấp nhất ($419893$) và hệ số xác định cao nhất ($0.5752$) trên tập kiểm thử. Tuy nhiên, mô hình này bị ảnh hưởng nặng nề bởi đa cộng tuyến. Mô hình tuyến tính chọn lọc sau khi loại bỏ hai thuộc tính trùng lặp chỉ ghi nhận mức giảm hiệu năng cực kỳ nhỏ (căn bậc hai sai số bình phương trung bình tăng lên $419984$ và hệ số xác định giảm về $0.5750$). Đây là một sự đánh đổi hoàn toàn xứng đáng để có được một mô hình ổn định, các hệ số hồi quy có ý nghĩa thống kê và độ tin cậy cao hơn.
- Vai trò của chính quy hóa: Mô hình hồi quy Ridge với siêu tham số tối ưu $lambda = 500.0$ (được tìm kiếm qua kiểm chứng chéo năm phần trên tập huấn luyện) đạt căn bậc hai sai số bình phương trung bình trên tập kiểm thử là $423830$. Việc hồi quy Ridge không vượt qua được hồi quy tuyến tính thông thường cho thấy sau khi loại bỏ đa cộng tuyến bằng lọc biến, ma trận thiết kế đã đủ tốt và không cần thêm cơ chế phạt bậc hai.
- Sự tương đồng của mô hình Bayes: Mô hình hồi quy Bayes cho kết quả dự báo và hệ số hồi quy gần như trùng khớp với hồi quy tuyến tính chọn lọc. Điều này phù hợp với lý thuyết: khi sử dụng phân phối tiên nghiệm yếu trên một mẫu dữ liệu có kích thước đủ lớn ($n = 9506$), phân phối hậu nghiệm sẽ hội tụ hoàn toàn về ước lượng bình phương bé nhất.
- Giới hạn của mô hình phi tuyến: Mô hình hồi quy Kernel Ridge sử dụng nhân RBF cho kết quả kém nhất ($432857$). Nguyên nhân có thể do hàm nhân RBF nhạy cảm với các điểm ngoại lai của giá nhà ở hai đầu phân phối (đuôi dày) hoặc các siêu tham số chưa được tối ưu hóa hoàn toàn cho tập dữ liệu lớn này.

== Tầm quan trọng của các đặc trưng

#figure(
  image("../images/feature_importance.png", width: 90%),
  caption: [Biểu đồ tầm quan trọng của các đặc trưng]
)

Nhìn vào biểu đồ hệ số hồi quy, có thể rút ra những hiểu biết về thị trường bất động sản:
- Khoảng cách đến trung tâm thành phố là yếu tố chi phối mạnh nhất: Hệ số có giá trị âm lớn nhất, phản ánh việc khoảng cách đến trung tâm tăng lên sẽ làm giảm giá trị bất động sản đáng kể.
- Quy mô không gian quyết định giá trị gia tăng: Các đặc trưng diện tích xây dựng và số phòng có hệ số dương lớn nhất, kéo giá nhà tăng cao.

== Tác động của chính quy hóa

#figure(
  image("../images/cv_error.png", width: 70%),
  caption: [Đường cong lỗi kiểm chứng chéo theo tham số phạt $lambda$]
)

Thông qua phương pháp kiểm chứng chéo năm phần, tham số phạt tối ưu được xác định là $lambda = 500.0$. Biểu đồ lỗi bình phương trung bình cho thấy khi tham số phạt tăng lên, sai số trên tập kiểm chứng có xu hướng đi ngang rồi tăng vọt nếu tham số phạt quá lớn do các hệ số hồi quy bị triệt tiêu quá mức. Hiệu năng của hồi quy Ridge không vượt trội so với hồi quy tuyến tính chọn lọc, chứng tỏ ma trận dữ liệu sau lọc biến đã ổn định và không cần chính quy hóa mạnh.

== Kiểm định giả định hồi quy tuyến tính và chẩn đoán mô hình

#figure(
  image("../images/residual_diagnostics.png", width: 100%),
  caption: [Biểu đồ chẩn đoán phần dư]
)

Dựa vào các biểu đồ chẩn đoán, có thể đánh giá về giới hạn của mô hình hồi quy tuyến tính thông thường:
- Hiện tượng phương sai thay đổi: Biểu đồ phần dư so với giá trị khớp có dạng hình phễu mở rộng sang phải. Điều này phản ánh sai số dự báo rất thấp đối với các căn nhà giá thấp, nhưng lại cực kỳ lớn đối với các bất động sản giá trị cao. Nguyên nhân do giá trị của phân khúc cao cấp chịu ảnh hưởng từ nhiều yếu tố vô hình (phong thủy, kiến trúc độc bản) không có trong dữ liệu quan trắc.
- Phân phối của phần dư: Biểu đồ phân vị cho thấy phần dư bám sát đường chéo ở đoạn giữa nhưng lệch mạnh ở hai đuôi, vi phạm giả định phân phối chuẩn, cho thấy dữ liệu có chứa nhiều điểm ngoại lai và phân phối đuôi dày.
- Điểm ảnh hưởng: Biểu đồ phần dư so với đòn bẩy cho thấy một vài điểm dữ liệu có đòn bẩy cao, có thể là các bất động sản có diện tích đất rất lớn làm lệch hướng đường hồi quy.

Để lượng hóa các vi phạm giả định trên, các kiểm định thống kê chính thức được tiến hành:
- Kiểm định Breusch-Pagan về phương sai thay đổi: Thu được giá trị thống kê $L M = 114.96$ với trị số $p approx 7.60 times 10^(-14) < 0.05$. Kết quả này bác bỏ giả thuyết phương sai sai số đồng nhất, xác nhận sự tồn tại của hiện tượng phương sai thay đổi. Do đó, các khoảng tin cậy của hệ số hồi quy cần được diễn giải thận trọng.
- Kiểm định Jarque-Bera về tính chuẩn: Thu được giá trị thống kê $J B = 344646.72$ với trị số $p = 0.0 < 0.05$. Kết quả bác bỏ hoàn toàn giả định về phân phối chuẩn của phần dư, xác nhận phân phối có đuôi rất dày do ảnh hưởng từ các giao dịch bất động sản có giá trị ngoại lai lớn.

Bài học rút ra: Để khắc phục các hạn chế này trong tương lai, một giải pháp khả thi là thực hiện biến đổi logarit cho biến mục tiêu giá nhà ($ln("Price")$) để đưa các giá trị giá bán siêu lớn về quy mô tuyến tính, giúp giảm bớt hiện tượng phương sai thay đổi và đưa phân phối phần dư về gần phân phối chuẩn hơn.

== Đề xuất hành động thực tiễn cho doanh nghiệp

Từ kết quả phân tích khám phá dữ liệu (EDA) và ước lượng hệ số của mô hình hồi quy tuyến tính chọn lọc, ta có thể rút ra một số đề xuất chiến lược có giá trị thực tiễn cao cho các doanh nghiệp đầu tư, phát triển và môi giới bất động sản tại Melbourne:

- Chiến lược đầu tư theo khoảng cách địa lý: Vì khoảng cách đến trung tâm thành phố (Distance) là yếu tố tác động tiêu cực mạnh nhất đến giá nhà, các doanh nghiệp đầu tư nên tập trung nguồn vốn phát triển phân khúc bất động sản cao cấp, nhà phố thông minh hoặc căn hộ cao tầng tại các vị trí trong bán kính dưới 10 km từ trung tâm. Tại đây, nhu cầu về vị trí đắc địa sẽ bảo đảm tính thanh khoản và biên lợi nhuận cao. Đối với các khu đất xa trung tâm (Distance > 15 - 20 km), chủ đầu tư nên định vị sản phẩm theo mô hình đô thị sinh thái xanh, tích hợp đầy đủ dịch vụ tiện ích để bù đắp cho điểm yếu về khoảng cách.
- Tối ưu hóa cấu trúc sản phẩm xây dựng: Hệ số dương lớn của biến số phòng (Rooms) và diện tích xây dựng (BuildingArea) so với hệ số tương đối nhỏ của diện tích đất trống (Landsize) cho thấy khách hàng ưu tiên không gian sống thực tế hơn là khuôn viên trống. Do đó, các nhà thiết kế và chủ đầu tư nên ưu tiên tối ưu hóa số lượng phòng chức năng trên cùng một diện tích sàn bằng thiết kế thông minh (ví dụ: căn hộ 2-3 phòng ngủ có gác lửng, tối đa hóa không gian sinh hoạt chung) thay vì mua những mảnh đất quá rộng nhưng chỉ xây dựng nhà nhỏ, giúp tối ưu hóa giá trị thương mại trên mỗi mét vuông đất.
- Quản trị rủi ro và định giá phân khúc ngoại lai: Việc phát hiện nhiều điểm ngoại lai cực đoan (nhà siêu đắt trên 3.3 triệu AUD, diện tích đất cực đại) và hiện tượng phương sai thay đổi lớn ở phân khúc cao cấp cho thấy các công cụ định giá tự động bằng tuyến tính thông thường chỉ hoạt động hiệu quả cho phân khúc nhà ở phổ thông (dưới 2 triệu AUD). Đối với phân khúc cao cấp và siêu sang, các doanh nghiệp môi giới và ngân hàng cần áp dụng quy trình thẩm định giá chuyên biệt kết hợp so sánh trực tiếp và định giá theo giá trị độc bản (kiến trúc, cảnh quan ven vịnh phía Đông Nam) để tránh rủi ro định giá sai lệch lớn.
- Khai thác lợi thế khu vực không gian địa lý: Bản đồ phân bố giá nhà cho thấy khu vực ven biển phía Đông Nam sở hữu mức giá vượt trội so với phía Tây và phía Bắc. Do đó, các doanh nghiệp môi giới nên định hướng tập trung đội ngũ tư vấn tại khu vực phía Đông Nam để nắm bắt phân khúc khách hàng thượng lưu có khả năng tài chính cao. Đồng thời, khu vực phía Tây và phía Bắc với chi phí quỹ đất rẻ là cơ hội vàng để phát triển các dự án nhà ở bình dân, nhà ở xã hội đáp ứng nhu cầu khổng lồ của tầng lớp người lao động và dân nhập cư.
