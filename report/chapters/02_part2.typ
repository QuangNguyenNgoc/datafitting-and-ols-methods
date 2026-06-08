#import "../theme.typ": *
= Ứng dụng mô hình hồi quy tuyến tính vào dữ liệu thực tế

== Thông tin bộ dữ liệu

Bộ dữ liệu được sử dụng là dữ liệu bất động sản Melbourne với mục tiêu dự đoán giá nhà.
- Kích thước gốc: 13.580 quan trắc và 21 đặc trưng.
- Đặc điểm dữ liệu: Đây là bộ dữ liệu thực tế phức tạp, chứa tỷ lệ khuyết thiếu tự nhiên lớn cùng nhiều biến phân loại đa dạng. Mẫu có quy mô quan trắc đủ lớn ($n >= 200$), số lượng đặc trưng đa dạng ($p >= 3$), biến mục tiêu dạng liên tục thích hợp cho bài toán hồi quy và tỷ lệ khuyết thiếu đáng kể nhằm thử nghiệm hiệu quả của các kỹ thuật nội suy.

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
1. Toán học (VIF > 10): Xác định các biến bị tương quan tuyến tính mạnh với các biến độc lập khác.
2. Thống kê (p-value > 0.05): Xem xét ý nghĩa thống kê của biến đó.
3. Ý nghĩa kinh tế: Đối chiếu giữa các cặp biến tương quan mạnh để quyết định giữ biến gốc hoặc biến bao quát hơn và loại bỏ biến phái sinh hoặc biến thứ cấp.

Quy trình chẩn đoán được thực hiện qua các vòng lặp chẩn đoán:
- Vòng chưa loại biến: Kết quả tính toán cho thấy hai hiện tượng đa cộng tuyến nghiêm trọng:
  - Cặp biến diện tích xây dựng và diện tích xây dựng trung bình mỗi phòng bị trùng lặp thông tin trầm trọng.
  - Cặp biến số phòng tổng và số phòng ngủ làm tăng nhân tử phóng đại phương sai vượt ngưỡng an toàn.
- Vòng thứ nhất (loại bỏ số phòng ngủ): Khi loại bỏ đặc trưng thứ cấp số phòng ngủ, chỉ số VIF của biến số phòng lập tức giảm mạnh xuống mức an toàn. Tuy nhiên, cặp biến diện tích vẫn duy trì chỉ số ở mức báo động.
- Vòng thứ hai (loại bỏ số phòng ngủ và diện tích xây dựng trung bình mỗi phòng): Quyết định giữ lại đặc trưng gốc diện tích xây dựng và loại bỏ đặc trưng phái sinh diện tích xây dựng trung bình mỗi phòng. Sau vòng lặp này, toàn bộ các đặc trưng còn lại đều có chỉ số VIF dưới ngưỡng an toàn 5.0 (với khu vực miền tây đạt tối đa 6.65, các biến khác đều nhỏ hơn 5.0).

Danh sách loại bỏ cuối cùng được xác lập bao gồm số phòng ngủ và diện tích xây dựng trung bình mỗi phòng. Sau khi loại bỏ các đặc trưng này, việc kiểm tra các điều kiện toán học của ma trận thiết kế được tiến hành:
- Kiểm tra hạng: Hạng của ma trận thiết kế bằng đúng số lượng cột ($"rank"(X) = 23$), xác nhận ma trận đạt đầy đủ hạng cột.
- Chỉ số điều kiện: Chỉ số đạt mức xấp xỉ 3.79, nhỏ hơn rất nhiều so với ngưỡng cảnh báo. Điều này bảo đảm ma trận hồi quy hoàn toàn khả nghịch và không bị suy biến về mặt số học.

== So sánh mô hình

Để đánh giá khách quan hiệu năng của các phương pháp hồi quy, phép đo lường chéo được thực hiện trên cả tập huấn luyện và tập kiểm thử thông qua các chỉ số: sai số tuyệt đối trung bình, căn bậc hai sai số bình phương trung bình và hệ số xác định ($R^2$). Quy trình huấn luyện và đánh giá được thực hiện nghiêm ngặt sau khi chia dữ liệu huấn luyện và kiểm thử để tránh hiện tượng rò rỉ thông tin.

Bảng @leaderboard tổng hợp kết quả hiệu năng của năm mô hình thực nghiệm và được sắp xếp theo thứ tự sai số bình phương trung bình tăng dần trên tập kiểm thử:

#figure(
  kind: table,
  [
    #set text(size: 10pt)
    #table(
      columns: (2.3fr, 1.1fr, 1.1fr, 0.9fr, 1.1fr, 0.9fr, 2.5fr),
      fill: (col, row) => if row == 0 { title_color } else if calc.odd(row) { rgb("#F4F7FB") } else { none },
      align: (col, row) => if row == 0 { center + horizon } else {
        (left, right, right, right, right, right, left).at(col) + horizon
      },
      stroke: 0.6pt + title_color,
      inset: (x: 5pt, y: 8pt),

      table.header(
        [*#text(white)[Mô hình]*],
        [*#text(white)[MAE (Kiểm thử)]*],
        [*#text(white)[RMSE (Kiểm thử)]*],
        [*#text(white)[$R^2$ (Kiểm thử)]*],
        [*#text(white)[RMSE (Huấn luyện)]*],
        [*#text(white)[$R^2$ (Huấn luyện)]*],
        [*#text(white)[Tham số / Ghi chú]*]
      ),

      [Hồi quy tuyến tính cơ bản], [271,915], [419,893], [0.5752], [393,991], [0.6176], [Toàn bộ 25 biến gốc],
      [Hồi quy bayes], [272,137], [419,984], [0.5750], [394,126], [0.6174], [23 biến sạch, tiên nghiệm yếu],
      [Hồi quy tuyến tính chọn lọc], [272,137], [419,984], [0.5750], [394,126], [0.6174], [23 biến đã lọc VIF],
      [Hồi quy ridge], [267,935], [423,830], [0.5672], [398,465], [0.6089], [23 biến, $lambda = 500.0$],
      [Hồi quy kernel ridge], [250,202], [432,857], [0.5486], [400,274], [0.6053], [Nhân rbf, $alpha = 1.0$]
    )
  ],
  caption: [Bảng so sánh hiệu năng các mô hình trên tập kiểm thử và tập huấn luyện]
) <leaderboard>

Từ bảng kết quả trên, có thể rút ra một số nhận xét thống kê quan trọng:
- Đánh đổi giữa hiệu năng dự báo và tính diễn giải: Mô hình hồi quy tuyến tính cơ bản đạt sai số bình phương trung bình thấp nhất ($419,893$) và hệ số xác định cao nhất ($0.5752$) trên tập kiểm thử. Tuy nhiên, mô hình này bị ảnh hưởng nặng nề bởi đa cộng tuyến. Mô hình tuyến tính chọn lọc sau khi loại bỏ hai thuộc tính trùng lặp chỉ ghi nhận mức giảm hiệu năng cực kỳ nhỏ (sai số bình phương trung bình tăng lên $419,984$ và hệ số xác định giảm về $0.5750$). Đây là một sự đánh đổi hoàn toàn xứng đáng để có được một mô hình ổn định, các hệ số hồi quy có ý nghĩa thống kê và độ tin cậy cao hơn.
- Vai trò của chính quy hóa: Mô hình hồi quy ridge với siêu tham số tối ưu $lambda = 500.0$ (được tìm kiếm qua kiểm chứng chéo năm phần trên tập huấn luyện) đạt sai số bình phương trung bình trên tập kiểm thử là $423,830$. Việc hồi quy ridge không vượt qua được hồi quy tuyến tính thông thường cho thấy sau khi loại bỏ đa cộng tuyến bằng lọc biến, ma trận thiết kế đã đủ tốt và không cần thêm cơ chế phạt bậc hai.
- Sự tương đồng của mô hình bayes: Mô hình hồi quy bayes cho kết quả dự báo và hệ số hồi quy gần như trùng khớp với hồi quy tuyến tính chọn lọc. Điều này phù hợp với lý thuyết: khi sử dụng phân phối tiên nghiệm yếu trên một mẫu dữ liệu có kích thước đủ lớn ($n = 9506$), phân phối hậu nghiệm sẽ hội tụ hoàn toàn về ước lượng bình phương bé nhất.
- Giới hạn của mô hình phi tuyến: Mô hình hồi quy kernel ridge sử dụng nhân rbf cho kết quả kém nhất ($432,857$). Nguyên nhân có thể do hàm nhân rbf nhạy cảm với các điểm ngoại lai của giá nhà ở hai đầu phân phối (đuôi dày) hoặc các siêu tham số chưa được tối ưu hóa hoàn toàn cho tập dữ liệu lớn này.

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

Thông qua phương pháp kiểm chứng chéo năm phần, tham số phạt tối ưu được xác định là $lambda = 500.0$. Biểu đồ lỗi bình phương trung bình cho thấy khi tham số phạt tăng lên, sai số trên tập kiểm chứng có xu hướng đi ngang rồi tăng vọt nếu tham số phạt quá lớn do các hệ số hồi quy bị triệt tiêu quá mức. Hiệu năng của hồi quy ridge không vượt trội so với hồi quy tuyến tính chọn lọc, chứng tỏ ma trận dữ liệu sau lọc biến đã ổn định và không cần chính quy hóa mạnh.

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
- Kiểm định breusch-pagan về phương sai thay đổi: Thu được giá trị thống kê $L M = 114.96$ với trị số $p approx 7.60 times 10^(-14) < 0.05$. Kết quả này bác bỏ giả thuyết phương sai sai số đồng nhất, xác nhận sự tồn tại của hiện tượng phương sai thay đổi. Do đó, các khoảng tin cậy của hệ số hồi quy cần được diễn giải thận trọng.
- Kiểm định jarque-bera về tính chuẩn: Thu được giá trị thống kê $J B = 344646.72$ với trị số $p = 0.0 < 0.05$. Kết quả bác bỏ hoàn toàn giả định về phân phối chuẩn của phần dư, xác nhận phân phối có đuôi rất dày do ảnh hưởng từ các giao dịch bất động sản có giá trị ngoại lai lớn.

Bài học rút ra: Để khắc phục các hạn chế này trong tương lai, một giải pháp khả thi là thực hiện biến đổi logarit cho biến mục tiêu giá nhà ($ln("Price")$) để đưa các giá trị giá bán siêu lớn về quy mô tuyến tính, giúp giảm bớt hiện tượng phương sai thay đổi và đưa phân phối phần dư về gần phân phối chuẩn hơn.
