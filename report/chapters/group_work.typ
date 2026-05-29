#import "../theme.typ": *
= Phân công công việc và đánh giá

#v(0.6cm)

#figure(
  kind: table,
  table(
    columns: (auto, auto, 1.2fr, 0.8fr, 2.7fr, auto),
    fill: (col, row) => if row == 0 { title_color } else if calc.odd(row) { rgb("#F4F7FB") } else { none },
    align: (col, row) => if row == 0 { center + horizon } else {
      (center, center, left, center, left, center).at(col) + horizon
    },
    stroke: 0.6pt + title_color,
    inset: (x: 7pt, y: 10pt),

    table.header(
      [*#text(white)[STT]*],
      [*#text(white)[MSSV]*],
      [*#text(white)[Họ và tên]*],
      [*#text(white)[Vai trò]*],
      [*#text(white)[Nhiệm vụ]*],
      [*#text(white)[Đánh giá]*],
    ),

    [1], [24120002], [Đinh Đức Hiếu], [Thành viên], [Phần 1: Xây dựng thuật toán cốt lõi cho OLS, Ridge, VIF], [100%],
    [2], [24120049], [Liên Trung Hiếu], [Thành viên], [Phần 2 (ML Modeler): Huấn luyện, tối ưu siêu tham số và so sánh các mô hình], [100%],
    [3], [24120064], [Trương Đình Nhật Huy], [Thành viên], [Phần 2 (Data Engineer): Làm sạch dữ liệu, EDA và xây dựng Pipeline tiền xử lý], [100%],
    [4], [24120127], [Nguyễn Ngọc Quang], [Nhóm trưởng], [Quản lý nhóm và Phần 1: Kiểm chứng, phân tích thống kê phương pháp OLS], [100%],
    [5], [24120149], [Đặng Quang Tiến], [Thành viên], [Phần 2 (Report Analyst): Phân tích chẩn đoán phần dư, trực quan hóa và tổng hợp báo cáo], [100%],
  ),
  caption: [Phân công công việc và đánh giá mức độ đóng góp của các thành viên]
)
