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

    [1], [24120002], [Đinh Đức Hiếu], [Thành viên], [Thực hiện phần 1 (Phép khử Gauss và các ứng dụng)], [100%],
    [2], [24120049], [Liên Trung Hiếu], [Thành viên], [Thực hiện phần 3 (Giải hệ phương trình và benchmark)], [100%],
    [3], [24120064], [Trương Đình Nhật Huy], [Thành viên], [Thực hiện phần 2 (Manim)], [100%],
    [4], [24120127], [Nguyễn Ngọc Quang], [Nhóm trưởng], [Quản lý đồ án, viết kiểm thử thực nghiệm và tổng hợp báo cáo], [100%],
    [5], [24120149], [Đặng Quang Tiến], [Thành viên], [Thực hiện phần 2 (Chéo hóa và phân rã SVD)], [100%],
  ),
  caption: [Phân công công việc và đánh giá mức độ đóng góp của các thành viên]
)
