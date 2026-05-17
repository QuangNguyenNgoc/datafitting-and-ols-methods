#let info-title(body) = text(weight: "semibold", 12pt, body)

// Áp dụng viền từ ảnh border.png
#place(top + left, dx: -1.5cm, dy: -1.5cm)[
  #image("../images/border.png", width: 100% + 3cm, height: 100% + 3cm, fit: "stretch")
]

// first page
#align(center)[
  #v(0.5cm)
  #text(16pt, weight: "bold")[ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH \
  TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN]
  #v(0.3cm)
  #text(14pt, weight: "bold")[KHOA CÔNG NGHỆ THÔNG TIN]
  #v(1.2cm)
  #image("../images/logo.png", width: 32%)
  #v(1.2cm)
  #text(24pt, weight: "bold", fill: rgb("#1E3778"))[BÁO CÁO ĐỒ ÁN 1]
  #v(0.4cm)
  #text(22pt, weight: "bold")[Ma Trận và Cơ Sở của Tính Toán Khoa Học]
  #v(0.4cm)
  #align(center)[
    #set text(15pt)
    #grid(
      columns: (auto, auto),
      column-gutter: 0.3cm,
      row-gutter: 0.2cm,
      align: (left, left),
      [*Môn học:*], [Toán Ứng Dụng và Thống Kê],
      [*Lớp:*], [24CTT3],
    )
  ]
  #v(1.5cm)
]

#align(center)[
  #block(width: 86%)[
    #line(length: 100%, stroke: 0.6pt + luma(160))
    #v(0.5cm)

    #grid(
      columns: (1.25fr, 1fr),
      column-gutter: 1.2cm,

      [
        #align(left)[
          #info-title[Nhóm 12 (CQ2024-3)]
          #v(0.15cm)
          #table(
            columns: (1fr, auto),
            stroke: (x, y) => if x == 1 { (left: 0.8pt + rgb("#1E3778")) } else { none },
            align: (left, right),
            inset: (col, row) => (
              left: if col == 0 { 0pt } else { 8pt },
              right: 8pt,
              y: 4.5pt,
            ),
            [Đinh Đức Hiếu], [24120002],
            [Liên Trung Hiếu], [24120049],
            [Trương Đình Nhật Huy], [24120064],
            [#box[Nguyễn Ngọc Quang]], [24120127],
            [Đặng Quang Tiến], [24120149]
          )
        ]
      ],

      [
        #align(left)[
          #info-title[Giảng viên hướng dẫn]
          #v(0.15cm)
          #table(
            columns: (auto,),
            stroke: none,
            inset: (left: 0pt, right: 0pt, y: 4.5pt),
            [ThS. Võ Nam Thục Đoan],
            [ThS. Lê Nhựt Nam],
          )
        ]
      ],
    )
  ]
]

#place(bottom + center)[
  #text(11pt)[Thành phố Hồ Chí Minh, ngày 20 tháng 4 năm 2026]
  #v(0.5cm)
]
