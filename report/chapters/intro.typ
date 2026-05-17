#import "../theme.typ": *
= Giới thiệu đồ án

== Mở đầu

Đồ án môn học này được xây dựng với mục tiêu giúp sinh viên không chỉ nắm vững các định lý toán học thuần túy, mà còn hiểu rõ cách những công thức đó vận hành bên trong kiến trúc máy tính.

Nội dung trọng tâm của đồ án được cấu trúc thành ba trục kiến thức chính:

_Phần 1: Phép khử Gauss và ứng dụng._ Tập trung nghiên cứu phép khử `Gauss` và chiến lược chọn phần tử chốt (`Partial Pivoting`). Đây là nền tảng quan trọng để giải quyết các bài toán kinh điển như giải hệ phương trình tuyến tính, tính định thức và tìm ma trận nghịch đảo.

_Phần 2: Phân rã ma trận và trực quan hóa với Manim._ Đi sâu vào cấu trúc nội tại của ma trận thông qua kỹ thuật chéo hóa và phân rã kỳ dị (`SVD`). Phân tích ma trận thành các phép biến đổi không gian (xoay, giãn, nén). Đồng thời, tích hợp thư viện `Manim` để xây dựng các hoạt cảnh trực quan, minh họa những khái niệm trừu tượng.

_Phần 3: Giải hệ phương trình và phân tích hiệu năng._ Thực hiện các bài đo đạc (`benchmark`) để so sánh các giải thuật trực tiếp (`Gauss`, `LU`) với phương pháp lặp (`Gauss-Seidel`). Thông qua việc khảo sát trên các lớp ma trận đặc biệt như `SPD` hay ma trận `Hilbert` (điều kiện kém), làm rõ mối quan hệ giữa số điều kiện (`Condition Number`) và độ tin cậy của nghiệm số.

Toàn bộ đồ án được thực hiện trên ngôn ngữ `Python`. Tự cài đặt lõi thuật toán, sau đó sử dụng các thư viện chuẩn như `NumPy`, `SciPy` để đối chiếu và kiểm chứng.

== Cấu trúc thư mục

Để đảm bảo tính module và dễ quản lý, mở rộng và bảo trì, mã nguồn của đồ án được tổ chức theo cấu trúc phân cấp như sau:

#codeblock((
  "Group_12/",
  "├── part1/           # Thuật toán khử Gauss và ứng dụng",
  "├── part2/           # Chéo hóa và phân rã ma trận SVD",
  "├── part3/           # Giải hệ phương trình và Benchmark",
  "├── report/          # Mã nguồn Typst và tài liệu báo cáo",
  "└── requirements.txt ",
  "└── README.md        ",
))
