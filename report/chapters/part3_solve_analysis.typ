#import "../theme.typ": *
= Giải hệ phương trình và phân tích hiệu năng <sec:benchmark>

== Mục tiêu

Tập trung vào việc nối từ cơ sở lý thuyết của đại số tuyến tính số đến hành vi thực tế của thuật toán khi chạy trên máy tính. Mục tiêu là đánh giá và so sánh ba phương pháp giải hệ phương trình tuyến tính gồm _khử Gauss có `partial pivoting`_, _phân rã `LU` có `partial pivoting`_, và _phương pháp lặp `Gauss-Seidel`_ trên ba phương diện chính:

- _Chi phí tính toán_: khảo sát thời gian thực thi khi kích thước ma trận tăng dần, từ đó đối chiếu với độ phức tạp lý thuyết của từng phương pháp.
- _Độ chính xác của nghiệm_: đo mức độ sai lệch của nghiệm tính được thông qua _residual error_ và _solution error_.
- _Độ ổn định số học_: quan sát phản ứng của thuật toán trên các lớp ma trận có bản chất rất khác nhau, đặc biệt là ma trận điều kiện tốt và điều kiện kém.

Thông qua phần này, đồ án hướng tới trả lời một câu hỏi thực tiễn hơn là chỉ mô tả thuật toán: _khi áp dụng vào dữ liệu cụ thể, phương pháp nào phù hợp hơn, nghiệm nào đáng tin cậy hơn, và chi phí tính toán thay đổi như thế nào khi kích thước bài toán tăng lên_.

== Thiết kế thực nghiệm và tiêu chí đánh giá

Để việc so sánh phản ánh đúng bản chất của từng phương pháp, hai nhóm thực nghiệm bổ sung cho nhau đã được xây dựng.

=== Nhóm thực nghiệm A: đo thời gian thực thi và sai số theo kích thước

Ở nhóm thực nghiệm thứ nhất, các ma trận vuông _chéo trội nghiêm ngặt_ được sinh ra với kích thước tăng dần
$ n in {50, 100, 200, 500, 1000}. $
Việc chọn ma trận chéo trội nhằm bảo đảm phương pháp `Gauss-Seidel` có điều kiện hội tụ rõ ràng, đồng thời tạo ra một điều kiện thực nghiệm tương đồng để so sánh với hai phương pháp trực tiếp là Gauss và `LU`.

Với mỗi kích thước $n$, nghiệm chuẩn được chọn là
$ x_"true" = (1,1,dots,1)^T $
và dựng vế phải theo công thức
$ b = A x_"true" $
Cách làm này cho phép kiểm tra trực tiếp chất lượng nghiệm trả về, vì nghiệm đúng đã được biết trước.

Mỗi thuật toán được chạy _5 lần_ trên cùng một kích thước để lấy _thời gian thực thi trung bình_. Cách lặp lại nhiều lần giúp giảm ảnh hưởng của dao động môi trường thực thi và làm cho số liệu benchmark ổn định hơn.

=== Nhóm thực nghiệm B: phân tích ổn định số học

Ở nhóm thực nghiệm thứ hai, mục tiêu không chỉ là tốc độ mà còn kiểm thử độ nhạy của nghiệm trước điều kiện của ma trận. Thực nghiệm sử dụng hai lớp dữ liệu:

- _Ma trận `Hilbert`_ với các kích thước
  $ n in {3, 5, 8, 10, 12, 15}, $
  đại diện cho lớp ma trận _điều kiện kém_ (_ill-conditioned_), trong đó số điều kiện tăng rất nhanh theo kích thước.
- _Ma trận `SPD` ngẫu nhiên_ với các kích thước
  $ n in {5, 10, 20, 50, 100}, $
  đóng vai trò nhóm đối chứng có điều kiện tốt hơn, giúp làm rõ sự khác biệt về ổn định số học.

Với mỗi ma trận, quy trình tiếp tục sử dụng nghiệm chuẩn `x_true` để dựng vế phải `b = A x_true`, sau đó giải lại bằng ba phương pháp và so sánh sai số.

=== Cài đặt và nguyên tắc thực thi

Các solver trong @sec:benchmark được cài đặt bằng _nested list thuần `Python`_, không sử dụng `NumPy` array trong phần lõi thuật toán. `NumPy` chỉ được dùng trong các bước sinh dữ liệu, tính chuẩn, tính số điều kiện và đối chiếu kết quả sau cùng. Điều này giúp giữ đúng tinh thần của đồ án: thư viện chỉ phục vụ _`benchmark` và verification_, không thay thế cho phần cài đặt thuật toán.

Trong ba solver được so sánh, `LU` được dùng như một phương pháp phân rã phục vụ bài toán giải hệ trong `benchmark`; mục tiêu của @sec:benchmark là đánh giá hiệu năng của các solver tiêu biểu, chứ không lặp lại toàn bộ phạm vi lý thuyết của @sec:decomposition.

Để đánh giá toàn diện, bốn chỉ số chính được sử dụng:

- _Average Runtime_: thời gian thực thi trung bình sau nhiều lần chạy.
- _Relative Residual Error_:
  $ (||A hat(x) - b||_2)/(||b||_2), $
 phản ánh mức độ thỏa mãn của nghiệm tính được đối với hệ phương trình ban đầu.
- _Solution Error_:
  $ (||hat(x) - x_"true"||_2)/(||x_"true"||_2), $
 phản ánh sai lệch thực sự của nghiệm tính toán so với nghiệm chuẩn.
- _Condition Number_:
  $ kappa(A), $
  dùng để giải thích mối quan hệ giữa cấu trúc ma trận và mức độ khuếch đại sai số.

Toàn bộ kết quả benchmark được lưu dưới dạng `.csv`, sau đó sử dụng notebook phân tích để dựng bảng tổng hợp và biểu đồ trực quan. Các bảng số liệu đầy đủ được đưa vào phụ lục, xem @tbl:runtime-full, @tbl:hilbert-full và @tbl:spd-full.
== Kết quả và thảo luận

Từ các số liệu benchmark thu được, có thể nhận thấy sự khác biệt rõ ràng giữa ba phương pháp cả về thời gian thực thi lẫn độ ổn định số học. Các quan sát dưới đây được rút ra từ dữ liệu tổng hợp và các biểu đồ phân tích đi kèm.

=== Thời gian thực thi và chi phí tính toán

Kết quả thực nghiệm A cho thấy hai phương pháp trực tiếp là _khử Gauss_ và `LU` đều có xu hướng thời gian thực thi tăng đáng kể khi kích thước ma trận tăng lên. Điều này phù hợp với phân tích lý thuyết rằng chi phí của hai phương pháp này tăng theo bậc xấp xỉ
$cal(O)(n^3). $

Ở các kích thước nhỏ, chênh lệch giữa ba phương pháp chưa rõ rệt. Tuy nhiên, khi $n$ tăng lên các mức lớn hơn như 500 hoặc 1000, thời gian của Gauss và `LU` tăng mạnh và trở thành yếu tố chi phối.
Mặt khác, dựa theo bộ dữ liệu, `Gauss-Seidel` không cho thời gian thực thi thấp hơn Gauss và `LU`. Dù mỗi vòng lặp chỉ có chi phí cỡ
$cal(O)(n^2). $

tổng thời gian của phương pháp lặp vẫn phụ thuộc mạnh vào số vòng lặp cần để hội tụ và vào chi phí của cài đặt list thuần. Điều này cho thấy lợi thế lý thuyết của phương pháp lặp không tự động chuyển thành lợi thế thời gian trong mọi môi trường chạy.

Kết quả trên cho ta biết rằng, khi đánh giá chi phí tính toán thì ta phải nhìn đồng thời vào cả độ phức tạp lý thuyết và chi phí thực nghiệm.

#figure(
  image("../images/loglog_timing.png", width: 80%),
  caption: [Đồ thị log-log thời gian thực thi theo kích thước $n$]
) <fig:timing>
Các giá trị thời gian thực thi chi tiết theo từng kích thước được tổng hợp trong @tbl:runtime-full.

=== Sai số trên dữ liệu chéo trội
Trên nhóm ma trận chéo trội dùng trong thực nghiệm A, cả ba phương pháp đều cho _Relative residual_ nhỏ. Điều này cho thấy các nghiệm tính được vẫn thỏa mãn khá tốt hệ phương trình ban đầu.

Tuy nhiên, việc residual nhỏ không đồng nghĩa với việc mọi phương pháp đều giống nhau về mặt hiệu năng. Trong bộ dữ liệu này, Gauss và `LU` vừa giữ residual rất thấp, vừa có thời gian thực thi tốt hơn `Gauss-Seidel` trong `benchmark` hiện tại. Vì vậy, dữ liệu chéo trội ở đây phù hợp để minh họa rằng việc đánh giá thuật toán không nên dựa trên một chỉ số duy nhất.

#figure(
  image("../images/error_vs_n.png", width: 80%),
  caption: [Sai số tương đối theo kích thước $n$ trên ma trận chéo trội]
)

=== Ảnh hưởng của số điều kiện: trường hợp ma trận Hilbert và SPD

Với _ma trận `Hilbert`_, khi kích thước tăng thì số điều kiện tăng rất nhanh, khiến bài toán trở nên cực kỳ nhạy với sai số làm tròn. Kết quả thực nghiệm cho thấy _solution error_ của `Gauss` và `LU` tăng rõ rệt theo `kappa(A)`, dù trong nhiều trường hợp _Relative residual_ vẫn còn nhỏ. Điều này phản ánh đúng bản chất của bài toán điều kiện kém: một nghiệm có thể gần thỏa mãn phương trình `A x = b`, nhưng vẫn sai khác đáng kể so với nghiệm thực `x_true`. Nói cách khác, residual nhỏ không đồng nghĩa với nghiệm đáng tin cậy khi `kappa(A)` lớn.

Với _ma trận `SPD` ngẫu nhiên_, `Gauss` và `LU` duy trì sai số rất thấp trên toàn bộ dải `condition number` được khảo sát. Trong khi đó, _`Gauss-Seidel` nhạy hơn rõ rệt_: ở các mức `condition number` lớn hơn trong bộ dữ liệu hiện tại, sai số của phương pháp này tăng đáng kể so với hai phương pháp trực tiếp. Điều đó cho thấy việc một phương pháp có cơ sở hội tụ về lý thuyết không đồng nghĩa với việc nó ổn định như nhau trên mọi bộ dữ liệu hữu hạn.

Đối với _`Gauss-Seidel`_ trên `Hilbert`, ma trận vẫn là đối xứng xác định dương nên về mặt lý thuyết phương pháp có cơ sở hội tụ. Tuy nhiên, thực nghiệm cho thấy tốc độ hội tụ và độ tin cậy của nghiệm suy giảm rõ rệt khi kích thước tăng. Trong các trường hợp khó, cần thực hiện phương pháp với nhiều vòng lặp hoặc dừng ở trạng thái mà sai số nghiệm vẫn còn lớn.

#figure(
  image("../images/stability_comparison.png", width: 90%),
  caption: [Ảnh hưởng của số điều kiện $kappa(A)$ đến sai số nghiệm trên hai lớp dữ liệu: Hilbert và SPD]
)

=== So sánh Hilbert và SPD: ý nghĩa về ổn định số học

Khi đặt hai nhóm ma trận cạnh nhau, một nhận xét quan trọng có thể rút ra: _độ ổn định của thuật toán không thể tách rời khỏi bản chất của dữ liệu đầu vào_.

Trên ma trận `SPD` ngẫu nhiên, các phương pháp trực tiếp cho kết quả rất ổn định và sai số nhỏ. Trên ma trận `Hilbert`, cùng một thuật toán vẫn có thể cho sai số nghiệm tăng mạnh chỉ vì số điều kiện lớn hơn nhiều. Điều này cho thấy khi đánh giá chất lượng nghiệm trong thực tế, không nên chỉ quan sát một chỉ số duy nhất. Việc theo dõi đồng thời `condition number`, `Relative residual` và `solution error` là cần thiết để tránh kết luận sai rằng bài toán đã được giải tốt chỉ vì residual nhỏ.


#figure(
  image("../images/hilbert_vs_spd_bar.png", width: 82%),
  caption: [So sánh sai số nghiệm giữa Hilbert và SPD đối với phương pháp Gauss]
)

== Kết luận của phần thực nghiệm

Từ toàn bộ benchmark, có thể nhận thấy không tồn tại một phương pháp tối ưu cho mọi loại dữ liệu.

- _Khử Gauss_ và `LU` phù hợp khi cần một lời giải trực tiếp, ổn định và dễ kiểm soát trên các ma trận khả nghịch thông thường.
- `LU` đặc biệt hữu ích nếu cùng một ma trận $A$ được dùng để giải nhiều hệ với các vector $b$ khác nhau.
- `Gauss-Seidel` cho thấy vai trò của phương pháp lặp trong việc khai thác cấu trúc của dữ liệu, nhưng hiệu quả thực tế còn phụ thuộc mạnh vào điều kiện hội tụ, số vòng lặp và chất lượng hiện thực thuật toán.

Quan trọng hơn, kết quả thực nghiệm khẳng định rằng việc lựa chọn thuật toán trong tính toán khoa học không thể chỉ dựa vào công thức lý thuyết, cần xét đồng thời đến _cấu trúc ma trận_, _độ ổn định số học_ và _chi phí tính toán quan sát được từ thực nghiệm_. Kết nối các khái niệm như số điều kiện, sai số dư và ổn định số với thực tiễn tính toán thay vì chỉ dừng lại ở lý thuyết trừu tượng.
