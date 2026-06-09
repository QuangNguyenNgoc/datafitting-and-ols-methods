import math
import pandas as pd


def _student_t_sf(t_val: float, df: int) -> float:
    """tính đuôi bên phải cho phân phối Student bằng xấp xỉ đa thức"""
    if df <= 0:
        return 0.0
    x = abs(t_val)
    # hiệu chỉnh của Wallace cho phân phối Student
    z = x * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + (x * x) / (2.0 * df))

    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def _student_t_ppf(df: int) -> float:
    """tìm giá trị t tới hạn cho khoảng tin cậy 95%"""
    if df <= 0:
        return 0.0
    z = 1.95996  # tương ứng với 95%
    # hiệu chỉnh Cornish-Fisher cho phân phối Student
    t_crit = (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
    )
    return t_crit


def _f_sf_paulson(f_stat: float, df1: int, df2: int) -> float:
    """
    Tính p-value cho phân phối Fisher-F bằng phép xấp xỉ chuẩn Paulson.
    """
    if f_stat <= 0.0:
        return 1.0

    cube_root_f = f_stat ** (1.0 / 3.0)

    # Hệ số hiệu chỉnh theo bậc tự do của tử số (df1) và mẫu số (df2)
    term1 = 1.0 - 2.0 / (9.0 * df1)
    term2 = 1.0 - 2.0 / (9.0 * df2)

    # Chuyển đổi F thành Z-score (Phân phối chuẩn tắc N(0,1))
    numerator = term2 * cube_root_f - term1
    denominator = math.sqrt(
        (2.0 / (9.0 * df1)) + (2.0 / (9.0 * df2)) * (cube_root_f**2)
    )

    z = numerator / denominator

    # Tính đuôi bên phải của Z-score bằng hàm sai số
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
