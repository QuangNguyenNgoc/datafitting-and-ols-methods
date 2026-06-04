import math
import pandas as pd


def _student_t_sf(t_val: float, df: int) -> float:
    """tính đuôi bên phải cho phân phối Student bằng xấp xỉ đa thức"""
    x = t_val
    # hiệu chỉnh của Wallace cho phân phối Student
    z = x * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + (x * x) / (2.0 * df))

    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def _student_t_ppf(prob: float, df: int) -> float:
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
