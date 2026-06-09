"""
Ridge Regression
=================
Cài đặt Ridge Regression và vẽ Ridge Trace.
"""

import matplotlib.pyplot as plt

from utils.matrix_utils import (
    mat_transpose,
    mat_mul,
    matrix_vector_multiply,
    _identity,
)
from utils.inverse import inverse


def ridge_fit(X, y, lam):
    """
    6. Cài đặt Ridge Regression từ scratch.

    Công thức toán học:
        beta_ridge = (X^T X + lam * I)^{-1} X^T y

    Ma trận (X^T X + lam * I) luôn khả nghịch khi lam > 0.
    Cài đặt thuần Python dùng utils/inverse.py (Gauss-Jordan).

    Tham số:
        X   : list of lists (n x p)
        y   : list (n,)
        lam : float, tham số điều chuẩn lambda >= 0

    Trả về:
        beta_ridge: list (p,)
    """
    X_list = [[float(v) for v in row] for row in X]
    y_list = [float(v) for v in y]
    p = len(X_list[0])

    X_T = mat_transpose(X_list)
    X_T_X = mat_mul(X_T, X_list)

    I_p = _identity(p)
    A = [
        [X_T_X[i][j] + float(lam) * I_p[i][j] for j in range(p)]
        for i in range(p)
    ]

    A_inv = inverse(A)
    X_T_y = matrix_vector_multiply(X_T, y_list)
    beta = matrix_vector_multiply(A_inv, X_T_y)

    return beta


if __name__ == "__main__":
    # TODO: Khởi tạo dữ liệu giả lập có hiện tượng đa cộng tuyến
    # TODO: Gọi hàm ridge_fit để minh họa
    # TODO: Kiểm chứng kết quả với sklearn.linear_model.Ridge
    print("Ridge Regression - Demo")
