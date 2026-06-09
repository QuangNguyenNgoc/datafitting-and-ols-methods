"""
Ridge Regression
=================
Cai dat Ridge Regression va ve Ridge Trace.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.matrix_utils import (
    mat_transpose,
    mat_mul,
    matrix_vector_multiply,
    _identity,
)
from utils.inverse import inverse


def ridge_fit(X: list[list[float]], y: list[float], lam: float) -> list[float]:
    """
    6. Cài đặt Ridge Regression.

    Công thức toán học:
        beta_ridge = (X^T X + lam * I)^{-1} X^T y

    Tham số:
        X : ma trận thiết kế (m x p), KHÔNG có cột hệ số chặn
        y: vector mục tiêu (m,)
        lam : tham số điều chuẩn lambda >= 0

    Trả về:
        beta_ridge: vector hệ số (list thuần Python)
    """
    X_list = [[float(v) for v in row] for row in X]
    y_list = [float(v) for v in y]

    p = len(X_list[0])

    X_T = mat_transpose(X_list)
    X_T_X = mat_mul(X_T, X_list)

    # ma trận A = X^T X + lam * I
    A = [
        [X_T_X[i][j] + (float(lam) if i == j else 0.0) for j in range(p)]
        for i in range(p)
    ]

    A_inv = inverse(A)
    X_T_y = matrix_vector_multiply(X_T, y_list)
    beta = matrix_vector_multiply(A_inv, X_T_y)

    return beta


def compute_ridge_trace(
    X: list[list[float]], y: list[float], lambdas: list[float]
) -> list[list[float]]:
    """
    Tính toán tập hợp các vector beta cho một mảng các giá trị lambda
    để phục vụ việc vẽ biểu đồ Ridge Trace.

    Trả về:
        Ma trận kích thước (len(lambdas) x p), mỗi hàng là một vector beta.
    """
    trace_path = []
    for lam in lambdas:
        beta_lam = ridge_fit(X, y, lam)
        trace_path.append(beta_lam)

    return trace_path
