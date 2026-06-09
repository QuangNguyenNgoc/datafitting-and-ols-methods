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


def ridge_fit(X, y, lam):
    """
    6. Tinh he so Ridge Regression.

    Cong thuc toan hoc:
        beta_ridge = (X^T X + lam * I)^{-1} X^T y

    Ma tran (X^T X + lam * I) luon kha nghich khi lam > 0.

    Tham so:
        X   : list of lists (n x p)
        y   : list (n,)
        lam : float, tham so dieu chuan lambda >= 0

    Tra ve:
        beta_ridge: np.ndarray (p,)
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

    return np.array(beta).flatten()


if __name__ == "__main__":
    # TODO: Khoi tao du lieu gia lap co hien tuong da cong tuyen
    # TODO: Goi ham ridge_fit de minh hoa
    # TODO: Kiem chung ket qua voi sklearn.linear_model.Ridge
    print("Ridge Regression - Demo")
