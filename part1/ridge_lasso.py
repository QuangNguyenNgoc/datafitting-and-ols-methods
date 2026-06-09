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
    6. Cai dat Ridge Regression tu scratch.

    Cong thuc toan hoc:
        beta_ridge = (X^T X + lam * I)^{-1} X^T y

    Tham so:
        X   : ma tran thiet ke (m x p), KHONG co cot he so chan
        y   : vector muc tieu (m,)
        lam : tham so dieu chuan lambda >= 0

    Tra ve:
        beta_ridge: numpy array (p,)
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float).flatten()

    X_list = [list(row) for row in X_arr]
    y_list = list(y_arr)
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
    return np.array(beta, dtype=float)


if __name__ == "__main__":
    print("Ridge Regression - Demo")
