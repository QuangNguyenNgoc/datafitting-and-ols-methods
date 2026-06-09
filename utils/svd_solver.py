"""
SVD Solver Utilities
====================
Cac tien ich tinh toan dua tren Singular Value Decomposition (SVD).

Dung svd_decomp tu utils/decomposition.py
de giai he OLS va tinh ma tran Hat H = U_r U_r^T.

Numpy chi dung de:
- Nhan dau vao dang array
- Kiem chung ket qua (trong notebook)
"""

from __future__ import annotations

import numpy as np

# Import svd_decomp thuan Python tu utils/decomposition.py
from utils.decomposition import svd_decomp

# Nguong bo qua gia tri ky di nho (dong bo voi _EPS trong diagonalization.py)
_EPS = 1e-9


def economic_svd(X):
    """
    Goi svd_decomp cho ma tran X.

    Tra ve:
        U     : list of lists (m x m), cac vector don vi trai
        s     : list cac gia tri ky di (giam dan), chieu dai min(m,n)
        V_T   : list of lists (n x n), V^T
    """
    X_list = [[float(v) for v in row] for row in X]
    U, Sigma, V_T = svd_decomp(X_list)
    k = min(len(Sigma), len(Sigma[0]))
    s = [Sigma[i][i] for i in range(k)]
    return U, s, V_T


def svd_solve(X, y):
    """
    Giai bai toan OLS bang Economic SVD.

    Cong thuc: beta = V Sigma^{+} U^T y
    Chi giu cac thanh phan voi sigma_i > _EPS.

    Tham so:
        X: ma tran thiet ke (m x n), list hoac numpy array
        y: vector muc tieu (m,)

    Tra ve:
        beta: numpy array (n,)
    """
    X_list = [[float(v) for v in row] for row in X]
    y_list = [float(v) for v in y]

    U, s, V_T = economic_svd(X_list)

    m = len(U)        # so hang cua X
    n = len(V_T)      # so cot cua X (so tham so)
    r = len(s)        # so gia tri ky di

    # Buoc 1: U^T y  =>  vector kich thuoc r
    # (U^T y)_i = sum_j U[j][i] * y[j]   (cot i cua U dot y)
    Ut_y = [
        sum(U[j][i] * y_list[j] for j in range(m))
        for i in range(r)
    ]

    # Buoc 2: Sigma^{+} (U^T y)  =>  chia cho sigma_i neu sigma_i > _EPS
    SigmaPlus_Ut_y = [
        Ut_y[i] / s[i] if s[i] > _EPS else 0.0
        for i in range(r)
    ]

    # Buoc 3: beta = V (Sigma^{+} U^T y)
    # V[j][i] = V_T[i][j]  =>  beta[j] = sum_i V_T[i][j] * SigmaPlus_Ut_y[i]
    beta = [
        sum(V_T[i][j] * SigmaPlus_Ut_y[i] for i in range(r))
        for j in range(n)
    ]

    return np.array(beta, dtype=float)


def hat_diagonal(X):
    """
    Tinh duong cheo h_ii cua Hat matrix H = U_r U_r^T bang SVD: h_ii = sum_{k: s_k > EPS} U[i][k]^2

    Tham so:
        X: ma tran thiet ke (m x n)

    Tra ve:
        h: numpy array (m,)
    """
    U, s, _ = economic_svd(X)
    m = len(U)
    h = [
        sum(U[i][k] ** 2 for k in range(len(s)) if s[k] > _EPS)
        for i in range(m)
    ]
    return np.array(h, dtype=float)
