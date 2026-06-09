"""
OLS Implementation and Inference
================================
Cài đặt các hàm OLS cơ bản, tính ma trận chiếu, metrics, suy diễn thống kê,
tính VIF và minh họa định lý Gauss-Markov.
"""

import math
import random

import numpy as np
import pandas as pd

from utils.matrix_utils import (
    mat_transpose,
    mat_mul,
    vector_subtract,
    vector_dot_product,
    matrix_vector_multiply,
)
from utils.inverse import inverse
from utils.decomposition import svd_decomp

from part1.utils_verif import _student_t_sf, _student_t_ppf

_EPS = 1e-9


def ols_fit(X, y):
    """
    1. Tinh vector he so beta_hat bang phuong phap OLS.

    Cong thuc toan hoc:
        beta_hat = (X^T X)^{-1} X^T y

    Phan ra SVD: X = U Sigma V^T  =>  beta_hat = V Sigma^{-1} U^T y

    Tham so:
        X: list of lists (n x p)
        y: list (n,)

    Tra ve:
        beta: np.ndarray (p,)
    """
    X_list = [[float(v) for v in row] for row in X]
    y_list = [float(v) for v in y]

    U, Sigma, V_T = svd_decomp(X_list)
    k = min(len(Sigma), len(Sigma[0]))
    s = [Sigma[i][i] for i in range(k)]

    m = len(U)
    n = len(V_T)
    r = len(s)

    # Buoc 1: U^T y
    Ut_y = [sum(U[j][i] * y_list[j] for j in range(m)) for i in range(r)]

    # Buoc 2: Sigma^+ (U^T y) — chi chia cho sigma_i > EPS
    sp_Uty = [Ut_y[i] / s[i] if s[i] > _EPS else 0.0 for i in range(r)]

    # Buoc 3: beta = V (Sigma^+ U^T y)
    beta = [sum(V_T[i][j] * sp_Uty[i] for i in range(r)) for j in range(n)]

    return np.array(beta)


def hat_matrix(X):
    """
    2. Tinh ma tran chieu H (Hat matrix).

    Cong thuc toan hoc:
        H = X (X^T X)^{-1} X^T

    Qua phan ra SVD: X = U Sigma V^T  =>  H = U_r U_r^T
    (U_r la cac cot ung voi sigma_i > EPS)

    Tinh chat: H^2 = H, H^T = H, tr(H) = rank(X).

    Tham so:
        X: list of lists (n x p)

    Tra ve:
        H: np.ndarray (n x n)
    """
    X_list = [[float(v) for v in row] for row in X]

    U, Sigma, _ = svd_decomp(X_list)
    k = min(len(Sigma), len(Sigma[0]))
    s = [Sigma[i][i] for i in range(k)]
    m = len(U)

    # H[a][b] = sum_{col: s[col]>EPS} U[a][col] * U[b][col]
    H = [[0.0] * m for _ in range(m)]
    for col in range(k):
        if s[col] > _EPS:
            for a in range(m):
                for b in range(m):
                    H[a][b] += U[a][col] * U[b][col]

    return np.array(H)


def model_metrics(y: list, y_hat: list, p: int) -> dict:
    """
    Tinh cac do do tong hop mo hinh bang 100% Python goc:
    - RSS (Residual Sum of Squares)
    - TSS (Total Sum of Squares)
    - R^2 (He so xac dinh)
    - R^2 hieu chinh (Adjusted R^2)
    - F-statistic
    """
    y_list = list(y)
    y_hat_list = list(y_hat)
    n = len(y_list)

    # RSS = e^T * e
    e = vector_subtract(y_list, y_hat_list)
    rss = vector_dot_product(e, e)

    # TSS = y_c^T * y_c
    y_mean = sum(y_list) / n if n > 0 else 0.0
    y_centered = [y_i - y_mean for y_i in y_list]
    tss = vector_dot_product(y_centered, y_centered)

    # R^2
    r2 = 1.0 - (rss / tss) if tss != 0.0 else 0.0

    # Adjusted R^2, F-statistic
    if n - p - 1 > 0:
        adj_r2 = 1.0 - ((rss / (n - p - 1)) / (tss / (n - 1)))

        if rss > 0.0:
            f_statistic = ((tss - rss) / p) / (rss / (n - p - 1))
        else:
            f_statistic = float("inf")
    else:
        adj_r2 = float("nan")
        f_statistic = float("nan")

    return {
        "RSS": rss,
        "TSS": tss,
        "R2": r2,
        "Adj_R2": adj_r2,
        "F_statistic": f_statistic,
    }


def coef_inference(X: list, y: list, beta_hat: list, sigma2: float) -> pd.DataFrame:
    """
    Suy dien thong ke cho cac he so:
    - Standard errors (sai so chuan cua tung he so)
    - t-statistics (gia tri t)
    - p-values
    - Khoang tin cay (Confidence Intervals) 95%
    """
    X_mat = list(X)
    beta_list = list(beta_hat)

    n = len(X_mat)
    k = len(X_mat[0])
    df = n - k

    # ma tran nghich dao
    X_T = mat_transpose(X_mat)
    X_T_X = mat_mul(X_T, X_mat)
    X_T_X_inv = inverse(X_T_X)

    # standard errors
    se = []
    for i in range(k):
        variance_beta_i = sigma2 * X_T_X_inv[i][i]
        se.append(math.sqrt(max(variance_beta_i, 0.0)))

    # t-statistics, p-values
    t_stats = [beta_list[i] / se[i] if se[i] > 0 else 0.0 for i in range(k)]
    p_values = [2.0 * _student_t_sf(abs(t_stats[i]), df) for i in range(k)]

    # khoang tin cay 95%
    t_critical = _student_t_ppf(0.975, df)
    ci_lower = [beta_list[i] - t_critical * se[i] for i in range(k)]
    ci_upper = [beta_list[i] + t_critical * se[i] for i in range(k)]

    inference_df = pd.DataFrame(
        {
            "Coefficient": beta_list,
            "Std_Error": se,
            "t_stat": t_stats,
            "p_value": p_values,
            "CI_95_Lower": ci_lower,
            "CI_95_Upper": ci_upper,
        }
    )

    return inference_df


def vif(X):
    """
    5. Tinh Variance Inflation Factor (VIF) cho tung bien doc lap
       de kiem tra hien tuong da cong tuyen.

    Cong thuc: VIF_j = 1 / (1 - R^2_j)

    R^2_j la he so xac dinh khi hoi quy bien thu j len tat ca
    cac bien con lai (co he so chan).

    Tham so:
        X: list of lists (n x p), KHONG bao gom cot he so chan

    Tra ve:
        DataFrame voi cot Feature va VIF_Score
    """
    X_df = pd.DataFrame(X)
    feature_names = X_df.columns
    X_list = [[float(v) for v in row] for row in X]
    n = len(X_list)
    p = len(X_list[0])

    vif_values = []
    for j in range(p):
        y_j = [X_list[i][j] for i in range(n)]

        other_cols = [c for c in range(p) if c != j]
        X_j = [[1.0] + [X_list[i][c] for c in other_cols] for i in range(n)]

        beta_j = ols_fit(X_j, y_j)
        y_hat_j = [
            sum(X_j[i][c] * beta_j[c] for c in range(len(beta_j)))
            for i in range(n)
        ]

        y_mean_j = sum(y_j) / n
        ss_res = sum((y_j[i] - y_hat_j[i]) ** 2 for i in range(n))
        ss_tot = sum((y_j[i] - y_mean_j) ** 2 for i in range(n))

        r2_j = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        vif_j = 1.0 / (1.0 - r2_j) if (1.0 - r2_j) > 1e-12 else float("inf")
        vif_values.append(vif_j)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF_Score"] = vif_values
    return vif_data


def gauss_markov_simulation(n_simulations=1000, n_samples=100):
    """
    Minh hoa dinh ly Gauss-Markov bang Monte Carlo:
    - Tinh khong chech: E[beta_hat] = beta
    - Tinh BLUE: Var(beta_OLS) < Var(beta_Alternative)
    """
    # khoi tao ma tran X co dinh va beta thuc
    # X co 1 cot bias va 2 cot dac trung ngau nhien
    random.seed(42)
    p = 2
    X = [[1.0, random.uniform(0, 10), random.uniform(-5, 5)] for _ in range(n_samples)]
    true_beta = [3.0, 1.5, -2.0]
    k = len(true_beta)

    # tien tinh toan ma tran anh xa cho OLS (100% data)
    X_T = mat_transpose(X)
    X_T_X_inv = inverse(mat_mul(X_T, X))
    # C_ols = (X^T X)^-1 X^T
    C_ols = mat_mul(X_T_X_inv, X_T)

    # tien tinh toan ma tran anh xa cho Alternative Estimator (60% data)
    n_alt = int(n_samples * 0.6)
    X_alt = X[:n_alt]
    X_alt_T = mat_transpose(X_alt)
    X_alt_T_X_alt_inv = inverse(mat_mul(X_alt_T, X_alt))
    C_alt = mat_mul(X_alt_T_X_alt_inv, X_alt_T)

    # luu tru ket qua cua 1000 vong lap
    beta_ols_results = {j: [] for j in range(k)}
    beta_alt_results = {j: [] for j in range(k)}

    # VONG LAP MONTE CARLO
    for _ in range(n_simulations):
        # tao nhieu e ~ N(0, sigma^2)
        sigma = 2.0
        e = [random.gauss(0, sigma) for _ in range(n_samples)]

        # sinh bien muc tieu y = X*beta + e
        y = [
            sum(X[i][j] * true_beta[j] for j in range(k)) + e[i]
            for i in range(n_samples)
        ]

        beta_ols = matrix_vector_multiply(C_ols, y)
        beta_alt = matrix_vector_multiply(C_alt, y[:n_alt])

        # Luu ket qua
        for j in range(k):
            beta_ols_results[j].append(beta_ols[j])
            beta_alt_results[j].append(beta_alt[j])

    # thong ke
    report = []
    for j in range(k):
        # E[beta_hat]
        e_ols = sum(beta_ols_results[j]) / n_simulations
        e_alt = sum(beta_alt_results[j]) / n_simulations

        # Var(beta_hat)
        var_ols = sum((b - e_ols) ** 2 for b in beta_ols_results[j]) / (
            n_simulations - 1
        )
        var_alt = sum((b - e_alt) ** 2 for b in beta_alt_results[j]) / (
            n_simulations - 1
        )

        report.append(
            {
                "beta_idx": j,
                "true_val": true_beta[j],
                "E_ols": e_ols,
                "E_alt": e_alt,
                "Var_ols": var_ols,
                "Var_alt": var_alt,
            }
        )

    return report
