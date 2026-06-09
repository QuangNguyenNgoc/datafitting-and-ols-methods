"""
OLS Implementation and Inference
================================
Cai dat cac ham OLS co ban, tinh ma tran chieu, metrics, suy dien thong ke,
tinh VIF va minh hoa dinh ly Gauss-Markov.
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
from utils.svd_solver import svd_solve, economic_svd

from part1.utils_verif import _student_t_sf, _student_t_ppf


def ols_fit(X, y):
    """
    1. Tinh vector he so beta_hat bang phuong phap OLS.

    Cong thuc toan hoc:
        beta_hat = (X^T X)^{-1} X^T y

    Cai dat bang Economic SVD (thuan Python, trong utils/svd_solver.py):
        X = U Sigma V^T  =>  beta_hat = V Sigma^{-1} U^T y

    Uu diem SVD so voi nghich dao truc tiep:
        - On dinh so hoc khi X^T X gan suy bien (da cong tuyen)
        - Xu ly duoc ma tran chu nhat (m > n hoac m < n)
    """
    X_list = [[float(v) for v in row] for row in X]
    y_list = [float(v) for v in np.array(y).flatten()]
    return svd_solve(X_list, y_list)


def hat_matrix(X):
    """
    2. Tinh ma tran chieu H (Hat matrix).

    Cong thuc toan hoc:
        H = X (X^T X)^{-1} X^T

    Cai dat bang Economic SVD (thuan Python): X = U Sigma V^T
        => H = U_r U_r^T
    trong do U_r la cac cot cua U tuong ung voi sigma_i > EPS.

    Tinh chat quan trong:
        - Luy dang (Idempotent): H^2 = H
        - Doi xung: H^T = H
        - tr(H) = rank(X) = p+1 (so tham so uoc luong)
    """
    X_list = [[float(v) for v in row] for row in X]
    U_mat, s, _ = economic_svd(X_list)

    m = len(U_mat)
    r = len(s)

    # H = sum_{k: sigma_k > EPS} u_k u_k^T  (thuan Python)
    H = [[0.0] * m for _ in range(m)]
    for k in range(r):
        if s[k] > 1e-9:
            for a in range(m):
                for b in range(m):
                    H[a][b] += U_mat[a][k] * U_mat[b][k]

    return np.array(H, dtype=float)


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

    e = vector_subtract(y_list, y_hat_list)
    rss = vector_dot_product(e, e)

    y_mean = sum(y_list) / n if n > 0 else 0.0
    y_centered = [y_i - y_mean for y_i in y_list]
    tss = vector_dot_product(y_centered, y_centered)

    r2 = 1.0 - (rss / tss) if tss != 0.0 else 0.0

    if n - p - 1 > 0:
        adj_r2 = 1.0 - ((rss / (n - p - 1)) / (tss / (n - 1)))
        f_statistic = ((tss - rss) / p) / (rss / (n - p - 1)) if rss > 0.0 else float("inf")
    else:
        adj_r2 = float("nan")
        f_statistic = float("nan")

    return {"RSS": rss, "TSS": tss, "R2": r2, "Adj_R2": adj_r2, "F_statistic": f_statistic}


def coef_inference(X: list, y: list, beta_hat: list, sigma2: float) -> pd.DataFrame:
    """
    Suy dien thong ke cho cac he so:
    - Standard errors, t-statistics, p-values, CI 95%
    """
    X_mat = list(X)
    beta_list = list(beta_hat)
    n = len(X_mat)
    k = len(X_mat[0])
    df = n - k

    X_T = mat_transpose(X_mat)
    X_T_X = mat_mul(X_T, X_mat)
    X_T_X_inv = inverse(X_T_X)

    se = [math.sqrt(max(sigma2 * X_T_X_inv[i][i], 0.0)) for i in range(k)]
    t_stats = [beta_list[i] / se[i] if se[i] > 0 else 0.0 for i in range(k)]
    p_values = [2.0 * _student_t_sf(abs(t_stats[i]), df) for i in range(k)]
    t_critical = _student_t_ppf(0.975, df)
    ci_lower = [beta_list[i] - t_critical * se[i] for i in range(k)]
    ci_upper = [beta_list[i] + t_critical * se[i] for i in range(k)]

    return pd.DataFrame({
        "Coefficient": beta_list, "Std_Error": se, "t_stat": t_stats,
        "p_value": p_values, "CI_95_Lower": ci_lower, "CI_95_Upper": ci_upper,
    })


def vif(X):
    """
    5. Tinh Variance Inflation Factor (VIF) cho tung bien doc lap.

    Cong thuc: VIF_j = 1 / (1 - R^2_j)
    R^2_j la he so xac dinh khi hoi quy bien j len cac bien con lai (co he so chan).

    Tham so:
        X: ma tran dac trung (m x p), KHONG bao gom cot he so chan

    Tra ve:
        DataFrame voi cot Feature va VIF_Score
    """
    X_arr = np.array(X, dtype=float)
    n, p = X_arr.shape

    vif_values = []
    for j in range(p):
        y_j = X_arr[:, j].tolist()
        other_cols = [k for k in range(p) if k != j]
        X_j_list = [
            [1.0] + [float(X_arr[i, k]) for k in other_cols]
            for i in range(n)
        ]

        beta_j = list(ols_fit(X_j_list, y_j))
        y_hat_j = [
            sum(X_j_list[i][k] * beta_j[k] for k in range(len(beta_j)))
            for i in range(n)
        ]

        y_mean_j = sum(y_j) / n
        ss_res = sum((y_j[i] - y_hat_j[i]) ** 2 for i in range(n))
        ss_tot = sum((y_j[i] - y_mean_j) ** 2 for i in range(n))

        r2_j = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        vif_j = 1.0 / (1.0 - r2_j) if (1.0 - r2_j) > 1e-12 else float("inf")
        vif_values.append(vif_j)

    X_df = pd.DataFrame(X_arr)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF_Score"] = vif_values
    return vif_data


def gauss_markov_simulation(n_simulations=1000, n_samples=100):
    """
    Minh hoa dinh ly Gauss-Markov bang Monte Carlo:
    - Tinh khong chech: E[beta_hat] = beta
    - Tinh BLUE: Var(beta_OLS) < Var(beta_Alternative)
    """
    random.seed(42)
    p = 2
    X = [[1.0, random.uniform(0, 10), random.uniform(-5, 5)] for _ in range(n_samples)]
    true_beta = [3.0, 1.5, -2.0]
    k = len(true_beta)

    X_T = mat_transpose(X)
    X_T_X_inv = inverse(mat_mul(X_T, X))
    C_ols = mat_mul(X_T_X_inv, X_T)

    n_alt = int(n_samples * 0.6)
    X_alt = X[:n_alt]
    X_alt_T = mat_transpose(X_alt)
    X_alt_T_X_alt_inv = inverse(mat_mul(X_alt_T, X_alt))
    C_alt = mat_mul(X_alt_T_X_alt_inv, X_alt_T)

    beta_ols_results = {j: [] for j in range(k)}
    beta_alt_results = {j: [] for j in range(k)}

    for _ in range(n_simulations):
        sigma = 2.0
        e = [random.gauss(0, sigma) for _ in range(n_samples)]
        y = [sum(X[i][j] * true_beta[j] for j in range(k)) + e[i] for i in range(n_samples)]
        beta_ols = matrix_vector_multiply(C_ols, y)
        beta_alt = matrix_vector_multiply(C_alt, y[:n_alt])
        for j in range(k):
            beta_ols_results[j].append(beta_ols[j])
            beta_alt_results[j].append(beta_alt[j])

    report = []
    for j in range(k):
        e_ols = sum(beta_ols_results[j]) / n_simulations
        e_alt = sum(beta_alt_results[j]) / n_simulations
        var_ols = sum((b - e_ols) ** 2 for b in beta_ols_results[j]) / (n_simulations - 1)
        var_alt = sum((b - e_alt) ** 2 for b in beta_alt_results[j]) / (n_simulations - 1)
        report.append({"beta_idx": j, "true_val": true_beta[j],
                       "E_ols": e_ols, "E_alt": e_alt, "Var_ols": var_ols, "Var_alt": var_alt})
    return report
