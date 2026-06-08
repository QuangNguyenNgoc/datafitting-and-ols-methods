"""
OLS Implementation and Inference
================================
Cài đặt các hàm OLS cơ bản, tính ma trận chiếu, metrics, suy diễn thống kê, tính VIF và minh họa định lý Gauss-Markov.
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
    1. Tính vector hệ số beta_hat bằng phương pháp OLS.

    Công thức toán học:
        beta_hat = (X^T X)^{-1} X^T y

    Cài đặt dùng Economic SVD (trong utils/svd_solver.py):
        X = U Sigma V^T  =>  beta_hat = V Sigma^{-1} U^T y

    Ưu điểm SVD so với nghịch đảo trực tiếp:
        - Ổn định số học khi X^T X gần suy biến
        - Xử lý được ma trận chữ nhật (m > n hoặc m < n)
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float).flatten()
    return svd_solve(X_arr, y_arr)


def hat_matrix(X):
    """
    2. Tính ma trận chiếu H (Hat matrix).

    Công thức toán học:
        H = X (X^T X)^{-1} X^T

    Cài đặt dùng Economic SVD: X = U Sigma V^T
        => H = U U^T  (vì X(X^TX)^{-1}X^T = UU^T)

    Tính chất quan trọng:
        - Lũy đẳng: H^2 = H
        - Đối xứng: H^T = H
        - tr(H) = rank(X) = p (số tham số ước lượng)
    """
    X_arr = np.array(X, dtype=float)
    U, _, _ = economic_svd(X_arr)
    return U @ U.T


def model_metrics(y: list, y_hat: list, p: int) -> dict:
    """
    Tính các độ đo tổng hợp mô hình:
    - RSS (Residual Sum of Squares)
    - TSS (Total Sum of Squares)
    - R^2 (Hệ số xác định)
    - R^2 hiệu chỉnh (Adjusted R^2)
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
    Suy diễn thống kê cho các hệ số:
    - Standard errors (sai số chuẩn của từng hệ số)
    - t-statistics (giá trị t)
    - p-values
    - Khoảng tin cậy 95%
    """
    X_mat = list(X)
    beta_list = list(beta_hat)

    n = len(X_mat)
    k = len(X_mat[0])
    df = n - k

    # ma trận nghịch đảo
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

    # khoảng tin cậy 95%
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
    5. Tính Variance Inflation Factor (VIF) cho từng biến độc lập
       để kiểm tra hiện tượng đa cộng tuyến.

    Công thức: VIF_j = 1 / (1 - R^2_j)

    Trong đó R^2_j là hệ số xác định khi hồi quy biến thứ j lên tất cả các biến còn lại (có hệ số chặn).

    Tham số:
        X: ma trận đặc trưng (m x p), KHÔNG bao gồm cột hệ số chặn

    Trả về:
        DataFrame với cột Feature và VIF_Score
    """
    X_arr = np.array(X, dtype=float)
    n, p = X_arr.shape

    vif_values = []
    for j in range(p):
        # Biến j làm mục tiêu
        y_j = X_arr[:, j].tolist()

        # Các biến còn lại + hệ số chặn làm đặc trưng
        other_cols = [k for k in range(p) if k != j]
        X_j_list = [
            [1.0] + [float(X_arr[i, k]) for k in other_cols]
            for i in range(n)
        ]

        # Hồi quy OLS (qua ols_fit, dùng SVD trong utils)
        beta_j = list(ols_fit(X_j_list, y_j))

        # Tính y_hat
        y_hat_j = [
            sum(X_j_list[i][k] * beta_j[k] for k in range(len(beta_j)))
            for i in range(n)
        ]

        # R^2 = 1 - RSS/TSS
        y_mean_j = sum(y_j) / n
        ss_res = sum((y_j[i] - y_hat_j[i]) ** 2 for i in range(n))
        ss_tot = sum((y_j[i] - y_mean_j) ** 2 for i in range(n))

        r2_j = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        vif_j = 1.0 / (1.0 - r2_j) if (1.0 - r2_j) > 1e-12 else float("inf")
        vif_values.append(vif_j)

    # Giữ nguyên định dạng output cũ (Feature = chỉ số cột)
    X_df = pd.DataFrame(X_arr)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF_Score"] = vif_values
    return vif_data


def gauss_markov_simulation(n_simulations=1000, n_samples=100):
    """
    Minh họa định lý Gauss-Markov bằng Monte Carlo:
    - Tính không chệch: E[beta_hat] = beta
    - Tính BLUE: Var(beta_OLS) < Var(beta_Alternative)
    """
    # khởi tạo ma trận X cố định và beta thực
    random.seed(42)
    p = 2
    X = [[1.0, random.uniform(0, 10), random.uniform(-5, 5)] for _ in range(n_samples)]
    true_beta = [3.0, 1.5, -2.0]
    k = len(true_beta)

    # tiền tính toán ma trận ánh xạ cho OLS (100% data)
    X_T = mat_transpose(X)
    X_T_X_inv = inverse(mat_mul(X_T, X))
    C_ols = mat_mul(X_T_X_inv, X_T)

    # tiền tính toán ma trận ánh xạ cho Alternative Estimator (60% data)
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
        y = [
            sum(X[i][j] * true_beta[j] for j in range(k)) + e[i]
            for i in range(n_samples)
        ]
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
        report.append({
            "beta_idx": j,
            "true_val": true_beta[j],
            "E_ols": e_ols,
            "E_alt": e_alt,
            "Var_ols": var_ols,
            "Var_alt": var_alt,
        })

    return report
