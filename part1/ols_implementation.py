"""
OLS Implementation and Inference
================================
Cài đặt các hàm OLS cơ bản, tính ma trận chiếu, metrics, suy diễn thống kê,
tính VIF và minh họa định lý Gauss-Markov.
"""

import math
import random

import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.matrix_utils import (
    mat_transpose,
    mat_mul,
    vector_subtract,
    vector_dot_product,
    matrix_vector_multiply,
)
from utils.inverse import inverse

from part1.utils_verif import _student_t_sf, _student_t_ppf


def ols_fit(X, y):
    """
    1. Tính vector hệ số beta_hat bằng phương pháp bình phương tối thiểu (OLS).

    Công thức toán học:
        beta_hat = (X^T X)^{-1} X^T y

    Cài đặt: dùng np.linalg.lstsq (giải qua SVD) để ổn định số,
    tránh trường hợp X^T X gần singular.
    """
    X_mat = np.array(X, dtype=float)
    y_vec = np.array(y, dtype=float).flatten()
    # lstsq giải hệ bình phương tối thiểu: argmin ||X beta - y||^2
    beta_hat, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    return beta_hat


def hat_matrix(X):
    """
    2. Tính ma trận chiếu H (Hat matrix) bằng Economic SVD.

    Công thức toán học:
        H = X(X^T X)^{-1} X^T = U U^T   (với X = U S V^T, dạng economic SVD)

    Lý do dùng SVD: tránh tràn bộ nhớ với ma trận lớn;
    chỉ cần U là ma trận trực giao (n x r) với r = rank(X).
    Điều kiện lũy đẳng: H @ H = H
    """
    X_mat = np.array(X, dtype=float)
    # Economic SVD: U có shape (n, r), S có shape (r,), Vt có shape (r, p)
    U, s, Vt = np.linalg.svd(X_mat, full_matrices=False)
    # Xác định rank số trị thực (loại singular values xấp xỉ 0)
    tol = np.finfo(float).eps * max(X_mat.shape) * s[0]
    rank = int(np.sum(s > tol))
    U_r = U[:, :rank]
    # H = U_r @ U_r^T
    return U_r @ U_r.T


def model_metrics(y: list, y_hat: list, p: int) -> dict:
    """
    Tính các độ đo tổng hợp mô hình bằng 100% Python gốc:
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

    # MAE và RMSE (bổ sung để tương thích với report generator)
    mae = sum(abs(e_i) for e_i in e) / n if n > 0 else 0.0
    rmse = math.sqrt(rss / n) if n > 0 else 0.0

    return {
        "RSS": rss,
        "TSS": tss,
        "R2": r2,
        "Adj_R2": adj_r2,
        "F_statistic": f_statistic,
        "MAE": mae,
        "RMSE": rmse,
    }


def coef_inference(X: list, y: list, beta_hat: list, sigma2: float) -> pd.DataFrame:
    """
    Suy diễn thống kê cho các hệ số:
    - Standard errors (sai số chuẩn của từng hệ số)
    - t-statistics (giá trị t)
    - p-values
    - Khoảng tin cậy (Confidence Intervals) 95%
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
    Trong đó R^2_j là R^2 của hồi quy phụ: X_j ~ X_{-j} (hồi quy X cột j lên tất cả cột còn lại).

    Cài đặt from-scratch bằng NumPy, không dùng statsmodels/sklearn.
    """
    X_mat = np.array(X, dtype=float)
    n, p = X_mat.shape

    vif_values = []
    for j in range(p):
        # Hồi quy phụ: X[:, j] ~ X[:, -j] (tất cả cột trừ cột j)
        X_other = np.delete(X_mat, j, axis=1)
        beta_j, _, _, _ = np.linalg.lstsq(X_other, X_mat[:, j], rcond=None)
        y_hat_j = X_other @ beta_j

        # Tính R^2_j
        ss_res = np.sum((X_mat[:, j] - y_hat_j) ** 2)
        ss_tot = np.sum((X_mat[:, j] - np.mean(X_mat[:, j])) ** 2)

        if ss_tot == 0.0:
            vif_j = np.inf
        else:
            r2_j = 1.0 - ss_res / ss_tot
            vif_j = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf

        vif_values.append(vif_j)

    X_df = pd.DataFrame(X_mat)
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
    # X có 1 cột bias và 2 cột đặc trưng ngẫu nhiên
    random.seed(42)
    p = 2
    X = [[1.0, random.uniform(0, 10), random.uniform(-5, 5)] for _ in range(n_samples)]
    true_beta = [3.0, 1.5, -2.0]
    k = len(true_beta)

    # tiền tính toán ma trận ánh xạ cho OLS (100% data)
    X_T = mat_transpose(X)
    X_T_X_inv = inverse(mat_mul(X_T, X))
    # C_ols = (X^T X)^-1 X^T
    C_ols = mat_mul(X_T_X_inv, X_T)

    # tiền tính toán ma trận ánh xạ cho Alternative Estimator (60% data)
    n_alt = int(n_samples * 0.6)
    X_alt = X[:n_alt]
    X_alt_T = mat_transpose(X_alt)
    X_alt_T_X_alt_inv = inverse(mat_mul(X_alt_T, X_alt))
    C_alt = mat_mul(X_alt_T_X_alt_inv, X_alt_T)

    # lưu trữ kết quả của 1000 vòng lặp
    beta_ols_results = {j: [] for j in range(k)}
    beta_alt_results = {j: [] for j in range(k)}

    # VÒNG LẶP MONTE CARLO
    for _ in range(n_simulations):
        # tạo nhiễu e ~ N(0, sigma^2)
        sigma = 2.0
        e = [random.gauss(0, sigma) for _ in range(n_samples)]

        # sinh biến mục tiêu y = X*beta + e
        y = [
            sum(X[i][j] * true_beta[j] for j in range(k)) + e[i]
            for i in range(n_samples)
        ]

        beta_ols = matrix_vector_multiply(C_ols, y)
        beta_alt = matrix_vector_multiply(C_alt, y[:n_alt])

        # Lưu kết quả
        for j in range(k):
            beta_ols_results[j].append(beta_ols[j])
            beta_alt_results[j].append(beta_alt[j])

    # thống kê
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
                "true_val": true_beta[j],
                "E_ols": e_ols,
                "E_alt": e_alt,
                "Var_ols": var_ols,
                "Var_alt": var_alt,
            }
        )

    return report
n report
