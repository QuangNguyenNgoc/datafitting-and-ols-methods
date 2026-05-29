"""
OLS Implementation and Inference
================================
Cài đặt các hàm OLS cơ bản, tính ma trận chiếu, metrics, suy diễn thống kê,
tính VIF và minh họa định lý Gauss-Markov.
"""

import math

import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
    1. Tính vector hệ số beta_hat và phương sai sai số sigma2_hat.

    Công thức toán học:
        beta_hat = (X^T X)^{-1} X^T y
        sigma2_hat = RSS / (n - p)
    """
    model = sm.OLS(y, X).fit()
    return np.array(model.params)


def hat_matrix(X):
    """
    2. Tính ma trận chiếu H (Hat matrix) và kiểm tra tính lũy đẳng (idempotent).

    Công thức toán học:
        H = X(X^T X)^{-1} X^T
    Điều kiện lũy đẳng: H @ H = H
    """
    X_mat = np.array(X)
    return X_mat @ np.linalg.pinv(X_mat)


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
    """
    X_df = pd.DataFrame(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns

    vif_values = []
    for i in range(X_df.shape[1]):
        try:
            val = variance_inflation_factor(X_df.values, i)
        except Exception:
            val = np.inf
        vif_values.append(val)

    vif_data["VIF_Score"] = vif_values
    return vif_data


def gauss_markov_simulation(n_simulations=1000, n_samples=100):
    """
    9. Minh họa định lý Gauss-Markov:
       Mô phỏng Monte Carlo để kiểm chứng:
       - Tính không chệch: E[beta_hat] = beta
       - OLS estimator có phương sai nhỏ nhất trong lớp các ước lượng tuyến tính không chệch (BLUE).
    """
    # TODO: Implement Monte Carlo simulation for Gauss-Markov theorem
    pass


if __name__ == "__main__":
    # TODO: Khởi tạo dữ liệu giả lập (dummy data)
    # TODO: Gọi các hàm trên để minh họa
    # TODO: Kiểm chứng kết quả với thư viện chuẩn (NumPy/sklearn)
    print("OLS Implementation - Demo")
