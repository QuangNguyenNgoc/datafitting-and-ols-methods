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

from part1.utils_verif import _student_t_sf, _student_t_ppf, _f_sf_paulson

_EPS = 1e-9


def ols_fit(X, y):
    """
    1. Tính vector hệ số beta_hat bằng phương pháp OLS.

    Công thức toán học:
        beta_hat = (X^T X)^{-1} X^T y

    Cài đặt bằng Economic SVD thuần Python (svd_decomp từ utils/decomposition.py):
        X = U Sigma V^T  =>  beta_hat = V Sigma^{-1} U^T y
    """
    X_list = [[float(v) for v in row] for row in X]
    y_list = [float(v) for v in y]

    n_samples = len(X_list)
    n_features = len(X_list[0])

    U, Sigma, V_T = svd_decomp(X_list)
    k_svd = min(len(Sigma), len(Sigma[0]))
    s = [Sigma[i][i] for i in range(k_svd)]

    m = len(U)
    n = len(V_T)
    r = len(s)

    # Bước 1 & 2 & 3: Tính beta_hat bằng SVD
    Ut_y = [sum(U[j][i] * y_list[j] for j in range(m)) for i in range(r)]
    sp_Uty = [Ut_y[i] / s[i] if s[i] > _EPS else 0.0 for i in range(r)]
    beta = [sum(V_T[i][j] * sp_Uty[i] for i in range(r)) for j in range(n)]

    # Bước 4: Tính y_hat = X * beta
    y_hat = []
    for i in range(n_samples):
        val = sum(X_list[i][j] * beta[j] for j in range(n_features))
        y_hat.append(val)

    # Bước 5: Tính RSS và sigma^2
    rss = sum((y_list[i] - y_hat[i]) ** 2 for i in range(n_samples))
    sigma2 = rss / (n_samples - n_features) if n_samples > n_features else 0.0

    return beta, sigma2


def hat_matrix(X: list[list[float]]) -> list[list[float]]:
    """
    2. Tính ma trận chiếu H (Hat matrix).

    Công thức toán học:
        H = X (X^T X)^{-1} X^T

    Cài đặt bằng SVD thuần Python: X = U Sigma V^T
        => H = U_r U_r^T
    trong đó U_r là các cột của U ứng với sigma_i > EPS.

    Tính chất quan trọng:
        - Lũy đẳng: H^2 = H
        - Đối xứng: H^T = H
        - tr(H) = rank(X) = p+1
    """
    # ===== Tìm H ====
    X_list = [[float(v) for v in row] for row in X]

    # Phân rã SVD
    U, Sigma, _ = svd_decomp(X_list)
    k = min(len(Sigma), len(Sigma[0]))
    s = [Sigma[i][i] for i in range(k)]

    m = len(U)

    # Khởi tạo ma trận H kích thước m x m với toàn số 0.0
    H = [[0.0] * m for _ in range(m)]

    # Tính H = sum_{col: s[col] > EPS} u_col u_col^T
    for col in range(k):
        if s[col] > _EPS:
            for a in range(m):
                for b in range(m):
                    H[a][b] += U[a][col] * U[b][col]

    # ==== kiểm tra idempotent ====
    is_idempotent = True
    H_square = [[0.0] * m for _ in range(m)]

    # Tính H_square = H x H
    for i in range(m):
        for j in range(m):
            for x in range(m):
                H_square[i][j] += H[i][x] * H[x][j]

    # So sánh H_square với H (sai số 1e-9)
    for i in range(m):
        for j in range(m):
            if abs(H_square[i][j] - H[i][j]) > 1e-9:
                is_idempotent = False
                break
        if not is_idempotent:
            break

    if is_idempotent:
        print("Ma trận H thỏa mãn tính lũy đẳng (H^2 = H).")
    else:
        print("Ma trận H bị vi phạm tính lũy đẳng!")
    return H


def model_metrics(y: list, y_hat: list, p: int) -> dict:
    """
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

    # p-value
    df_model = p
    df_resid = n - p - 1
    # Tính F-stat
    f_stat = ((tss - rss) / df_model) / (rss / df_resid) if rss > 0 else 0.0

    p_value = _f_sf_paulson(f_stat, df_model, df_resid)

    return {
        "RSS": rss,
        "TSS": tss,
        "R2": r2,
        "Adj_R2": adj_r2,
        "F_statistic": f_statistic,
        "p_value": p_value,
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
    t_critical = _student_t_ppf(df)
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

    Trong đó R^2_j là hệ số xác định khi hồi quy biến thứ j lên tất cả
    các biến còn lại (có hệ số chặn).

    Tham số:
        X: ma trận đặc trưng (m x p), KHÔNG bao gồm cột hệ số chặn

    Trả về:
        DataFrame với cột Feature và VIF_Score
    """
    # Nếu đầu vào là DataFrame, tự động lấy tên cột
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_list = X.values.tolist()
    else:
        X_list = [[float(v) for v in row] for row in X]
        p = len(X_list[0])
        # Nếu truyền list thuần, tự động đánh số thứ tự
        feature_names = [f"Feature_{i}" for i in range(p)]

    n = len(X_list)
    p = len(X_list[0])

    # Nếu không có tên cột, fallback về 0, 1, 2...
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(p)]

    vif_values = []
    for j in range(p):
        y_j = [X_list[i][j] for i in range(n)]
        other_cols = [c for c in range(p) if c != j]

        # Thêm hệ số chặn [1.0] vào đặc trưng
        X_j_list = [[1.0] + [X_list[i][c] for c in other_cols] for i in range(n)]

        try:
            beta_j, _ = ols_fit(X_j_list, y_j)

            y_hat_j = [
                sum(X_j_list[i][c] * beta_j[c] for c in range(len(beta_j)))
                for i in range(n)
            ]

            y_mean_j = sum(y_j) / n
            ss_res = sum((y_j[i] - y_hat_j[i]) ** 2 for i in range(n))
            ss_tot = sum((y_j[i] - y_mean_j) ** 2 for i in range(n))

            r2_j = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            vif_j = 1.0 / (1.0 - r2_j) if (1.0 - r2_j) > 1e-12 else float("inf")

        except Exception:
            # ma trận suy biến
            vif_j = float("inf")

        vif_values.append(vif_j)

    vif_data = pd.DataFrame(
        {
            "Feature": feature_names,
            "VIF_Score": vif_values,
        }
    )
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
