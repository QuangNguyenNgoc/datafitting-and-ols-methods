"""
OLS Implementation and Inference
================================
Cài đặt các hàm OLS cơ bản, tính ma trận chiếu, metrics, suy diễn thống kê,
tính VIF và minh họa định lý Gauss-Markov.
"""

import numpy as np
import scipy.stats
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

def model_metrics(y, y_hat, p):
    """
    3. Tính các độ đo tổng hợp mô hình:
       - RSS (Residual Sum of Squares)
       - TSS (Total Sum of Squares)
       - R^2 (Hệ số xác định)
       - R^2 hiệu chỉnh (Adjusted R^2)
       - F-statistic
    """
    y_arr = np.array(y).flatten()
    y_hat_arr = np.array(y_hat).flatten()
    n = len(y_arr)
    
    mae = mean_absolute_error(y_arr, y_hat_arr)
    rmse = np.sqrt(mean_squared_error(y_arr, y_hat_arr))
    r2 = r2_score(y_arr, y_hat_arr)
    
    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan
        
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Adj_R2': adj_r2
    }

def coef_inference(X, y, beta_hat, sigma2):
    """
    4. Suy diễn thống kê cho các hệ số:
       - Standard errors (sai số chuẩn của từng hệ số)
       - t-statistics (giá trị t)
       - p-values
       - Khoảng tin cậy (Confidence Intervals) 95%
    """
    model = sm.OLS(y, X).fit()
    
    inference_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std_Error': model.bse,
        't_stat': model.tvalues,
        'p_value': model.pvalues
    })
    
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
