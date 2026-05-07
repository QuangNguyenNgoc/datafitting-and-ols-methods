"""
OLS Implementation and Inference
================================
Cài đặt các hàm OLS cơ bản, tính ma trận chiếu, metrics, suy diễn thống kê,
tính VIF và minh họa định lý Gauss-Markov.
"""

import numpy as np
import scipy.stats

def ols_fit(X, y):
    """
    1. Tính vector hệ số beta_hat và phương sai sai số sigma2_hat.
    
    Công thức toán học:
        beta_hat = (X^T X)^{-1} X^T y
        sigma2_hat = RSS / (n - p)
    """
    # TODO: Implement OLS fitting logic
    pass

def hat_matrix(X):
    """
    2. Tính ma trận chiếu H (Hat matrix) và kiểm tra tính lũy đẳng (idempotent).
    
    Công thức toán học:
        H = X(X^T X)^{-1} X^T
    Điều kiện lũy đẳng: H @ H = H
    """
    # TODO: Implement hat matrix calculation and idempotency check
    pass

def model_metrics(y, y_hat, p):
    """
    3. Tính các độ đo tổng hợp mô hình:
       - RSS (Residual Sum of Squares)
       - TSS (Total Sum of Squares)
       - R^2 (Hệ số xác định)
       - R^2 hiệu chỉnh (Adjusted R^2)
       - F-statistic
    """
    # TODO: Implement metrics calculation
    pass

def coef_inference(X, y, beta_hat, sigma2):
    """
    4. Suy diễn thống kê cho các hệ số:
       - Standard errors (sai số chuẩn của từng hệ số)
       - t-statistics (giá trị t)
       - p-values
       - Khoảng tin cậy (Confidence Intervals) 95%
    """
    # TODO: Implement coefficient inference logic
    pass

def vif(X):
    """
    5. Tính Variance Inflation Factor (VIF) cho từng biến độc lập
       để kiểm tra hiện tượng đa cộng tuyến.
       
    Công thức: VIF_j = 1 / (1 - R^2_j)
    """
    # TODO: Implement VIF calculation
    pass

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
