"""
Ridge Regression
=================
Cài đặt Ridge Regression và vẽ Ridge Trace.
"""

import numpy as np
import matplotlib.pyplot as plt


def ridge_fit(X, y, lam):
    """
    6. Cài đặt Ridge Regression from-scratch bằng NumPy.

    Công thức toán học:
        beta_ridge = (X^T X + lam * I)^{-1} X^T y

    Tham số:
        X   : numpy array shape (n, p) - ma trận đặc trưng (đã bao gồm cột bias nếu cần)
        y   : numpy array shape (n,)   - vector mục tiêu
        lam : float                    - tham số regularization lambda (>= 0)

    Ghi chú:
        Khi lam = 0, kết quả trùng với OLS thông thường.
        Lặp qua nhiều giá trị lambda để vẽ đồ thị Ridge Trace.
    """
    X_mat = np.array(X, dtype=float)
    y_vec = np.array(y, dtype=float).flatten()
    p = X_mat.shape[1]
    # (X^T X + lam * I)^{-1} X^T y
    A = X_mat.T @ X_mat + lam * np.eye(p)
    b = X_mat.T @ y_vec
    beta_ridge = np.linalg.solve(A, b)
    return beta_ridge


if __name__ == "__main__":
    # TODO: Khởi tạo dữ liệu giả lập có hiện tượng đa cộng tuyến
    # TODO: Gọi hàm ridge_fit để minh họa
    # TODO: Kiểm chứng kết quả với sklearn.linear_model.Ridge
    print("Ridge Regression - Demo")
