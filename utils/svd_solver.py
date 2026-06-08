"""
SVD Solver Utilities
====================
Các tính toán dựa trên Singular Value Decomposition (SVD).

Dùng np.linalg.svd (NumPy) để:
- Giải hệ OLS (tránh kỳ dị khi X^T X gần suy biến)
- Tính ma trận Hat H = U U^T hiệu quả bộ nhớ (Economic SVD)

  - (X^T X)^{-1} kém ổn định khi X có cột gần tuyến tính phụ thuộc.
  - SVD luôn tồn tại, xử lý được cả ma trận chữ nhật và suy biến.
  - H = U U^T (U từ Economic SVD của X) cho leverage h_ii chính xác mà không cần lưu toàn bộ H (dùng np.sum(U**2, axis=1)).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def economic_svd(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phân rã SVD kinh tế (thin SVD) của X (m x n): X = U Sigma V^T.

    Tham số:
        X: ma trận kích thước (m, n)

    Trả về:
        U     : ma trận (m, k), k = min(m,n), cột trực chuẩn
        s     : vector giá trị kỳ dị kích thước (k,), giảm dần
        Vt    : ma trận (k, n), V^T
    """
    return np.linalg.svd(X, full_matrices=False)


def svd_solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Giải bài toán OLS bằng Economic SVD.

    Công thức: beta = V diag(1/s) U^T y
    Chỉ giữ lại các giá trị kỳ dị lớn hơn ngưỡng (tránh chia cho 0).

    Tham số:
        X: ma trận thiết kế (m, n), đã bao gồm cột hệ số chặn nếu cần
        y: vector mục tiêu (m,)

    Trả về:
        beta: vector hệ số OLS (n,)
    """
    U, s, Vt = economic_svd(X)
    # Ngưỡng loại bỏ giá trị kỳ dị nhỏ (tương tự np.linalg.lstsq)
    tol = max(X.shape) * np.finfo(float).eps * s[0]
    mask = s > tol
    # beta = V[:,mask] @ diag(1/s[mask]) @ U[:,mask].T @ y
    beta = (Vt[mask].T) @ (U[:, mask].T @ y / s[mask])
    return beta


def hat_diagonal(X: np.ndarray) -> np.ndarray:
    """
    Tính đường chéo h_ii của ma trận Hat H = X(X^T X)^{-1} X^T
    hiệu quả bộ nhớ bằng Economic SVD: h_ii = sum(U_i^2).

    Tham số:
        X: ma trận thiết kế (m, n)

    Trả về:
        h: vector leverage h_ii kích thước (m,)
    """
    U, _, _ = economic_svd(X)
    return np.sum(U ** 2, axis=1)
