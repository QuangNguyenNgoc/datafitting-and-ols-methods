from __future__ import annotations
from collections.abc import Sequence

from .utils import Matrix, Number, _to_matrix, _shape, _identity, _augment, _swap_rows, _clean_small_entries

"""
Tính ma trận nghịch đảo bằng Gauss-Jordan.
"""

def inverse(A: Sequence[Sequence[Number]], eps: float = 1e-12) -> Matrix:
    """
    Đầu vào:
    - A: ma trận vuông
    - eps: ngưỡng nhận diện số gần bằng 0

    Đầu ra:
    - Ma trận nghịch đảo A^{-1}
    
    * Nếu A không khả nghịch thì hàm báo lỗi.
    """
    matrix = _to_matrix(A)
    n, m = _shape(matrix)
    if n != m:
        raise ValueError("inverse(A) requires a square matrix.")

    augmented = [matrix[i] + _identity(n)[i] for i in range(n)]

    pivot_row = 0
    for col in range(n):
        best_row = max(range(pivot_row, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[best_row][col]) <= eps:
            raise ValueError("Matrix is singular, so it does not have an inverse.")

        if best_row != pivot_row:
            _swap_rows(augmented, pivot_row, best_row)

        pivot = augmented[pivot_row][col]
        for j in range(2 * n):
            augmented[pivot_row][j] /= pivot
        augmented[pivot_row][col] = 1.0

        for r in range(n):
            if r == pivot_row:
                continue
            factor = augmented[r][col]
            if abs(factor) <= eps:
                augmented[r][col] = 0.0
                continue
            for j in range(2 * n):
                augmented[r][j] -= factor * augmented[pivot_row][j]
            augmented[r][col] = 0.0

        pivot_row += 1

    inverse_matrix = [row[n:] for row in _clean_small_entries(augmented, eps)]
    return inverse_matrix