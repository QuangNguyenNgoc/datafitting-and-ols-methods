from __future__ import annotations
from collections.abc import Sequence

from .utils import Number, _shape, _to_matrix
from .gaussian import _forward_elimination_ref

"""
Tính định thức của ma trận vuông bằng phép khử Gauss.
"""

def determinant(A: Sequence[Sequence[Number]], eps: float = 1e-12) -> float:
    """
    Đầu vào:
    - A: ma trận vuông
    - eps: ngưỡng nhận diện số gần bằng 0

    Đầu ra:
    - Giá trị định thức của A

    Ý nghĩa:
    - Nếu kết quả bằng 0 hoặc gần 0 thì ma trận suy biến.
    """
    matrix = _to_matrix(A)
    rows, cols = _shape(matrix)
    if rows != cols:
        raise ValueError("determinant(A) requires a square matrix.")

    ref_matrix, swap_count, _, _ = _forward_elimination_ref(matrix, eps=eps, pivot_limit=cols)
    det = -1.0 if swap_count % 2 else 1.0
    for i in range(rows):
        det *= ref_matrix[i][i]

    return 0.0 if abs(det) <= eps else det
