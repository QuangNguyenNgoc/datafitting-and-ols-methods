from __future__ import annotations

from typing import Any, Dict, List, Sequence
from .utils import Number, Vector, _to_matrix, _shape
from .gaussian import _rref

"""
Tính hạng ma trận và các cơ sở liên quan.
"""

def rank_and_basis(A: Sequence[Sequence[Number]], eps: float = 1e-12) -> Dict[str, Any]:
    """
    Tính hạng của ma trận và trả về cơ sở của:
    - không gian cột
    - không gian dòng
    - không gian nghiệm

    Đầu vào:
    - A: ma trận bất kỳ
    - eps: ngưỡng nhận diện số gần bằng 0

    Đầu ra:
    - Một dict chứa rank, pivot columns và các basis tương ứng
    """
    matrix = _to_matrix(A)
    rows, cols = _shape(matrix)
    rref_matrix, pivot_columns = _rref(matrix, eps=eps)
    rank = len(pivot_columns)
    free_columns = [j for j in range(cols) if j not in pivot_columns]

    column_space_basis = []
    for pivot_col in pivot_columns:
        column_space_basis.append([matrix[i][pivot_col] for i in range(rows)])

    row_space_basis = []
    for row in rref_matrix:
        if any(abs(value) > eps for value in row):
            row_space_basis.append(row[:])

    null_space_basis: List[Vector] = []
    for free_col in free_columns:
        vector = [0.0] * cols
        vector[free_col] = 1.0
        for row_index, pivot_col in enumerate(pivot_columns):
            vector[pivot_col] = -rref_matrix[row_index][free_col]
        null_space_basis.append([0.0 if abs(v) <= eps else v for v in vector])

    return {
        "rank": rank,
        "pivot_columns": pivot_columns,
        "free_columns": free_columns,
        "rref": rref_matrix,
        "column_space_basis": column_space_basis,
        "row_space_basis": row_space_basis,
        "null_space_basis": null_space_basis,
    }