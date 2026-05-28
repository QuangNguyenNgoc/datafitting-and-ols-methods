from __future__ import annotations

from typing import Any, List, Sequence, Union, Tuple

Number = Union[int, float]
Matrix = List[List[float]]
Vector = List[float]

# Các hàm phụ trợ để chuẩn hóa dữ liệu đầu vào và thao tác ma trận.

def _to_matrix(A: Sequence[Sequence[Number]]) -> Matrix:
    if not A:
        raise ValueError("Matrix A must not be empty.")
    matrix = [list(map(float, row)) for row in A]
    row_length = len(matrix[0])
    if row_length == 0:
        raise ValueError("Matrix A must have at least one column.")
    for row in matrix:
        if len(row) != row_length:
            raise ValueError("Matrix A must be rectangular.")
    return matrix


def _to_vector(b: Sequence[Number]) -> Vector:
    if not b:
        raise ValueError("Vector b must not be empty.")
    return [float(value) for value in b]


def _shape(A: Matrix) -> Tuple[int, int]:
    return len(A), len(A[0])


def _copy_matrix(A: Matrix) -> Matrix:
    return [row[:] for row in A]


def _identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _clean_small_entries(A: Matrix, eps: float = 1e-12) -> Matrix:
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j]) <= eps:
                A[i][j] = 0.0
    return A


def _augment(A: Matrix, b: Vector) -> Matrix:
    if len(A) != len(b):
        raise ValueError("A and b must have the same number of rows.")
    return [A[i][:] + [b[i]] for i in range(len(A))]


def _swap_rows(M: Matrix, i: int, j: int) -> None:
    M[i], M[j] = M[j], M[i]