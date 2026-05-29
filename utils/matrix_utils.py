from __future__ import annotations

from typing import List, Sequence, Tuple, Union

Number = Union[int, float]
Matrix = List[List[float]]
Vector = List[float]


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


def _identity(n: int) -> Matrix:
    """
    Tạo ma trận đơn vị kích thước n x n.

    Đầu vào:
    - n: kích thước ma trận

    Đầu ra:
    - ma trận đơn vị kích thước n x n
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_mul(A: Matrix, B: Matrix) -> Matrix:
    """
    Nhân hai ma trận A (m x k) và B (k x n).

    Đầu vào:
    - A: ma trận kích thước m x k
    - B: ma trận kích thước k x n

    Đầu ra:
    - ma trận tích kích thước m x n
    """
    m, k, n = len(A), len(A[0]), len(B[0])
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)] for i in range(m)
    ]


def mat_transpose(M: Matrix) -> Matrix:
    """
    Chuyển vị ma trận M (m x n) thành ma trận n x m.

    Đầu vào:
    - M: ma trận kích thước m x n

    Đầu ra:
    - ma trận chuyển vị kích thước n x m
    """
    m, n = len(M), len(M[0])
    return [[M[i][j] for i in range(m)] for j in range(n)]


def vector_subtract(u: Vector, v: Vector) -> Vector:
    """Trừ hai vector 1 chiều: u - v"""
    return [u_i - v_i for u_i, v_i in zip(u, v)]


def vector_dot_product(u: Vector, v: Vector) -> float:
    """Nhân vô hướng hai vector 1 chiều (tương đương u.T @ v hoặc e.T @ e)"""
    return sum(u_i * v_i for u_i, v_i in zip(u, v))


def matrix_vector_multiply(A: Matrix, v: Vector) -> Vector:
    """Nhân ma trận 2 chiều với vector 1 chiều: A * v (Phục vụ tính y_hat = X @ beta)"""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _copy_matrix(A: Matrix) -> Matrix:
    return [row[:] for row in A]


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