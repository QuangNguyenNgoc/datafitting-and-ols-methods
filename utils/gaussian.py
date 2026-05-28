from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .utils import (
    Matrix,
    Number,
    Vector,
    _augment,
    _clean_small_entries,
    _copy_matrix,
    _shape,
    _swap_rows,
    _to_matrix,
    _to_vector,
)

"""
Các nội dung trong file này:
- khử tiến để đưa ma trận tăng cường về REF
- thế ngược cho hệ tam giác trên
- dựng RREF để phân loại hệ
- biểu diễn nghiệm tổng quát khi hệ có vô số nghiệm
"""

_NEAR_ZERO_PIVOT_WARNING = "pivot is near zero; the system may be ill-conditioned"


def back_substitution(U: Sequence[Sequence[Number]], c: Sequence[Number], eps: float = 1e-12) -> Vector:
    """
    Giải hệ tam giác trên Ux = c.

    Đầu vào:
    - U: ma trận tam giác trên
    - c: vector vế phải
    - eps: ngưỡng nhận diện số gần bằng 0

    Đầu ra:
    - vector nghiệm x
    """
    upper = _to_matrix(U)
    rhs = _to_vector(c)
    n, m = _shape(upper)
    if n != m:
        raise ValueError("U must be a square matrix.")
    if len(rhs) != n:
        raise ValueError("U and c must have compatible dimensions.")

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(upper[i][i]) <= eps:
            raise ValueError("Back substitution failed because a diagonal pivot is zero.")
        subtotal = sum(upper[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (rhs[i] - subtotal) / upper[i][i]

    return [0.0 if abs(value) <= eps else value for value in x]


def _append_warning_once(warnings: List[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def _forward_elimination_ref(
    M: Matrix,
    eps: float = 1e-12,
    *,
    pivot_limit: int | None = None,
) -> Tuple[Matrix, int, List[int], List[str]]:
    ref = _copy_matrix(M)
    rows, cols = _shape(ref)
    if pivot_limit is None:
        pivot_limit = cols

    pivot_columns: List[int] = []
    warnings: List[str] = []
    swap_count = 0
    pivot_row = 0

    matrix_scale = max((abs(value) for row in ref for value in row), default=1.0)
    near_zero_threshold = max(1_000.0 * eps * max(1.0, matrix_scale), eps)

    for col in range(min(pivot_limit, cols)):
        if pivot_row >= rows:
            break

        # Partial pivoting: chọn dòng có phần tử lớn nhất theo trị tuyệt đối trong cột hiện tại
        best_row = max(range(pivot_row, rows), key=lambda r: abs(ref[r][col]))
        best_value = ref[best_row][col]
        if abs(best_value) <= eps:
            continue

        if best_row != pivot_row:
            _swap_rows(ref, pivot_row, best_row)
            swap_count += 1

        pivot_columns.append(col)
        pivot_value = ref[pivot_row][col]
        if abs(pivot_value) <= near_zero_threshold:
            _append_warning_once(warnings, _NEAR_ZERO_PIVOT_WARNING)

        for r in range(pivot_row + 1, rows):
            if abs(ref[r][col]) <= eps:
                ref[r][col] = 0.0
                continue
            factor = ref[r][col] / pivot_value
            for c in range(col, cols):
                ref[r][c] -= factor * ref[pivot_row][c]
            ref[r][col] = 0.0

        pivot_row += 1

    return _clean_small_entries(ref, eps), swap_count, pivot_columns, warnings


def _rref(M: Matrix, eps: float = 1e-12) -> Tuple[Matrix, List[int]]:
    rref = _copy_matrix(M)
    rows, cols = _shape(rref)

    pivot_columns: List[int] = []
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        best_row = max(range(pivot_row, rows), key=lambda r: abs(rref[r][col]))
        if abs(rref[best_row][col]) <= eps:
            continue

        if best_row != pivot_row:
            _swap_rows(rref, pivot_row, best_row)

        pivot = rref[pivot_row][col]
        for c in range(col, cols):
            rref[pivot_row][c] /= pivot
        rref[pivot_row][col] = 1.0

        for r in range(rows):
            if r == pivot_row:
                continue
            factor = rref[r][col]
            if abs(factor) <= eps:
                rref[r][col] = 0.0
                continue
            for c in range(col, cols):
                rref[r][c] -= factor * rref[pivot_row][c]
            rref[r][col] = 0.0

        pivot_columns.append(col)
        pivot_row += 1

    return _clean_small_entries(rref, eps), pivot_columns


def _build_general_solution_from_rref(
    rref_augmented: Matrix,
    variable_count: int,
    pivot_columns: List[int],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    free_columns = [j for j in range(variable_count) if j not in pivot_columns]
    particular = [0.0] * variable_count

    for row_index, pivot_col in enumerate(pivot_columns):
        particular[pivot_col] = rref_augmented[row_index][-1]

    nullspace_basis: List[Vector] = []
    for free_col in free_columns:
        vector = [0.0] * variable_count
        vector[free_col] = 1.0
        for row_index, pivot_col in enumerate(pivot_columns):
            vector[pivot_col] = -rref_augmented[row_index][free_col]
        nullspace_basis.append([0.0 if abs(v) <= eps else v for v in vector])

    parameter_names = [f"t{i + 1}" for i in range(len(free_columns))]
    basis_terms = []
    for name, basis_vector in zip(parameter_names, nullspace_basis):
        basis_terms.append(f"{name}*{basis_vector}")

    if basis_terms:
        general_form = f"x = {particular} + " + " + ".join(basis_terms)
    else:
        general_form = f"x = {particular}"

    return {
        "type": "infinite",
        "particular": particular,
        "nullspace_basis": nullspace_basis,
        "pivot_columns": pivot_columns,
        "free_columns": free_columns,
        "parameters": parameter_names,
        "general_form": general_form,
    }


def _extract_upper_system(
    ref_augmented: Matrix,
    variable_count: int,
    eps: float = 1e-12,
) -> Tuple[Matrix, Vector]:
    """
    Tách ma trận tăng cường ở dạng REF thành ma trận tam giác trên U và vector vế phải c để dùng cho back_substitution.
    """
    rows, cols = _shape(ref_augmented)

    if cols != variable_count + 1:
        raise ValueError("The augmented matrix must have exactly variable_count + 1 columns.")
    if rows < variable_count:
        raise ValueError("Not enough rows to extract a square upper-triangular system.")

    U: Matrix = []
    c_vector: Vector = []

    for i in range(variable_count):
        row = [
            0.0 if abs(ref_augmented[i][j]) <= eps else ref_augmented[i][j]
            for j in range(variable_count)
        ]
        rhs_value = 0.0 if abs(ref_augmented[i][-1]) <= eps else ref_augmented[i][-1]

        U.append(row)
        c_vector.append(rhs_value)

    return U, c_vector


def _build_unique_solution_from_rref(
    rref_augmented: Matrix,
    variable_count: int,
    pivot_columns: List[int],
    eps: float = 1e-12,
) -> Vector:
    x = [0.0] * variable_count
    for row_index, pivot_col in enumerate(pivot_columns):
        if pivot_col >= variable_count:
            continue
        x[pivot_col] = rref_augmented[row_index][-1]
    return [0.0 if abs(value) <= eps else value for value in x]


def gaussian_eliminate(
    A: Sequence[Sequence[Number]],
    b: Sequence[Number],
    eps: float = 1e-12,
) -> Tuple[Matrix, Dict[str, Any], int]:
    """
    Giải hệ Ax = b bằng phép khử Gauss.

    Đầu vào:
    - A: ma trận hệ số kích thước m x n
    - b: vector vế phải kích thước m
    - eps: ngưỡng nhận diện số gần bằng 0

    Đầu ra:
    - ref_augmented: ma trận tăng cường sau khi khử về REF
    - solution_info: thông tin mô tả loại nghiệm
    - swap_count: số lần đổi dòng

    solution_info có thể thuộc một trong ba dạng:
    1. unique:
       {"type": "unique", "x": [...], "pivot_columns": [...], ...}
    2. infinite:
       {"type": "infinite", "particular": [...], "nullspace_basis": [...], ...}
    3. inconsistent:
       {"type": "inconsistent", "message": "...", "pivot_columns": [...], ...}
    """
    matrix = _to_matrix(A)
    rhs = _to_vector(b)
    rows, cols = _shape(matrix)
    if len(rhs) != rows:
        raise ValueError("A and b must have the same number of rows.")

    augmented = _augment(matrix, rhs)
    ref_augmented, swap_count, pivot_columns, warnings = _forward_elimination_ref(
        augmented,
        eps=eps,
        pivot_limit=cols,
    )

    rank_A = len(pivot_columns)

    for row in ref_augmented:
        if all(abs(row[j]) <= eps for j in range(cols)) and abs(row[-1]) > eps:
            solution_info: Dict[str, Any] = {
                "type": "inconsistent",
                "pivot_columns": pivot_columns,
                "message": "The system is inconsistent, so it has no solution.",
            }
            if warnings:
                solution_info["warnings"] = warnings
            return ref_augmented, solution_info, swap_count

    if rank_A == cols:
        if cols == rows:
            # Tách hệ tam giác trên Ux = c từ ma trận tăng cường sau khi khử
            U, c_vector = _extract_upper_system(ref_augmented, cols, eps=eps)
            x = back_substitution(U, c_vector, eps=eps)
        else:
            rref_augmented, rref_pivot_columns = _rref(augmented, eps=eps)
            x = _build_unique_solution_from_rref(
                rref_augmented,
                cols,
                [col for col in rref_pivot_columns if col < cols],
                eps=eps,
            )

        solution_info = {
            "type": "unique",
            "x": x,
            "pivot_columns": pivot_columns,
        }
        if warnings:
            solution_info["warnings"] = warnings
        return ref_augmented, solution_info, swap_count

    rref_augmented, rref_pivot_columns = _rref(augmented, eps=eps)
    solution_info = _build_general_solution_from_rref(
        rref_augmented,
        cols,
        [col for col in rref_pivot_columns if col < cols],
        eps=eps,
    )
    solution_info["rref_augmented"] = rref_augmented
    if warnings:
        solution_info["warnings"] = warnings
    return ref_augmented, solution_info, swap_count
