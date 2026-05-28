from __future__ import annotations

import cmath
import math
from typing import List, Sequence, Tuple, Union

import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part1.inverse import inverse
from part1.rank_basis import rank_and_basis

Number = Union[int, float]
Matrix = List[List[float]]
Vector = List[float]

_EPS = 1e-9

# ──────────────────────────────────────────────

def _copy(M: Matrix) -> Matrix:
    """
    Tạo bản sao của ma trận M.

    Đầu vào:
    - M: ma trận đầu vào

    Đầu ra:
    - bản sao của ma trận M
    """
    return [row[:] for row in M]

def _identity(n: int) -> Matrix:
    """
    Tạo ma trận đơn vị kích thước n x n.

    Đầu vào:
    - n: kích thước ma trận

    Đầu ra:
    - ma trận đơn vị kích thước n x n
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def get_column(M: Matrix, col: int) -> Vector:
    """
    Trả về cột thứ `col` của ma trận M.

    Đầu vào:
    - M: ma trận kích thước m x n
    - col: chỉ số cột

    Đầu ra:
    - vector cột dưới dạng list
    """
    return [row[col] for row in M]

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
        [sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
        for i in range(m)
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

def _validate_matrix(M: Sequence[Sequence[Number]], label: str = "Ma trận") -> Matrix:
    """
    Kiểm tra và chuyển đổi ma trận đầu vào về dạng chuẩn.

    Đầu vào:
    - M: ma trận đầu vào
    - label: nhãn ma trận

    Đầu ra:
    - ma trận chuẩn
    """
    if not M:
        raise ValueError(f"{label} không được rỗng.")
    mat = [list(map(float, row)) for row in M]
    ncols = len(mat[0])
    if ncols == 0:
        raise ValueError(f"{label} phải có ít nhất một cột.")
    if any(len(row) != ncols for row in mat):
        raise ValueError(f"{label} phải là ma trận hình chữ nhật.")
    for row in mat:
        for v in row:
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"{label} chứa giá trị NaN hoặc Inf không hợp lệ.")
    return mat

# ──────────────────────────────────────────────

def gauss_inverse(M: Matrix) -> Matrix:
    try:
        return inverse(M, eps=_EPS)
    except ValueError as e:
        raise ValueError("Ma trận suy biến, không tồn tại ma trận nghịch đảo.") from e

# ──────────────────────────────────────────────

def char_poly(M: Matrix) -> List[float]:
    """
    Tính các hệ số đa thức đặc trưng của ma trận M bằng thuật toán Faddeev-LeVerrier.

    Đầu vào:
    - M: ma trận vuông kích thước n x n

    Đầu ra:
    - list hệ số [c_n, c_{n-1}, ..., c_0]
    """
    n = len(M)
    c = [1.0]
    B = _identity(n)
    for k in range(1, n + 1):
        MB = [row[:] for row in M] if k == 1 else mat_mul(M, B)
        trace = sum(MB[i][i] for i in range(n))
        ak = -trace / k
        c.append(ak)
        B = [[MB[i][j] + (ak if i == j else 0.0) for j in range(n)] for i in range(n)]
    return c

def find_roots(coeffs: List[float]) -> List[float]:
    """
    Tìm nghiệm thực của đa thức bằng phương pháp Durand-Kerner.

    Đầu vào:
    - coeffs: list hệ số [c_n, c_{n-1}, ..., c_0]

    Đầu ra:
    - list các nghiệm thực
    """
    n = len(coeffs) - 1
    roots = [cmath.rect(1.5, 2 * math.pi * k / n) for k in range(n)]
    for _ in range(200):
        for k in range(n):
            p_val = sum(coeffs[j] * roots[k] ** (n - j) for j in range(n + 1))
            denom = 1.0
            for j in range(n):
                if j != k:
                    denom *= roots[k] - roots[j]
            if abs(denom) > 1e-15:
                roots[k] -= p_val / denom
    return [float(r.real) for r in roots if abs(r.imag) < 1e-3]

# ──────────────────────────────────────────────

def _gram_schmidt(vectors: List[Vector]) -> List[Vector]:
    """Trực chuẩn hóa danh sách vector bằng Gram-Schmidt."""
    ortho: List[Vector] = []
    for v in vectors:
        for u in ortho:
            dot = sum(x * y for x, y in zip(v, u))
            v = [x - dot * y for x, y in zip(v, u)]
        norm = math.sqrt(sum(x * x for x in v))
        if norm > _EPS:
            ortho.append([x / norm for x in v])
    return ortho

def eigenspace_basis(M: Matrix, eigenvalue: float) -> List[Vector]:
    """
    Tìm cơ sở trực chuẩn của không gian riêng (M - lambda*I)x = 0.

    Đầu vào:
    - M: ma trận vuông kích thước n x n
    - eigenvalue: giá trị riêng lambda

    - list các vector cơ sở trực chuẩn của không gian riêng tương ứng
    """
    n = len(M)
    shifted = [[M[i][j] - (eigenvalue if i == j else 0.0) for j in range(n)] for i in range(n)]
    
    info = rank_and_basis(shifted, eps=1e-4)
    basis = info["null_space_basis"]

    if not basis:
        rref = info["rref"]
        v = [0.0] * n
        v[-1] = 1.0
        for pr, pc in enumerate(info["pivot_columns"][:-1]):
            if pr < len(rref):
                v[pc] = -rref[pr][-1]
        basis.append(v)

    return _gram_schmidt(basis)

# ──────────────────────────────────────────────

def find_eigen(M: Matrix) -> Tuple[List[float], Matrix]:
    """
    Tìm giá trị riêng và ma trận vector riêng của ma trận vuông M.

    Đầu vào:
    - M: ma trận vuông kích thước n x n

    Đầu ra:
    - (eigenvalues, P): list giá trị riêng sắp xếp giảm dần và ma trận P
      gồm các cột là vector riêng tương ứng
    """
    n = len(M)

    is_diag = True
    for i in range(n):
        for j in range(n):
            if i != j and abs(M[i][j]) > _EPS:
                is_diag = False
                break
        if not is_diag:
            break

    if is_diag:
        return [float(M[i][i]) for i in range(n)], _identity(n)

    if n <= 4:
        coeffs = char_poly(M)
        raw_evs = find_roots(coeffs)
        raw_evs.sort(reverse=True)

        unique_evs: List[float] = []
        for ev in raw_evs:
            if not any(abs(ev - u) < 1e-4 for u in unique_evs):
                unique_evs.append(ev)

        ev_list: List[float] = []
        vec_list: List[Vector] = []
        for ev in unique_evs:
            for vec in eigenspace_basis(M, ev):
                ev_list.append(ev)
                vec_list.append(vec)

        pairs = sorted(zip(ev_list, vec_list), key=lambda p: p[0], reverse=True)[:n]
        ev_list = [p[0] for p in pairs]
        vec_list = [p[1] for p in pairs]

        while len(ev_list) < n:
            ev_list.append(ev_list[0] if ev_list else 0.0)
            vec_list.append([0.0] * n)

        return ev_list, mat_transpose(vec_list)
    else:
        arr = np.array(M, dtype=float)
        vals, vecs = np.linalg.eig(arr)
        vals, vecs = vals.real, vecs.real
        idx = np.argsort(vals)[::-1]
        return vals[idx].tolist(), vecs[:, idx].tolist()

# ──────────────────────────────────────────────

def check_diagonalizable(evs: List[float], P: Matrix) -> Tuple[bool, str]:
    """
    Kiểm tra ma trận có chéo hóa được không dựa vào ma trận vector riêng P.

    Đầu vào:
    - evs: list giá trị riêng
    - P: ma trận vector riêng (các cột là vector riêng)

    Đầu ra:
    - (is_diag, message): True nếu chéo hóa được, kèm thông báo giải thích
    """
    n = len(evs)
    if all(abs(evs[i] - evs[j]) > 1e-4 for i in range(n) for j in range(i + 1, n)):
        return True, "Ma trận thỏa mãn điều kiện đủ: Có n giá trị riêng phân biệt."

    info = rank_and_basis(P, eps=_EPS)
    if info["rank"] == n:
        return True, "Ma trận có đủ n vector riêng độc lập tuyến tính."
    return False, "Ma trận không chéo hóa được: Không đủ n vector riêng độc lập tuyến tính."

def diagonalize(A: Sequence[Sequence[Number]]) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Chéo hóa ma trận A thành dạng A = P * D * P^-1.

    Đầu vào:
    - A: ma trận vuông kích thước n x n

    Đầu ra:
    - (P, D, P_inv): ma trận đổi cơ sở, ma trận đường chéo và nghịch đảo của P
    """
    mat = _validate_matrix(A)
    n = len(mat)
    if len(mat[0]) != n:
        raise ValueError("Ma trận phải là ma trận vuông.")

    evs, P = find_eigen(mat)
    D = [[evs[i] if i == j else 0.0 for j in range(n)] for i in range(n)]

    ok, msg = check_diagonalizable(evs, P)
    if not ok:
        raise ValueError(f"Lỗi: {msg}")

    P_inv = gauss_inverse(P)
    return P, D, P_inv

# ──────────────────────────────────────────────

def verify(A: Matrix, P: Matrix, D: Matrix, P_inv: Matrix) -> bool:
    """
    Kiểm chứng kết quả chéo hóa: tính sai số ||A - P*D*P^-1||max và so sánh giá trị
    riêng với NumPy.

    Đầu vào:
    - A: ma trận gốc
    - P: ma trận đổi cơ sở
    - D: ma trận đường chéo
    - P_inv: nghịch đảo của P

    Đầu ra:
    - True nếu cả hai sai số nhỏ hơn 1e-4, ngược lại False
    """
    arrA = np.array(A, dtype=float)
    rebuilt = np.array(P) @ np.array(D) @ np.array(P_inv)
    err_recon = float(np.max(np.abs(arrA - rebuilt)))
    print("* Kiểm chứng chéo hóa")
    print(f"Sai số tái cấu trúc ||A - P*D*P^-1||max = {err_recon:.2e}")
    print("Cài đặt chéo hóa đúng (A ~ P*D*P^-1)." if err_recon < 1e-4
          else "Cài đặt chéo hóa có thể có sai số cao.")

    np_evs = np.sort(np.linalg.eigvals(arrA).real)[::-1]
    my_evs = np.sort(np.diag(np.array(D)))[::-1]
    err_ev = float(np.max(np.abs(my_evs - np_evs)))
    print(f"Sai số giá trị riêng so với NumPy = {err_ev:.2e}")
    return err_recon < 1e-4 and err_ev < 1e-4

def demo_diagonalize(A: Sequence[Sequence[Number]]) -> bool:
    """
    Thực hiện chéo hóa ma trận A và in kết quả ra màn hình.

    Đầu vào:
    - A: ma trận vuông kích thước n x n

    Đầu ra:
    - True nếu kết quả chéo hóa đúng (sai số nhỏ hơn ngưỡng), ngược lại False
    """
    P, D, P_inv = diagonalize(A)
    mat = _validate_matrix(A)
    print("\n* Kết quả chéo hóa")
    print("Ma trận đổi cơ sở P:")
    for row in P:
        print("  " + " ".join(f"{v:8.4f}" for v in row))
    print("\nMa trận đường chéo D:")
    for row in D:
        print("  " + " ".join(f"{v:8.4f}" for v in row))
    print("\nMa trận P^-1:")
    for row in P_inv:
        print("  " + " ".join(f"{v:8.4f}" for v in row))
    return verify(mat, P, D, P_inv)

# ──────────────────────────────────────────────

def main() -> None:
    try:
        size = int(input("Nhập kích thước ma trận vuông n = "))
        print(f"Nhập ma trận {size}x{size} (mỗi dòng cách nhau bởi dấu Enter, "
              "các phần tử cách nhau bởi khoảng trắng):")
        rows = []
        for i in range(size):
            row = list(map(float, input().strip().split()))
            if len(row) != size:
                print(f"Lỗi: Dòng {i + 1} không có đủ {size} phần tử.")
                return
            rows.append(row)
        demo_diagonalize(rows)
    except ValueError as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
