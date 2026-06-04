"""
Advanced Methods
================
List-first implementations for Part 2 advanced regression methods.
"""

from __future__ import annotations

import math


def _to_2d_list(X):
    if hasattr(X, "tolist"):
        X = X.tolist()
    if not X:
        return []
    if isinstance(X[0], (int, float)):
        return [[float(value)] for value in X]
    return [[float(value) for value in row] for row in X]


def _to_1d_list(y):
    if hasattr(y, "tolist"):
        y = y.tolist()
    if y and isinstance(y[0], list):
        return [float(row[0]) for row in y]
    return [float(value) for value in y]


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _transpose(matrix):
    return [list(col) for col in zip(*matrix)] if matrix else []


def _matmul(left, right):
    right_t = _transpose(right)
    return [[_dot(row, col) for col in right_t] for row in left]


def _matvec(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _add_diagonal(matrix, value):
    result = [row[:] for row in matrix]
    for idx in range(len(result)):
        result[idx][idx] += value
    return result


def _cholesky_decompose(matrix):
    n = len(matrix)
    lower = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            correction = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                value = matrix[i][i] - correction
                if value <= 0:
                    value = 1e-12
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = (matrix[i][j] - correction) / lower[j][j]

    return lower


def _forward_substitution(lower, rhs):
    result = [0.0 for _ in rhs]
    for i, value in enumerate(rhs):
        correction = sum(lower[i][j] * result[j] for j in range(i))
        result[i] = (value - correction) / lower[i][i]
    return result


def _backward_substitution_from_lower(lower, rhs):
    n = len(rhs)
    result = [0.0 for _ in rhs]
    for i in range(n - 1, -1, -1):
        correction = sum(lower[j][i] * result[j] for j in range(i + 1, n))
        result[i] = (rhs[i] - correction) / lower[i][i]
    return result


def _cholesky_solve(matrix, rhs):
    lower = _cholesky_decompose(matrix)
    middle = _forward_substitution(lower, rhs)
    return _backward_substitution_from_lower(lower, middle)


def _cholesky_inverse(matrix):
    lower = _cholesky_decompose(matrix)
    inverse_columns = []
    for unit in _identity(len(matrix)):
        middle = _forward_substitution(lower, unit)
        inverse_columns.append(_backward_substitution_from_lower(lower, middle))
    return _transpose(inverse_columns)


def _with_intercept(X):
    return [[1.0, *row] for row in X]


def _kernel_value(
    left,
    right,
    kernel: str = "rbf",
    gamma: float | None = None,
    degree: int = 3,
    coef0: float = 1.0,
):
    gamma = float(gamma) if gamma is not None else 1.0 / max(len(left), 1)

    if kernel == "linear":
        return _dot(left, right)

    if kernel in {"poly", "polynomial"}:
        return (gamma * _dot(left, right) + coef0) ** degree

    if kernel == "rbf":
        squared_distance = sum((a - b) ** 2 for a, b in zip(left, right))
        return math.exp(-gamma * squared_distance)

    raise ValueError("kernel must be one of: 'rbf', 'linear', 'poly', 'polynomial'.")


def _kernel_matrix(
    X_left,
    X_right,
    kernel: str = "rbf",
    gamma: float | None = None,
    degree: int = 3,
    coef0: float = 1.0,
):
    return [
        [
            _kernel_value(row_left, row_right, kernel, gamma, degree, coef0)
            for row_right in X_right
        ]
        for row_left in X_left
    ]


class KernelRegression:
    """Kernel Ridge Regression using the Gram matrix formula."""

    def __init__(
        self,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        bandwidth: float | None = None,
    ):
        self.alpha = float(alpha)
        self.kernel = kernel
        self.gamma = gamma if bandwidth is None else 1.0 / (2.0 * bandwidth**2)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.X_fit_ = None
        self.dual_coef_ = None

    def fit(self, X, y):
        X_rows = _to_2d_list(X)
        y_values = _to_1d_list(y)

        if len(X_rows) != len(y_values):
            raise ValueError("X and y must have the same number of rows.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive for Kernel Ridge Regression.")

        gram = _kernel_matrix(
            X_rows,
            X_rows,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )
        system = _add_diagonal(gram, self.alpha)
        self.dual_coef_ = _cholesky_solve(system, y_values)
        self.X_fit_ = X_rows
        return self

    def predict(self, X):
        if self.X_fit_ is None or self.dual_coef_ is None:
            raise ValueError("KernelRegression is not fitted.")

        gram = _kernel_matrix(
            _to_2d_list(X),
            self.X_fit_,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )
        return _matvec(gram, self.dual_coef_)


def kernel_ridge_fit(
    X_train,
    y_train,
    X_test=None,
    lambda_kernel: float | None = None,
    alpha: float | None = None,
    kernel: str = "rbf",
    gamma: float | None = None,
    degree: int = 3,
    coef0: float = 1.0,
):
    """Fit Kernel Ridge Regression and return handover artifacts."""
    regularization = 1.0 if alpha is None and lambda_kernel is None else (
        alpha if alpha is not None else lambda_kernel
    )
    model = KernelRegression(
        alpha=float(regularization),
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
    ).fit(X_train, y_train)

    return {
        "model": model,
        "predictions_train": model.predict(X_train),
        "predictions_test": None if X_test is None else model.predict(X_test),
        "params": {
            "alpha": float(regularization),
            "lambda_kernel": float(regularization),
            "kernel": kernel,
            "gamma": gamma,
            "degree": int(degree),
            "coef0": float(coef0),
        },
    }


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with Gaussian prior.

    This implements the conjugate posterior:
    S_n = (S_0^-1 + beta X^T X)^-1
    m_n = S_n(S_0^-1 m_0 + beta X^T y)
    """

    def __init__(
        self,
        prior_precision: float = 1.0,
        noise_precision: float = 1.0,
        fit_intercept: bool = True,
    ):
        self.prior_precision = float(prior_precision)
        self.noise_precision = float(noise_precision)
        self.fit_intercept = bool(fit_intercept)
        self.posterior_mean = None
        self.posterior_cov = None

    def fit(self, X, y, prior_mean=None):
        X_rows = _to_2d_list(X)
        y_values = _to_1d_list(y)
        X_design = _with_intercept(X_rows) if self.fit_intercept else X_rows

        if len(X_design) != len(y_values):
            raise ValueError("X and y must have the same number of rows.")

        n_features = len(X_design[0]) if X_design else 0
        if prior_mean is None:
            prior_mean = [0.0 for _ in range(n_features)]
        else:
            prior_mean = _to_1d_list(prior_mean)
        if len(prior_mean) != n_features:
            raise ValueError("prior_mean must match the number of coefficients.")

        X_t = _transpose(X_design)
        xtx = _matmul(X_t, X_design)
        precision = [
            [
                self.prior_precision * (1.0 if row == col else 0.0)
                + self.noise_precision * xtx[row][col]
                for col in range(n_features)
            ]
            for row in range(n_features)
        ]
        xty = _matvec(X_t, y_values)
        rhs = [
            self.prior_precision * prior_mean[idx] + self.noise_precision * xty[idx]
            for idx in range(n_features)
        ]

        self.posterior_mean = _cholesky_solve(precision, rhs)
        self.posterior_cov = _cholesky_inverse(precision)
        return self

    def predict(self, X, return_std: bool = False):
        if self.posterior_mean is None or self.posterior_cov is None:
            raise ValueError("BayesianLinearRegression is not fitted.")

        X_rows = _to_2d_list(X)
        X_design = _with_intercept(X_rows) if self.fit_intercept else X_rows
        y_pred = _matvec(X_design, self.posterior_mean)

        if not return_std:
            return y_pred

        std_values = []
        for row in X_design:
            cov_row = _matvec(self.posterior_cov, row)
            variance = 1.0 / self.noise_precision + _dot(row, cov_row)
            std_values.append(math.sqrt(max(variance, 0.0)))

        return y_pred, std_values
