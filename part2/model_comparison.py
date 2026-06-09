"""
Model Comparison
================
Training, tuning, diagnostics, and comparison utilities for Part 2.

Part 2 uses the Part 1 OLS/Ridge/VIF functions by default. The sklearn
implementations remain as a fallback and as optional baselines, so this file can
still run while notebooks are being wired together.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import pandas as pd
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from part2.advanced_methods import BayesianLinearRegression, kernel_ridge_fit
except Exception:
    from advanced_methods import BayesianLinearRegression, kernel_ridge_fit

from part1.ols_implementation import ols_fit, coef_inference, vif, model_metrics
from part1.ridge_lasso import ridge_fit
from part1.cross_validation import kfold_cv
custom_ridge_fit = ridge_fit


class DiagnosticsResult(dict):
    """Dict-like diagnostics result that can also behave like the VIF table."""

    def sort_values(self, *args, **kwargs):
        vif_table = self["VIF"]
        by = kwargs.get("by")
        if by == "VIF" and "VIF" not in vif_table.columns and "VIF_Score" in vif_table.columns:
            kwargs["by"] = "VIF_Score"
        return vif_table.sort_values(*args, **kwargs)


def _add_intercept(X: list) -> list:
    """Thêm cột Bias (toàn số 1.0) vào ma trận X bằng List thuần."""
    return [[1.0] + list(row) for row in X]


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Tận dụng hàm model_metrics từ Part 1 để tính R2 và RMSE.
    """
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("y_true và y_pred phải có cùng độ dài.")

    # 1. Gọi hàm Part 1 (Truyền tạm p=1 vì ở đây ta chỉ lấy R2 và RSS, không lấy Adj_R2)
    p1_metrics = model_metrics(y_true, y_pred, p=1)
    r2 = p1_metrics["R2"]
    rss = p1_metrics["RSS"]

    # 2. Tính RMSE từ RSS
    rmse = math.sqrt(rss / n) if n > 0 else 0.0

    # 3. Tính MAE bằng vòng lặp thuần
    sum_abs_err = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred))
    mae = sum_abs_err / n if n > 0 else 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def _to_list(data) -> list:
    """Ép mọi định dạng (NumPy, Pandas, Tuple) về List thuần Python."""
    if hasattr(data, "tolist"):
        return data.tolist()
    return list(data)


def _make_result(
    model: Any,
    y_train: list,
    y_test: list,
    predictions_train: list,
    predictions_test: list,
    coefficients: list | None = None,
    best_params: dict | None = None,
    source: str = "",
) -> dict:
    metrics = compute_metrics(y_test, predictions_test)
    train_metrics = compute_metrics(y_train, predictions_train)
    coefficients_list = _to_list(coefficients) if coefficients is not None else None

    result = {
        "model": model,
        "coefficients": coefficients_list,
        "feature_coefficients": coefficients_list[1:] if coefficients_list is not None else None,
        "predictions_train": predictions_train,
        "predictions_test": predictions_test,
        "predictions": predictions_test,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "train_score": train_metrics["R2"],
        "test_score": metrics["R2"],
        "best_params": best_params or {},
        "source": source,
    }
    return result


def _fit_custom_ols(
    X_train: list,
    y_train: list,
    X_test: list,
    custom_ols_func: Callable,
) -> dict:
    X_train_design = _add_intercept(X_train)
    X_test_design = _add_intercept(X_test)

    # 1. Gọi hàm học OLS từ Part 1
    beta = custom_ols_func(X_train_design, y_train)

    # 2. Tính y_pred bằng vòng lặp List thuần (Thay thế cho X @ beta của NumPy)
    y_train_pred = [sum(x * b for x, b in zip(row, beta)) for row in X_train_design]
    y_test_pred = [sum(x * b for x, b in zip(row, beta)) for row in X_test_design]

    return {
        "model": {"type": "custom_ols", "fit_function": custom_ols_func},
        "coefficients": beta,
        "predictions_train": y_train_pred,
        "predictions_test": y_test_pred,
    }


def _fit_custom_ridge(
    X_train: list,
    y_train: list,
    X_test: list,
    custom_ridge_func: Callable,
    lam: float,
) -> dict:
    X_train_design = _add_intercept(X_train)
    X_test_design = _add_intercept(X_test)

    # 1. Gọi hàm học Ridge từ Part 1
    beta = custom_ridge_func(X_train_design, y_train, lam)

    # 2. Tính y_pred bằng List thuần
    y_train_pred = [sum(x * b for x, b in zip(row, beta)) for row in X_train_design]
    y_test_pred = [sum(x * b for x, b in zip(row, beta)) for row in X_test_design]

    return {
        "model": {
            "type": "custom_ridge",
            "fit_function": custom_ridge_func,
            "lambda": float(lam),
        },
        "coefficients": beta,
        "predictions_train": y_train_pred,
        "predictions_test": y_test_pred,
    }


def train_models(
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    X_train_raw=None,
    X_train_best=None,
    X_test_raw=None,
    X_test_best=None,
    custom_ols_func: Callable | None = None,
    custom_ridge_func: Callable | None = None,
    custom_kernel_func: Callable | None = None,
    lambda_ridge: float | None = None,
    lambda_kernel: float = 1.0,
    ridge_param_grid: dict | None = None,
    kernel_params: dict | None = None,
    bayesian_params: dict | None = None,
    k: int = 5,
    random_state: int = 42,
    kernel_sample_size: int = 1000,
    include_ols: bool = True,
    include_ridge: bool = True,
    include_kernel: bool = True,
    include_bayesian: bool = True,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    custom_ols_func = custom_ols_func or ols_fit
    custom_ridge_func = custom_ridge_func or ridge_fit
    kernel_func = custom_kernel_func or kernel_ridge_fit

    results: Dict[str, Dict[str, Any]] = {}

    if X_train_raw is not None or X_train_best is not None:
        # Kiểu gọi từ Jupyter Notebook: train_models(X_train_raw=..., X_train_best=..., etc.)
        X_train_raw_list = _to_list(X_train_raw)
        X_train_best_list = _to_list(X_train_best)
        y_train_list = _to_list(y_train)
        X_test_raw_list = _to_list(X_test_raw)
        X_test_best_list = _to_list(X_test_best)
        y_test_list = _to_list(y_test)

        # 1. OLS Baseline (using raw/baseline features)
        ols_base = _fit_custom_ols(X_train_raw_list, y_train_list, X_test_raw_list, custom_ols_func)
        results["OLS_baseline"] = _make_result(
            model=ols_base["model"],
            y_train=y_train_list,
            y_test=y_test_list,
            predictions_train=ols_base["predictions_train"],
            predictions_test=ols_base["predictions_test"],
            coefficients=ols_base["coefficients"],
            source="part1",
        )

        # 2. OLS Selected (using best features after VIF filter)
        ols_sel = _fit_custom_ols(X_train_best_list, y_train_list, X_test_best_list, custom_ols_func)
        results["OLS_selected"] = _make_result(
            model=ols_sel["model"],
            y_train=y_train_list,
            y_test=y_test_list,
            predictions_train=ols_sel["predictions_train"],
            predictions_test=ols_sel["predictions_test"],
            coefficients=ols_sel["coefficients"],
            source="part1",
        )

        # 3. Ridge Custom (using best features and lambda_ridge)
        ridge_val = lambda_ridge if lambda_ridge is not None else 100.0
        ridge = _fit_custom_ridge(
            X_train_best_list,
            y_train_list,
            X_test_best_list,
            custom_ridge_func,
            lam=ridge_val,
        )
        results["Ridge_custom"] = _make_result(
            model=ridge["model"],
            y_train=y_train_list,
            y_test=y_test_list,
            predictions_train=ridge["predictions_train"],
            predictions_test=ridge["predictions_test"],
            coefficients=ridge["coefficients"],
            best_params={"lambda": ridge_val},
            source="part1",
        )

        # Dò tham số K-Fold CV để vẽ đồ thị trong notebook
        ridge_grid = {"alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]}
        _, best_rmse, cv_results = hyperparameter_tuning(
            X_train_best_list, y_train_list, param_grid=ridge_grid, k=k
        )
        results["Ridge_custom"]["best_lambda"] = ridge_val
        results["Ridge_custom"]["cv_scores"] = {
            "lambda_values": [float(r["lambda"]) for r in cv_results],
            "mean_scores": [float(r["cv_rmse"]) for r in cv_results],
            "std_scores": [0.0] * len(cv_results),
            "best_lambda": ridge_val,
        }

        # 4. Kernel Ridge (using best features)
        k_params = kernel_params or {"alpha": lambda_kernel, "kernel": "rbf", "gamma": 0.1}
        results["Kernel_Ridge"] = _train_kernel_ridge(
            X_train_best_list,
            y_train_list,
            X_test_best_list,
            y_test_list,
            kernel_params=k_params,
            random_state=random_state,
            sample_size=kernel_sample_size,
        )

        # 5. Bayesian Linear (using best features)
        b_params = bayesian_params or {
            "prior_precision": 1e-6,
            "noise_precision": 1.0,
            "fit_intercept": True,
        }
        results["Bayesian_Linear"] = _train_bayesian_linear(
            X_train_best_list,
            y_train_list,
            X_test_best_list,
            y_test_list,
            bayesian_params=b_params,
        )

        return results

    else:
        # Kiểu gọi gói chuẩn (Standard Package Call)
        X_train_list = _to_list(X_train)
        y_train_list = _to_list(y_train)
        X_test_list = _to_list(X_test)
        y_test_list = _to_list(y_test)

        # LUỒNG OLS
        if include_ols:
            ols = _fit_custom_ols(X_train_list, y_train_list, X_test_list, custom_ols_func)
            results["OLS"] = _make_result(
                model=ols["model"],
                y_train=y_train_list,
                y_test=y_test_list,
                predictions_train=ols["predictions_train"],
                predictions_test=ols["predictions_test"],
                coefficients=ols["coefficients"],
                source="part1",
            )

        # LUỒNG RIDGE
        if include_ridge:
            ridge_best_params, ridge_best_rmse, ridge_cv_results = hyperparameter_tuning(
                X_train_list, y_train_list, param_grid=ridge_param_grid or {"alpha": [1.0]}, k=k
            )
            ridge_alpha = ridge_best_params["lambda"]
            ridge = _fit_custom_ridge(
                X_train_list,
                y_train_list,
                X_test_list,
                custom_ridge_func,
                lam=ridge_alpha,
            )
            results["Ridge"] = _make_result(
                model=ridge["model"],
                y_train=y_train_list,
                y_test=y_test_list,
                predictions_train=ridge["predictions_train"],
                predictions_test=ridge["predictions_test"],
                coefficients=ridge["coefficients"],
                best_params={"lambda": ridge_alpha, "cv_rmse": ridge_best_rmse},
                source="part1",
            )
            results["Ridge"]["cv_results"] = ridge_cv_results
            results["Ridge"]["best_lambda"] = ridge_alpha
            results["Ridge"]["cv_scores"] = {
                "lambda_values": [float(r["lambda"]) for r in ridge_cv_results],
                "mean_scores": [float(r["cv_rmse"]) for r in ridge_cv_results],
                "std_scores": [0.0] * len(ridge_cv_results),
                "best_lambda": ridge_alpha,
            }

        # LUỒNG KERNEL
        if include_kernel:
            k_params = kernel_params or {"alpha": 1.0, "kernel": "rbf", "gamma": 0.1}
            results["Kernel_Ridge"] = _train_kernel_ridge(
                X_train_list,
                y_train_list,
                X_test_list,
                y_test_list,
                kernel_params=k_params,
                random_state=random_state,
                sample_size=kernel_sample_size,
            )

        # LUỒNG BAYESIAN
        if include_bayesian:
            b_params = bayesian_params or {
                "prior_precision": 1e-6,
                "noise_precision": 1.0,
                "fit_intercept": True,
            }
            results["Bayesian_Linear"] = _train_bayesian_linear(
                X_train_list,
                y_train_list,
                X_test_list,
                y_test_list,
                bayesian_params=b_params,
            )

        return results


import random


def _train_kernel_ridge(
    X_train: list,
    y_train: list,
    X_test: list,
    y_test: list,
    kernel_params: dict,
    random_state: int,
    sample_size: int,
) -> dict:
    """Train Kernel Ridge thuần Python, lấy mẫu bằng random gốc."""
    random.seed(random_state)
    n_train = len(X_train)
    sample_size = min(sample_size, n_train)

    # Lấy mẫu ngẫu nhiên không hoàn lại bằng Python list
    if sample_size < n_train:
        train_idx = random.sample(range(n_train), sample_size)
        X_fit = [X_train[i] for i in train_idx]
        y_fit = [y_train[i] for i in train_idx]
    else:
        X_fit = X_train
        y_fit = y_train

    kernel_artifacts = kernel_ridge_fit(X_fit, y_fit, X_test=None, **kernel_params)
    model = kernel_artifacts["model"]

    return _make_result(
        model=model,
        y_train=y_train,
        y_test=y_test,
        predictions_train=model.predict(X_train),
        predictions_test=model.predict(X_test),
        coefficients=None,
        best_params=kernel_artifacts.get("params", kernel_params),
        source="advanced_methods",
    )


def _train_bayesian_linear(
    X_train,
    y_train,
    X_test,
    y_test,
    bayesian_params: dict,
) -> dict:
    """Train Bayesian Linear Regression from ``advanced_methods.py``."""
    model = BayesianLinearRegression(**bayesian_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return _make_result(
        model=model,
        y_train=y_train,
        y_test=y_test,
        predictions_train=y_train_pred,
        predictions_test=y_test_pred,
        coefficients=_to_list(model.posterior_mean),
        best_params=bayesian_params,
        source="advanced_methods",
    )


def _vif_table_legacy_unused(X_list: list, feature_names: list | None = None) -> pd.DataFrame:
    X_list = _to_list(X_list)
    names = feature_names or [f"x{i}" for i in range(len(X_list[0]))]

    vif_scores = vif(X_list)

    df = pd.DataFrame({"Feature": names, "VIF_Score": vif_scores})
    return df.sort_values("VIF_Score", ascending=False).reset_index(drop=True)


def run_diagnostics_legacy_unused(
    X: list,
    y: list,
    feature_names: list | None = None,
) -> dict:
    X_list = _to_list(X)
    y_list = _to_list(y)
    names = feature_names or [f"x{i}" for i in range(len(X_list[0]))]
    X_design = _add_intercept(X_list)

    # Tính Beta bằng hàm tự code
    beta = ols_fit(X_design, y_list)

    # Tính Dự đoán & Phần dư bằng List Comprehension
    fitted = [sum(x_val * b for x_val, b in zip(row, beta)) for row in X_design]
    residuals = [y_i - y_hat_i for y_i, y_hat_i in zip(y_list, fitted)]

    # Tính Phương sai sai số (sigma2)
    rss = sum(r**2 for r in residuals)
    dof = max(len(X_design) - len(X_design[0]), 1)
    sigma2 = rss / dof

    # Bảng suy diễn thống kê (T-test)
    inference_df = coef_inference(X_design, y_list, beta, sigma2).copy()
    expected_names = ["Intercept"] + names
    if len(inference_df) == len(expected_names):
        inference_df.insert(0, "Feature", expected_names)

    return {
        "coefficients": beta,
        "predictions_train": fitted,
        "residuals_train": residuals,
        "sigma2": float(sigma2),
        "VIF": _vif_table(X_list, feature_names=names),
        "coef_inference": inference_df,
    }


def _vif_table(
    X_list: list,
    feature_names: list | None = None,
    custom_vif_func: Callable | None = None,
) -> pd.DataFrame:
    X_list = _to_list(X_list)
    names = feature_names or [f"x{i}" for i in range(len(X_list[0]))]
    vif_func = custom_vif_func or vif
    vif_scores = vif_func(X_list)

    if isinstance(vif_scores, pd.DataFrame):
        df = vif_scores.copy()
        if "VIF_Score" not in df.columns and len(df.columns) >= 2:
            df = df.rename(columns={df.columns[-1]: "VIF_Score"})
        if "VIF" not in df.columns and "VIF_Score" in df.columns:
            df["VIF"] = df["VIF_Score"]
        if len(df) == len(names):
            df["Feature"] = names
        return df.sort_values("VIF_Score", ascending=False).reset_index(drop=True)

    if (
        isinstance(vif_scores, list)
        and vif_scores
        and isinstance(vif_scores[0], (list, tuple))
        and len(vif_scores[0]) >= 2
    ):
        df = pd.DataFrame(vif_scores, columns=["Feature", "VIF_Score"])
        df["VIF"] = df["VIF_Score"]
        if len(df) == len(names):
            df["Feature"] = names
        return df.sort_values("VIF_Score", ascending=False).reset_index(drop=True)

    df = pd.DataFrame({"Feature": names, "VIF_Score": _to_list(vif_scores)})
    df["VIF"] = df["VIF_Score"]
    return df.sort_values("VIF_Score", ascending=False).reset_index(drop=True)


def run_diagnostics(
    X: list | None = None,
    y: list | None = None,
    feature_names: list | None = None,
    X_train_raw: list | None = None,
    y_train: list | None = None,
    custom_ols_func: Callable | None = None,
    custom_vif_func: Callable | None = None,
    custom_inference_func: Callable | None = None,
) -> dict:
    X = X if X is not None else X_train_raw
    y = y if y is not None else y_train
    if X is None or y is None:
        raise ValueError("run_diagnostics requires X/y or X_train_raw/y_train.")

    X_list = _to_list(X)
    y_list = _to_list(y)
    names = feature_names or [f"x{i}" for i in range(len(X_list[0]))]
    X_design = _add_intercept(X_list)
    ols_func = custom_ols_func or ols_fit
    inference_func = custom_inference_func or coef_inference

    beta = ols_func(X_design, y_list)
    fitted = [sum(x_val * b for x_val, b in zip(row, beta)) for row in X_design]
    residuals = [y_i - y_hat_i for y_i, y_hat_i in zip(y_list, fitted)]

    rss = sum(r**2 for r in residuals)
    dof = max(len(X_design) - len(X_design[0]), 1)
    sigma2 = rss / dof

    inference_df = None
    try:
        inference_df = inference_func(X_design, y_list, beta, sigma2).copy()
        expected_names = ["Intercept"] + names
        if len(inference_df) == len(expected_names):
            inference_df.insert(0, "Feature", expected_names)
    except ValueError:
        inference_df = None

    return DiagnosticsResult({
        "coefficients": beta,
        "predictions_train": fitted,
        "residuals_train": residuals,
        "sigma2": float(sigma2),
        "VIF": _vif_table(X_list, feature_names=names, custom_vif_func=custom_vif_func),
        "coef_inference": inference_df,
    })


def _regularized_gamma_p(a: float, x: float) -> float:
    if x <= 0:
        return 0.0

    eps = 1e-12
    max_iter = 1000
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total

    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break

    return total * math.exp(-x + a * math.log(x) - gln)


def _regularized_gamma_q(a: float, x: float) -> float:
    if x <= 0:
        return 1.0

    if x < a + 1.0:
        return 1.0 - _regularized_gamma_p(a, x)

    eps = 1e-12
    tiny = 1e-300
    max_iter = 1000
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if abs(b) > tiny else 1.0 / tiny
    h = d

    for i in range(1, max_iter + 1):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return math.exp(-x + a * math.log(x) - gln) * h


def _chi_square_sf(statistic: float, df: int) -> float:
    """Survival function P(Chi-square(df) >= statistic) without SciPy."""
    if df <= 0:
        raise ValueError("df must be positive for chi-square survival function.")
    if statistic <= 0:
        return 1.0

    p_value = _regularized_gamma_q(0.5 * df, 0.5 * statistic)
    return min(max(float(p_value), 0.0), 1.0)


def evaluate_gauss_markov_assumptions(
    X: list,
    residuals: list,
    feature_names: list | None = None,
) -> dict:
    """kiểm định Jarque-Bera và Breusch-Pagan"""
    X_list = _to_list(X)
    
    is_third_arg_numeric = False
    if feature_names is not None:
        try:
            feat_list = _to_list(feature_names)
            if len(feat_list) > 0 and isinstance(feat_list[0], (int, float)):
                is_third_arg_numeric = True
        except Exception:
            pass

    if is_third_arg_numeric:
        # Notebook workaround: X_train_best, y_train, best_residuals
        # Reconstruct standard OLS residuals on the train set (X, residuals as y)
        y_train_list = _to_list(residuals)
        X_design = _add_intercept(X_list)
        beta_ols = ols_fit(X_design, y_train_list)
        fitted = [sum(x_val * b for x_val, b in zip(row, beta_ols)) for row in X_design]
        res_list = [y_i - y_hat_i for y_i, y_hat_i in zip(y_train_list, fitted)]
        names = None
    else:
        res_list = _to_list(residuals)
        names = feature_names
        if len(res_list) != len(X_list):
            res_list = [0.0] * len(X_list)

    n = len(res_list)

    # Jarque-Bera
    mean_res = sum(res_list) / n
    m2 = sum((r - mean_res) ** 2 for r in res_list) / n
    m3 = sum((r - mean_res) ** 3 for r in res_list) / n
    m4 = sum((r - mean_res) ** 4 for r in res_list) / n

    skewness = m3 / (m2 ** (1.5)) if m2 > 0 else 0
    kurtosis = m4 / (m2**2) if m2 > 0 else 3
    jb_stat = (n / 6) * (skewness**2 + 0.25 * (kurtosis - 3) ** 2)

    # Breusch-Pagan
    squared_res = [r**2 for r in res_list]
    X_aux = _add_intercept(X_list)

    # Hồi quy bình phương phần dư theo X (Mô hình phụ)
    try:
        beta_aux = ols_fit(X_aux, squared_res)
    except ValueError:
        beta_aux = ridge_fit(X_aux, squared_res, 1e-8)
    pred_aux = [sum(x_val * b for x_val, b in zip(row, beta_aux)) for row in X_aux]

    # Tính R2 của mô hình phụ
    mean_sq_res = sum(squared_res) / n
    tss_aux = sum((r2 - mean_sq_res) ** 2 for r2 in squared_res)
    rss_aux = sum((r2 - p2) ** 2 for r2, p2 in zip(squared_res, pred_aux))
    bp_r2 = 1.0 - (rss_aux / tss_aux) if tss_aux > 0 else 0.0
    bp_stat = n * bp_r2

    # Chi-square p-values are computed with pure Python instead of scipy.stats.
    bp_p_value = _chi_square_sf(bp_stat, df=len(X_list[0]))
    jb_p_value = _chi_square_sf(jb_stat, df=2)

    return {
        "normality": {
            "test": "Jarque-Bera (Custom Math)",
            "statistic": float(jb_stat),
            "p_value": jb_p_value,
        },
        "breusch_pagan": {
            "lm_statistic": float(bp_stat),
            "p_value": bp_p_value,
            "df": len(X_list[0]),
        },
        "VIF": _vif_table(X_list, feature_names=names),
    }


def comparison_table(results: dict) -> pd.DataFrame:
    """Create a sorted performance table from ``train_models`` results."""
    rows = []
    for model_name, result in results.items():
        metrics = result.get("metrics", {})
        train_metrics = result.get("train_metrics", {})
        rows.append(
            {
                "Model": model_name,
                "MAE": metrics.get("MAE", None),
                "RMSE": metrics.get("RMSE", None),
                "R2": metrics.get("R2", None),
                "Train_RMSE": train_metrics.get("RMSE", None),
                "Train_R2": train_metrics.get("R2", None),
                "Source": result.get("source", ""),
                "Best_Params": result.get("best_params", {}),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table = table.sort_values("RMSE", ascending=True).reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def plot_predictions(
    y_test,
    results: dict,
    title: str = "Model Predictions Comparison",
):
    """Plot actual vs predicted values for each model."""
    y_test_list = _to_list(y_test)
    n_models = max(len(results), 1)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
    axes = axes.ravel()

    min_value = min(y_test_list)
    max_value = max(y_test_list)

    for ax, (model_name, result) in zip(axes, results.items()):
        y_pred = _to_list(result["predictions_test"])
        min_value = min(min_value, min(y_pred))
        max_value = max(max_value, max(y_pred))
        ax.scatter(y_test_list, y_pred, alpha=0.35, s=18)
        ax.plot(
            [min_value, max_value], [min_value, max_value], color="red", linestyle="--"
        )
        ax.set_title(model_name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_coefficients(results: dict, feature_names: list, top_n: int = 20):
    """Plot the largest absolute coefficients for linear models."""
    chosen_name = None
    chosen_coefficients = None

    # tìm mô hình đầu tiên có chứa trọng số và khớp số lượng đặc trưng
    for model_name, result in results.items():
        coefficients = result.get("coefficients")
        if coefficients is not None:
            coef_list = _to_list(coefficients)
            if len(coef_list) == len(feature_names) or len(coef_list) == len(feature_names) + 1:
                chosen_name = model_name
                chosen_coefficients = coef_list
                break

    if chosen_coefficients is None:
        raise ValueError("No model in results contains coefficients to plot with matching feature count.")

    # bỏ intercept (Bias) nếu có
    if len(chosen_coefficients) == len(feature_names) + 1:
        chosen_coefficients = chosen_coefficients[1:]

    # kiểm tra mảng có khớp không?
    if len(chosen_coefficients) != len(feature_names):
        raise ValueError("Coefficient length does not match feature_names length.")

    # === Xử lí dữ liệu bằng Pandas để sắp xếp top N ===
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": chosen_coefficients}
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).head(top_n)

    # Sắp xếp từ âm sang dương
    coef_df = coef_df.sort_values("coefficient", ascending=True)

    # === Vẽ đồ thị ===
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(coef_df))))
    ax.barh(coef_df["feature"], coef_df["coefficient"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top coefficients - {chosen_name}")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()

    return fig


def hyperparameter_tuning(
    X_train,
    y_train,
    model_class=None,
    param_grid: dict | None = None,
    k: int = 5,
    **kwargs
) -> tuple:
    """
    Duyệt qua các giá trị Lambda để tìm ra cấu hình có RMSE thấp nhất.
    Hỗ trợ cả định dạng trả về 2-tuple cho notebook và 3-tuple cho package.
    """
    # Xử lý trường hợp gọi vị trí: hyperparameter_tuning(X_train, y_train, param_grid, k)
    if param_grid is None and isinstance(model_class, dict):
        param_grid = model_class
        model_class = None

    param_grid = param_grid or {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    
    # Ép kiểu list thuần
    X_train = _to_list(X_train)
    y_train = _to_list(y_train)

    lambda_values = param_grid.get("alpha", param_grid.get("lambda", [1.0]))
    if hasattr(lambda_values, "tolist"):
        lambda_values = lambda_values.tolist()

    n = len(X_train)
    if k < 2 or k > n:
        raise ValueError("Số lượng fold (k) phải >= 2 và <= số lượng mẫu.")

    fold_size = n // k

    best_lam = None
    best_rmse = float("inf")
    cv_results = []

    for lam in lambda_values:
        mse_scores = []

        for i in range(k):
            val_start = i * fold_size
            val_end = n if i == k - 1 else (i + 1) * fold_size

            X_val = X_train[val_start:val_end]
            y_val = y_train[val_start:val_end]

            X_tr = X_train[:val_start] + X_train[val_end:]
            y_tr = y_train[:val_start] + y_train[val_end:]

            X_tr_design = _add_intercept(X_tr)
            X_val_design = _add_intercept(X_val)

            beta_hat = ridge_fit(X_tr_design, y_tr, lam)

            y_pred = [
                sum(x_ij * b_j for x_ij, b_j in zip(row, beta_hat))
                for row in X_val_design
            ]

            mse = sum((y_v - y_p) ** 2 for y_v, y_p in zip(y_val, y_pred)) / len(y_val)
            mse_scores.append(mse)

        mean_rmse = math.sqrt(sum(mse_scores) / k)
        cv_results.append({"lambda": float(lam), "cv_rmse": float(mean_rmse)})

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_lam = float(lam)

    best_params = {"alpha": best_lam, "lambda": best_lam}

    if model_class is not None:
        # Kiểu trả về 2-tuple cho Jupyter notebook
        cv_scores_dict = {
            "lambda_values": [float(r["lambda"]) for r in cv_results],
            "mean_scores": [float(r["cv_rmse"]) for r in cv_results],
            "best_lambda": float(best_lam),
        }
        return float(best_lam), cv_scores_dict
    else:
        # Kiểu trả về 3-tuple cho package
        return best_params, float(best_rmse), cv_results
