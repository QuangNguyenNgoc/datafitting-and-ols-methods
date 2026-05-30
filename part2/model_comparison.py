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
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from part2.advanced_methods import BayesianLinearRegression, kernel_ridge_fit
except Exception:  # pragma: no cover - supports direct script execution.
    from advanced_methods import BayesianLinearRegression, kernel_ridge_fit

from part1.ols_implementation import ols_fit, coef_inference, vif, model_metrics
from part1.ridge_lasso import ridge_fit
from part1.cross_validation import kfold_cv

try:
    from part1.ridge_lasso import ridge_fit as PART1_RIDGE_FIT
except Exception as exc:  # pragma: no cover - fallback depends on runtime path.
    PART1_RIDGE_FIT = None
    PART1_RIDGE_IMPORT_ERROR = exc
else:
    PART1_RIDGE_IMPORT_ERROR = None

import math


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

    result = {
        "model": model,
        "coefficients": coefficients,
        "feature_coefficients": coefficients[1:] if coefficients else None,
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
    X_train,
    y_train,
    X_test,
    y_test,
    custom_ols_func: Callable | None = None,
    custom_ridge_func: Callable | None = None,
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
) -> Dict[str, Dict[str, Any]]:
    # ép kiểu chắc chắn ma trận dạng list
    X_train_list = _to_list(X_train)
    y_train_list = _to_list(y_train)
    X_test_list = _to_list(X_test)
    y_test_list = _to_list(y_test)

    results: Dict[str, Dict[str, Any]] = {}

    custom_ols_func = custom_ols_func or ols_fit
    custom_ridge_func = custom_ridge_func or ridge_fit

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
            X_train_list, y_train_list, param_grid=ridge_param_grid, k=k
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

    # LUỒNG KERNEL
    if include_kernel:
        kernel_params = kernel_params or {"alpha": 1.0, "kernel": "rbf", "gamma": 0.1}
        results["Kernel_Ridge"] = _train_kernel_ridge(
            X_train_list,
            y_train_list,
            X_test_list,
            y_test_list,
            kernel_params=kernel_params,
            random_state=random_state,
            sample_size=kernel_sample_size,
        )

    # LUỒNG BAYESIAN
    if include_bayesian:
        bayesian_params = bayesian_params or {
            "prior_precision": 1e-6,
            "noise_precision": 1.0,
            "fit_intercept": True,
        }
        results["Bayesian_Linear"] = _train_bayesian_linear(
            X_train_list,
            y_train_list,
            X_test_list,
            y_test_list,
            bayesian_params=bayesian_params,
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
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
        coefficients=np.asarray(model.posterior_mean, dtype=float),
        best_params=bayesian_params,
        source="advanced_methods",
    )


def _vif_table(X_list: list, feature_names: list | None = None) -> pd.DataFrame:
    X_list = _to_list(X_list)
    names = feature_names or [f"x{i}" for i in range(len(X_list[0]))]

    vif_scores = vif(X_list)

    df = pd.DataFrame({"Feature": names, "VIF_Score": vif_scores})
    return df.sort_values("VIF_Score", ascending=False).reset_index(drop=True)


def run_diagnostics(
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


def evaluate_gauss_markov_assumptions(
    X: list,
    residuals: list,
    feature_names: list | None = None,
) -> dict:
    """Tự lập trình kiểm định Jarque-Bera và Breusch-Pagan thuần Toán học."""
    X_list = _to_list(X)
    res_list = _to_list(residuals)
    n = len(res_list)

    # 1. Tự code Jarque-Bera (Kiểm định phân phối chuẩn)
    mean_res = sum(res_list) / n
    m2 = sum((r - mean_res) ** 2 for r in res_list) / n
    m3 = sum((r - mean_res) ** 3 for r in res_list) / n
    m4 = sum((r - mean_res) ** 4 for r in res_list) / n

    skewness = m3 / (m2 ** (1.5)) if m2 > 0 else 0
    kurtosis = m4 / (m2**2) if m2 > 0 else 3
    jb_stat = (n / 6) * (skewness**2 + 0.25 * (kurtosis - 3) ** 2)

    # 2. Tự code Breusch-Pagan bằng hàm OLS Part 1 (Kiểm định phương sai sai số không đổi)
    squared_res = [r**2 for r in res_list]
    X_aux = _add_intercept(X_list)

    # Hồi quy bình phương phần dư theo X (Mô hình phụ)
    beta_aux = ols_fit(X_aux, squared_res)
    pred_aux = [sum(x_val * b for x_val, b in zip(row, beta_aux)) for row in X_aux]

    # Tính R2 của mô hình phụ
    mean_sq_res = sum(squared_res) / n
    tss_aux = sum((r2 - mean_sq_res) ** 2 for r2 in squared_res)
    rss_aux = sum((r2 - p2) ** 2 for r2, p2 in zip(squared_res, pred_aux))
    bp_r2 = 1.0 - (rss_aux / tss_aux) if tss_aux > 0 else 0.0
    bp_stat = n * bp_r2

    # NOTE: chỉ dùng dể soi bảng thống kê
    try:
        import scipy.stats as st

        bp_p_value = float(st.chi2.sf(bp_stat, df=len(X_list[0])))
        jb_p_value = float(st.chi2.sf(jb_stat, df=2))
    except ImportError:
        bp_p_value = None
        jb_p_value = None

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
        "VIF": _vif_table(X_list, feature_names=feature_names),
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
                "MAE": metrics.get("MAE", np.nan),
                "RMSE": metrics.get("RMSE", np.nan),
                "R2": metrics.get("R2", np.nan),
                "Train_RMSE": train_metrics.get("RMSE", np.nan),
                "Train_R2": train_metrics.get("R2", np.nan),
                "Source": result.get("source", ""),
                "Best_Params": result.get("best_params", {}),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table = table.sort_values("RMSE", ascending=True).reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table


def plot_predictions(
    y_test: np.ndarray,
    results: dict,
    title: str = "Model Predictions Comparison",
):
    """Plot actual vs predicted values for each model."""
    y_test = _as_1d_array(y_test)
    n_models = max(len(results), 1)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)
    axes = axes.ravel()

    min_value = y_test.min()
    max_value = y_test.max()

    for ax, (model_name, result) in zip(axes, results.items()):
        y_pred = _as_1d_array(result["predictions_test"])
        min_value = min(min_value, y_pred.min())
        max_value = max(max_value, y_pred.max())
        ax.scatter(y_test, y_pred, alpha=0.35, s=18)
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

    for model_name, result in results.items():
        coefficients = result.get("coefficients")
        if coefficients is not None:
            chosen_name = model_name
            chosen_coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
            break

    if chosen_coefficients is None:
        raise ValueError("No model in results contains coefficients to plot.")

    if chosen_coefficients.shape[0] == len(feature_names) + 1:
        chosen_coefficients = chosen_coefficients[1:]

    if chosen_coefficients.shape[0] != len(feature_names):
        raise ValueError("Coefficient length does not match feature_names length.")

    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": chosen_coefficients}
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).head(top_n)
    coef_df = coef_df.sort_values("coefficient", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(coef_df))))
    ax.barh(coef_df["feature"], coef_df["coefficient"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top coefficients - {chosen_name}")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    return fig


def hyperparameter_tuning(
    X_train: list, y_train: list, param_grid: dict, k: int = 5
) -> tuple:
    """
    Thuật toán GridSearch thuần Python bằng K-Fold CV.
    Duyệt qua các giá trị Lambda để tìm ra cấu hình có RMSE thấp nhất.
    Thay thế hoàn toàn GridSearchCV và KFold của sklearn.
    """
    # 1. Trích xuất danh sách các Lambda cần thử nghiệm
    # (Hỗ trợ ép kiểu về list nếu người dùng vô tình truyền NumPy Array)
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

    # 2. VÒNG LẶP GRID SEARCH: Duyệt qua từng siêu tham số
    for lam in lambda_values:
        mse_scores = []

        # 3. VÒNG LẶP K-FOLD: Chia cắt dữ liệu thuần Python
        for i in range(k):
            val_start = i * fold_size
            val_end = n if i == k - 1 else (i + 1) * fold_size

            # Tách tập Validation
            X_val = X_train[val_start:val_end]
            y_val = y_train[val_start:val_end]

            # Tách tập Train (Bằng cách nối List phần đầu và phần đuôi)
            X_tr = X_train[:val_start] + X_train[val_end:]
            y_tr = y_train[:val_start] + y_train[val_end:]

            # Tiền xử lý: Chèn cột Bias (Intercept)
            X_tr_design = _add_intercept(X_tr)
            X_val_design = _add_intercept(X_val)

            # 4. GỌI THUẬT TOÁN LÕI TỪ PART 1
            if PART1_RIDGE_FIT is None:
                raise ImportError("Không tìm thấy hàm ridge_fit từ Part 1.")
            beta_hat = PART1_RIDGE_FIT(X_tr_design, y_tr, lam)

            # Dự đoán trên tập Validation
            y_pred = [
                sum(x_ij * b_j for x_ij, b_j in zip(row, beta_hat))
                for row in X_val_design
            ]

            # Tính MSE của fold này
            mse = sum((y_v - y_p) ** 2 for y_v, y_p in zip(y_val, y_pred)) / len(y_val)
            mse_scores.append(mse)

        # Tính trung bình RMSE cho giá trị lambda hiện tại
        mean_rmse = math.sqrt(sum(mse_scores) / k)
        cv_results.append({"lambda": float(lam), "cv_rmse": float(mean_rmse)})

        # 5. Cập nhật Kỷ lục (Best Params)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_lam = float(lam)

    # Đóng gói kết quả (giữ key 'alpha' để tương thích ngược với luồng code cũ)
    best_params = {"alpha": best_lam, "lambda": best_lam}
    return best_params, float(best_rmse), cv_results


if __name__ == "__main__":
    try:
        from data_pipeline import DataPipeline, load_data, train_test_split
    except ImportError:
        from part2.data_pipeline import DataPipeline, load_data, train_test_split

    df = load_data("part2/data/melb_data.csv")
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    pipeline = DataPipeline(drop_columns=["Bedroom2"])
    X_train_demo, y_train_demo = pipeline.fit_transform(df_train)
    X_test_demo, y_test_demo = pipeline.transform(df_test)

    demo_results = train_models(
        X_train_demo,
        y_train_demo,
        X_test_demo,
        y_test_demo,
        kernel_sample_size=300,
    )
    print(comparison_table(demo_results))
