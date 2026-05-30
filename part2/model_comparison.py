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
from scipy import stats
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from part2.advanced_methods import BayesianLinearRegression, kernel_ridge_fit
except Exception:  # pragma: no cover - supports direct script execution.
    from advanced_methods import BayesianLinearRegression, kernel_ridge_fit

from part1.ols_implementation import ols_fit, coef_inference, vif, model_metrics
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


def _make_result(
    model: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
    predictions_train: np.ndarray,
    predictions_test: np.ndarray,
    coefficients: np.ndarray | None = None,
    best_params: dict | None = None,
    source: str = "",
) -> dict:
    metrics = compute_metrics(y_test, predictions_test)
    train_metrics = compute_metrics(y_train, predictions_train)

    result = {
        "model": model,
        "coefficients": coefficients,
        "feature_coefficients": (
            None if coefficients is None else np.asarray(coefficients).reshape(-1)[1:]
        ),
        "predictions_train": _as_1d_array(predictions_train),
        "predictions_test": _as_1d_array(predictions_test),
        "predictions": _as_1d_array(predictions_test),
        "metrics": metrics,
        "train_metrics": train_metrics,
        "train_score": train_metrics["R2"],
        "test_score": metrics["R2"],
        "best_params": best_params or {},
        "source": source,
    }
    return result


def _fit_custom_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    custom_ols_func: Callable,
) -> dict:
    X_train_design = _add_intercept(X_train)
    X_test_design = _add_intercept(X_test)
    beta = np.asarray(custom_ols_func(X_train_design, y_train), dtype=float).reshape(-1)

    return {
        "model": {"type": "custom_ols", "fit_function": custom_ols_func},
        "coefficients": beta,
        "predictions_train": X_train_design @ beta,
        "predictions_test": X_test_design @ beta,
    }


def _fit_custom_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    custom_ridge_func: Callable,
    lam: float,
) -> dict:
    X_train_design = _add_intercept(X_train)
    X_test_design = _add_intercept(X_test)
    beta = np.asarray(
        custom_ridge_func(X_train_design, y_train, lam), dtype=float
    ).reshape(-1)

    return {
        "model": {
            "type": "custom_ridge",
            "fit_function": custom_ridge_func,
            "lambda": float(lam),
        },
        "coefficients": beta,
        "predictions_train": X_train_design @ beta,
        "predictions_test": X_test_design @ beta,
    }


def _fit_sklearn_model(
    model: Any, X_train, y_train, X_test
) -> tuple[Any, np.ndarray, np.ndarray]:
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)
    return fitted_model, fitted_model.predict(X_train), fitted_model.predict(X_test)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.sqrt(mean_squared_error(_as_1d_array(y_true), _as_1d_array(y_pred)))
    )


def _ridge_cv_with_part1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambda_values: np.ndarray,
    custom_ridge_func: Callable | None,
    k: int,
    random_state: int,
) -> tuple[dict, float, pd.DataFrame]:
    """Tune Ridge lambda without requiring changes to Part 1 implementation."""
    rows = []

    for lam in np.asarray(lambda_values, dtype=float).reshape(-1):
        if custom_ridge_func is not None:

            def fit_func(X_fold, y_fold, lam=lam):
                return np.asarray(
                    custom_ridge_func(_add_intercept(X_fold), y_fold, lam),
                    dtype=float,
                ).reshape(-1)

            def predict_func(beta, X_valid):
                return _add_intercept(X_valid) @ np.asarray(beta, dtype=float).reshape(
                    -1
                )

            cv_source = "part2.KFold+part1.ridge_fit"
        else:

            def fit_func(X_fold, y_fold, lam=lam):
                model = Ridge(alpha=lam)
                model.fit(X_fold, y_fold)
                return model

            def predict_func(model, X_valid):
                return model.predict(X_valid)

            cv_source = "part2.KFold+sklearn_ridge"

        cv = KFold(
            n_splits=min(int(k), X_train.shape[0]),
            shuffle=True,
            random_state=random_state,
        )
        fold_scores = []
        for train_idx, valid_idx in cv.split(X_train):
            model = fit_func(X_train[train_idx], y_train[train_idx])
            fold_scores.append(
                _rmse(y_train[valid_idx], predict_func(model, X_train[valid_idx]))
            )
        mean_rmse = float(np.mean(fold_scores))
        std_rmse = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

        rows.append(
            {
                "param_alpha": float(lam),
                "mean_test_score": -float(mean_rmse),
                "std_test_score": float(std_rmse),
                "mean_rmse": float(mean_rmse),
                "fold_scores": fold_scores,
                "cv_source": cv_source,
            }
        )

    cv_results = pd.DataFrame(rows)
    best_idx = int(cv_results["mean_rmse"].idxmin())
    best_alpha = float(cv_results.loc[best_idx, "param_alpha"])
    best_rmse = float(cv_results.loc[best_idx, "mean_rmse"])
    return {"alpha": best_alpha}, best_rmse, cv_results


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    custom_ols_func: Callable | None = None,
    custom_ridge_func: Callable | None = None,
    sklearn_models: dict | None = None,
    ridge_param_grid: dict | None = None,
    kernel_params: dict | None = None,
    bayesian_params: dict | None = None,
    k: int = 5,
    random_state: int = 42,
    kernel_sample_size: int = 1000,
    include_ridge: bool = True,
    include_kernel: bool = True,
    include_bayesian: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Train Part 1 OLS/Ridge, Kernel Ridge, and optional sklearn baselines."""
    X_train, y_train = _validate_xy(X_train, y_train, "train")
    X_test, y_test = _validate_xy(X_test, y_test, "test")
    results: Dict[str, Dict[str, Any]] = {}

    custom_ols_func = custom_ols_func or PART1_OLS_FIT
    custom_ridge_func = custom_ridge_func or PART1_RIDGE_FIT

    if custom_ols_func is not None:
        ols = _fit_custom_ols(X_train, y_train, X_test, custom_ols_func)
        results["OLS"] = _make_result(
            model=ols["model"],
            y_train=y_train,
            y_test=y_test,
            predictions_train=ols["predictions_train"],
            predictions_test=ols["predictions_test"],
            coefficients=ols["coefficients"],
            source="part1",
        )
    else:
        ols_model, y_train_pred, y_test_pred = _fit_sklearn_model(
            LinearRegression(), X_train, y_train, X_test
        )
        results["OLS"] = _make_result(
            model=ols_model,
            y_train=y_train,
            y_test=y_test,
            predictions_train=y_train_pred,
            predictions_test=y_test_pred,
            coefficients=_coef_with_intercept(ols_model),
            best_params={
                "fallback_reason": f"Part 1 OLS unavailable: {PART1_OLS_IMPORT_ERROR}"
            },
            source="sklearn_fallback",
        )

    if include_ridge:
        ridge_param_grid = ridge_param_grid or {"alpha": np.logspace(-3, 5, 17)}
        ridge_best_params, ridge_best_rmse, ridge_cv_results = hyperparameter_tuning(
            X_train,
            y_train,
            Ridge,
            ridge_param_grid,
            k=k,
            random_state=random_state,
            return_cv_results=True,
            custom_ridge_func=custom_ridge_func,
        )
        ridge_alpha = float(ridge_best_params.get("alpha", 1.0))
        ridge_cv_scores = cv_scores_from_results(ridge_cv_results, param_name="alpha")
        ridge_handover_params = {
            **ridge_best_params,
            "lambda": ridge_alpha,
            "cv_rmse": ridge_best_rmse,
        }

        if custom_ridge_func is not None:
            ridge = _fit_custom_ridge(
                X_train,
                y_train,
                X_test,
                custom_ridge_func,
                lam=ridge_alpha,
            )
            results["Ridge"] = _make_result(
                model=ridge["model"],
                y_train=y_train,
                y_test=y_test,
                predictions_train=ridge["predictions_train"],
                predictions_test=ridge["predictions_test"],
                coefficients=ridge["coefficients"],
                best_params=ridge_handover_params,
                source="part1",
            )
        else:
            ridge_model, y_train_pred, y_test_pred = _fit_sklearn_model(
                Ridge(alpha=ridge_alpha), X_train, y_train, X_test
            )
            results["Ridge"] = _make_result(
                model=ridge_model,
                y_train=y_train,
                y_test=y_test,
                predictions_train=y_train_pred,
                predictions_test=y_test_pred,
                coefficients=_coef_with_intercept(ridge_model),
                best_params={
                    **ridge_handover_params,
                    "fallback_reason": f"Part 1 Ridge unavailable: {PART1_RIDGE_IMPORT_ERROR}",
                },
                source="sklearn_fallback",
            )

        results["Ridge"]["best_lambda"] = ridge_alpha
        results["Ridge"]["cv_scores"] = ridge_cv_scores
        results["Ridge"]["cv_results"] = ridge_cv_results

    if include_kernel:
        kernel_params = kernel_params or {"alpha": 1.0, "kernel": "rbf", "gamma": 0.1}
        kernel_result = _train_kernel_ridge(
            X_train,
            y_train,
            X_test,
            y_test,
            kernel_params=kernel_params,
            random_state=random_state,
            sample_size=kernel_sample_size,
        )
        results["Kernel_Ridge"] = kernel_result

    if include_bayesian:
        bayesian_params = bayesian_params or {
            "prior_precision": 1e-6,
            "noise_precision": 1.0,
            "fit_intercept": True,
        }
        results["Bayesian_Linear"] = _train_bayesian_linear(
            X_train,
            y_train,
            X_test,
            y_test,
            bayesian_params=bayesian_params,
        )

    for name, model in (sklearn_models or {}).items():
        fitted, y_train_pred, y_test_pred = _fit_sklearn_model(
            model, X_train, y_train, X_test
        )
        result_name = name if name not in results else f"{name}_extra"
        results[result_name] = _make_result(
            model=fitted,
            y_train=y_train,
            y_test=y_test,
            predictions_train=y_train_pred,
            predictions_test=y_test_pred,
            coefficients=_coef_with_intercept(fitted),
            source="sklearn_extra",
        )

    return results


def _train_kernel_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kernel_params: dict,
    random_state: int,
    sample_size: int,
) -> dict:
    """Train Kernel Ridge from ``advanced_methods.py`` on a deterministic subset."""
    rng = np.random.default_rng(random_state)
    n_train = X_train.shape[0]
    sample_size = min(sample_size, n_train)

    if sample_size < n_train:
        train_idx = rng.choice(n_train, size=sample_size, replace=False)
        X_fit = X_train[train_idx]
        y_fit = y_train[train_idx]
    else:
        X_fit = X_train
        y_fit = y_train

    kernel_artifacts = kernel_ridge_fit(X_fit, y_fit, X_test=None, **kernel_params)
    model = kernel_artifacts["model"]

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    result = _make_result(
        model=model,
        y_train=y_train,
        y_test=y_test,
        predictions_train=y_train_pred,
        predictions_test=y_test_pred,
        coefficients=None,
        best_params=kernel_artifacts.get("params", kernel_params),
        source="advanced_methods",
    )
    result["training_rows_used"] = int(sample_size)
    return result


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


def _vif_table(X: np.ndarray, feature_names: list[str] | None = None) -> pd.DataFrame:
    X = _as_2d_array(X)
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]

    if PART1_VIF is not None:
        try:
            vif_df = PART1_VIF(X).copy()
            if "Feature" in vif_df.columns and len(vif_df) == len(names):
                vif_df["Feature"] = names
            if "VIF_Score" in vif_df.columns:
                return vif_df.sort_values("VIF_Score", ascending=False).reset_index(
                    drop=True
                )
        except Exception:
            pass

    rows = []

    for idx, name in enumerate(names):
        target = X[:, idx]
        if np.std(target) < 1e-12:
            vif_value = np.inf
        else:
            others = np.delete(X, idx, axis=1)
            if others.shape[1] == 0:
                vif_value = 1.0
            else:
                aux_model = LinearRegression()
                aux_model.fit(others, target)
                r2 = aux_model.score(others, target)
                vif_value = np.inf if r2 >= 1.0 else 1.0 / max(1.0 - r2, 1e-12)

        rows.append({"Feature": name, "VIF_Score": float(vif_value)})

    return (
        pd.DataFrame(rows)
        .sort_values("VIF_Score", ascending=False)
        .reset_index(drop=True)
    )


def run_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Run the Member B Phase 1 diagnostics before final model selection.

    This returns VIF for multicollinearity plus coefficient inference from the
    Part 1 implementation when available. It is intended as the selection gate
    artifact that Member C can review before deciding whether to drop features.
    """
    X, y = _validate_xy(X, y, "run_diagnostics")
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]
    X_design = _add_intercept(X)

    if PART1_OLS_FIT is not None:
        beta = np.asarray(PART1_OLS_FIT(X_design, y), dtype=float).reshape(-1)
    else:
        model = LinearRegression(fit_intercept=False)
        model.fit(X_design, y)
        beta = np.asarray(model.coef_, dtype=float).reshape(-1)

    fitted = X_design @ beta
    residuals = y - fitted
    rss = float(np.sum(residuals**2))
    dof = max(X_design.shape[0] - X_design.shape[1], 1)
    sigma2 = rss / dof

    inference_df = None
    if PART1_COEF_INFERENCE is not None:
        try:
            inference_df = PART1_COEF_INFERENCE(X_design, y, beta, sigma2).copy()
            expected_names = ["Intercept", *names]
            if len(inference_df) == len(expected_names):
                inference_df.insert(0, "Feature", expected_names)
        except Exception:
            inference_df = None

    return {
        "coefficients": beta,
        "predictions_train": fitted,
        "residuals_train": residuals,
        "sigma2": float(sigma2),
        "VIF": _vif_table(X, feature_names=names),
        "coef_inference": inference_df,
    }


def evaluate_gauss_markov_assumptions(
    X: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """Evaluate lightweight OLS diagnostics for the real dataset."""
    X, y = _validate_xy(X, y, "diagnostics")
    residuals = _as_1d_array(residuals)

    if residuals.shape[0] != y.shape[0]:
        raise ValueError("residuals must have the same length as y.")

    squared_residuals = residuals**2
    X_aux = _add_intercept(X)
    aux_model = LinearRegression(fit_intercept=False)
    aux_model.fit(X_aux, squared_residuals)
    aux_pred = aux_model.predict(X_aux)
    bp_r2 = max(r2_score(squared_residuals, aux_pred), 0.0)
    bp_stat = X.shape[0] * bp_r2
    bp_p_value = stats.chi2.sf(bp_stat, df=X.shape[1])

    jb_stat, jb_p_value = stats.jarque_bera(residuals)

    return {
        "residual_summary": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals, ddof=1)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
        },
        "normality": {
            "test": "Jarque-Bera",
            "statistic": float(jb_stat),
            "p_value": float(jb_p_value),
        },
        "breusch_pagan": {
            "lm_statistic": float(bp_stat),
            "p_value": float(bp_p_value),
            "df": int(X.shape[1]),
        },
        "VIF": _vif_table(X, feature_names=feature_names),
    }


def cv_scores_from_results(
    cv_results: pd.DataFrame,
    param_name: str = "alpha",
) -> dict:
    """Convert GridSearchCV rows to the cv_scores contract used for handover."""
    param_column = f"param_{param_name}"
    if param_column not in cv_results.columns:
        raise ValueError(f"Missing column in cv_results: {param_column}")

    lambda_values = cv_results[param_column].astype(float).to_numpy()
    mean_scores = (-cv_results["mean_test_score"].astype(float)).to_numpy()
    std_scores = cv_results["std_test_score"].astype(float).to_numpy()
    best_idx = int(np.argmin(mean_scores))

    return {
        "lambda_values": lambda_values.tolist(),
        "mean_scores": mean_scores.tolist(),
        "std_scores": std_scores.tolist(),
        "best_lambda": float(lambda_values[best_idx]),
        "best_cv_rmse": float(mean_scores[best_idx]),
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_class: Callable,
    param_grid: dict,
    k: int = 5,
    random_state: int = 42,
    return_cv_results: bool = False,
    custom_ridge_func: Callable | None = None,
) -> tuple:
    """Tune a sklearn-compatible estimator with K-fold CV and RMSE scoring."""
    X_train, y_train = _validate_xy(X_train, y_train, "hyperparameter_tuning")

    if model_class is Ridge and "alpha" in param_grid:
        best_params, best_score, cv_results = _ridge_cv_with_part1(
            X_train,
            y_train,
            lambda_values=param_grid["alpha"],
            custom_ridge_func=custom_ridge_func,
            k=k,
            random_state=random_state,
        )
        if return_cv_results:
            return best_params, best_score, cv_results
        return best_params, best_score

    if isinstance(model_class, type):
        estimator = model_class()
    else:
        estimator = clone(model_class)

    n_splits = min(int(k), X_train.shape[0])
    if n_splits < 2:
        raise ValueError("k must be at least 2 and no larger than the number of rows.")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
    )
    grid.fit(X_train, y_train)

    best_params = dict(grid.best_params_)
    best_score = float(-grid.best_score_)

    if return_cv_results:
        cv_results = pd.DataFrame(grid.cv_results_)
        return best_params, best_score, cv_results

    return best_params, best_score


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
