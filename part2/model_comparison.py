"""
Model Comparison
=================
So sánh hiệu năng các mô hình regression trên dữ liệu thực.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any


def run_diagnostics(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    custom_ols_func: Callable,
    custom_vif_func: Callable,
    custom_inference_func: Callable,
) -> pd.DataFrame:
    """
    Thực thi Phase 1 (Chẩn đoán Đa cộng tuyến và Ý nghĩa Thống kê).
    Hàm này gom tất cả các hàm của Phần 1 lại, xuất ra một bảng DataFrame duy nhất
    để Data Analyst đọc và ra quyết định loại biến (Sync Point 2).
    """

    # 1. Chạy OLS để lấy hệ số (beta_hat) và phương sai nhiễu (sigma2)
    # Lưu ý: Cấu trúc output của hàm custom_ols_func phụ thuộc vào cách nhóm code ở Phần 1
    beta_hat, sigma2 = custom_ols_func(X_train_raw, y_train)

    # 2. Tính hệ số phóng đại phương sai (VIF)
    vif_values = custom_vif_func(X_train_raw)

    # 3. Chạy kiểm định thống kê để lấy Sai số chuẩn, t-stat, p-value và Khoảng tin cậy
    se, t_stats, p_values, ci_lower, ci_upper = custom_inference_func(
        X_train_raw, y_train, beta_hat, sigma2
    )

    # 4. Gom toàn bộ "kết quả xét nghiệm máu" vào một bảng duy nhất cho Bác sĩ (DA) dễ đọc
    diagnostics_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": beta_hat,
            "Std_Error": se,
            "t_statistic": t_stats,
            "p_value": p_values,
            "CI_95_Lower": ci_lower,
            "CI_95_Upper": ci_upper,
            "VIF": vif_values,
        }
    )

    print("--- HOÀN TẤT CHẨN ĐOÁN. CHỜ DATA ANALYST REVIEW ---")
    return diagnostics_df


def train_models(
    X_train_raw: np.ndarray,
    X_train_best: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    X_test_best: np.ndarray,
    y_test: np.ndarray,
    custom_ols_func: Callable,
    custom_ridge_func: Callable,
    custom_kernel_func: Callable,
    lambda_ridge: float,
    lambda_kernel: float,
) -> Dict[str, Dict[str, Any]]:
    """Trains and evaluates the four required regression models using custom code.

    Args:
        X_train_raw (np.ndarray): Full training feature matrix.
        X_train_best (np.ndarray): Training feature matrix after removing collinear variables.
        y_train (np.ndarray): Training target vector.
        X_test_raw (np.ndarray): Full testing feature matrix.
        X_test_best (np.ndarray): Testing feature matrix after removing collinear variables.
        y_test (np.ndarray): Testing target vector.
        custom_ols_func (Callable): The custom `ols_fit` function from Part 1.
        custom_ridge_func (Callable): The custom `ridge_fit` function from Part 1.
        custom_kernel_func (Callable): The custom `kernel_ridge_fit` function from advanced methods.
        lambda_ridge (float): Optimal Ridge regularization parameter found via K-fold CV.
        lambda_kernel (float): Optimal Kernel Ridge regularization parameter.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing the handoff results for the Data Analyst.

    Raises:
        NotImplementedError: If one of the custom fitting functions is not callable or not implemented.
    """

    results = {}

    # 1. Execute OLS on the full raw feature matrix.
    # beta_raw = custom_ols_func(X_train_raw, y_train)
    # y_pred_raw = X_test_raw @ beta_raw
    # Compute MAE, RMSE, R2 on y_test

    # 2. Execute OLS on the selected-feature matrix.
    # beta_best = custom_ols_func(X_train_best, y_train)
    # y_pred_best = X_test_best @ beta_best
    # Compute MAE, RMSE, R2 on y_test

    # 3. Execute Ridge Regression on the selected feature matrix.
    # beta_ridge = custom_ridge_func(X_train_best, y_train, lambda_ridge)
    # y_pred_ridge = X_test_best @ beta_ridge
    # Compute MAE, RMSE, R2 on y_test

    # 4. Execute Kernel Ridge Regression on the selected feature matrix.
    # y_pred_kernel = custom_kernel_func(X_train_best, y_train, X_test_best, lambda_kernel)
    # Compute MAE, RMSE, R2 on y_test

    return results


def evaluate_gauss_markov_assumptions(
    X: np.ndarray, y: np.ndarray, residuals: np.ndarray
) -> dict:
    """Evaluates Gauss-Markov assumptions on real data using statistical tests.

    Performs tests such as the Breusch-Pagan test for heteroscedasticity,
    Variance Inflation Factor (VIF) checks for multicollinearity, and evaluates
    normality of residuals to ensure OLS assumptions hold on the real dataset.

    Args:
        X (np.ndarray): The design matrix (features).
        y (np.ndarray): The true target vector.
        residuals (np.ndarray): The residual vector (y - y_hat) from the fitted model.

    Returns:
        dict: A dictionary containing the results of various statistical tests
            (e.g., {'Breusch-Pagan': p_value, 'VIF': dataframe, ...}).

    Raises:
        NotImplementedError: If the method is not yet implemented.
    """
    raise NotImplementedError


def comparison_table(results: dict) -> pd.DataFrame:
    """Creates a comparison table summarizing the results of various models.

    Args:
        results (dict): The dictionary of results returned by `train_models`.

    Returns:
        pd.DataFrame: A formatted pandas DataFrame comparing model metrics.

    Raises:
        NotImplementedError: If the method is not yet implemented.
    """
    raise NotImplementedError


def plot_predictions(
    y_test: np.ndarray, results: dict, title: str = "Model Predictions Comparison"
) -> None:
    """Plots a comparison of predictions made by different models against the true values.

    Args:
        y_test (np.ndarray): The true testing target vector.
        results (dict): The dictionary of results containing model predictions.
        title (str, optional): The title of the plot. Defaults to "Model Predictions Comparison".

    Raises:
        NotImplementedError: If the method is not yet implemented.
    """
    raise NotImplementedError


def plot_coefficients(results: dict, feature_names: list) -> None:
    """Plots a comparison of the learned coefficients across different models.

    Args:
        results (dict): The dictionary of results containing model coefficients.
        feature_names (list): A list of strings representing the feature names.

    Raises:
        NotImplementedError: If the method is not yet implemented.
    """
    raise NotImplementedError


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_class: Callable,
    param_grid: dict,
    k: int = 5,
) -> tuple:
    """Performs k-fold cross-validation to find the best hyperparameters.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        model_class (Callable): The class of the model to tune.
        param_grid (dict): A dictionary mapping parameter names to lists of values to evaluate.
        k (int, optional): The number of folds for cross-validation. Defaults to 5.

    Returns:
        tuple: A tuple containing the best parameters (dict) and the best cross-validation score (float).

    Raises:
        NotImplementedError: If the method is not yet implemented.
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("Model Comparison - Skeleton Demo")
    # TODO: Thêm demo code
