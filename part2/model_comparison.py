"""
Model Comparison
=================
So sánh hiệu năng các mô hình regression trên dữ liệu thực.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any


def train_models(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    custom_ols_func: Callable,
    custom_ridge_func: Callable,
    sklearn_models: dict = None
) -> Dict[str, Dict[str, Any]]:
    """Trains and evaluates both custom from-scratch models and baseline sklearn models.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_test (np.ndarray): Testing feature matrix.
        y_test (np.ndarray): Testing target vector.
        custom_ols_func (Callable): The custom `ols_fit` function from Part 1.
        custom_ridge_func (Callable): The custom `ridge_fit` function from Part 1.
        sklearn_models (dict, optional): A dictionary of instantiated scikit-learn models 
            to use as baselines (e.g., {'sklearn_OLS': LinearRegression()}).

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing evaluation results for each model.
            Format: {model_name: {'train_score': ..., 'test_score': ..., 'predictions': ..., 'coefficients': ...}}

    Raises:
        NotImplementedError: If the method is not yet implemented.
    """
    raise NotImplementedError


def evaluate_gauss_markov_assumptions(X: np.ndarray, y: np.ndarray, residuals: np.ndarray) -> dict:
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


def plot_predictions(y_test: np.ndarray, results: dict, title: str = "Model Predictions Comparison") -> None:
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


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray, model_class: Callable, param_grid: dict, k: int = 5) -> tuple:
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
