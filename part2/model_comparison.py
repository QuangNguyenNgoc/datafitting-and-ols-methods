"""
Model Comparison
=================
So sánh hiệu năng các mô hình regression trên dữ liệu thực.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_models(X_train, y_train, X_test, y_test):
    """
    Huấn luyện và đánh giá nhiều mô hình.

    Models bao gồm:
    - OLS (from scratch, Part 1)
    - Ridge Regression
    - Lasso Regression
    - scikit-learn LinearRegression (baseline)

    Returns
    -------
    results : dict
        {model_name: {'train_score': ..., 'test_score': ..., 'predictions': ...}}
    """
    # TODO: Implement model training and evaluation
    raise NotImplementedError


def comparison_table(results):
    """
    Tạo bảng so sánh kết quả các mô hình.

    Parameters
    ----------
    results : dict

    Returns
    -------
    df : pd.DataFrame
    """
    # TODO: Implement comparison table
    raise NotImplementedError


def plot_predictions(y_test, results, title="Model Predictions Comparison"):
    """
    Vẽ biểu đồ so sánh dự đoán của các mô hình.
    """
    # TODO: Implement prediction plots
    raise NotImplementedError


def plot_coefficients(results, feature_names):
    """
    Vẽ biểu đồ so sánh hệ số (coefficients) giữa các mô hình.
    """
    # TODO: Implement coefficient comparison plot
    raise NotImplementedError


def hyperparameter_tuning(X_train, y_train, model_class, param_grid, k=5):
    """
    Tìm hyperparameter tốt nhất bằng cross-validation.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    model_class : class
    param_grid : dict
    k : int

    Returns
    -------
    best_params : dict
    best_score : float
    """
    # TODO: Implement hyperparameter tuning
    raise NotImplementedError


if __name__ == "__main__":
    print("Model Comparison - Demo")
    # TODO: Thêm demo code
