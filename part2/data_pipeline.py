"""
Data Pipeline
==============
Tiền xử lý dữ liệu cho Part 2.
"""

import numpy as np
import pandas as pd


def load_data(filepath):
    """
    Đọc dữ liệu từ file CSV.

    Parameters
    ----------
    filepath : str
        Đường dẫn tới file CSV.

    Returns
    -------
    df : pd.DataFrame
    """
    # TODO: Implement data loading
    raise NotImplementedError


def explore_data(df):
    """
    Khám phá dữ liệu: shape, dtypes, missing values, statistics.

    Parameters
    ----------
    df : pd.DataFrame
    """
    # TODO: Implement EDA
    raise NotImplementedError


def handle_missing_values(df, strategy="mean"):
    """
    Xử lý giá trị thiếu.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        Chiến lược: 'mean', 'median', 'mode', 'drop'.

    Returns
    -------
    df_clean : pd.DataFrame
    """
    # TODO: Implement missing value handling
    raise NotImplementedError


def feature_scaling(X, method="standardize"):
    """
    Chuẩn hóa features.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
    method : str
        'standardize' (z-score) hoặc 'normalize' (min-max).

    Returns
    -------
    X_scaled : np.ndarray
    scaler_params : dict
    """
    # TODO: Implement feature scaling
    raise NotImplementedError


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Chia dữ liệu thành tập train và test.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    test_size : float
    random_state : int or None

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of np.ndarray
    """
    # TODO: Implement train/test split
    raise NotImplementedError


def build_pipeline(filepath, target_column, test_size=0.2):
    """
    Pipeline hoàn chỉnh: load -> clean -> scale -> split.

    Returns
    -------
    X_train, X_test, y_train, y_test, metadata : tuple
    """
    # TODO: Implement full pipeline
    raise NotImplementedError


if __name__ == "__main__":
    print("Data Pipeline - Demo")
    # TODO: Thêm demo code
