"""
Data Pipeline
==============
Tiền xử lý dữ liệu cho Part 2 (OOP Design).
"""

import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        NotImplementedError: If the method is not yet implemented.
    """
    raise NotImplementedError


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None
) -> tuple:
    """Splits the data into training and testing sets.

    Args:
        X (np.ndarray): The features array.
        y (np.ndarray): The target array.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.

    Raises:
        NotImplementedError: If the method is not yet implemented.
    """
    raise NotImplementedError


class DataPipeline:
    def __init__(self, drop_columns: list = None):
        """
        Tham số drop_columns nhận danh sách cột cần loại bỏ từ DA (Sync Point 2)
        """
        self.drop_columns = drop_columns if drop_columns is not None else []
        self.scalers = {}
        self.imputation_values = {}
        self.encoded_columns = []

    def _prepare_xy(self, df: pd.DataFrame):
        """Hàm nội bộ tách biệt X (đặc trưng) và y (mục tiêu), đồng thời loại bỏ cột xấu"""
        # 1. Loại bỏ các cột đa cộng tuyến theo yêu cầu của DA
        df_clean = df.drop(
            columns=[col for col in self.drop_columns if col in df.columns]
        )

        # 2. Tách biệt X và y
        if "Price" in df_clean.columns:
            X_df = df_clean.drop(columns=["Price"])
            y_arr = df_clean["Price"].to_numpy()
        else:
            X_df = df_clean
            y_arr = None
        return X_df, y_arr

    def _impute_missing(self, X_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        # Thực thi KNN Imputer dựa trên trạng thái is_train
        return X_df

    def _engineer_and_encode(self, X_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        # Thực thi One-Hot Encoding + Biến đổi phi tuyến (Log/Age)
        return X_df

    def _scale_features(self, X_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        # Thực thi Z-score Standardization
        return X_df

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """Chỉ chạy trên tập Train thô"""
        X_df, y_train = self._prepare_xy(df)

        # Chạy xuyên qua hệ thống ống dẫn xử lý
        X_df = self._impute_missing(X_df, is_train=True)
        X_df = self._engineer_and_encode(X_df, is_train=True)
        X_df = self._scale_features(X_df, is_train=True)

        # Lưu lại danh sách cột sau khi One-Hot để transform áp dụng theo
        self.encoded_columns = X_df.columns.tolist()
        return X_df.to_numpy(), y_train

    def transform(self, df: pd.DataFrame) -> tuple:
        """Chỉ chạy trên tập Test thô - Tuyệt đối đóng băng trạng thái"""
        X_df, y_test = self._prepare_xy(df)

        X_df = self._impute_missing(X_df, is_train=False)
        X_df = self._engineer_and_encode(X_df, is_train=False)

        # Đồng bộ hóa số lượng cột của tập Test dựa trên tập Train cũ
        X_df = X_df.reindex(columns=self.encoded_columns, fill_value=0)

        X_df = self._scale_features(X_df, is_train=False)
        return X_df.to_numpy(), y_test


if __name__ == "__main__":
    print("Data Pipeline - OOP Skeleton Demo")
    # TODO: Thêm demo code
