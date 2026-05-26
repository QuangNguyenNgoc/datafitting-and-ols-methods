"""
Data Pipeline
==============
Tiền xử lý dữ liệu cho Part 2 (OOP Design).
"""

import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Đọc file CSV thành DataFrame."""
    return pd.read_csv(filepath)


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = None,
) -> tuple:
    """Chia DataFrame thô thành train/test split cố định."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(df))
    n_test = int(np.ceil(len(df) * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test


class DataPipeline:
    def __init__(
        self,
        drop_columns: list = None,
        target_name: str = "Price",
        categorical_columns: list = None,
        n_neighbors: int = 5,
    ):
        """Khởi tạo cấu hình pipeline và trạng thái học từ train data."""
        default_drop_columns = [
            "Address",
            "SellerG",
            "Suburb",
            "Date",
            "Postcode",
            "Method",
            "CouncilArea",
        ]
        extra_drop_columns = drop_columns or []

        self.target_name = target_name
        self.drop_columns = list(dict.fromkeys(default_drop_columns + extra_drop_columns))
        self.categorical_columns = categorical_columns or ["Type", "Regionname"]
        self.n_neighbors = n_neighbors

        self.scalers = {}
        self.imputation_values = {}
        self.encoded_columns = []
        self.feature_names = []
        self.knn_imputer = None
        self.knn_columns = []
        self.numeric_fallbacks = {}

        self.required_columns = [
            "Rooms",
            "Distance",
            "Bedroom2",
            "Bathroom",
            "Car",
            "Landsize",
            "BuildingArea",
            "YearBuilt",
            "Lattitude",
            "Longtitude",
            "Propertycount",
            "Type",
            "Regionname",
        ]

    def _prepare_xy(self, df: pd.DataFrame) -> tuple:
        """Tách X và y từ DataFrame thô."""
        if self.target_name not in df.columns:
            raise ValueError(f"Missing target column: {self.target_name}")

        X_df = df.drop(columns=[self.target_name]).copy()
        y_arr = df[self.target_name].to_numpy()
        return X_df, y_arr

    def _validate_schema(self, X_df: pd.DataFrame) -> None:
        """Kiểm tra các cột đầu vào bắt buộc."""
        missing = [
            column
            for column in self.required_columns
            if column not in X_df.columns and column not in self.drop_columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _repair_invalid_values(self, X_df: pd.DataFrame) -> None:
        """Chuyển giá trị không hợp lệ thành missing value."""
        if "BuildingArea" in X_df.columns:
            missing = X_df["BuildingArea"].isna() | (X_df["BuildingArea"] <= 0)
            X_df["BuildingArea_missing"] = missing.astype(float)
            X_df.loc[X_df["BuildingArea"] <= 0, "BuildingArea"] = np.nan

        if "YearBuilt" in X_df.columns:
            missing = X_df["YearBuilt"].isna() | (X_df["YearBuilt"] < 1800)
            X_df["YearBuilt_missing"] = missing.astype(float)
            X_df.loc[X_df["YearBuilt"] < 1800, "YearBuilt"] = np.nan

        if "Landsize" in X_df.columns:
            missing = X_df["Landsize"].isna() | (X_df["Landsize"] <= 0)
            X_df["Landsize_zero_or_missing"] = missing.astype(float)
            X_df.loc[X_df["Landsize"] <= 0, "Landsize"] = np.nan

    def _impute_missing(self, X_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Fit hoặc áp dụng KNN Imputer trên các cột numeric."""
        try:
            from sklearn.impute import KNNImputer
        except ImportError as exc:
            raise ImportError("scikit-learn is required for KNNImputer") from exc

        result = X_df.copy()
        numeric_columns = result.select_dtypes(include=np.number).columns.tolist()

        if is_train:
            self.knn_columns = numeric_columns
            self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            imputed = self.knn_imputer.fit_transform(result[self.knn_columns])
            self.imputation_values = {
                "method": "knn_imputer",
                "n_neighbors": self.n_neighbors,
                "columns": self.knn_columns,
            }
        else:
            missing_columns = [column for column in self.knn_columns if column not in result.columns]
            if missing_columns:
                raise ValueError(f"Missing KNN imputation columns: {missing_columns}")
            imputed = self.knn_imputer.transform(result[self.knn_columns])

        result[self.knn_columns] = imputed
        return result

    def _engineer_and_encode(self, X_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Tạo feature mới và one-hot encode categorical columns."""
        result = X_df.copy()

        if "YearBuilt" in result.columns:
            result["Age"] = (2026 - result["YearBuilt"]).clip(lower=0)

        rooms = result["Rooms"].replace(0, np.nan) if "Rooms" in result.columns else np.nan

        if {"Rooms", "Bathroom"}.issubset(result.columns):
            result["OtherRooms"] = (result["Rooms"] - result["Bathroom"]).clip(lower=0)

        if "BuildingArea" in result.columns:
            result["BuildingArea_per_Room"] = result["BuildingArea"] / rooms

        if {"BuildingArea", "Landsize"}.issubset(result.columns):
            result["BuildingCoverage"] = result["BuildingArea"] / result["Landsize"]
            result["BuildingCoverage"] = result["BuildingCoverage"].replace([np.inf, -np.inf], np.nan)
            result.loc[result["BuildingCoverage"] < 0, "BuildingCoverage"] = np.nan

        for column in self.categorical_columns:
            if column in result.columns:
                result[column] = result[column].fillna("Unknown").astype(str)

        existing_drop_columns = [column for column in self.drop_columns if column in result.columns]
        result = result.drop(columns=existing_drop_columns)

        active_categorical_columns = [
            column for column in self.categorical_columns if column in result.columns
        ]
        result = pd.get_dummies(result, columns=active_categorical_columns, dtype=float)

        if is_train:
            self.encoded_columns = result.columns.tolist()
        else:
            result = result.reindex(columns=self.encoded_columns, fill_value=0.0)

        return result.replace([np.inf, -np.inf], np.nan)

    def _scale_features(self, X_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Chuẩn hóa Z-score bằng trạng thái học từ train data."""
        non_numeric_columns = X_df.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric_columns:
            raise ValueError(f"Unexpected non-numeric columns before scaling: {non_numeric_columns}")

        result = X_df.copy()

        if is_train:
            self.numeric_fallbacks = result.median().fillna(0.0).to_dict()

        result = result.fillna(self.numeric_fallbacks).fillna(0.0)

        if is_train:
            self.scalers = {}
            for column in result.columns:
                mean = result[column].mean()
                std = result[column].std(ddof=0)

                if pd.isna(std) or std == 0:
                    std = 1.0

                self.scalers[column] = {"mean": float(mean), "std": float(std)}

        for column, params in self.scalers.items():
            result[column] = (result[column] - params["mean"]) / params["std"]

        return result

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """Chỉ chạy trên tập train thô."""
        X_df, y_train = self._prepare_xy(df)
        self._validate_schema(X_df)
        self._repair_invalid_values(X_df)
        X_df = self._impute_missing(X_df, is_train=True)
        X_df = self._engineer_and_encode(X_df, is_train=True)
        X_df = self._scale_features(X_df, is_train=True)

        self.feature_names = X_df.columns.tolist()
        return X_df.to_numpy(dtype=float), y_train

    def transform(self, df: pd.DataFrame) -> tuple:
        """Chỉ chạy trên tập test thô bằng trạng thái đã fit."""
        if self.knn_imputer is None or not self.encoded_columns or not self.scalers:
            raise ValueError("Pipeline is not fitted")

        X_df, y_test = self._prepare_xy(df)
        self._validate_schema(X_df)
        self._repair_invalid_values(X_df)
        X_df = self._impute_missing(X_df, is_train=False)
        X_df = self._engineer_and_encode(X_df, is_train=False)
        X_df = self._scale_features(X_df, is_train=False)

        return X_df.to_numpy(dtype=float), y_test
