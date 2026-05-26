import numpy as np
import pandas as pd


# Đọc file CSV thành DataFrame.
def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


# Chia features và target thành tập train/test cố định.
def train_test_split(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = None,
) -> tuple:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(y))
    n_test = int(np.ceil(len(y) * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
    else:
        X_arr = np.asarray(X)
        X_train = X_arr[train_idx]
        X_test = X_arr[test_idx]

    y_arr = np.asarray(y)
    return X_train, X_test, y_arr[train_idx], y_arr[test_idx]


class DataPipeline:
    # Khởi tạo trạng thái đã fit và cấu hình tiền xử lý.
    def __init__(self):
        self.scalers = {}
        self.imputation_values = {}
        self.encoded_columns = []
        self.feature_names = []
        self.categorical_columns = ["Type", "Method", "Regionname", "CouncilArea"]
        self.group_impute_columns = ["BuildingArea", "YearBuilt", "Car"]
        self.drop_columns = ["Address", "SellerG", "Suburb", "Date", "Postcode"]
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
            "Method",
            "Regionname",
            "CouncilArea",
            "Date",
        ]
        self.sale_year = None

    # Fit trạng thái tiền xử lý trên train data và trả về ma trận đã biến đổi.
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        features = self._prepare_features(df, fit=True)
        encoded = pd.get_dummies(features, columns=self.categorical_columns, dtype=float)
        self.encoded_columns = encoded.columns.tolist()
        encoded = self._fill_remaining_numeric(encoded, fit=True)
        self._fit_scalers(encoded)
        self.feature_names = encoded.columns.tolist()
        return self._scale(encoded).to_numpy(dtype=float)

    # Biến đổi dữ liệu mới bằng trạng thái tiền xử lý đã fit.
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.encoded_columns or not self.scalers:
            raise ValueError("Pipeline is not fitted")

        features = self._prepare_features(df, fit=False)
        encoded = pd.get_dummies(features, columns=self.categorical_columns, dtype=float)
        encoded = encoded.reindex(columns=self.encoded_columns, fill_value=0.0)
        encoded = self._fill_remaining_numeric(encoded, fit=False)
        return self._scale(encoded).to_numpy(dtype=float)

    # Thực hiện sửa dữ liệu, imputation, tạo feature và drop cột.
    def _prepare_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        data = df.copy()

        if "Price" in data.columns:
            data = data.drop(columns=["Price"])

        self._validate_schema(data)
        self._repair_invalid_values(data)
        self._prepare_categoricals(data)

        if fit:
            self._fit_imputation(data)

        self._apply_imputation(data)
        self._add_engineered_features(data, fit)

        existing_drop_columns = [col for col in self.drop_columns if col in data.columns]
        data = data.drop(columns=existing_drop_columns)

        return data.replace([np.inf, -np.inf], np.nan)

    # Kiểm tra các cột đầu vào bắt buộc trước khi tiền xử lý.
    def _validate_schema(self, data: pd.DataFrame) -> None:
        missing = [column for column in self.required_columns if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # Chuyển giá trị gốc không hợp lệ thành missing value và missing flag.
    def _repair_invalid_values(self, data: pd.DataFrame) -> None:
        if "BuildingArea" in data.columns:
            missing = data["BuildingArea"].isna() | (data["BuildingArea"] <= 0)
            data["BuildingArea_missing"] = missing.astype(float)
            data.loc[data["BuildingArea"] <= 0, "BuildingArea"] = np.nan

        if "YearBuilt" in data.columns:
            missing = data["YearBuilt"].isna() | (data["YearBuilt"] < 1800)
            data["YearBuilt_missing"] = missing.astype(float)
            data.loc[data["YearBuilt"] < 1800, "YearBuilt"] = np.nan

        if "Landsize" in data.columns:
            missing = data["Landsize"].isna() | (data["Landsize"] <= 0)
            data["Landsize_zero_or_missing"] = missing.astype(float)
            data.loc[data["Landsize"] <= 0, "Landsize"] = np.nan

    # Điền missing cho categorical và chuẩn hóa về chuỗi.
    def _prepare_categoricals(self, data: pd.DataFrame) -> None:
        for column in self.categorical_columns:
            if column not in data.columns:
                data[column] = "Unknown"
            data[column] = data[column].fillna("Unknown").astype(str)

    # Lưu median theo nhóm chỉ tính trên train data để impute numeric.
    def _fit_imputation(self, data: pd.DataFrame) -> None:
        self.imputation_values = {}

        for column in self.group_impute_columns:
            if column not in data.columns:
                continue

            global_median = data[column].median()
            if pd.isna(global_median):
                global_median = 0.0

            by_type = data.groupby("Type")[column].median().dropna().to_dict()
            self.imputation_values[column] = {
                "global": float(global_median),
                "by_type": {str(key): float(value) for key, value in by_type.items()},
            }

    # Áp dụng giá trị imputation đã lưu mà không fit lại.
    def _apply_imputation(self, data: pd.DataFrame) -> None:
        for column in self.group_impute_columns:
            values = self.imputation_values.get(column)
            if column not in data.columns or not isinstance(values, dict):
                continue

            fill_values = data["Type"].map(values["by_type"]).fillna(values["global"])
            data[column] = data[column].fillna(fill_values)

    # Tạo các feature nhà ở đơn giản từ cột gốc đã làm sạch.
    def _add_engineered_features(self, data: pd.DataFrame, fit: bool) -> None:
        years = self._sale_years(data, fit)

        if "YearBuilt" in data.columns:
            data["PropertyAge"] = (years - data["YearBuilt"]).clip(lower=0)

        rooms = data["Rooms"].replace(0, np.nan) if "Rooms" in data.columns else np.nan

        if {"Rooms", "Bathroom"}.issubset(data.columns):
            data["OtherRooms"] = (data["Rooms"] - data["Bathroom"]).clip(lower=0)

        if "BuildingArea" in data.columns:
            data["BuildingArea_per_Room"] = data["BuildingArea"] / rooms

        if {"BuildingArea", "Landsize"}.issubset(data.columns):
            data["BuildingCoverage"] = data["BuildingArea"] / data["Landsize"]
            data["BuildingCoverage"] = data["BuildingCoverage"].replace([np.inf, -np.inf], np.nan)
            data.loc[data["BuildingCoverage"] < 0, "BuildingCoverage"] = np.nan

    # Parse năm bán và lưu năm fallback từ train data.
    def _sale_years(self, data: pd.DataFrame, fit: bool) -> pd.Series:
        if "Date" in data.columns:
            years = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce").dt.year
        else:
            years = pd.Series(np.nan, index=data.index)

        if fit:
            median_year = years.median()
            self.sale_year = int(median_year) if not pd.isna(median_year) else 2017

        return years.fillna(self.sale_year)

    # Điền missing numeric còn lại bằng median đã fit.
    def _fill_remaining_numeric(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        non_numeric_columns = data.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric_columns:
            raise ValueError(f"Unexpected non-numeric columns before scaling: {non_numeric_columns}")

        numeric_data = data.copy()

        if fit:
            values = numeric_data.median().fillna(0.0).to_dict()
            self.imputation_values["numeric_fallbacks"] = {
                column: float(value) for column, value in values.items()
            }

        fallbacks = self.imputation_values.get("numeric_fallbacks", {})
        return numeric_data.fillna(fallbacks).fillna(0.0)

    # Lưu mean và standard deviation của từng feature chỉ từ train data.
    def _fit_scalers(self, data: pd.DataFrame) -> None:
        self.scalers = {}

        for column in data.columns:
            mean = data[column].mean()
            std = data[column].std(ddof=0)

            if pd.isna(std) or std == 0:
                std = 1.0

            self.scalers[column] = {"mean": float(mean), "std": float(std)}

    # Standardize feature bằng scaler đã fit.
    def _scale(self, data: pd.DataFrame) -> pd.DataFrame:
        scaled = data.copy()

        for column, params in self.scalers.items():
            scaled[column] = (scaled[column] - params["mean"]) / params["std"]

        return scaled
