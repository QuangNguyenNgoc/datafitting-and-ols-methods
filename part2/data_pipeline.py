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


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None) -> tuple:
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
    """A scikit-learn style data preprocessing pipeline.
    
    This class handles missing values, feature scaling, and categorical encoding.
    It maintains state from the training data to ensure no data leakage occurs
    when transforming the test data.
    
    Attributes:
        scalers (dict): Stored parameters for scaling (e.g., mean and std for each column).
        imputation_values (dict): Stored values for filling missing data (e.g., column means or modes).
        encoded_columns (list): Stored column names after one-hot encoding to ensure consistency.
    """

    def __init__(self):
        """Initializes the DataPipeline with empty state dictionaries."""
        self.scalers = {}
        self.imputation_values = {}
        self.encoded_columns = []

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fits the pipeline to the data and then transforms it.
        
        Calculates and stores imputation values, scaling parameters, and 
        encoding mapping based on the provided dataframe, and then applies 
        these transformations to the dataframe.

        Args:
            df (pd.DataFrame): The training dataframe to fit and transform.

        Returns:
            np.ndarray: The transformed training data as a NumPy array.

        Raises:
            NotImplementedError: If the method is not yet implemented.
        """
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms the data using the previously fitted state.
        
        Applies imputation, scaling, and encoding using the parameters
        stored in `self.imputation_values`, `self.scalers`, and `self.encoded_columns`
        to prevent data leakage into the test set.

        Args:
            df (pd.DataFrame): The test/unseen dataframe to transform.

        Returns:
            np.ndarray: The transformed data as a NumPy array.

        Raises:
            NotImplementedError: If the method is not yet implemented.
        """
        raise NotImplementedError


if __name__ == "__main__":
    print("Data Pipeline - OOP Skeleton Demo")
    # TODO: Thêm demo code
