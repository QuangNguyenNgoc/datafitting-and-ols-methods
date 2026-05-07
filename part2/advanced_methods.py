"""
Advanced Methods
=================
Phương pháp nâng cao: Kernel Regression, Bayesian Regression (nếu có).
"""

import numpy as np


# ============================================================
# Kernel Regression
# ============================================================

class KernelRegression:
    """Kernel Regression sử dụng Nadaraya-Watson estimator."""

    def __init__(self, kernel="rbf", bandwidth=1.0):
        """
        Parameters
        ----------
        kernel : str
            Loại kernel: 'rbf', 'polynomial', 'linear'.
        bandwidth : float
            Bandwidth parameter (h).
        """
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit(self, X, y):
        """Lưu trữ dữ liệu training (non-parametric)."""
        # TODO: Implement
        raise NotImplementedError

    def predict(self, X):
        """Dự đoán sử dụng kernel weights."""
        # TODO: Implement
        raise NotImplementedError


# ============================================================
# Bayesian Linear Regression
# ============================================================

class BayesianLinearRegression:
    """Bayesian Linear Regression với prior Gaussian."""

    def __init__(self, alpha=1.0, beta=1.0):
        """
        Parameters
        ----------
        alpha : float
            Precision of the prior (inverse variance).
        beta : float
            Precision of the noise (inverse variance).
        """
        self.alpha = alpha
        self.beta = beta
        self.posterior_mean = None
        self.posterior_cov = None

    def fit(self, X, y):
        """
        Tính posterior distribution của weights.
        """
        # TODO: Implement Bayesian fitting
        raise NotImplementedError

    def predict(self, X, return_std=False):
        """
        Dự đoán với uncertainty estimation.

        Returns
        -------
        y_pred : np.ndarray
        y_std : np.ndarray (if return_std=True)
        """
        # TODO: Implement prediction with uncertainty
        raise NotImplementedError


if __name__ == "__main__":
    print("Advanced Methods - Demo")
    # TODO: Thêm demo code
