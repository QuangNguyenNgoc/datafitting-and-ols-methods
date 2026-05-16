import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

def ols_fit(X, y):
    """
    Mock implementation of Ordinary Least Squares fit.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        Design matrix (assumed to include a constant column if an intercept is desired).
    y : array-like, shape (n_samples,)
        Target vector.
        
    Returns:
    beta_hat : numpy array, shape (n_features,)
        Estimated coefficients.
    """
    model = sm.OLS(y, X).fit()
    # Ensure it returns a numpy array for consistency
    return np.array(model.params)

def hat_matrix(X):
    """
    Mock implementation of the Hat matrix: H = X(X^T X)^{-1} X^T
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
    
    Returns:
    H : numpy array, shape (n_samples, n_samples)
        The projection matrix.
    """
    X_mat = np.array(X)
    # Using pseudo-inverse for numerical stability and mock simplicity
    return X_mat @ np.linalg.pinv(X_mat)

def model_metrics(y, y_hat, p):
    """
    Calculate and return MAE, RMSE, R2, and Adjusted R2.
    
    Parameters:
    y : array-like, shape (n_samples,)
        True target values.
    y_hat : array-like, shape (n_samples,)
        Predicted target values.
    p : int
        Number of predictors (excluding the intercept).
        
    Returns:
    metrics : dict
        Dictionary containing MAE, RMSE, R2, and Adj_R2.
    """
    y = np.array(y).flatten()
    y_hat = np.array(y_hat).flatten()
    n = len(y)
    
    mae = mean_absolute_error(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    r2 = r2_score(y, y_hat)
    
    # Adjusted R2 calculation
    if n - p - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan
        
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Adj_R2': adj_r2
    }

def coef_inference(X, y, beta_hat, sigma2):
    """
    Mock implementation for coefficient statistical inference.
    Returns Standard Errors, t-statistics, and p-values.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    beta_hat : array-like (unused in mock)
    sigma2 : float (unused in mock)
    
    Returns:
    inference_df : pandas DataFrame
        DataFrame containing Coefficient, Std_Error, t_stat, and p_value.
    """
    # Using statsmodels to quickly mock the inference table
    model = sm.OLS(y, X).fit()
    
    inference_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std_Error': model.bse,
        't_stat': model.tvalues,
        'p_value': model.pvalues
    })
    
    return inference_df

def vif(X):
    """
    Mock implementation of Variance Inflation Factor calculation.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
    
    Returns:
    vif_df : pandas DataFrame
        DataFrame containing feature index/names and their VIF scores.
    """
    X_df = pd.DataFrame(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    
    vif_values = []
    for i in range(X_df.shape[1]):
        try:
            val = variance_inflation_factor(X_df.values, i)
        except Exception:
            val = np.inf
        vif_values.append(val)
        
    vif_data["VIF_Score"] = vif_values
    return vif_data

def ridge_fit(X, y, lam):
    """
    Mock implementation of Ridge Regression fit.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    lam : float
        Regularization parameter (lambda).
        
    Returns:
    beta_hat_ridge : numpy array, shape (n_features,)
        Estimated coefficients for Ridge regression.
    """
    # fit_intercept=False ensures it aligns with math models where X has a constant
    model = Ridge(alpha=lam, fit_intercept=False)
    model.fit(X, y)
    return np.array(model.coef_).flatten()

def residual_plots(X, y, beta_hat):
    """
    Mock implementation of residual diagnostic plots.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    beta_hat : array-like, shape (n_features,)
    """
    X_mat = np.array(X)
    y_arr = np.array(y).flatten()
    beta_arr = np.array(beta_hat).flatten()
    
    y_hat = X_mat @ beta_arr
    residuals = y_arr - y_hat
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Residuals vs Fitted Plot
    sns.scatterplot(x=y_hat, y=residuals, ax=axes[0], alpha=0.6)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals vs Fitted')
    axes[0].set_xlabel('Fitted Values (y_hat)')
    axes[0].set_ylabel('Residuals (e)')
    
    # 2. Histogram of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color='blue')
    axes[1].set_title('Distribution of Residuals')
    axes[1].set_xlabel('Residuals (e)')
    
    plt.tight_layout()
    plt.show()
