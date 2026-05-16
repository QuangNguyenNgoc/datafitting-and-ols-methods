import numpy as np

def generate_synthetic_data(n=50, p=3, noise_std=1.5, random_state=42):
    """
    Generates a synthetic dataset specifically for unit-testing OLS functions.
    
    Parameters:
    n : int
        Number of samples (rows). Default is 50.
    p : int
        Number of features (excluding the intercept). Default is 3.
    noise_std : float
        Standard deviation of the Gaussian noise added to y.
    random_state : int
        Random seed for reproducibility during testing.
        
    Returns:
    X : numpy.ndarray, shape (n, p + 1)
        The design matrix, including a leading column of ones for the bias/intercept.
    y : numpy.ndarray, shape (n,)
        The continuous target variable.
    true_beta : numpy.ndarray, shape (p + 1,)
        The true coefficients used to generate y (useful for verifying model accuracy).
    """
    np.random.seed(random_state)
    
    # 1. Generate random continuous features X (size: n x p)
    X_features = np.random.randn(n, p)
    
    # 2. Append a column of ones at the beginning for the intercept (bias)
    intercept = np.ones((n, 1))
    X = np.hstack((intercept, X_features))
    
    # 3. Define the "true" beta coefficients (including the intercept at index 0)
    # Using small integers/floats for readability during testing
    true_beta = np.random.uniform(-5, 5, size=(p + 1,))
    
    # 4. Generate the target variable y = X * beta + error (noise)
    noise = np.random.randn(n) * noise_std
    y = X @ true_beta + noise
    
    return X, y, true_beta

if __name__ == "__main__":
    # Quick execution to verify shapes and outputs
    X, y, true_beta = generate_synthetic_data(n=50, p=3)
    
    print("=== Synthetic Data Generated for Unit Testing ===")
    print(f"X shape : {X.shape} (Includes {X.shape[1]-1} features + 1 intercept)")
    print(f"y shape : {y.shape}")
    print(f"Beta    : {true_beta.shape}")
    
    print("\n--- First 3 rows of Design Matrix X ---")
    print(np.round(X[:3], 3))
    
    print("\n--- First 3 values of Target y ---")
    print(np.round(y[:3], 3))
    
    print("\n--- True Beta coefficients (Intercept is the first value) ---")
    print(np.round(true_beta, 3))
