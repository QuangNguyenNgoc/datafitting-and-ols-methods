"""
Residual Analysis
==================
Vẽ các biểu đồ phân tích phần dư để đánh giá các giả định của mô hình.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def residual_plots(X, y, beta_hat):
    """
    7. Vẽ 4 biểu đồ phân tích phần dư (tương tự plot() trong R cho lm object):
       - Residuals vs Fitted values (Kiểm tra tính tuyến tính và phương sai không đổi)
       - Normal Q-Q plot của residuals (Kiểm tra phân phối chuẩn)
       - Scale-Location plot (Kiểm tra phương sai sai số không đổi - homoscedasticity)
       - Residuals vs Leverage plot (Phát hiện điểm ảnh hưởng/outliers)
    """
    X_mat = np.array(X)
    y_arr = np.array(y).flatten()
    beta_arr = np.array(beta_hat).flatten()
    
    y_hat = X_mat @ beta_arr
    residuals = y_arr - y_hat
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.scatterplot(x=y_hat, y=residuals, ax=axes[0], alpha=0.6)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals vs Fitted')
    axes[0].set_xlabel('Fitted Values (y_hat)')
    axes[0].set_ylabel('Residuals (e)')
    
    sns.histplot(residuals, kde=True, ax=axes[1], color='blue')
    axes[1].set_title('Distribution of Residuals')
    axes[1].set_xlabel('Residuals (e)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # TODO: Khởi tạo dữ liệu giả lập và mô hình OLS
    # TODO: Gọi hàm residual_plots để minh họa
    print("Residual Analysis - Demo")
