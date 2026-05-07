"""
Residual Analysis
==================
Vẽ các biểu đồ phân tích phần dư để đánh giá các giả định của mô hình.
"""

import numpy as np
import matplotlib.pyplot as plt

def residual_plots(X, y, beta_hat):
    """
    7. Vẽ 4 biểu đồ phân tích phần dư (tương tự plot() trong R cho lm object):
       - Residuals vs Fitted values (Kiểm tra tính tuyến tính và phương sai không đổi)
       - Normal Q-Q plot của residuals (Kiểm tra phân phối chuẩn)
       - Scale-Location plot (Kiểm tra phương sai sai số không đổi - homoscedasticity)
       - Residuals vs Leverage plot (Phát hiện điểm ảnh hưởng/outliers)
    """
    # TODO: Tính phần dư (residuals), giá trị dự báo (fitted values), leverage
    # TODO: Vẽ 4 subplot tương ứng
    pass

if __name__ == "__main__":
    # TODO: Khởi tạo dữ liệu giả lập và mô hình OLS
    # TODO: Gọi hàm residual_plots để minh họa
    print("Residual Analysis - Demo")
