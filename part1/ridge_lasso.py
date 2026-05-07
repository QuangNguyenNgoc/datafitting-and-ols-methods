"""
Ridge Regression
=================
Cài đặt Ridge Regression và vẽ Ridge Trace.
"""

import numpy as np
import matplotlib.pyplot as plt

def ridge_fit(X, y, lam):
    """
    6. Cài đặt Ridge Regression và vẽ ridge trace.
    
    Công thức toán học:
        beta_ridge = (X^T X + lam * I)^{-1} X^T y
        
    Ghi chú:
        Cần lặp qua các giá trị lambda khác nhau để vẽ đồ thị Ridge Trace,
        thể hiện sự thay đổi của các hệ số hồi quy theo lambda.
    """
    # TODO: Implement Ridge fitting logic and plotting
    pass

if __name__ == "__main__":
    # TODO: Khởi tạo dữ liệu giả lập có hiện tượng đa cộng tuyến
    # TODO: Gọi hàm ridge_fit để minh họa
    # TODO: Kiểm chứng kết quả với sklearn.linear_model.Ridge
    print("Ridge Regression - Demo")
