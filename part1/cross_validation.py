"""
Cross-Validation
=================
Cài đặt k-fold cross-validation từ đầu.
"""

import numpy as np

def kfold_cv(X, y, k):
    """
    8. Cài đặt k-fold cross-validation và tính CV score.
    
    Quy trình:
        - Chia tập dữ liệu thành k fold (phần) bằng nhau.
        - Với mỗi fold i (từ 1 đến k):
            + Dùng fold i làm tập validation.
            + Dùng k-1 folds còn lại làm tập train.
            + Huấn luyện mô hình (tính beta) trên tập train.
            + Tính error/score trên tập validation.
        - Trả về điểm trung bình của k folds (CV score).
    """
    # TODO: Implement k-fold splitting and scoring logic
    pass

if __name__ == "__main__":
    # TODO: Khởi tạo dữ liệu giả lập
    # TODO: Gọi hàm kfold_cv
    # TODO: Kiểm chứng với sklearn.model_selection.KFold và sklearn.model_selection.cross_val_score
    print("Cross-Validation - Demo")
