from part1.ols_implementation import ols_fit
from utils.matrix_utils import matrix_vector_multiply


def kfold_cv(X: list, y: list, k: int = 5) -> float:
    """
    Cài đặt k-fold cross-validation và tính CV score
    Quy trình:
    - Chia tập dữ liệu thành k fold (phần) bằng nhau
    - Với mỗi fold i (từ 1 đến k):
        + Dùng fold i làm tập validation
        + Dùng k-1 folds còn lại làm tập train
        + Huấn luyện mô hình (tính beta) trên tập train
        + Tính error/score trên tập validation
    - Trả về điểm trung bình của k folds (CV score).
    """
    n = len(X)
    if k <= 1 or k > n:
        raise ValueError(
            "Số lượng fold (k) phải lớn hơn 1 và nhỏ hơn hoặc bằng số lượng mẫu."
        )
    indices = list(range(n))
    fold_size = n // k

    mse_scores = []

    for i in range(k):
        val_start = i * fold_size
        # nếu là fold cuối cùng, vét toàn bộ các mẫu còn lại
        val_end = n if i == k - 1 else (i + 1) * fold_size

        # tách tập Train và Validation
        val_indices = set(indices[val_start:val_end])
        train_indices = [idx for idx in indices if idx not in val_indices]

        # chuyển đổi dữ liệu thành list
        X_train = [X[idx] for idx in train_indices]
        y_train = [y[idx] for idx in train_indices]
        X_val = [X[idx] for idx in val_indices]
        y_val = [y[idx] for idx in val_indices]

        # huấn luyện mô hình trên tập Train
        beta_hat = ols_fit(X_train, y_train)

        # dự đoán trên tập Validation
        y_pred = matrix_vector_multiply(X_val, beta_hat)

        # MSE
        m = len(y_val)
        fold_mse = sum((y_val[j] - y_pred[j]) ** 2 for j in range(m)) / m
        mse_scores.append(fold_mse)

    # CV score
    cv_score = sum(mse_scores) / k
    return cv_score
