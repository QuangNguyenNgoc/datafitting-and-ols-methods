import math
import matplotlib.pyplot as plt
import seaborn as sns

from utils.inverse import inverse
from utils.matrix_utils import mat_mul, mat_transpose


def residual_plots(X, y, beta_hat):
    """
    Vẽ 4 biểu đồ phân tích phần dư (tương tự plot() trong R cho lm object):
    - Residuals vs Fitted values (Kiểm tra tính tuyến tính và phương sai không đổi)
    - Normal Q-Q plot của residuals (Kiểm tra phân phối chuẩn)
    - Scale-Location plot (Kiểm tra phương sai sai số không đổi - homoscedasticity)
    - Residuals vs Leverage plot (Phát hiện điểm ảnh hưởng/outliers)
    """
    # 1. Chuyển đổi dữ liệu sang List thuần để tính toán lõi
    X_mat = list(X)
    y_list = list(y)
    beta_list = list(beta_hat)

    n = len(X_mat)
    k = len(X_mat[0])  # Số hệ số bao gồm cả bias

    # 2. Tính toán các đại lượng cơ bản bằng toán thuần Python
    # y_hat = X @ beta
    y_hat = [sum(X_mat[i][j] * beta_list[j] for j in range(k)) for i in range(n)]
    # residuals (e) = y - y_hat
    residuals = [y_list[i] - y_hat[i] for i in range(n)]

    # 3. Tính toán các đại lượng nâng cao (Leverage và Studentized Residuals)
    # Công thức ma trận đòn bẩy H (Hat Matrix): H = X @ (X^T @ X)^(-1) @ X^T
    # Giá trị đòn bẩy h_ii của mẫu i chính là phần tử nằm trên đường chéo chính của ma trận H
    X_T = mat_transpose(X_mat)
    X_T_X_inv = inverse(mat_mul(X_T, X_mat))

    # Tính nhanh đường chéo h_ii mà không cần sinh toàn bộ ma trận vuông n x n
    leverage = []
    for i in range(n):
        # dòng_i_of_X @ X_T_X_inv @ cột_i_of_X_T (chính là dòng_i_of_X)
        row_i = X_mat[i]
        # temp = row_i @ X_T_X_inv
        temp = [sum(row_i[j] * X_T_X_inv[j][m] for j in range(k)) for m in range(k)]
        # h_ii = temp @ row_i
        h_ii = sum(temp[m] * row_i[m] for m in range(k))
        leverage.append(h_ii)

    # Tính phương sai sai số ước lượng (sigma^2)
    rss = sum(e_i**2 for e_i in residuals)
    sigma2 = rss / (n - k) if (n - k) > 0 else 1e-12
    sigma = math.sqrt(sigma2)

    # Tính phần dư chuẩn hóa (Standardized/Studentized Residuals): r_i = e_i / (sigma * sqrt(1 - h_ii))
    standardized_residuals = []
    for i in range(n):
        denom = sigma * math.sqrt(max(1.0 - leverage[i], 1e-12))
        standardized_residuals.append(residuals[i] / denom)

    # 4. KHỞI TẠO KHUNG ĐỒ HỌA 2x2 ĐÚNG CHUẨN R
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ----------------------------------------------------
    # BIỂU ĐỒ 1: Residuals vs Fitted
    # ----------------------------------------------------
    sns.scatterplot(x=y_hat, y=residuals, ax=axes[0, 0], alpha=0.6, color="darkblue")
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    # Đường gợn sóng LOWESS xấp xỉ bằng xu hướng trung bình trượt thô sơ để thấy độ cong
    axes[0, 0].set_title("Residuals vs Fitted", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")

    # ----------------------------------------------------
    # BIỂU ĐỒ 2: Normal Q-Q plot
    # ----------------------------------------------------
    # Sắp xếp phần dư chuẩn hóa tăng dần để tìm phân vị thực nghiệm
    sorted_std_res = sorted(standardized_residuals)
    # Tính phân vị lý thuyết của phân phối chuẩn Blom's method
    theoretical_quantiles = []
    for i in range(1, n + 1):
        alpha = 0.375
        p_val = (i - alpha) / (n - 2 * alpha + 1)
        # Hàm ngược phân phối chuẩn tích lũy xấp xỉ (Inverse CDF)
        # Sử dụng xấp xỉ tỉ lệ phân vị chuẩn tắc
        q = 4.91 * (p_val**0.14 - (1.0 - p_val) ** 0.14)
        theoretical_quantiles.append(q)

    sns.scatterplot(
        x=theoretical_quantiles,
        y=sorted_std_res,
        ax=axes[0, 1],
        alpha=0.6,
        color="darkgreen",
    )
    # Vẽ đường thẳng 45 độ lý thuyết đi qua gốc
    axes[0, 1].plot(
        [min(theoretical_quantiles), max(theoretical_quantiles)],
        [min(theoretical_quantiles), max(theoretical_quantiles)],
        color="red",
        linestyle="--",
        linewidth=1.5,
    )
    axes[0, 1].set_title("Normal Q-Q", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Theoretical Quantiles")
    axes[0, 1].set_ylabel("Standardized Residuals")

    # ----------------------------------------------------
    # BIỂU ĐỒ 3: Scale-Location plot
    # ----------------------------------------------------
    # Trục y là căn bậc hai của trị tuyệt đối phần dư chuẩn hóa: sqrt(|r_i|)
    sqrt_abs_std_res = [math.sqrt(abs(r)) for r in standardized_residuals]
    sns.scatterplot(
        x=y_hat, y=sqrt_abs_std_res, ax=axes[1, 0], alpha=0.6, color="purple"
    )
    axes[1, 0].set_title("Scale-Location", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel(r"$\sqrt{|\text{Standardized Residuals}|}$")

    # ----------------------------------------------------
    # BIỂU ĐỒ 4: Residuals vs Leverage plot
    # ----------------------------------------------------
    sns.scatterplot(
        x=leverage,
        y=standardized_residuals,
        ax=axes[1, 1],
        alpha=0.6,
        color="darkorange",
    )
    axes[1, 1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1, 1].set_title("Residuals vs Leverage", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Leverage")
    axes[1, 1].set_ylabel("Standardized Residuals")

    # Khống chế giới hạn trục x của đòn bẩy để tránh các điểm quá dị biệt làm méo đồ thị
    axes[1, 1].set_xlim(0, max(leverage) * 1.1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # TODO: Khởi tạo dữ liệu giả lập và mô hình OLS
    # TODO: Gọi hàm residual_plots để minh họa
    print("Residual Analysis - Demo")
