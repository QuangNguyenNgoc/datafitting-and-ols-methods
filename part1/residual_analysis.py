import math
import matplotlib.pyplot as plt
import seaborn as sns

from utils.matrix_utils import matrix_vector_multiply, vector_subtract

from part1.ols_implementation import hat_matrix, model_metrics


def residual_plots(X: list, y: list, beta_hat: list):
    """
    Vẽ 4 biểu đồ phân tích phần dư (tương tự plot() trong R cho lm object):
    - Residuals vs Fitted values (Kiểm tra tính tuyến tính và phương sai không đổi)
    - Normal Q-Q plot của residuals (Kiểm tra phân phối chuẩn)
    - Scale-Location plot (Kiểm tra phương sai sai số không đổi - homoscedasticity)
    - Residuals vs Leverage plot (Phát hiện điểm ảnh hưởng/outliers)
    """
    X_mat = list(X)
    y_list = list(y)
    beta_list = list(beta_hat)

    n = len(X_mat)
    k = len(X_mat[0])

    # y_hat, residuals
    y_hat = matrix_vector_multiply(X_mat, beta_list)
    residuals = vector_subtract(y_list, y_hat)

    # Giá trị đòn bẩy h_ii
    H = hat_matrix(X_mat)
    leverage = [H[i][i] for i in range(n)]

    # RSS
    metrics = model_metrics(y_list, y_hat, p=k - 1)

    # sigma: phương sai sai số ước lượng
    sigma = math.sqrt(metrics["RSS"] / (n - k)) if n > k else 1e-12

    # Standardized/Studentized Residuals
    standardized_residuals = []
    for i in range(n):
        denom = sigma * math.sqrt(max(1.0 - leverage[i], 1e-12))
        standardized_residuals.append(residuals[i] / denom)

    # =========================================================================
    # khởi tạo khung đồ họa
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # BIỂU ĐỒ 1: Residuals vs Fitted
    sns.scatterplot(x=y_hat, y=residuals, ax=axes[0, 0], alpha=0.6, color="darkblue")
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Residuals vs Fitted", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")

    # BIỂU ĐỒ 2: Normal Q-Q plot
    sorted_std_res = sorted(standardized_residuals)
    theoretical_quantiles = []
    for i in range(1, n + 1):
        alpha = 0.375
        p_val = (i - alpha) / (n - 2 * alpha + 1)
        q = 4.91 * (p_val**0.14 - (1.0 - p_val) ** 0.14)
        theoretical_quantiles.append(q)

    sns.scatterplot(
        x=theoretical_quantiles,
        y=sorted_std_res,
        ax=axes[0, 1],
        alpha=0.6,
        color="darkgreen",
    )
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

    # BIỂU ĐỒ 3: Scale-Location plot
    sqrt_abs_std_res = [math.sqrt(abs(r)) for r in standardized_residuals]
    sns.scatterplot(
        x=y_hat, y=sqrt_abs_std_res, ax=axes[1, 0], alpha=0.6, color="purple"
    )
    axes[1, 0].set_title("Scale-Location", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel(r"$\sqrt{|\text{Standardized Residuals}|}$")

    # BIỂU ĐỒ 4: Residuals vs Leverage plot
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

    axes[1, 1].set_xlim(0, max(leverage) * 1.1)

    plt.tight_layout()
    plt.show()
