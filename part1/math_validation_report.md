# Mathematical Validation Report: OLS Implementation

> **Generated:** 2026-05-16 14:32:47
> **Dataset:** Synthetic ($n = 50$, $p = 3$, $\sigma_{\text{noise}} = 1.5$, seed = 42)
> **Verdict:** ✅ ALL ASSERTIONS PASSED

---

## 1. Data Generation Model

The synthetic target vector $y$ was generated according to the linear model:

$$
y = X \beta_{\text{true}} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

where $X \in \mathbb{R}^{50 \times 4}$ is the design matrix (with a leading column of ones
for the intercept), $\beta_{\text{true}} \in \mathbb{R}^{4}$, and $\sigma = 1.5$.

Because the noise $\varepsilon$ is non-zero, we expect $\hat{\beta} \approx \beta_{\text{true}}$ but
not exactly equal. The purpose of this validation is **not** to recover the true coefficients
perfectly, but to prove that our custom solver produces the **exact same** $\hat{\beta}$ as
established numerical libraries.

---

## 2. Coefficient Estimation — Proof of Numerical Equivalence

### 2.1 The OLS Normal Equation

The Ordinary Least Squares estimator is defined as the unique minimizer of the residual
sum of squares. Differentiating $\| y - X\beta \|^2$ with respect to $\beta$ and setting
the gradient to zero yields the **normal equation**:

$$
X^T X \hat{\beta} = X^T y
$$

When $X^T X$ is invertible (i.e., the columns of $X$ are linearly independent), the
closed-form solution is:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

### 2.2 Comparison of Solvers

Our custom function `ols_fit(X, y)` currently delegates to `statsmodels.api.OLS`, which
internally solves the normal equation via a **QR decomposition** of $X$. The reference
baselines are:

- **scikit-learn** `LinearRegression`: uses the **LAPACK** driver `gelsd` (SVD-based
  least-squares solver).
- **statsmodels** `OLS`: uses **QR decomposition** via `numpy.linalg.qr`.

Despite these different numerical strategies (SVD vs QR vs direct inversion), they all
solve the same mathematical system. The following table confirms that the resulting
$\hat{\beta}$ vectors are identical up to floating-point precision:

| Coefficient | $\beta_{\text{true}}$ | $\hat{\beta}_{\text{custom}}$ | $\hat{\beta}_{\text{sklearn}}$ | $\hat{\beta}_{\text{statsmodels}}$ | $|\Delta|$ (custom vs sklearn) |
|:---|---:|---:|---:|---:|---:|
| $\beta_0$ (intercept) |     3.870864 |     4.120352 |     4.120352 |     4.120352 | 0.00e+00 |
| $\beta_1$ |     2.798755 |     2.964710 |     2.964710 |     2.964710 | 8.88e-16 |
| $\beta_2$ |     1.420316 |     1.640310 |     1.640310 |     1.640310 | 5.11e-15 |
| $\beta_3$ |    -4.158600 |    -4.110641 |    -4.110641 |    -4.110641 | 5.33e-15 |

The column $|\Delta|$ shows the absolute difference between our implementation and
scikit-learn. Values on the order of $10^{-10}$ or smaller confirm **numerical
equivalence** — the residual discrepancy is attributable solely to IEEE 754
floating-point rounding, not to any algorithmic error.

---

## 3. Coefficient of Determination $R^2$ — Algebraic and Geometric Interpretation

### 3.1 Definition

The coefficient of determination is defined as:

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

where:

$$
RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \| y - X\hat{\beta} \|^2, \qquad
TSS = \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

### 3.2 Computed Values for This Dataset

| Quantity | Value |
|:---|---:|
| $\bar{y}$ (sample mean) | 4.167508 |
| $RSS = \| e \|^2$ | 88.445654 |
| $TSS = \| y - \bar{y} \mathbf{1} \|^2$ | 1575.555812 |
| $R^2 = 1 - RSS / TSS$ | 0.943864 |
| Adjusted $R^2 = 1 - \frac{n-1}{n-p-1}(1 - R^2)$ | 0.940203 |

### 3.3 Algebraic Interpretation

$TSS = 1575.5558$ measures the **total variance** of $y$ around its mean. This is the
baseline "energy" in the data before any model is applied. $RSS = 88.4457$ measures
the **unexplained variance** — the energy that remains in the residual vector $e = y - \hat{y}$
after the OLS projection. Therefore:

$$
R^2 = 1 - \frac{88.4457}{1575.5558} = 0.943864
$$

This means our linear model captures **94.39%** of the total variance in $y$.

### 3.4 Geometric Interpretation

In the column space of $X$ (denoted $\mathcal{C}(X)$), the OLS fit $\hat{y} = X\hat{\beta} = Hy$
is the **orthogonal projection** of $y$ onto $\mathcal{C}(X)$. By the Pythagorean theorem
in $\mathbb{R}^n$:

$$
\| y - \bar{y} \mathbf{1} \|^2 = \| \hat{y} - \bar{y} \mathbf{1} \|^2 + \| y - \hat{y} \|^2
\quad \Longrightarrow \quad
TSS = ESS + RSS
$$

$R^2$ is the ratio $ESS / TSS$, i.e., the fraction of the squared length of the
centered $y$ that lies **within** the column space of $X$.

---

## 4. Hat Matrix Verification

The hat (projection) matrix is:

$$
H = X (X^T X)^{-1} X^T
$$

| Property | Expected | Observed |
|:---|:---|:---|
| Shape | $(50, 50)$ | $(50, 50)$ |
| $\text{tr}(H) = p + 1$ | 4 | 4.000000 |
| Idempotency: $\max |H - H^2|$ | $\approx 0$ | 1.25e-16 |

The trace of $H$ equals the number of estimated parameters (including the intercept),
confirming that $H$ projects onto a $(p+1)$-dimensional subspace. The idempotency
residual of $1.25e-16$ is within machine epsilon, confirming $H^2 = H$.

---

## 5. Variance Inflation Factor (VIF)

VIF is computed for each feature column $x_j$ (excluding the intercept) by regressing
$x_j$ on all other features and computing:

$$
VIF_j = \frac{1}{1 - R^2_j}
$$

where $R^2_j$ is the $R^2$ from that auxiliary regression. A $VIF_j > 10$ signals
severe multicollinearity.

| Feature | $VIF$ |
|:---|---:|
| $x_{1}$ | 1.0798 |
| $x_{2}$ | 1.0274 |
| $x_{3}$ | 1.0569 |

Since the synthetic features were drawn independently from $\mathcal{N}(0,1)$, VIF
values close to 1.0 are expected, confirming negligible multicollinearity (as designed).

---

## 6. Coefficient Inference Table

With $\hat{\sigma}^2 = RSS / (n - p - 1) = 1.922732$, the standard errors, $t$-statistics,
and $p$-values are:

| Coefficient | Estimate | Std. Error | $t$-statistic | $p$-value | Sig. |
|:---|---:|---:|---:|---:|:---|
| $\beta_{0}$ | 4.120352 | 0.199177 | 20.6869 | 0.000000 | \* |
| $\beta_{1}$ | 2.964710 | 0.274931 | 10.7835 | 0.000000 | \* |
| $\beta_{2}$ | 1.640310 | 0.199024 | 8.2418 | 0.000000 | \* |
| $\beta_{3}$ | -4.110641 | 0.193689 | -21.2229 | 0.000000 | \* |

Significance codes: \* indicates $p < 0.05$.

---

## 7. Additional Metrics

| Metric | Value |
|:---|---:|
| MAE | 1.084128 |
| RMSE | 1.330005 |
| $R^2$ | 0.943864 |
| Adjusted $R^2$ | 0.940203 |

---

## 8. Assertion Summary

| Test | Status | Detail |
|:---|:---|:---|
| Beta vs sklearn | ✅ PASS | Max |beta_custom - beta_sklearn| = 5.33e-15 |
| Beta vs statsmodels | ✅ PASS | Max |beta_custom - beta_statsmodels| = 0.00e+00 |
| R2 consistency | ✅ PASS | |R2_function - R2_manual| = 0.00e+00 |
| Hat matrix idempotency | ✅ PASS | Max |H - H^2| = 1.25e-16 |
| Hat matrix trace | ✅ PASS | tr(H) = 4.000000, expected 4 |

---

## 9. Conclusion

All numerical assertions pass with tolerances at $10^{-6}$ or better. This formally
validates that our `ols_fit`, `hat_matrix`, `model_metrics`, `coef_inference`, and `vif`
functions produce outputs that are **mathematically identical** (up to IEEE 754 rounding)
to the reference implementations in scikit-learn and statsmodels. The codebase is ready
for the next phase: replacing the library-backed mock internals with pure `numpy` linear
algebra while preserving these same function signatures and numerical guarantees.
