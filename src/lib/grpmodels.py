
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import minimize

def find_bad_design_columns(
    X: pd.DataFrame,
    *,
    exclude_cols=("GROUP_ID",),
    sample_pct: float | None = None,
    random_state: int = 123,
    threshold: float = 1e-8,
    min_nonzero_frac: float | None = 0.0005,
    min_unique_values: int | None = None,
    check_collinearity: bool = True,
    debug: bool = False
) -> list[str]:
    """
    Identify design-matrix columns likely to cause convergence slowness
    or numerical instability.

    Parameters
    ----------
    X:
        Input dataframe containing numeric design columns.

    exclude_cols:
        Columns to ignore, e.g. group_id, offset, response columns.

    sample_pct:
        Optional percentage of rows to sample, e.g. 10.0 for 10%.
        If None, uses all rows.

    random_state:
        Random seed for sampling.

    threshold:
        Eagerness threshold.
        Larger values are more aggressive.
        Used for near-zero variance and QR rank diagnostics.

    min_nonzero_frac:
        Minimum fraction of rows that must be nonzero.
        If None, defaults to max(threshold, 1 / n_rows).

    min_unique_values:
        Minimum number of unique values required.
        Columns with fewer unique values are dropped.

    check_collinearity:
        If True, performs pivoted QR to find near-linearly-dependent columns.

    Returns
    -------
    dropped_cols:
        List of columns recommended for dropping.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold should usually be between 0 and 1.")

    if sample_pct is not None:
        if not 0 < sample_pct <= 100:
            raise ValueError("sample_pct must be in (0, 100].")

        n_sample = max(1, int(len(X) * sample_pct / 100))
        X_work = X.sample(n=n_sample, random_state=random_state)
    else:
        X_work = X

    exclude_cols = set(exclude_cols)

    candidate_cols = [
        c for c in X_work.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(X_work[c])
    ]

    if not candidate_cols:
        return []

    X_num = X_work[candidate_cols].copy()

    # Check for infinities or NaNs and raise an error (avoid silently masking them)
    A = X_num.to_numpy(dtype=float)
    inf_cols = X_num.columns[np.isinf(A).any(axis=0)].tolist()
    nan_cols = X_num.columns[np.isnan(A).any(axis=0)].tolist()
    errors = []
    if inf_cols:
        errors.append(f"infinite values in columns: {inf_cols}")
    if nan_cols:
        errors.append(f"NaNs in columns: {nan_cols}")
    if errors:
        raise ValueError("Invalid values detected in design matrix: " + "; ".join(errors))

    n_rows = len(X_num)
    if min_nonzero_frac is None:
        min_nonzero_frac = max(threshold, 1.0 / max(n_rows, 1))

    dropped = set()

    # 1. Unique value / constant checks
    nunique = X_num.nunique(dropna=False)

    if min_unique_values is not None:
        for col in candidate_cols:
            if nunique[col] < min_unique_values:
                if debug: print(f"\tdebug: dropping {col} (min_unique_values)")
                dropped.add(col)

    # 2. Sparsity / rare nonzero checks
    nonzero_frac = (X_num != 0).mean(axis=0)

    if min_nonzero_frac is not None:
        for col in candidate_cols:
            if nonzero_frac[col] < min_nonzero_frac:
                if debug: print(f"\tdebug: dropping {col} (min_nonzero_frac)")
                dropped.add(col)

    # 3. Near-zero variance after centering/scaling logic
    # Use relative variance so large-scale columns are not unfairly penalized.
    means = X_num.mean(axis=0)
    stds = X_num.std(axis=0, ddof=0)

    abs_scale = np.maximum(np.abs(means), 1.0)
    relative_std = stds / abs_scale

    for col in candidate_cols:
        if relative_std[col] < threshold:
            if debug: print(f"\tdebug: dropping {col} (relative_std[col] < threshold)")
            dropped.add(col)

    # 4. QR-based approximate linear dependency check
    if check_collinearity:
        remaining_cols = [c for c in candidate_cols if c not in dropped]

        if len(remaining_cols) >= 2:
            A = X_num[remaining_cols].to_numpy(dtype=float)

            # Standardize for conditioning diagnostics.
            col_mean = A.mean(axis=0)
            col_std = A.std(axis=0)
            col_std = np.where(col_std > 0, col_std, 1.0)

            A_std = (A - col_mean) / col_std

            # Drop zero columns again defensively
            active = np.std(A_std, axis=0) > 0
            A_std_active = A_std[:, active]
            active_cols = [c for c, keep in zip(remaining_cols, active) if keep]

            if A_std_active.shape[1] >= 2:
                _, R, piv = qr(A_std_active, mode="economic", pivoting=True)

                diag = np.abs(np.diag(R))

                if len(diag) > 0:
                    # Relative to the strongest pivot.
                    rel_diag = diag / max(diag[0], 1e-300)

                    dependent_positions = np.where(rel_diag < threshold)[0]

                    # Pivoted QR puts stronger independent columns first.
                    # Columns appearing after weak pivots are the unstable ones.
                    for pos in dependent_positions:
                        bad_original_position = piv[pos]
                        bad_col = active_cols[bad_original_position]
                        if debug: print(f"\tdebug: dropping {bad_col} (dependent_positions)")
                        dropped.add(bad_col)

    return sorted(dropped)


class GroupedPoissonGLM:
    """
    Grouped Poisson GLM.

    Individual rows:
        eta_i = intercept + x_i @ beta + offset_i
        mu_i = exp(eta_i)

    Group likelihood:
        Mu_g = sum_{i in g} mu_i
        y_g ~ Poisson(Mu_g)

    The grouped response dataframe must have one row per group_id.
    """

    def __init__(
        self,
        group_col="GROUP_ID",
        y_col="y",
        offset_col=None,
        feature_cols=None,
        normalize=True,
        maxiter=500,
        tol=1e-8,
        bounds=None,
    ):
        self.group_col = group_col
        self.y_col = y_col
        self.offset_col = offset_col
        self.feature_cols = feature_cols
        self.normalize = normalize
        self.maxiter = maxiter
        self.tol = tol
        self.bounds = bounds

    def fit(self, X_df: pd.DataFrame, y_df: pd.DataFrame):
        
        group_col = self.group_col

        if y_df[group_col].duplicated().any():
            raise ValueError(f"`y_df` must have one row per `{group_col}`.")

        y_lookup = y_df.set_index(group_col)[self.y_col]
        group_index = y_lookup.index

        group_codes = group_index.get_indexer(X_df[group_col])
        if np.any(group_codes < 0):
            missing = X_df.loc[group_codes < 0, group_col].unique()[:10]
            raise ValueError(f"Some X groups are missing from y_df: {missing}")

        y_g = y_lookup.to_numpy(dtype=float)
        n_groups = len(y_g)

        if np.any(y_g < 0):
            raise ValueError("Poisson response must be nonnegative.")

        if self.feature_cols is None:
            exclude = {group_col}
            if self.offset_col is not None:
                exclude.add(self.offset_col)
            feature_cols = [c for c in X_df.columns if c not in exclude]
        else:
            feature_cols = list(self.feature_cols)

        X_raw = X_df[feature_cols].to_numpy(dtype=float)

        if self.offset_col is None:
            offset = np.zeros(len(X_df), dtype=float)
        else:
            offset = X_df[self.offset_col].to_numpy(dtype=float)

        if self.normalize:
            x_mean = X_raw.mean(axis=0)
            x_scale = X_raw.std(axis=0)
            x_scale = np.where(x_scale > 0, x_scale, 1.0)
            X = (X_raw - x_mean) / x_scale
        else:
            x_mean = np.zeros(X_raw.shape[1])
            x_scale = np.ones(X_raw.shape[1])
            X = X_raw

        n_features = X.shape[1]

        # Good intercept-only initializer:
        # total expected = exp(intercept) * sum(exp(offset))
        denom = np.exp(offset).sum()
        init_intercept = np.log((y_g.sum() + 1e-12) / (denom + 1e-12))
        theta0 = np.zeros(n_features + 1)
        theta0[0] = init_intercept

        if self.bounds is None:
            # unconstrained
            bounds = [(None, None)] * len(theta0)
        else:
            bounds = self.bounds

        last_theta = None
        last_nll = None
        last_grad = None
        
        def objective_and_grad(theta):
            
            nonlocal last_theta, last_nll, last_grad
            
            intercept = theta[0]
            beta = theta[1:]
            
            if np.all(theta == last_theta):
                return last_nll, last_grad
            last_theta = theta

            eta = intercept + X @ beta + offset

            # Prevent catastrophic overflow during early optimizer steps.
            # If this is hit often, consider tighter bounds or better initialization.
            eta = np.clip(eta, -745, 700)

            mu_i = np.exp(eta)
            mu_g = np.bincount(group_codes, weights=mu_i, minlength=n_groups)

            eps = 1e-30
            mu_g_safe = np.maximum(mu_g, eps)

            # Negative log-likelihood, dropping constant log(y!)
            nll = np.sum(mu_g_safe - y_g * np.log(mu_g_safe))

            # d nll / d mu_g = 1 - y_g / mu_g
            group_factor = 1.0 - y_g / mu_g_safe

            # Chain rule back to individual rows
            row_factor = mu_i * group_factor[group_codes]

            grad_intercept = row_factor.sum()
            grad_beta = X.T @ row_factor

            grad = np.empty_like(theta)
            grad[0] = grad_intercept
            grad[1:] = grad_beta

            last_nll = nll
            last_grad = grad
            
            return nll, grad

        result = minimize(
            fun=lambda th: objective_and_grad(th)[0],
            x0=theta0,
            jac=lambda th: objective_and_grad(th)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": self.tol},
        )

        theta_hat = result.x
        intercept_scaled = theta_hat[0]
        beta_scaled = theta_hat[1:]

        # Back-transform:
        # eta = a_s + sum_j beta_s_j * (x_j - mean_j) / scale_j
        #     = a_orig + sum_j beta_orig_j * x_j
        beta_orig = beta_scaled / x_scale
        intercept_orig = intercept_scaled - np.sum(beta_scaled * x_mean / x_scale)

        self.result_ = result
        self.feature_cols_ = feature_cols
        self.x_mean_ = x_mean
        self.x_scale_ = x_scale
        self.intercept_ = float(intercept_orig)
        self.coef_ = pd.Series(beta_orig, index=feature_cols)
        self.intercept_scaled_ = float(intercept_scaled)
        self.coef_scaled_ = pd.Series(beta_scaled, index=feature_cols)

        return self

    def predict_individual_mu(self, X_df: pd.DataFrame) -> np.ndarray:
        X = X_df[self.feature_cols_].to_numpy(dtype=float)

        if self.offset_col is None:
            offset = np.zeros(len(X_df), dtype=float)
        else:
            offset = X_df[self.offset_col].to_numpy(dtype=float)

        eta = self.intercept_ + X @ self.coef_.to_numpy() + offset
        eta = np.clip(eta, -745, 700)
        return np.exp(eta)

    def predict_group_mu(self, X_df: pd.DataFrame) -> pd.DataFrame:
        mu_i = self.predict_individual_mu(X_df)

        out = (
            pd.DataFrame({
                self.group_col: X_df[self.group_col].to_numpy(),
                "mu_individual": mu_i,
            })
            .groupby(self.group_col, as_index=False)["mu_individual"]
            .sum()
            .rename(columns={"mu_individual": "mu_group"})
        )

        return out
