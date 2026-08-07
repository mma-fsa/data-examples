
"""Grouped Poisson GLM helpers and stepwise selection."""

# isort: skip_file

import formulaic as frm
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.linalg import qr
from scipy.optimize import minimize
from typing import cast

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
    column_index = X_num.columns.to_numpy()
    inf_cols = column_index[np.isinf(A).any(axis=0)].tolist()
    nan_cols = column_index[np.isnan(A).any(axis=0)].tolist()
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
                qr_result = cast(tuple[np.ndarray, np.ndarray, np.ndarray], qr(A_std_active, mode="economic", pivoting=True))
                R = qr_result[1]
                piv = qr_result[2]

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


def _coerce_warm_start_theta(
    warm_start: pd.Series | np.ndarray | list[float] | tuple[float, ...] | None,
    *,
    feature_cols: list[str],
    initial_intercept: float,
) -> np.ndarray | None:
    if warm_start is None:
        return None

    param_names = ["intercept", *feature_cols]
    theta0 = np.zeros(len(param_names), dtype=float)
    theta0[0] = initial_intercept

    if isinstance(warm_start, pd.Series):
        aligned = warm_start.reindex(param_names)
        for idx, value in enumerate(aligned.to_numpy()):
            if pd.notna(value):
                theta0[idx] = float(value)
        return theta0

    theta_array = np.asarray(warm_start, dtype=float)
    if theta_array.ndim != 1:
        raise ValueError("warm_start must be a 1D sequence or named pd.Series.")

    if theta_array.shape[0] != len(param_names):
        raise ValueError(
            "warm_start sequence must have length equal to intercept plus feature count."
        )

    return theta_array.copy()


def _pack_coef_series(intercept: float, coef: pd.Series) -> pd.Series:
    return pd.concat([
        pd.Series({"intercept": float(intercept)}),
        coef.astype(float),
    ])


def _profile_intercept(y_total: float, linear_predictor: np.ndarray) -> float:
    y_total_safe = max(float(y_total), 1e-300)
    return float(np.log(y_total_safe) - logsumexp(linear_predictor))


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
        self.theta_: pd.Series | None = None
        self.log_likelihood_: float | None = None
        self.n_params_: int | None = None
        self.n_obs_: int | None = None
        self.bic_: float | None = None
        self.model_spec_: object | None = None
        self.formula_: str = ""
        self.base_feature_cols_: list[str] | None = None
        self.selected_terms_: list[str] | None = None
        self.context_: dict[str, object] | None = None
        self.step_records_: list[dict[str, object]] | None = None
        self.screening_records_: list[dict[str, object]] | None = None
        self._steps_df: pd.DataFrame = pd.DataFrame()
        self.final_fit_: GroupedPoissonGLM | None = None

    def _compute_bic(self, *, nll: float, n_params: int, n_obs: int) -> float:
        return float(2.0 * nll + n_params * np.log(max(n_obs, 1)))

    def fit(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame,
        warm_start: pd.Series | np.ndarray | list[float] | tuple[float, ...] | None = None,
    ):
        
        group_col = self.group_col

        if y_df[group_col].duplicated().any():
            raise ValueError(f"`y_df` must have one row per `{group_col}`.")

        y_lookup = y_df.set_index(group_col)[self.y_col]
        group_index = y_lookup.index

        group_codes = group_index.get_indexer(pd.Index(X_df[group_col]))
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

        # Good beta initializer:
        # intercept is profiled out of the optimization entirely.
        theta0 = _coerce_warm_start_theta(
            warm_start,
            feature_cols=feature_cols,
            initial_intercept=0.0,
        )
        if theta0 is None:
            theta0 = np.zeros(n_features + 1)
        beta0 = np.asarray(theta0[1:], dtype=float)

        if self.bounds is None:
            # unconstrained
            bounds = [(None, None)] * len(beta0)
        else:
            if len(self.bounds) == len(theta0):
                bounds = list(self.bounds[1:])
            elif len(self.bounds) == len(beta0):
                bounds = list(self.bounds)
            else:
                raise ValueError(
                    "bounds must have either one entry per feature or one entry per feature plus intercept."
                )

        last_theta = None
        last_nll = None
        last_grad = None
        
        def objective_and_grad(theta):
            
            nonlocal last_theta, last_nll, last_grad
            
            beta = theta
            
            if last_theta is not None and np.array_equal(theta, last_theta):
                return last_nll, last_grad
            last_theta = theta

            eta_no_intercept = X @ beta + offset
            intercept = _profile_intercept(y_g.sum(), eta_no_intercept)
            eta = intercept + eta_no_intercept

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

            grad_beta = X.T @ row_factor

            grad = np.empty_like(theta)
            grad[:] = grad_beta

            last_nll = nll
            last_grad = grad
            
            return nll, grad

        result = minimize(
            fun=lambda th: objective_and_grad(th)[0],
            x0=beta0,
            jac=lambda th: objective_and_grad(th)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": self.tol},
        )

        beta_scaled = result.x
        intercept_scaled = _profile_intercept(y_g.sum(), X @ beta_scaled + offset)

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
        final_nll = float(result.fun)
        self.log_likelihood_ = float(-final_nll)
        self.n_params_ = int(len(feature_cols) + 1)
        self.n_obs_ = int(n_groups)
        self.bic_ = self._compute_bic(nll=final_nll, n_params=self.n_params_, n_obs=self.n_obs_)
        self.theta_ = _pack_coef_series(self.intercept_, self.coef_)

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

        return (
            pd.DataFrame({
                self.group_col: X_df[self.group_col].to_numpy(),
                "mu_individual": mu_i,
            })
            .groupby(self.group_col, as_index=False)
            .agg(mu_group=("mu_individual", "sum"))
        )


class StepWiseGroupedPoissonGLM(GroupedPoissonGLM):
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
        eval_terms: list[str] | tuple[str, ...] | None = None,
        context: dict[str, object] | None = None,
        pct_retained: float = 100.0,
        selection_metric: str = "bic",
        min_selection_change: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(
            group_col=group_col,
            y_col=y_col,
            offset_col=offset_col,
            feature_cols=feature_cols,
            normalize=normalize,
            maxiter=maxiter,
            tol=tol,
            bounds=bounds,
        )
        self.eval_terms = list(dict.fromkeys(eval_terms or []))
        self.context = {} if context is None else dict(context)
        self.pct_retained = pct_retained
        metric = str(selection_metric).strip().lower()
        if metric not in {"bic", "aic", "nll"}:
            raise ValueError("selection_metric must be one of: 'bic', 'aic', 'nll'.")
        self.selection_metric = metric
        self.min_selection_change = float(min_selection_change)
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _base_feature_cols(self, X_df: pd.DataFrame) -> list[str]:
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in X_df.columns]
            if missing:
                raise ValueError(
                    "X_df is missing required base feature columns: "
                    f"{missing[:10]}"
                )
            return list(self.feature_cols)

        exclude = {self.group_col}
        if self.offset_col is not None:
            exclude.add(self.offset_col)
        if self.y_col in X_df.columns:
            exclude.add(self.y_col)
        return [c for c in X_df.columns if c not in exclude]

    def _formula_for_terms(self, base_feature_cols: list[str], selected_terms: list[str]) -> str:
        formula_terms = [*base_feature_cols, *selected_terms]
        if not formula_terms:
            return ""
        return "0 + " + " + ".join(formula_terms)

    def _build_model_matrix(
        self,
        X_df: pd.DataFrame,
        selected_terms: list[str],
    ) -> tuple[pd.DataFrame, object | None, list[str]]:
        base_feature_cols = self._base_feature_cols(X_df)
        base_design = X_df.loc[:, base_feature_cols].copy()

        if not selected_terms:
            return base_design, None, base_feature_cols

        term_formula = "0 + " + " + ".join(selected_terms)
        term_matrix = frm.model_matrix(term_formula, X_df, context=self.context)
        term_design = pd.DataFrame(term_matrix)

        overlap = [c for c in term_design.columns if c in base_design.columns]
        if overlap:
            term_design = term_design.drop(columns=overlap)

        design_df = pd.concat(
            [base_design.reset_index(drop=True), term_design.reset_index(drop=True)],
            axis=1,
        )
        return design_df, term_matrix.model_spec, base_feature_cols

    def _current_theta(self, fit_result: GroupedPoissonGLM | None) -> pd.Series | None:
        if fit_result is None:
            return None
        return _pack_coef_series(fit_result.intercept_, fit_result.coef_)

    def _fit_terms(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame,
        selected_terms: list[str],
        warm_start: pd.Series | np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> tuple[GroupedPoissonGLM, pd.DataFrame, object | None]:
        design_df, model_spec, base_feature_cols = self._build_model_matrix(X_df, selected_terms)
        fit_input = pd.concat(
            [
                X_df.loc[:, [self.group_col]].reset_index(drop=True),
                (
                    X_df.loc[:, [self.offset_col]].reset_index(drop=True)
                    if self.offset_col is not None
                    else pd.DataFrame(index=X_df.index)
                ),
                design_df.reset_index(drop=True),
            ],
            axis=1,
        )

        model = GroupedPoissonGLM(
            group_col=self.group_col,
            y_col=self.y_col,
            offset_col=self.offset_col,
            feature_cols=list(design_df.columns),
            normalize=self.normalize,
            maxiter=self.maxiter,
            tol=self.tol,
            bounds=self.bounds,
        )
        model.fit(fit_input, y_df, warm_start=warm_start)
        model.model_spec_ = model_spec
        model.formula_ = self._formula_for_terms(base_feature_cols, selected_terms)
        model.base_feature_cols_ = base_feature_cols
        model.selected_terms_ = list(selected_terms)
        model.context_ = dict(self.context)
        return model, design_df, model_spec

    def _retain_limit(self) -> int:
        if not 0 < self.pct_retained <= 100:
            raise ValueError("pct_retained must be in (0, 100].")
        return max(1, int(np.ceil(len(self.eval_terms) * (self.pct_retained / 100.0))))

    def _selection_score(self, fit_result: GroupedPoissonGLM) -> float:
        metric = self.selection_metric

        if fit_result.log_likelihood_ is None or fit_result.n_params_ is None:
            raise RuntimeError("Fitted model is missing likelihood or parameter metadata.")

        nll = -float(fit_result.log_likelihood_)

        if metric == "nll":
            return float(nll)

        if metric == "aic":
            return float(2.0 * nll + 2.0 * int(fit_result.n_params_))

        if fit_result.bic_ is None:
            raise RuntimeError("Fitted model did not produce a BIC value.")
        return float(fit_result.bic_)

    def fit(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame,
        warm_start: pd.Series | np.ndarray | list[float] | tuple[float, ...] | None = None,
        n_round: int | None = None,
    ):
        if n_round is not None and n_round < 1:
            raise ValueError("n_round must be at least 1 when provided.")

        remaining_terms = list(self.eval_terms)
        selected_terms: list[str] = []
        retained_limit = self._retain_limit() if remaining_terms else 0
        self.step_records_ = []
        self.screening_records_ = []
        self._steps_df = pd.DataFrame(columns=[
            "round",
            "term",
            "selection_metric",
            "current_score",
            "candidate_score",
            "score_change",
            "optimizer_success",
            "n_terms_selected",
            "n_candidates_evaluated",
            "n_candidates_retained",
        ])

        current_fit, _, _ = self._fit_terms(X_df, y_df, selected_terms, warm_start=warm_start)
        current_score = self._selection_score(current_fit)
        metric_label = self.selection_metric.upper()
        self._log(f"Initial model {metric_label}: {current_score:.6f}")

        round_index = 0
        while remaining_terms:
            if n_round is not None and round_index >= n_round:
                self._log(f"Stopping: reached n_round={n_round}.")
                break

            round_index += 1
            warm_theta = self._current_theta(current_fit)
            candidate_results: list[tuple[float, str, GroupedPoissonGLM]] = []
            best_term: str | None = None
            best_fit: GroupedPoissonGLM | None = None
            best_change: float | None = None

            for term in remaining_terms:
                trial_terms = [*selected_terms, term]
                trial_fit, _, _ = self._fit_terms(X_df, y_df, trial_terms, warm_start=warm_theta)

                if trial_fit.bic_ is None:
                    raise RuntimeError("Candidate model did not produce a BIC value.")

                trial_score = self._selection_score(trial_fit)
                trial_success = bool(getattr(trial_fit.result_, "success", True))

                # Warm starts can occasionally land in numerically poor regions.
                # Retry with a cold start if convergence failed, BIC is non-finite,
                # or the candidate is implausibly worse than the current model.
                retry_cold_start = (
                    (not trial_success)
                    or (not np.isfinite(trial_score))
                    or ((trial_score - current_score) > 1e4)
                )

                if retry_cold_start:
                    cold_fit, _, _ = self._fit_terms(X_df, y_df, trial_terms, warm_start=None)
                    if cold_fit.bic_ is None:
                        raise RuntimeError("Cold-start candidate model did not produce a BIC value.")

                    cold_score = self._selection_score(cold_fit)
                    cold_success = bool(getattr(cold_fit.result_, "success", True))

                    if np.isfinite(cold_score) and (cold_success or not trial_success or not np.isfinite(trial_score) or cold_score < trial_score):
                        trial_fit = cold_fit
                        trial_score = cold_score
                        trial_success = cold_success

                if trial_fit.bic_ is None:
                    raise RuntimeError("Candidate model did not produce a BIC value.")
                score_change = current_score - trial_score
                candidate_results.append((score_change, term, trial_fit))
                row = {
                    "round": round_index,
                    "term": term,
                    "selection_metric": self.selection_metric,
                    "current_score": current_score,
                    "candidate_score": trial_score,
                    "score_change": score_change,
                    "optimizer_success": trial_success,
                    "n_terms_selected": len(trial_terms),
                    "n_candidates_evaluated": len(remaining_terms),
                    "n_candidates_retained": retained_limit,
                }
                self.screening_records_.append(row)
                if best_change is None or score_change > best_change:
                    best_change = score_change
                    best_term = term
                    best_fit = trial_fit

            if not candidate_results or best_term is None or best_fit is None or best_change is None:
                break

            candidate_results.sort(key=lambda row: row[0], reverse=True)
            retained_candidates = [term for _, term, _ in candidate_results[:retained_limit]]
            self._log(
                f"Round {round_index}: best term={best_term} {self.selection_metric}_change={best_change:.6f} retained={retained_candidates}"
            )

            if best_change < self.min_selection_change:
                self._log(
                    f"Stopping: best {self.selection_metric.upper()} change {best_change:.6f} below threshold {self.min_selection_change:.6f}."
                )
                break

            selected_terms.append(best_term)
            self.step_records_.append({
                "round": round_index,
                "term": best_term,
                "selection_metric": self.selection_metric,
                "current_score": current_score,
                "candidate_score": self._selection_score(best_fit),
                "score_change": float(best_change),
                "optimizer_success": bool(getattr(best_fit.result_, "success", True)),
                "n_terms_selected": len(selected_terms),
                "n_candidates_evaluated": len(remaining_terms),
                "n_candidates_retained": retained_limit,
            })
            current_fit = best_fit
            if current_fit.bic_ is None:
                raise RuntimeError("Accepted model did not produce a BIC value.")
            current_score = self._selection_score(current_fit)
            remaining_terms = [term for term in retained_candidates if term not in selected_terms]
            self._log(f"Accepted term {selected_terms[-1]}; current {metric_label}={current_score:.6f}")

        self.final_fit_ = current_fit
        self.selected_terms_ = selected_terms
        self._steps_df = pd.DataFrame(self.step_records_)

        self.result_ = current_fit.result_
        self.feature_cols_ = current_fit.feature_cols_
        self.x_mean_ = current_fit.x_mean_
        self.x_scale_ = current_fit.x_scale_
        self.intercept_ = current_fit.intercept_
        self.coef_ = current_fit.coef_
        self.intercept_scaled_ = current_fit.intercept_scaled_
        self.coef_scaled_ = current_fit.coef_scaled_
        self.log_likelihood_ = current_fit.log_likelihood_
        self.n_params_ = current_fit.n_params_
        self.n_obs_ = current_fit.n_obs_
        self.bic_ = current_fit.bic_
        self.theta_ = current_fit.theta_
        self.formula_ = getattr(current_fit, "formula_", "")
        self.model_spec_ = getattr(current_fit, "model_spec_", None)
        self.base_feature_cols_ = getattr(current_fit, "base_feature_cols_", self._base_feature_cols(X_df))
        self.context_ = dict(self.context)
        return self

    def steps(self) -> pd.DataFrame:
        if not hasattr(self, "_steps_df"):
            return pd.DataFrame(columns=[
                "round",
                "term",
                "selection_metric",
                "current_score",
                "candidate_score",
                "score_change",
                "optimizer_success",
                "n_terms_selected",
                "n_candidates_evaluated",
                "n_candidates_retained",
            ])
        return self._steps_df.copy()

    def _design_from_input(self, X_df: pd.DataFrame) -> pd.DataFrame:
        base_cols = list(getattr(self, "base_feature_cols_", []) or [])
        model_spec = getattr(self, "model_spec_", None)

        if not base_cols:
            return X_df[self.feature_cols_].copy()

        base_design = X_df.loc[:, base_cols].copy()

        if model_spec is None or not getattr(self, "selected_terms_", None):
            return base_design.loc[:, self.feature_cols_].copy()

        term_matrix = model_spec.get_model_matrix(X_df, context=self.context)
        term_design = pd.DataFrame(term_matrix)
        overlap = [c for c in term_design.columns if c in base_design.columns]
        if overlap:
            term_design = term_design.drop(columns=overlap)

        design_df = pd.concat(
            [base_design.reset_index(drop=True), term_design.reset_index(drop=True)],
            axis=1,
        )
        return design_df.loc[:, self.feature_cols_].copy()

    def predict_individual_mu(self, X_df: pd.DataFrame) -> np.ndarray:
        design_df = self._design_from_input(X_df)

        if self.offset_col is None:
            offset = np.zeros(len(X_df), dtype=float)
        else:
            offset = X_df[self.offset_col].to_numpy(dtype=float)

        eta = self.intercept_ + design_df.to_numpy(dtype=float) @ self.coef_.to_numpy() + offset
        eta = np.clip(eta, -745, 700)
        return np.exp(eta)

    def predict_group_mu(self, X_df: pd.DataFrame) -> pd.DataFrame:
        mu_i = self.predict_individual_mu(X_df)

        return (
            pd.DataFrame({
                self.group_col: X_df[self.group_col].to_numpy(),
                "mu_individual": mu_i,
            })
            .groupby(self.group_col, as_index=False)
            .agg(mu_group=("mu_individual", "sum"))
        )
