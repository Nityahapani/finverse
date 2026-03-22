"""
finverse.macro.var_model — Vector Autoregression (VAR) model.

Models the joint dynamics of multiple macro variables simultaneously.
Computes impulse response functions (IRF) to show how a shock to one
variable propagates through the system over time.

Pure numpy/scipy — no statsmodels required (though supported if available).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class VARResult:
    variables: list[str]
    n_lags: int
    coefficients: dict[str, np.ndarray]     # VAR coefficient matrices
    residuals: pd.DataFrame
    aic: float
    bic: float
    impulse_responses: dict[str, pd.DataFrame]   # shock → responses over time
    forecast: pd.DataFrame                        # h-step ahead forecasts
    granger_causality: pd.DataFrame              # pairwise Granger table

    def summary(self):
        from finverse.utils.display import console
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold blue]VAR({self.n_lags}) Model[/bold blue]")
        console.print(f"[dim]Variables: {', '.join(self.variables)}  |  AIC: {self.aic:.2f}  |  BIC: {self.bic:.2f}[/dim]\n")

        if not self.forecast.empty:
            table = Table(title="Forecast (h steps ahead)", box=box.SIMPLE_HEAD, header_style="bold blue")
            table.add_column("Horizon")
            for v in self.variables:
                table.add_column(v, justify="right")
            for h, row in self.forecast.iterrows():
                table.add_row(str(h), *[f"{row[v]:.4f}" for v in self.variables if v in row])
            console.print(table)

        if not self.granger_causality.empty:
            console.print("\n[dim]Granger causality (p-values — significant < 0.05):[/dim]")
            console.print(self.granger_causality.round(4).to_string())

        console.print()

    def irf(self, shock: str, response: str) -> pd.Series:
        """Get impulse response of 'response' to a unit shock in 'shock'."""
        key = f"{shock}→{response}"
        if shock in self.impulse_responses:
            df = self.impulse_responses[shock]
            if response in df.columns:
                return df[response]
        return pd.Series(dtype=float)

    def plot_irf(self, shock: str):
        """Plot all impulse responses to a shock in 'shock' variable."""
        try:
            import matplotlib.pyplot as plt

            if shock not in self.impulse_responses:
                print(f"No IRF for shock '{shock}'")
                return

            irf_df = self.impulse_responses[shock]
            n_vars = len(irf_df.columns)
            fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4))
            if n_vars == 1:
                axes = [axes]

            for ax, var in zip(axes, irf_df.columns):
                irf = irf_df[var]
                ax.plot(irf.index, irf.values, color="#185FA5", linewidth=2)
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                ax.fill_between(irf.index, irf.values * 0.8, irf.values * 1.2,
                               alpha=0.2, color="#185FA5")
                ax.set_title(f"Response of {var}", fontsize=11)
                ax.set_xlabel("Periods")
                ax.grid(alpha=0.3)

            fig.suptitle(f"Impulse Response to {shock} shock", fontsize=13)
            plt.tight_layout()
            plt.show()
        except ImportError:
            from finverse.utils.display import console
            console.print("[yellow]matplotlib required: pip install matplotlib[/yellow]")


def _ols_var(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate VAR(p) by OLS (equation-by-equation).
    Returns coefficient matrix B and residuals.
    """
    T, K = Y.shape
    X_list = [np.ones(T - p)]
    for lag in range(1, p + 1):
        X_list.append(Y[p - lag:T - lag])

    X = np.column_stack(X_list)
    Y_reg = Y[p:]

    B = np.linalg.lstsq(X, Y_reg, rcond=None)[0]
    residuals = Y_reg - X @ B
    return B, residuals


def _compute_irf(
    B: np.ndarray,
    Sigma: np.ndarray,
    K: int,
    p: int,
    horizon: int,
) -> np.ndarray:
    """
    Compute orthogonalized IRF using Cholesky decomposition.
    Returns array of shape (horizon+1, K, K).
    """
    B_coef = B[1:].reshape(p, K, K)

    Phi = np.zeros((horizon + 1, K, K))
    Phi[0] = np.eye(K)

    for h in range(1, horizon + 1):
        for j in range(min(h, p)):
            Phi[h] += Phi[h - j - 1] @ B_coef[j].T

    try:
        P = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        P = np.diag(np.sqrt(np.diag(Sigma)))

    Psi = np.array([Phi[h] @ P for h in range(horizon + 1)])
    return Psi


def _granger_test(Y: np.ndarray, cause_idx: int, effect_idx: int, p: int) -> float:
    """
    Granger causality test: does 'cause' help predict 'effect'?
    Returns p-value (small = significant causality).
    """
    from scipy.stats import f as f_dist

    T, K = Y.shape
    X_list_full = [np.ones(T - p)]
    X_list_restricted = [np.ones(T - p)]

    for lag in range(1, p + 1):
        lagged = Y[p - lag:T - lag]
        X_list_full.append(lagged)
        cols = [c for c in range(K) if c != cause_idx]
        X_list_restricted.append(lagged[:, cols])

    X_full = np.column_stack(X_list_full)
    X_rest = np.column_stack(X_list_restricted)
    y_effect = Y[p:, effect_idx]

    rss_full = float(np.sum((y_effect - X_full @ np.linalg.lstsq(X_full, y_effect, rcond=None)[0])**2))
    rss_rest = float(np.sum((y_effect - X_rest @ np.linalg.lstsq(X_rest, y_effect, rcond=None)[0])**2))

    df1 = X_full.shape[1] - X_rest.shape[1]
    df2 = len(y_effect) - X_full.shape[1]

    if df1 <= 0 or df2 <= 0 or rss_full <= 0:
        return 1.0

    F = ((rss_rest - rss_full) / df1) / (rss_full / df2)
    p_val = float(1 - f_dist.cdf(F, df1, df2))
    return max(0.0, min(1.0, p_val))


def fit(
    data: pd.DataFrame,
    n_lags: int = 2,
    forecast_horizon: int = 8,
    irf_horizon: int = 20,
) -> VARResult:
    """
    Fit a VAR model to multivariate time series data.

    Parameters
    ----------
    data             : pd.DataFrame — columns are variables, rows are observations
                       (e.g. from pull.fred("GDP", "UNRATE", "FEDFUNDS"))
    n_lags           : int — VAR order p (default 2)
    forecast_horizon : int — steps ahead to forecast (default 8)
    irf_horizon      : int — periods for impulse response functions (default 20)

    Returns
    -------
    VARResult

    Example
    -------
    >>> from finverse import pull
    >>> from finverse.macro import var_model
    >>> macro = pull.fred("UNRATE", "FEDFUNDS", "CPIAUCSL")
    >>> # Resample to quarterly
    >>> quarterly = macro.resample("QE").last().pct_change().dropna()
    >>> result = var_model.fit(quarterly, n_lags=2)
    >>> result.summary()
    >>> result.plot_irf("FEDFUNDS")  # shock to fed funds rate
    """
    from finverse.utils.display import console

    variables = list(data.columns)
    K = len(variables)
    console.print(f"[dim]Fitting VAR({n_lags}) with {K} variables: {', '.join(variables)}...[/dim]")

    df_clean = data.dropna()
    if len(df_clean) < n_lags + K + 5:
        console.print(f"[yellow]Warning: only {len(df_clean)} obs — increasing lags or reducing variables recommended[/yellow]")

    Y = df_clean.values.astype(float)
    T, _ = Y.shape

    B, residuals = _ols_var(Y, n_lags)

    Sigma = residuals.T @ residuals / (T - n_lags - K * n_lags - 1)
    n_params = K * (1 + K * n_lags)
    ll = -0.5 * (T - n_lags) * (K * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma) + 1e-10) + K)
    aic = float(-2 * ll + 2 * n_params)
    bic = float(-2 * ll + np.log(T - n_lags) * n_params)

    Psi = _compute_irf(B, Sigma, K, n_lags, irf_horizon)
    irf_dict = {}
    for shock_idx, shock_var in enumerate(variables):
        irf_df = pd.DataFrame(
            Psi[:, :, shock_idx],
            index=range(irf_horizon + 1),
            columns=variables,
        )
        irf_dict[shock_var] = irf_df

    last_obs = Y[-n_lags:]
    forecasts = []
    current = list(Y)

    for h in range(1, forecast_horizon + 1):
        X_new = [1.0]
        for lag in range(1, n_lags + 1):
            X_new.extend(current[-lag])
        X_new = np.array(X_new)
        y_new = X_new @ B
        current.append(y_new)
        forecasts.append(y_new)

    forecast_df = pd.DataFrame(
        forecasts,
        index=range(1, forecast_horizon + 1),
        columns=variables,
    ).round(6)

    granger_pvals = np.ones((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                granger_pvals[i, j] = _granger_test(Y, cause_idx=i, effect_idx=j, p=n_lags)

    granger_df = pd.DataFrame(
        granger_pvals,
        index=[f"{v} causes →" for v in variables],
        columns=variables,
    ).round(4)

    residuals_df = pd.DataFrame(residuals, columns=variables)

    coefs = {f"Lag {p+1}": B[1 + p*K:1 + (p+1)*K] for p in range(n_lags)}

    console.print(
        f"[green]✓[/green] VAR({n_lags}) fitted — "
        f"AIC: {aic:.2f}, BIC: {bic:.2f}"
    )

    return VARResult(
        variables=variables,
        n_lags=n_lags,
        coefficients=coefs,
        residuals=residuals_df,
        aic=round(aic, 4),
        bic=round(bic, 4),
        impulse_responses=irf_dict,
        forecast=forecast_df,
        granger_causality=granger_df,
    )


def select_lag_order(data: pd.DataFrame, max_lags: int = 6) -> pd.DataFrame:
    """
    Select optimal VAR lag order by AIC and BIC.

    Returns a DataFrame comparing information criteria across lag orders.

    Example
    -------
    >>> table = var_model.select_lag_order(quarterly_data, max_lags=6)
    >>> print(table)
    >>> # Use the lag with lowest AIC
    """
    from finverse.utils.display import console
    console.print(f"[dim]Selecting lag order (1–{max_lags})...[/dim]")

    results = []
    for p in range(1, max_lags + 1):
        try:
            r = fit(data, n_lags=p, forecast_horizon=1, irf_horizon=1)
            results.append({"lags": p, "AIC": r.aic, "BIC": r.bic})
        except Exception:
            pass

    df = pd.DataFrame(results).set_index("lags")
    best_aic = int(df["AIC"].idxmin())
    best_bic = int(df["BIC"].idxmin())
    console.print(f"[green]Best:[/green] AIC→ p={best_aic}, BIC→ p={best_bic}")
    return df.round(4)
