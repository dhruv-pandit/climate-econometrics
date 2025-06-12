import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Sequence, Optional, Union, List

__all__ = ["pvarfeols", "PvarFeols"]


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _panel_lag(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    """
    Create backward lags for each variable in *df* (except the 'period' column).
    """
    if "period" not in df.columns:
        raise ValueError("'period' column required for panel lagging.")
    df_sorted = df.sort_values("period")
    out = pd.DataFrame(index=df_sorted.index)
    for var in [c for c in df_sorted.columns if c != "period"]:
        for l in range(1, lags + 1):
            out[f"lag{l}_{var}"] = df_sorted[var].shift(l)
    return out


def _panel_forward_lag(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    """
    Create forward lags (leads) for each variable in *df* (except the 'period' column).
    Only the lead *lags* is returned.
    """
    if "period" not in df.columns:
        raise ValueError("'period' column required for panel forward lagging.")
    df_sorted = df.sort_values("period")
    out = pd.DataFrame(index=df_sorted.index)
    for var in [c for c in df_sorted.columns if c != "period"]:
        out[var] = df_sorted[var].shift(-lags)
    return out


def _panel_demean(df: pd.DataFrame) -> pd.DataFrame:
    """Demean every numeric column (including duplicates) within each category.
    A fresh DataFrame with ``demeaned_`` prefixed columns is returned; the
    original order of observations is preserved. Duplicate source columns are
    permitted and handled individually.
    """
    cat = df["category"]
    out = pd.DataFrame(index=df.index)

    for idx, col in enumerate(df.columns):
        if col in ("category", "period"):
            continue
        series = df.iloc[:, idx]
        # Compute group‐wise mean for this specific column instance
        mean_by_cat = series.groupby(cat).transform("mean")
        out[f"demeaned_{col}"] = series - mean_by_cat

    return out


# --------------------------------------------------------------------------- #
# Main result container
# --------------------------------------------------------------------------- #


class PvarFeols:
    """
    Python analogue of the R `pvarfeols` S3 object.
    """

    def __init__(
        self,
        *,
        dependent_vars: Sequence[str],
        lags: int,
        transformation: str,
        Set_Vars: pd.DataFrame,
        Set_Vars_with_NAs: pd.DataFrame,
        panel_identifier: Sequence[Union[int, str]],
        nof_observations: int,
        obs_per_group_avg: float,
        obs_per_group_min: int,
        obs_per_group_max: int,
        nof_groups: int,
        coef_: np.ndarray,
        se_: np.ndarray,
        pvals_: np.ndarray,
        residuals: np.ndarray,
        lagged_var_names: List[str],
    ):
        self.dependent_vars = list(dependent_vars)
        self.lags = lags
        self.transformation = transformation
        self.Set_Vars = Set_Vars
        self.Set_Vars_with_NAs = Set_Vars_with_NAs
        self.panel_identifier = list(panel_identifier)
        self.nof_observations = nof_observations
        self.obs_per_group_avg = obs_per_group_avg
        self.obs_per_group_min = obs_per_group_min
        self.obs_per_group_max = obs_per_group_max
        self.nof_groups = nof_groups
        self.OLS = {"coef": coef_, "se": se_, "pvalues": pvals_}
        self.residuals = residuals
        self.lagged_var_names = lagged_var_names

    # --------------------------------------------------------------------- #
    # Convenience accessors
    # --------------------------------------------------------------------- #
    def coef(self) -> np.ndarray:
        return self.OLS["coef"]

    def se(self) -> np.ndarray:
        return self.OLS["se"]

    def pvalue(self) -> np.ndarray:
        return self.OLS["pvalues"]

    # --------------------------------------------------------------------- #
    # vcov
    # --------------------------------------------------------------------- #
    def vcov(self) -> np.ndarray:
        res = self.residuals
        sigma_hat = (res.T @ res) / (res.shape[0] - self.coef().shape[1])
        return sigma_hat

    # --------------------------------------------------------------------- #
    # String representation
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        header = "Fixed Effects OLS Panel VAR estimation\n"
        info = (
            f"Transformation: {self.transformation}\n"
            f"Group variable: {self.panel_identifier[0]}\n"
            f"Time variable: {self.panel_identifier[1]}\n"
            f"Number of observations = {self.nof_observations}\n"
            f"Number of groups      = {self.nof_groups}\n"
            f"Obs per group: min = {self.obs_per_group_min}, "
            f"avg = {self.obs_per_group_avg:.2f}, "
            f"max = {self.obs_per_group_max}\n"
        )
        coef_df = pd.DataFrame(
            self.coef(), index=self.dependent_vars, columns=self.lagged_var_names
        )
        return header + info + "\nCoefficients:\n" + coef_df.to_string()


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #

def pvarfeols(
    *,
    dependent_vars: Sequence[str],
    lags: int,
    exog_vars: Optional[Sequence[str]] = None,
    transformation: str = "demean",
    data: pd.DataFrame,
    panel_identifier: Sequence[Union[int, str]] = (1, 2),
) -> PvarFeols:
    """
    Fixed‑effects estimator for a stationary Panel VAR (Python translation of the
    R function `pvarfeols` from the **panelvar** package).
    """
    # --------------------------------------------------------------------- #
    # Pre‑processing and variable selection
    # --------------------------------------------------------------------- #
    if data is None:
        raise ValueError("`data` must be provided.")

    data = data.copy()

    for col in data.select_dtypes("category").columns:
        data[col] = data[col].cat.remove_unused_categories()

    required_vars = list(dependent_vars)
    if exog_vars is not None:
        required_vars.extend(exog_vars)

    # Panel identifiers
    if all(isinstance(i, int) for i in panel_identifier):
        id_cols = [data.columns[i - 1] for i in panel_identifier]  # 1‑based
    else:
        id_cols = list(panel_identifier)

    Set_Vars = data[id_cols + required_vars].copy()

    # Sorting by identifiers
    name_category, name_period = id_cols
    Set_Vars = (
        Set_Vars.rename(columns={name_category: "category", name_period: "period"})
        .sort_values(["category", "period"])
    )
    Set_Vars["category"] = Set_Vars["category"].astype("category")
    Set_Vars["period"] = Set_Vars["period"].astype("category")

    categories = np.sort(Set_Vars["category"].unique())

    # --------------------------------------------------------------------- #
    # Descriptive statistics
    # --------------------------------------------------------------------- #
    nof_observations = len(data)
    counts = data.groupby(id_cols[0]).size()
    obs_per_group_avg = counts.mean()
    obs_per_group_min = counts.min()
    obs_per_group_max = counts.max()
    nof_groups = counts.size

    # --------------------------------------------------------------------- #
    # Lagged dependent variables
    # --------------------------------------------------------------------- #
    lagged_dep = []
    for cat in categories:
        mask = Set_Vars["category"] == cat
        lagged_dep.append(
            _panel_lag(Set_Vars.loc[mask, ["period"] + list(dependent_vars)], lags)
        )
    Set_Vars = pd.concat([Set_Vars] + lagged_dep, axis=1)

    # --------------------------------------------------------------------- #
    # Forward‑lag exogenous variables
    # --------------------------------------------------------------------- #
    if exog_vars is not None:
        forward_exog = []
        for cat in categories:
            mask = Set_Vars["category"] == cat
            forward_exog.append(
                _panel_forward_lag(
                    Set_Vars.loc[mask, ["period"] + list(exog_vars)], lags
                )
            )
        Set_Vars = pd.concat([Set_Vars] + forward_exog, axis=1)

    # --------------------------------------------------------------------- #
    # Demean
    # --------------------------------------------------------------------- #
    demeaned = []
    for cat in categories:
        mask = Set_Vars["category"] == cat
        demeaned.append(_panel_demean(Set_Vars.loc[mask]))
    Set_Vars = pd.concat(demeaned).sort_index()

    # --------------------------------------------------------------------- #
    # Regression matrices
    # --------------------------------------------------------------------- #
    dep_dm = [f"demeaned_{v}" for v in dependent_vars]
    lagged_vars = [
        f"demeaned_lag{l}_{v}" for l in range(1, lags + 1) for v in dependent_vars
    ]
    if exog_vars is not None:
        lagged_vars.extend([f"demeaned_{v}" for v in exog_vars])

    Set_Vars_with_NAs = Set_Vars.copy()
    Set_Vars = Set_Vars.dropna(subset=dep_dm + lagged_vars)

    X = Set_Vars[lagged_vars].to_numpy()
    Y = Set_Vars[dep_dm].to_numpy()

    XTX = X.T @ X
    XTY = X.T @ Y
    beta = np.linalg.solve(XTX, XTY)

    # Residuals and variance
    residuals = Y - X @ beta
    sigma2 = (residuals.T @ residuals) / (residuals.shape[0] - beta.shape[0])
    var_beta = np.kron(sigma2, np.linalg.inv(XTX))
    se_beta = np.sqrt(np.diag(var_beta)).reshape(beta.shape, order="F")
    pvals = 2 * norm.sf(np.abs(beta / se_beta))

    # --------------------------------------------------------------------- #
    # Build result object
    # --------------------------------------------------------------------- #
    result = PvarFeols(
        dependent_vars=dependent_vars,
        lags=lags,
        transformation=transformation,
        Set_Vars=Set_Vars,
        Set_Vars_with_NAs=Set_Vars_with_NAs,
        panel_identifier=id_cols,
        nof_observations=int(nof_observations),
        obs_per_group_avg=float(obs_per_group_avg),
        obs_per_group_min=int(obs_per_group_min),
        obs_per_group_max=int(obs_per_group_max),
        nof_groups=int(nof_groups),
        coef_=beta.T,
        se_=se_beta.T,
        pvals_=pvals.T,
        residuals=residuals,
        lagged_var_names=lagged_vars,
    )

    return result
