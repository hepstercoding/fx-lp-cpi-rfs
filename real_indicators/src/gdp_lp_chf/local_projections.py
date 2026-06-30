from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class LPConfig:
    horizons: int = 12
    lags: int = 4
    ci_level: float = 0.9
    include_forward_shocks: bool = False


def z_value(ci_level: float) -> float:
    if np.isclose(ci_level, 0.68):
        return 0.994457883209753
    if np.isclose(ci_level, 0.9):
        return 1.6448536269514722
    if np.isclose(ci_level, 0.95):
        return 1.959963984540054
    alpha = 1.0 - ci_level
    return NormalDist().inv_cdf(1 - alpha / 2)


def prepare_lp_design(
    data: pd.DataFrame,
    response: str,
    shock: str,
    controls: list[str],
    config: LPConfig,
    cumulative_response: bool,
) -> pd.DataFrame:
    out = data.copy()
    required = {response, shock, *controls}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    derived: dict[str, pd.Series] = {}
    for horizon in range(config.horizons + 1):
        if cumulative_response:
            derived[f"{response}_lead_{horizon}"] = out[response].shift(-horizon) - out[response].shift(1)
        else:
            derived[f"{response}_lead_{horizon}"] = out[response].shift(-horizon)

    for lag in range(1, config.lags + 1):
        derived[f"{response}_lag_{lag}"] = out[response].shift(lag)
        derived[f"{shock}_lag_{lag}"] = out[shock].shift(lag)
        for control in controls:
            derived[f"{control}_lag_{lag}"] = out[control].shift(lag)

    for lead in range(1, config.horizons + 1):
        derived[f"{shock}_lead_{lead}"] = out[shock].shift(-lead)

    return pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)


def estimate_lp(
    data: pd.DataFrame,
    response: str,
    shock: str,
    controls: list[str],
    config: LPConfig,
    response_label: str,
    shock_label: str,
    cumulative_response: bool,
) -> pd.DataFrame:
    design = prepare_lp_design(data, response, shock, controls, config, cumulative_response)
    z = z_value(config.ci_level)
    lag_regressors = []
    for lag in range(1, config.lags + 1):
        lag_regressors.extend([f"{response}_lag_{lag}", f"{shock}_lag_{lag}"])
        lag_regressors.extend(f"{control}_lag_{lag}" for control in controls)

    results = []
    for horizon in range(config.horizons + 1):
        dep_var = f"{response}_lead_{horizon}"
        forward_shocks = (
            [f"{shock}_lead_{lead}" for lead in range(1, horizon + 1)]
            if config.include_forward_shocks
            else []
        )
        regressors = [shock, *controls, *forward_shocks, *lag_regressors]
        sample = design.loc[:, [dep_var, *regressors]].dropna()
        if sample.empty or len(sample) <= len(regressors) + 1:
            results.append(
                {
                    "response": response_label,
                    "shock": shock_label,
                    "horizon": horizon,
                    "beta": np.nan,
                    "std_error": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "nobs": len(sample),
                    "r_squared": np.nan,
                    "num_forward_shocks": len(forward_shocks),
                }
            )
            continue

        y = sample[dep_var]
        x = sm.add_constant(sample[regressors], has_constant="add")
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon + 1, 1)})
        coef = model.params[shock]
        se = model.bse[shock]
        results.append(
            {
                "response": response_label,
                "shock": shock_label,
                "horizon": horizon,
                "beta": coef,
                "std_error": se,
                "ci_lower": coef - z * se,
                "ci_upper": coef + z * se,
                "nobs": int(model.nobs),
                "r_squared": model.rsquared,
                "num_forward_shocks": len(forward_shocks),
            }
        )

    return pd.DataFrame(results)


def estimate_asymmetric_lp(
    data: pd.DataFrame,
    response: str,
    shock: str,
    controls: list[str],
    config: LPConfig,
    response_label: str,
    shock_label: str,
    cumulative_response: bool,
) -> pd.DataFrame:
    design = prepare_lp_design(data, response, shock, controls, config, cumulative_response)
    positive_shock = f"{shock}_positive"
    negative_shock = f"{shock}_negative"
    design[positive_shock] = design[shock].clip(lower=0)
    design[negative_shock] = design[shock].clip(upper=0)

    z = z_value(config.ci_level)
    lag_regressors = []
    for lag in range(1, config.lags + 1):
        lag_regressors.extend([f"{response}_lag_{lag}", f"{shock}_lag_{lag}"])
        lag_regressors.extend(f"{control}_lag_{lag}" for control in controls)

    results = []
    for horizon in range(config.horizons + 1):
        dep_var = f"{response}_lead_{horizon}"
        forward_shocks = (
            [f"{shock}_lead_{lead}" for lead in range(1, horizon + 1)]
            if config.include_forward_shocks
            else []
        )
        regressors = [positive_shock, negative_shock, *controls, *forward_shocks, *lag_regressors]
        sample = design.loc[:, [dep_var, *regressors]].dropna()
        if sample.empty or len(sample) <= len(regressors) + 1:
            for regime in ["Appreciation (+1 pp)", "Depreciation (-1 pp)", "Appreciation minus depreciation"]:
                results.append(
                    {
                        "response": response_label,
                        "shock": shock_label,
                        "regime": regime,
                        "horizon": horizon,
                        "beta": np.nan,
                        "raw_coefficient": np.nan,
                        "display_multiplier": np.nan,
                        "std_error": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "nobs": len(sample),
                        "r_squared": np.nan,
                        "num_forward_shocks": len(forward_shocks),
                    }
                )
            continue

        y = sample[dep_var]
        x = sm.add_constant(sample[regressors], has_constant="add")
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon + 1, 1)})
        cov = model.cov_params()

        app_coef = float(model.params[positive_shock])
        app_se = float(model.bse[positive_shock])
        raw_dep_coef = float(model.params[negative_shock])
        dep_coef = -raw_dep_coef
        dep_se = float(model.bse[negative_shock])
        diff_coef = float(model.params[positive_shock] + model.params[negative_shock])
        diff_var = (
            cov.loc[positive_shock, positive_shock]
            + cov.loc[negative_shock, negative_shock]
            + 2 * cov.loc[positive_shock, negative_shock]
        )
        diff_se = float(np.sqrt(max(diff_var, 0.0)))

        for regime, coef, raw_coef, multiplier, se in [
            ("Appreciation (+1 pp)", app_coef, app_coef, 1.0, app_se),
            ("Depreciation (-1 pp)", dep_coef, raw_dep_coef, -1.0, dep_se),
            ("Appreciation minus depreciation", diff_coef, diff_coef, 1.0, diff_se),
        ]:
            results.append(
                {
                    "response": response_label,
                    "shock": shock_label,
                    "regime": regime,
                    "horizon": horizon,
                    "beta": coef,
                    "raw_coefficient": raw_coef,
                    "display_multiplier": multiplier,
                    "std_error": se,
                    "ci_lower": coef - z * se,
                    "ci_upper": coef + z * se,
                    "nobs": int(model.nobs),
                    "r_squared": model.rsquared,
                    "num_forward_shocks": len(forward_shocks),
                }
            )

    return pd.DataFrame(results)


def maintained_response_from_irfs(
    response_irf: pd.DataFrame,
    shock_path_irf: pd.DataFrame,
    ci_level: float,
    label: str,
) -> pd.DataFrame:
    """Layer one-shock IRFs to keep the CHF appreciation path at 1 percent."""
    response = response_irf.sort_values("horizon").reset_index(drop=True)
    path = shock_path_irf.sort_values("horizon").reset_index(drop=True)
    n = min(len(response), len(path))
    response = response.iloc[:n]
    path = path.iloc[:n]

    path_beta = path["beta"].to_numpy(dtype=float)
    response_beta = response["beta"].to_numpy(dtype=float)
    response_se = response["std_error"].to_numpy(dtype=float)
    if n == 0 or not np.isfinite(path_beta[0]) or np.isclose(path_beta[0], 0.0):
        return pd.DataFrame(columns=response_irf.columns)

    shocks = np.zeros(n)
    maintained_path = np.zeros(n)
    layered_beta = np.zeros(n)
    layered_var = np.zeros(n)
    for horizon in range(n):
        inherited_path = sum(shocks[k] * path_beta[horizon - k] for k in range(horizon))
        shocks[horizon] = (1.0 - inherited_path) / path_beta[0]
        maintained_path[horizon] = sum(shocks[k] * path_beta[horizon - k] for k in range(horizon + 1))
        layered_beta[horizon] = sum(shocks[k] * response_beta[horizon - k] for k in range(horizon + 1))
        layered_var[horizon] = sum((shocks[k] ** 2) * (response_se[horizon - k] ** 2) for k in range(horizon + 1))

    std_error = np.sqrt(layered_var)
    z = z_value(ci_level)
    out = response.copy()
    out["response"] = label
    out["beta"] = layered_beta
    out["std_error"] = std_error
    out["ci_lower"] = layered_beta - z * std_error
    out["ci_upper"] = layered_beta + z * std_error
    out["maintenance_shock"] = shocks
    out["maintained_exchange_rate_path"] = maintained_path
    return out
