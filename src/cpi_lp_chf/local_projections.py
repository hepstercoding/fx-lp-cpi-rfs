from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class LPConfig:
    horizons: int = 36
    lags: int = 12
    ci_level: float = 0.9
    include_forward_shocks: bool = False


def z_value(ci_level: float) -> float:
    if np.isclose(ci_level, 0.9):
        return 1.6448536269514722
    alpha = 1.0 - ci_level
    return NormalDist().inv_cdf(1 - alpha / 2)


def estimate_simple_lp(
    data: pd.DataFrame,
    dep_prefix: str,
    shock_name: str,
    config: LPConfig,
) -> pd.DataFrame:
    """Estimate horizon-by-horizon LPs with lags of the response and shock."""
    results = []
    z = z_value(config.ci_level)

    for horizon in range(config.horizons + 1):
        dep_var = f"{dep_prefix}_lead_{horizon}"
        regressors = [shock_name]
        regressors.extend(f"{dep_prefix}_lag_{lag}" for lag in range(1, config.lags + 1))
        regressors.extend(f"{shock_name}_lag_{lag}" for lag in range(1, config.lags + 1))

        sample = data[[dep_var, *regressors]].dropna()
        y = sample[dep_var]
        x = sm.add_constant(sample[regressors])
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon + 1, 1)})
        coef = model.params[shock_name]
        se = model.bse[shock_name]

        results.append(
            {
                "horizon": horizon,
                "beta": coef,
                "std_error": se,
                "ci_lower": coef - z * se,
                "ci_upper": coef + z * se,
                "nobs": int(model.nobs),
                "r_squared": model.rsquared,
            }
        )

    return pd.DataFrame(results)


def prepare_lp_design(
    data: pd.DataFrame,
    response: str,
    shock_name: str,
    controls: list[str] | None,
    config: LPConfig,
    cumulative_response: bool = False,
    dummy_controls: list[str] | None = None,
) -> pd.DataFrame:
    """Create leads, lags, and optional forward-shock columns for dashboard LPs."""
    controls = controls or []
    dummy_controls = dummy_controls or []
    out = data.copy()
    required = {response, shock_name, *controls, *dummy_controls}
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
        derived[f"{shock_name}_lag_{lag}"] = out[shock_name].shift(lag)
        for control in controls:
            derived[f"{control}_lag_{lag}"] = out[control].shift(lag)

    for lead in range(1, config.horizons + 1):
        derived[f"{shock_name}_lead_{lead}"] = out[shock_name].shift(-lead)

    return pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)


def estimate_dashboard_lp(
    data: pd.DataFrame,
    response: str,
    shock_name: str,
    controls: list[str],
    config: LPConfig,
    response_label: str,
    cumulative_response: bool = False,
    dummy_controls: list[str] | None = None,
) -> pd.DataFrame:
    """Estimate horizon-by-horizon LPs used by the dashboard."""
    dummy_controls = dummy_controls or []
    design = prepare_lp_design(
        data=data,
        response=response,
        shock_name=shock_name,
        controls=controls,
        config=config,
        cumulative_response=cumulative_response,
        dummy_controls=dummy_controls,
    )
    results = []
    z = z_value(config.ci_level)

    lag_regressors = []
    for lag in range(1, config.lags + 1):
        lag_regressors.append(f"{response}_lag_{lag}")
        lag_regressors.append(f"{shock_name}_lag_{lag}")
        for control in controls:
            lag_regressors.append(f"{control}_lag_{lag}")

    for horizon in range(config.horizons + 1):
        dep_var = f"{response}_lead_{horizon}"
        forward_shocks = (
            [f"{shock_name}_lead_{lead}" for lead in range(1, horizon + 1)]
            if config.include_forward_shocks
            else []
        )
        regressors = [shock_name, *controls, *dummy_controls, *forward_shocks, *lag_regressors]
        sample = design.loc[:, [dep_var, *regressors]].dropna()
        if sample.empty or len(sample) <= len(regressors) + 1:
            results.append(
                {
                    "response": response_label,
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
        coef = model.params[shock_name]
        se = model.bse[shock_name]
        results.append(
            {
                "response": response_label,
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


def estimate_asymmetric_dashboard_lp(
    data: pd.DataFrame,
    response: str,
    shock_name: str,
    controls: list[str],
    config: LPConfig,
    response_label: str,
    cumulative_response: bool = False,
    dummy_controls: list[str] | None = None,
) -> pd.DataFrame:
    """Estimate LPs with separate coefficients for positive and negative shocks."""
    dummy_controls = dummy_controls or []
    positive_shock = f"{shock_name}_positive"
    negative_shock = f"{shock_name}_negative"

    asym_data = data.copy()
    asym_data[positive_shock] = asym_data[shock_name].clip(lower=0)
    asym_data[negative_shock] = (-asym_data[shock_name].clip(upper=0))

    design = prepare_lp_design(
        data=asym_data,
        response=response,
        shock_name=shock_name,
        controls=[positive_shock, negative_shock, *controls],
        config=config,
        cumulative_response=cumulative_response,
        dummy_controls=dummy_controls,
    )
    split_leads = {}
    for lead in range(1, config.horizons + 1):
        split_leads[f"{positive_shock}_lead_{lead}"] = design[positive_shock].shift(-lead)
        split_leads[f"{negative_shock}_lead_{lead}"] = design[negative_shock].shift(-lead)
    if split_leads:
        design = pd.concat([design, pd.DataFrame(split_leads, index=design.index)], axis=1)
    results = []
    z = z_value(config.ci_level)

    lag_regressors = []
    for lag in range(1, config.lags + 1):
        lag_regressors.append(f"{response}_lag_{lag}")
        lag_regressors.append(f"{positive_shock}_lag_{lag}")
        lag_regressors.append(f"{negative_shock}_lag_{lag}")
        for control in controls:
            lag_regressors.append(f"{control}_lag_{lag}")

    for horizon in range(config.horizons + 1):
        dep_var = f"{response}_lead_{horizon}"
        if config.include_forward_shocks:
            forward_shocks = [
                item
                for lead in range(1, horizon + 1)
                for item in (f"{positive_shock}_lead_{lead}", f"{negative_shock}_lead_{lead}")
            ]
        else:
            forward_shocks = []
        regressors = [
            positive_shock,
            negative_shock,
            *controls,
            *dummy_controls,
            *forward_shocks,
            *lag_regressors,
        ]
        sample = design.loc[:, [dep_var, *regressors]].dropna()
        if sample.empty or len(sample) <= len(regressors) + 1:
            for component, label in [
                (positive_shock, "Positive CHF shock"),
                (negative_shock, "Negative CHF shock"),
            ]:
                results.append(
                    {
                        "response": response_label,
                        "shock_component": label,
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
        for component, label in [
            (positive_shock, "Positive CHF shock"),
            (negative_shock, "Negative CHF shock"),
        ]:
            coef = model.params[component]
            se = model.bse[component]
            results.append(
                {
                    "response": response_label,
                    "shock_component": label,
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
