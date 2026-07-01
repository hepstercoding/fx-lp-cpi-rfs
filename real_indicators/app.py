from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gdp_lp_chf.local_projections import (
    LPConfig,
    estimate_asymmetric_lp,
    estimate_lp,
    estimate_split_shock_lp,
    maintained_response_from_irfs,
    z_value,
)


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "swiss_gdp_macro_real.csv"
MONTHLY_LABOR_PATH = PROJECT_ROOT / "data" / "raw" / "swiss_labor_market_monthly.csv"
COMPONENT_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "seco_component_metadata.csv"
SOURCE_METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "source_metadata.csv"

SHOCK_OPTIONS = {
    "CHF NEER appreciation": {
        "column": "chf_neer_appreciation_qoq",
        "level_column": "log_chf_neer_pct",
        "label": "CHF NEER appreciation",
        "level_label": "CHF NEER appreciation level",
        "unit": "1 percentage-point quarterly log appreciation",
    },
    "EURCHF appreciation": {
        "column": "eur_chf_appreciation_qoq",
        "level_column": "log_eur_chf_appreciation_pct",
        "label": "EURCHF-implied CHF appreciation",
        "level_label": "EURCHF-implied CHF appreciation level",
        "unit": "1 percentage-point quarterly log appreciation against EUR",
    },
    "USDCHF appreciation": {
        "column": "usd_chf_appreciation_qoq",
        "level_column": "log_usd_chf_appreciation_pct",
        "label": "USDCHF-implied CHF appreciation",
        "level_label": "USDCHF-implied CHF appreciation level",
        "unit": "1 percentage-point quarterly log appreciation against USD",
    },
    "SNB CHF NEER appreciation": {
        "column": "snb_chf_neer_appreciation_qoq",
        "level_column": "log_snb_chf_neer_pct",
        "label": "SNB CHF NEER appreciation",
        "level_label": "SNB CHF NEER appreciation level",
        "unit": "1 percentage-point quarterly log appreciation",
    },
}

MONTHLY_SHOCK_OPTIONS = {
    "CHF NEER appreciation": {
        "column": "chf_neer_appreciation_mom",
        "level_column": "log_chf_neer_pct",
        "label": "CHF NEER appreciation",
        "level_label": "CHF NEER appreciation level",
        "unit": "1 percentage-point monthly log appreciation",
    },
    "EURCHF appreciation": {
        "column": "eur_chf_appreciation_mom",
        "level_column": "log_eur_chf_appreciation_pct",
        "label": "EURCHF-implied CHF appreciation",
        "level_label": "EURCHF-implied CHF appreciation level",
        "unit": "1 percentage-point monthly log appreciation against EUR",
    },
    "USDCHF appreciation": {
        "column": "usd_chf_appreciation_mom",
        "level_column": "log_usd_chf_appreciation_pct",
        "label": "USDCHF-implied CHF appreciation",
        "level_label": "USDCHF-implied CHF appreciation level",
        "unit": "1 percentage-point monthly log appreciation against USD",
    },
    "SNB CHF NEER appreciation": {
        "column": "snb_chf_neer_appreciation_mom",
        "level_column": "log_snb_chf_neer_pct",
        "label": "SNB CHF NEER appreciation",
        "level_label": "SNB CHF NEER appreciation level",
        "unit": "1 percentage-point monthly log appreciation",
    },
}

LP_RESPONSE_OPTIONS = {
    "Cumulative GDP level": {
        "column": "gdp_log_level",
        "label": "Cumulative real GDP level",
        "cumulative": True,
        "unit": "percent log points",
    },
    "Quarterly GDP growth": {
        "column": "gdp_growth_qoq",
        "label": "Real GDP q/q growth",
        "cumulative": False,
        "unit": "percentage points",
    },
    "Output gap, production function": {
        "column": "snb_output_gap",
        "label": "SNB output gap, production function",
        "cumulative": False,
        "unit": "percentage points of potential output",
    },
    "Output gap, HP filter": {
        "column": "snb_output_gap_hp",
        "label": "SNB output gap, HP filter",
        "cumulative": False,
        "unit": "percentage points of potential output",
    },
    "Output gap, multivariate filter": {
        "column": "snb_output_gap_multivariate",
        "label": "SNB output gap, multivariate filter",
        "cumulative": False,
        "unit": "percentage points of potential output",
    },
}

MONTHLY_LABOR_RESPONSE_OPTIONS = {
    "% change in number of unemployed": {
        "column": "log_registered_unemployed_sa_pct",
        "label": "Registered unemployed, seasonally adjusted",
        "cumulative": True,
        "unit": "percent log points",
    },
    "% change in number of job seekers": {
        "column": "log_registered_job_seekers_pct",
        "label": "Registered job seekers",
        "cumulative": True,
        "unit": "percent log points",
    },
    "pp change in unemployment rate": {
        "column": "unemployment_rate_sa",
        "label": "Unemployment rate, seasonally adjusted",
        "cumulative": True,
        "unit": "percentage points",
    },
    "pp change in job seeker rate": {
        "column": "job_seeker_rate",
        "label": "Job seeker rate",
        "cumulative": True,
        "unit": "percentage points",
    },
}

CONTROL_SETS = {
    "None": [],
    "Foreign demand": ["ea_real_gdp_growth_qoq"],
    "Oil": ["brent_oil_change_qoq"],
    "Output gap": ["snb_output_gap"],
    "Full": ["ea_real_gdp_growth_qoq", "brent_oil_change_qoq", "snb_output_gap"],
}

CONTROL_OPTIONS = {
    "Euro area real GDP growth": "ea_real_gdp_growth_qoq",
    "Brent oil q/q log change": "brent_oil_change_qoq",
    "SNB output gap": "snb_output_gap",
}


st.set_page_config(page_title="Swiss GDP and CHF Data Audit", layout="wide")


@st.cache_data(show_spinner=False)
def load_data_csv(path: str, file_mtime: float | None = None) -> pd.DataFrame:
    _ = file_mtime
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metadata_csv(path: str, file_mtime: float | None = None) -> pd.DataFrame:
    _ = file_mtime
    return pd.read_csv(path)


def safe_load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_data_csv(str(path), path.stat().st_mtime)


def safe_load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_metadata_csv(str(path), path.stat().st_mtime)


def date_bounds(data: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return data["date"].min(), data["date"].max()


def quarter_label(date: pd.Timestamp) -> str:
    period = pd.Timestamp(date).to_period("Q")
    return f"{period.year} Q{period.quarter}"


def pct_missing(series: pd.Series) -> float:
    return 100 * float(series.isna().mean())


def coverage_table(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metadata.iterrows():
        column = row["structure"]
        if column not in data:
            rows.append(
                {
                    "Group": row["group"],
                    "Code": column,
                    "Label": row["label"],
                    "Available": False,
                    "First": pd.NaT,
                    "Last": pd.NaT,
                    "Missing %": np.nan,
                    "Observations": 0,
                }
            )
            continue
        sample = data.loc[data[column].notna(), ["date", column]]
        rows.append(
            {
                "Group": row["group"],
                "Code": column,
                "Label": row["label"],
                "Available": bool(len(sample)),
                "First": sample["date"].min() if len(sample) else pd.NaT,
                "Last": sample["date"].max() if len(sample) else pd.NaT,
                "Missing %": pct_missing(data[column]),
                "Observations": int(sample[column].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def component_options(metadata: pd.DataFrame, group: str, data: pd.DataFrame) -> dict[str, str]:
    if metadata.empty:
        return {}
    current = metadata.loc[(metadata["group"] == group) & (metadata["available"])].copy()
    current = current.loc[current["structure"].isin(data.columns)]
    return {f"{row['label']} ({row['structure']})": row["structure"] for _, row in current.iterrows()}


def line_chart(data: pd.DataFrame, columns: list[str], title: str, y_label: str) -> None:
    plot_data = data.loc[:, ["date", *columns]].set_index("date")
    st.caption(title)
    st.line_chart(plot_data, y_label=y_label)


def lp_chart(
    results: pd.DataFrame,
    y_label: str,
    horizon_unit: str = "quarters",
    reference_value: float = 0.0,
) -> None:
    chart_data = results.loc[:, ["horizon", "beta", "ci_lower", "ci_upper"]].dropna()
    if chart_data.empty:
        st.info("No estimated horizons to plot.")
        return
    horizon_title = f"Horizon, {horizon_unit}"
    x_axis = alt.X("horizon:Q", title=horizon_title)
    band = (
        alt.Chart(chart_data)
        .mark_area(opacity=0.22, color="#7A9E9F")
        .encode(
            x=x_axis,
            y=alt.Y("ci_lower:Q", title=y_label),
            y2="ci_upper:Q",
        )
    )
    line = (
        alt.Chart(chart_data)
        .mark_line(color="#1F2A2E", strokeWidth=2.5)
        .encode(
            x=x_axis,
            y=alt.Y("beta:Q", title=y_label),
        )
    )
    zero_data = pd.DataFrame(
        {
            "x_start": [chart_data["horizon"].min()],
            "x_end": [chart_data["horizon"].max()],
            "y": [reference_value],
        }
    )
    zero = (
        alt.Chart(zero_data)
        .mark_rule(color="#4B5563", strokeWidth=1.6)
        .encode(x=alt.X("x_start:Q", title=horizon_title), x2="x_end:Q", y="y:Q")
    )
    chart = band + line + zero
    st.altair_chart(
        chart.properties(height=420),
        use_container_width=True,
    )


def selected_valid_controls(
    sample: pd.DataFrame,
    selected_control_labels: list[str],
    excluded_columns: list[str],
) -> tuple[list[str], list[str]]:
    selected = []
    omitted = []
    excluded = set(excluded_columns)
    for label in selected_control_labels:
        column = CONTROL_OPTIONS[label]
        if column in excluded or column not in sample.columns or not sample[column].notna().any():
            omitted.append(column)
        else:
            selected.append(column)
    return selected, omitted


def asymmetric_lp_chart(
    results: pd.DataFrame,
    y_label: str,
    regimes: list[str],
    horizon_unit: str = "quarters",
    reference_value: float = 0.0,
) -> None:
    chart_data = results.loc[results["regime"].isin(regimes), ["horizon", "regime", "beta", "ci_lower", "ci_upper"]].dropna()
    if chart_data.empty:
        st.info("No estimated horizons to plot.")
        return
    horizon_title = f"Horizon, {horizon_unit}"
    x_axis = alt.X("horizon:Q", title=horizon_title)
    palette = ["#1F2A2E", "#B35C44", "#4C6B8A"]
    band = (
        alt.Chart(chart_data)
        .mark_area(opacity=0.16)
        .encode(
            x=x_axis,
            y=alt.Y("ci_lower:Q", title=y_label),
            y2="ci_upper:Q",
            color=alt.Color("regime:N", scale=alt.Scale(domain=regimes, range=palette[: len(regimes)]), title="Shock"),
        )
    )
    line = (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=2.4)
        .encode(
            x=x_axis,
            y=alt.Y("beta:Q", title=y_label),
            color=alt.Color("regime:N", scale=alt.Scale(domain=regimes, range=palette[: len(regimes)]), title="Shock"),
        )
    )
    zero_data = pd.DataFrame(
        {
            "x_start": [chart_data["horizon"].min()],
            "x_end": [chart_data["horizon"].max()],
            "y": [reference_value],
        }
    )
    zero = (
        alt.Chart(zero_data)
        .mark_rule(color="#4B5563", strokeWidth=1.6)
        .encode(x=alt.X("x_start:Q", title=horizon_title), x2="x_end:Q", y="y:Q")
    )
    chart = band + line + zero
    st.altair_chart(
        chart.properties(height=420),
        use_container_width=True,
    )


def split_lp_chart(
    split_results: pd.DataFrame,
    symmetric_results: pd.DataFrame,
    y_label: str,
    regimes: list[str],
    horizon_unit: str = "quarters",
    reference_value: float = 0.0,
) -> None:
    chart_data = split_results.loc[
        split_results["regime"].isin(regimes),
        ["horizon", "regime", "beta", "ci_lower", "ci_upper"],
    ].dropna()
    symmetric_data = symmetric_results.loc[:, ["horizon", "beta"]].dropna() if not symmetric_results.empty else pd.DataFrame()
    if chart_data.empty and symmetric_data.empty:
        st.info("No estimated horizons to plot.")
        return

    horizon_title = f"Horizon, {horizon_unit}"
    x_axis = alt.X("horizon:Q", title=horizon_title)
    palette = ["#0F766E", "#B45309"]
    layers = []
    if not chart_data.empty:
        layers.append(
            alt.Chart(chart_data)
            .mark_area(opacity=0.15)
            .encode(
                x=x_axis,
                y=alt.Y("ci_lower:Q", title=y_label),
                y2="ci_upper:Q",
                color=alt.Color("regime:N", scale=alt.Scale(domain=regimes, range=palette[: len(regimes)]), title="Shock"),
            )
        )
        layers.append(
            alt.Chart(chart_data)
            .mark_line(strokeWidth=2.4)
            .encode(
                x=x_axis,
                y=alt.Y("beta:Q", title=y_label),
                color=alt.Color("regime:N", scale=alt.Scale(domain=regimes, range=palette[: len(regimes)]), title="Shock"),
            )
        )
    if not symmetric_data.empty:
        layers.append(
            alt.Chart(symmetric_data)
            .mark_line(color="#111827", strokeDash=[6, 4], strokeWidth=2.1)
            .encode(x=x_axis, y=alt.Y("beta:Q", title=y_label))
        )

    horizon_min = chart_data["horizon"].min() if not chart_data.empty else symmetric_data["horizon"].min()
    horizon_max = chart_data["horizon"].max() if not chart_data.empty else symmetric_data["horizon"].max()
    reference_data = pd.DataFrame({"x_start": [horizon_min], "x_end": [horizon_max], "y": [reference_value]})
    layers.append(
        alt.Chart(reference_data)
        .mark_rule(color="#4B5563", strokeWidth=1.2)
        .encode(x=alt.X("x_start:Q", title=horizon_title), x2="x_end:Q", y="y:Q")
    )
    st.altair_chart(alt.layer(*layers).properties(height=460), use_container_width=True)
    if not symmetric_data.empty:
        st.caption("Black dashed line: symmetric IRF estimated on the same sample and controls.")


def large_appreciation_definition_chart(
    sample: pd.DataFrame,
    shock_level_column: str,
    shock_column: str,
    shock_label: str,
    threshold: float,
    frequency_label: str,
) -> None:
    if shock_level_column not in sample.columns or shock_column not in sample.columns or np.isnan(threshold):
        st.info("Not enough exchange-rate data to show the large-appreciation definition chart.")
        return
    plot_data = sample.loc[:, ["date", shock_level_column, shock_column]].dropna(subset=[shock_column]).copy()
    if plot_data.empty:
        st.info("No exchange-rate observations are available in the selected sample.")
        return
    plot_data["large_appreciation"] = plot_data[shock_column].gt(0) & plot_data[shock_column].ge(threshold)
    level = (
        alt.Chart(plot_data)
        .mark_line(color="#2563EB", strokeWidth=1.8)
        .encode(x=alt.X("date:T", title=""), y=alt.Y(f"{shock_level_column}:Q", title="Log level"))
    )
    level_points = (
        alt.Chart(plot_data.loc[plot_data["large_appreciation"]])
        .mark_point(color="#0F766E", filled=True, size=55)
        .encode(x="date:T", y=f"{shock_level_column}:Q")
    )
    moves = (
        alt.Chart(plot_data)
        .mark_line(color="#4B5563", strokeWidth=1.3)
        .encode(x=alt.X("date:T", title=""), y=alt.Y(f"{shock_column}:Q", title="Percent log points"))
    )
    move_points = (
        alt.Chart(plot_data.loc[plot_data["large_appreciation"]])
        .mark_point(color="#0F766E", filled=True, size=65)
        .encode(x="date:T", y=f"{shock_column}:Q")
    )
    rules = pd.DataFrame({"value": [0.0, threshold], "label": ["Zero", "Large threshold"]})
    move_rules = (
        alt.Chart(rules)
        .mark_rule(strokeDash=[5, 4], strokeWidth=1.3)
        .encode(
            y="value:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=["Zero", "Large threshold"], range=["#4B5563", "#0F766E"]),
                title="",
            ),
        )
    )
    chart = alt.vconcat(
        (level + level_points).properties(title=f"{shock_label}: appreciation-positive FX level", height=220),
        (moves + move_points + move_rules).properties(title=f"{frequency_label} FX move used for the large-appreciation split", height=260),
    ).resolve_scale(color="independent")
    st.altair_chart(chart, use_container_width=True)


@st.cache_data(show_spinner=False)
def run_headline_lp(
    data: pd.DataFrame,
    response_column: str,
    response_label: str,
    shock_column: str,
    shock_label: str,
    controls: list[str],
    horizons: int,
    lags: int,
    ci_level: float,
    cumulative_response: bool,
    include_forward_shocks: bool,
) -> pd.DataFrame:
    config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        include_forward_shocks=include_forward_shocks,
    )
    return estimate_lp(
        data=data,
        response=response_column,
        shock=shock_column,
        controls=controls,
        config=config,
        response_label=response_label,
        shock_label=shock_label,
        cumulative_response=cumulative_response,
    )


@st.cache_data(show_spinner=False)
def run_asymmetric_lp(
    data: pd.DataFrame,
    response_column: str,
    response_label: str,
    shock_column: str,
    shock_label: str,
    controls: list[str],
    horizons: int,
    lags: int,
    ci_level: float,
    cumulative_response: bool,
    include_forward_shocks: bool,
) -> pd.DataFrame:
    config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        include_forward_shocks=include_forward_shocks,
    )
    return estimate_asymmetric_lp(
        data=data,
        response=response_column,
        shock=shock_column,
        controls=controls,
        config=config,
        response_label=response_label,
        shock_label=shock_label,
        cumulative_response=cumulative_response,
    )


@st.cache_data(show_spinner=False)
def run_large_appreciation_lp(
    data: pd.DataFrame,
    response_column: str,
    response_label: str,
    shock_column: str,
    shock_label: str,
    shock_level_column: str,
    shock_level_label: str,
    controls: list[str],
    horizons: int,
    lags: int,
    ci_level: float,
    cumulative_response: bool,
    include_forward_shocks: bool,
    appreciation_percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float, int, int]:
    sample = data.copy()
    positive_moves = sample.loc[sample[shock_column].gt(0), shock_column].dropna()
    if positive_moves.empty:
        empty = pd.DataFrame(
            columns=[
                "response",
                "shock",
                "regime",
                "horizon",
                "beta",
                "std_error",
                "ci_lower",
                "ci_upper",
                "nobs",
                "r_squared",
                "num_forward_shocks",
            ]
        )
        return empty, empty.copy(), float("nan"), 0, 0

    threshold = float(np.nanpercentile(positive_moves, appreciation_percentile))
    large_component = f"{shock_column}_large_appreciation"
    other_component = f"{shock_column}_all_other_moves"
    large_mask = sample[shock_column].gt(0) & sample[shock_column].ge(threshold)
    sample[large_component] = sample[shock_column].where(large_mask, 0.0)
    sample[other_component] = sample[shock_column].where(~large_mask, 0.0)
    shock_components = {
        large_component: "Large appreciations",
        other_component: "All other FX moves",
    }
    config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        include_forward_shocks=include_forward_shocks,
    )
    response_results = estimate_split_shock_lp(
        data=sample,
        response=response_column,
        shock=shock_column,
        shock_components=shock_components,
        controls=controls,
        config=config,
        response_label=response_label,
        shock_label=shock_label,
        cumulative_response=cumulative_response,
    )
    fx_results = pd.DataFrame()
    if shock_level_column in sample.columns:
        fx_results = estimate_split_shock_lp(
            data=sample,
            response=shock_level_column,
            shock=shock_column,
            shock_components=shock_components,
            controls=controls,
            config=config,
            response_label=shock_level_label,
            shock_label=shock_label,
            cumulative_response=True,
        )
    return response_results, fx_results, threshold, int(large_mask.sum()), int(positive_moves.size)


def layered_split_response_from_irfs(response_irf: pd.DataFrame, shock_path_irf: pd.DataFrame, ci_level: float) -> pd.DataFrame:
    layered_parts = []
    for regime, response_part in response_irf.groupby("regime", sort=False):
        path_part = shock_path_irf.loc[shock_path_irf["regime"].eq(regime)].copy()
        if response_part.empty or path_part.empty:
            continue
        layered = maintained_response_from_irfs(
            response_part,
            path_part,
            ci_level=ci_level,
            label=response_part["response"].iloc[0] if "response" in response_part else regime,
        )
        if layered.empty:
            continue
        layered["regime"] = regime
        layered_parts.append(layered)
    return pd.concat(layered_parts, ignore_index=True) if layered_parts else pd.DataFrame(columns=response_irf.columns)


def layered_asymmetric_response_from_irfs(
    response_irf: pd.DataFrame,
    shock_path_irf: pd.DataFrame,
    ci_level: float,
    label: str,
) -> pd.DataFrame:
    layered_parts = []
    for regime in ["Appreciation (+1 pp)", "Depreciation (-1 pp)"]:
        response_part = response_irf.loc[response_irf["regime"].eq(regime)].copy()
        path_part = shock_path_irf.loc[shock_path_irf["regime"].eq(regime)].copy()
        if response_part.empty or path_part.empty:
            continue

        if regime == "Depreciation (-1 pp)":
            path_part = path_part.copy()
            path_part["beta"] = path_part["beta"].abs()

        layered = maintained_response_from_irfs(
            response_part,
            path_part,
            ci_level=ci_level,
            label=label,
        )
        if layered.empty:
            continue
        layered["regime"] = regime
        layered_parts.append(layered)

    if not layered_parts:
        return pd.DataFrame(columns=response_irf.columns)

    out = pd.concat(layered_parts, ignore_index=True)
    app = out.loc[out["regime"].eq("Appreciation (+1 pp)")].sort_values("horizon").reset_index(drop=True)
    dep = out.loc[out["regime"].eq("Depreciation (-1 pp)")].sort_values("horizon").reset_index(drop=True)
    if not app.empty and not dep.empty:
        n = min(len(app), len(dep))
        diff = app.iloc[:n].copy()
        diff["regime"] = "Appreciation minus depreciation"
        diff["beta"] = app["beta"].iloc[:n].to_numpy() - dep["beta"].iloc[:n].to_numpy()
        diff["std_error"] = np.sqrt(app["std_error"].iloc[:n].to_numpy() ** 2 + dep["std_error"].iloc[:n].to_numpy() ** 2)
        z = z_value(ci_level)
        diff["ci_lower"] = diff["beta"] - z * diff["std_error"]
        diff["ci_upper"] = diff["beta"] + z * diff["std_error"]
        if "maintenance_shock" in diff:
            diff["maintenance_shock"] = np.nan
        if "maintained_exchange_rate_path" in diff:
            diff["maintained_exchange_rate_path"] = np.nan
        out = pd.concat([out, diff], ignore_index=True)
    return out


def component_panel(data: pd.DataFrame, metadata: pd.DataFrame, group: str, title: str) -> None:
    st.subheader(title)
    options = component_options(metadata, group, data)
    if not options:
        st.info("No available component series found for this group.")
        return
    defaults = list(options)[: min(4, len(options))]
    selected_labels = st.multiselect(
        "Series",
        options=list(options),
        default=defaults,
        key=f"{group}_series",
    )
    selected = [options[label] for label in selected_labels]
    transform = st.radio(
        "View",
        ["Level", "Quarterly growth", "Year-over-year growth"],
        horizontal=True,
        key=f"{group}_transform",
    )
    suffix = {
        "Level": "",
        "Quarterly growth": "_growth_qoq",
        "Year-over-year growth": "_growth_yoy",
    }[transform]
    chart_columns = [f"{column}{suffix}" for column in selected if f"{column}{suffix}" in data.columns]
    if chart_columns:
        line_chart(data, chart_columns, title, "Index / percent log points")
    else:
        st.info("Selected transform is not available for these series.")


def render_headline_data_tab(sample: pd.DataFrame) -> None:
    st.subheader("Headline Real GDP")
    chart_columns = [column for column in ["gdp", "gdp_growth_qoq", "gdp_growth_yoy"] if column in sample.columns]
    if "gdp" in chart_columns:
        line_chart(sample, ["gdp"], "Real GDP, SECO real/cssa", "Level")
    growth_cols = [column for column in ["gdp_growth_qoq", "gdp_growth_yoy"] if column in sample.columns]
    if growth_cols:
        line_chart(sample, growth_cols, "GDP growth transforms", "Percent log points")
    st.dataframe(sample.tail(12), use_container_width=True)


def render_headline_lp_tab(sample: pd.DataFrame) -> None:
    st.subheader("Headline GDP Local Projection")
    col1, col2, col3 = st.columns(3)
    with col1:
        response_name = st.selectbox("Response", list(LP_RESPONSE_OPTIONS), index=0)
        shock_name = st.selectbox("Shock", list(SHOCK_OPTIONS), index=0)
    with col2:
        selected_control_labels = st.multiselect(
            "Controls",
            options=list(CONTROL_OPTIONS),
            default=[],
        )
        ci_level = st.selectbox("Confidence band", [0.68, 0.90, 0.95], index=1, format_func=lambda x: f"{int(x * 100)}%")
    with col3:
        fx_path_label = st.selectbox("FX path", ["Standard", "Layered 1% move", "Forward shocks"], index=0)
        horizons = st.slider("Horizon", min_value=4, max_value=16, value=12, step=1)
        lags = st.slider("Lags", min_value=1, max_value=8, value=4, step=1)

    response_settings = LP_RESPONSE_OPTIONS[response_name]
    shock_settings = SHOCK_OPTIONS[shock_name]
    include_forward_shocks = fx_path_label == "Forward shocks"
    use_layered_shocks = fx_path_label == "Layered 1% move"
    selected_controls, missing_controls = selected_valid_controls(
        sample,
        selected_control_labels,
        [response_settings["column"], shock_settings["column"]],
    )
    required_columns = [response_settings["column"], shock_settings["column"], *selected_controls]
    lp_sample = sample.loc[:, ["date", *required_columns]].dropna().copy()

    if missing_controls:
        st.warning(f"Unavailable controls omitted: {', '.join(missing_controls)}")

    if len(lp_sample) <= (2 * lags + len(selected_controls) * lags + horizons + 5):
        st.warning("The selected model has a thin usable sample. Consider fewer lags, fewer controls, or a wider date range.")

    if lp_sample.empty:
        st.info("No usable observations for the selected LP setup.")
    else:
        results = run_headline_lp(
            data=lp_sample,
            response_column=response_settings["column"],
            response_label=response_settings["label"],
            shock_column=shock_settings["column"],
            shock_label=shock_settings["label"],
            controls=selected_controls,
            horizons=horizons,
            lags=lags,
            ci_level=ci_level,
            cumulative_response=bool(response_settings["cumulative"]),
            include_forward_shocks=include_forward_shocks,
        )
        chf_results = pd.DataFrame()
        shock_level_column = shock_settings["level_column"]
        if shock_level_column in sample.columns:
            chf_required = [shock_level_column, shock_settings["column"], *selected_controls]
            chf_sample = sample.loc[:, ["date", *chf_required]].dropna().copy()
            if not chf_sample.empty:
                chf_results = run_headline_lp(
                    data=chf_sample,
                    response_column=shock_level_column,
                    response_label=shock_settings["level_label"],
                    shock_column=shock_settings["column"],
                    shock_label=shock_settings["label"],
                    controls=selected_controls,
                    horizons=horizons,
                    lags=lags,
                    ci_level=ci_level,
                    cumulative_response=True,
                    include_forward_shocks=include_forward_shocks,
                )
        if use_layered_shocks:
            standard_results = run_headline_lp(
                data=lp_sample,
                response_column=response_settings["column"],
                response_label=response_settings["label"],
                shock_column=shock_settings["column"],
                shock_label=shock_settings["label"],
                controls=selected_controls,
                horizons=horizons,
                lags=lags,
                ci_level=ci_level,
                cumulative_response=bool(response_settings["cumulative"]),
                include_forward_shocks=False,
            )
            standard_chf_results = pd.DataFrame()
            if shock_level_column in sample.columns:
                chf_required = [shock_level_column, shock_settings["column"], *selected_controls]
                chf_sample = sample.loc[:, ["date", *chf_required]].dropna().copy()
                if not chf_sample.empty:
                    standard_chf_results = run_headline_lp(
                        data=chf_sample,
                        response_column=shock_level_column,
                        response_label=shock_settings["level_label"],
                        shock_column=shock_settings["column"],
                        shock_label=shock_settings["label"],
                        controls=selected_controls,
                        horizons=horizons,
                        lags=lags,
                        ci_level=ci_level,
                        cumulative_response=True,
                        include_forward_shocks=False,
                    )
            if not standard_chf_results.empty:
                results = maintained_response_from_irfs(
                    standard_results,
                    standard_chf_results,
                    ci_level=ci_level,
                    label=f"{response_settings['label']} under maintained 1% CHF appreciation",
                )
                chf_results = maintained_response_from_irfs(
                    standard_chf_results,
                    standard_chf_results,
                    ci_level=ci_level,
                    label=f"{shock_settings['level_label']} under maintained 1% CHF appreciation",
                )

        if results.empty:
            st.info("No usable LP estimates are available for the selected setup.")
            return

        metric_cols = st.columns(4)
        valid_nobs = results["nobs"].replace(0, np.nan)
        metric_cols[0].metric("Usable rows", f"{len(lp_sample):,}")
        metric_cols[1].metric("First row", quarter_label(lp_sample["date"].min()))
        metric_cols[2].metric("Last row", quarter_label(lp_sample["date"].max()))
        metric_cols[3].metric("Minimum nobs", f"{int(valid_nobs.min()) if valid_nobs.notna().any() else 0:,}")

        st.caption(
            f"Response to a {shock_settings['unit']}. Controls: "
            f"{', '.join(selected_controls) if selected_controls else 'none'}."
        )
        if fx_path_label == "Standard":
            st.caption("FX path: one initial CHF move followed by the estimated CHF path.")
        elif fx_path_label == "Layered 1% move":
            st.caption(
                "FX path: one-shock IRFs are layered so the CHF appreciation path is maintained at 1%. "
                "Confidence bands are approximate and ignore uncertainty in the estimated FX-path weights."
            )
        else:
            st.caption("FX path: future CHF moves enter the regression, isolating the initial move conditional on later CHF changes.")
        lp_chart(results, response_settings["unit"])
        st.dataframe(
            results.loc[:, ["horizon", "beta", "std_error", "ci_lower", "ci_upper", "nobs", "r_squared"]],
            use_container_width=True,
            hide_index=True,
        )

        if not chf_results.empty:
            st.subheader("FX Reaction")
            st.caption(
                f"Estimated path of {shock_settings['level_label']} after the selected "
                f"{shock_settings['unit']}. Same controls and sample range."
            )
            lp_chart(chf_results, "percent log points", reference_value=1.0)
            with st.expander("CHF reaction estimates", expanded=False):
                display_columns = [
                    column
                    for column in [
                        "horizon",
                        "beta",
                        "std_error",
                        "ci_lower",
                        "ci_upper",
                        "nobs",
                        "r_squared",
                        "maintenance_shock",
                        "maintained_exchange_rate_path",
                    ]
                    if column in chf_results.columns
                ]
                st.dataframe(
                    chf_results.loc[:, display_columns],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No usable FX reaction estimates are available for the selected setup.")


def render_asymmetric_irf_tab(sample: pd.DataFrame) -> None:
    st.subheader("Asymmetric Local Projections")
    col1, col2, col3 = st.columns(3)
    with col1:
        response_name = st.selectbox("Response", list(LP_RESPONSE_OPTIONS), index=0, key="asym_response")
        shock_name = st.selectbox("Shock", list(SHOCK_OPTIONS), index=0, key="asym_shock")
    with col2:
        selected_control_labels = st.multiselect(
            "Controls",
            options=list(CONTROL_OPTIONS),
            default=[],
            key="asym_controls",
        )
        ci_level = st.selectbox(
            "Confidence band",
            [0.68, 0.90, 0.95],
            index=1,
            format_func=lambda x: f"{int(x * 100)}%",
            key="asym_ci",
        )
    with col3:
        dynamic_mode = st.selectbox("Dynamics", ["Standard", "Layered 1% move", "Forward shocks"], index=0, key="asym_dynamics")
        horizons = st.slider("Horizon", min_value=4, max_value=16, value=12, step=1, key="asym_horizon")
        lags = st.slider("Lags", min_value=1, max_value=8, value=4, step=1, key="asym_lags")

    response_settings = LP_RESPONSE_OPTIONS[response_name]
    shock_settings = SHOCK_OPTIONS[shock_name]
    include_forward_shocks = dynamic_mode == "Forward shocks"
    use_layered_shocks = dynamic_mode == "Layered 1% move"
    selected_controls, missing_controls = selected_valid_controls(
        sample,
        selected_control_labels,
        [response_settings["column"], shock_settings["column"]],
    )
    required_columns = [response_settings["column"], shock_settings["column"], *selected_controls]
    lp_sample = sample.loc[:, ["date", *required_columns]].dropna().copy()

    if missing_controls:
        st.warning(f"Unavailable controls omitted: {', '.join(missing_controls)}")
    if len(lp_sample) <= (3 + 2 * lags + len(selected_controls) * lags + horizons + 5):
        st.warning("The asymmetric specification has a thin usable sample. Consider fewer lags, fewer controls, or a wider date range.")
    if lp_sample.empty:
        st.info("No usable observations for the selected asymmetric LP setup.")
        return

    results = run_asymmetric_lp(
        data=lp_sample,
        response_column=response_settings["column"],
        response_label=response_settings["label"],
        shock_column=shock_settings["column"],
        shock_label=shock_settings["label"],
        controls=selected_controls,
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        cumulative_response=bool(response_settings["cumulative"]),
        include_forward_shocks=include_forward_shocks,
    )
    fx_results = pd.DataFrame()
    shock_level_column = shock_settings["level_column"]
    fx_required = [shock_level_column, shock_settings["column"], *selected_controls]
    fx_sample = sample.loc[:, ["date", *fx_required]].dropna().copy() if shock_level_column in sample.columns else pd.DataFrame()
    if not fx_sample.empty:
        fx_results = run_asymmetric_lp(
            data=fx_sample,
            response_column=shock_level_column,
            response_label=shock_settings["level_label"],
            shock_column=shock_settings["column"],
            shock_label=shock_settings["label"],
            controls=selected_controls,
            horizons=horizons,
            lags=lags,
            ci_level=ci_level,
            cumulative_response=True,
            include_forward_shocks=include_forward_shocks,
        )
    if use_layered_shocks:
        standard_fx_results = pd.DataFrame()
        if not fx_sample.empty:
            standard_fx_results = run_asymmetric_lp(
                data=fx_sample,
                response_column=shock_level_column,
                response_label=shock_settings["level_label"],
                shock_column=shock_settings["column"],
                shock_label=shock_settings["label"],
                controls=selected_controls,
                horizons=horizons,
                lags=lags,
                ci_level=ci_level,
                cumulative_response=True,
                include_forward_shocks=False,
            )
            fx_results = layered_asymmetric_response_from_irfs(
                standard_fx_results,
                standard_fx_results,
                ci_level=ci_level,
                label=f"{shock_settings['level_label']} under maintained 1% CHF appreciation/depreciation",
            )
        if not standard_fx_results.empty:
            results = layered_asymmetric_response_from_irfs(
                results,
                standard_fx_results,
                ci_level=ci_level,
                label=f"{response_settings['label']} under maintained 1% CHF appreciation/depreciation",
            )
    if results.empty:
        st.info("No usable asymmetric LP estimates are available for the selected setup.")
        return

    shown_regimes = st.multiselect(
        "IRFs",
        options=["Appreciation (+1 pp)", "Depreciation (-1 pp)", "Appreciation minus depreciation"],
        default=["Appreciation (+1 pp)", "Depreciation (-1 pp)"],
        key="asym_regimes",
    )

    metric_cols = st.columns(4)
    valid_nobs = results["nobs"].replace(0, np.nan)
    metric_cols[0].metric("Usable rows", f"{len(lp_sample):,}")
    metric_cols[1].metric("First row", quarter_label(lp_sample["date"].min()))
    metric_cols[2].metric("Last row", quarter_label(lp_sample["date"].max()))
    metric_cols[3].metric("Minimum nobs", f"{int(valid_nobs.min()) if valid_nobs.notna().any() else 0:,}")

    st.caption(
        f"The shock is split into positive CHF appreciations and negative CHF depreciations in one regression. "
        f"The depreciation line is reported as the response to a 1 percentage-point CHF depreciation. "
        f"For presentation, the raw negative-shock coefficient is multiplied by -1. "
        f"Controls: {', '.join(selected_controls) if selected_controls else 'none'}."
    )
    if dynamic_mode == "Forward shocks":
        st.caption("Dynamics: future CHF moves enter the regression, so the sign-split effect is conditional on later CHF changes.")
    elif dynamic_mode == "Layered 1% move":
        st.caption(
            "Dynamics: sign-specific one-shock IRFs are layered so the appreciation and depreciation paths are maintained at 1%. "
            "Confidence bands are approximate and ignore uncertainty in the estimated FX-path weights."
        )
    else:
        st.caption("Dynamics: standard sign-split local projection with one initial CHF move.")

    if shown_regimes:
        if not fx_results.empty:
            st.subheader("FX Reaction")
            st.caption(
                f"Sign-specific estimated path of {shock_settings['level_label']} after the selected "
                f"{shock_settings['unit']}. Same controls and sample range."
            )
            asymmetric_lp_chart(
                fx_results,
                "percent log points",
                shown_regimes,
                reference_value=1.0,
            )
            with st.expander("FX reaction estimates", expanded=False):
                fx_display = fx_results.loc[fx_results["regime"].isin(shown_regimes), :]
                fx_display_columns = [
                    column
                    for column in [
                        "regime",
                        "horizon",
                        "beta",
                        "raw_coefficient",
                        "display_multiplier",
                        "std_error",
                        "ci_lower",
                        "ci_upper",
                        "nobs",
                        "r_squared",
                        "maintenance_shock",
                        "maintained_exchange_rate_path",
                    ]
                    if column in fx_display.columns
                ]
                st.dataframe(fx_display.loc[:, fx_display_columns], use_container_width=True, hide_index=True)
        else:
            st.info("No usable FX reaction estimates are available for the selected asymmetric setup.")

        st.subheader("GDP Response")
        asymmetric_lp_chart(results, response_settings["unit"], shown_regimes)
        display = results.loc[results["regime"].isin(shown_regimes), :]
        display_columns = [
            column
            for column in [
                "regime",
                "horizon",
                "beta",
                "raw_coefficient",
                "display_multiplier",
                "std_error",
                "ci_lower",
                "ci_upper",
                "nobs",
                "r_squared",
            ]
            if column in display.columns
        ]
        st.dataframe(
            display.loc[:, display_columns],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Select at least one asymmetric IRF.")


def render_large_appreciation_lp_section(
    data: pd.DataFrame,
    response_options: dict[str, dict],
    shock_options: dict[str, dict],
    title: str,
    key_prefix: str,
    horizon_unit: str,
    horizon_min: int,
    horizon_max: int,
    horizon_default: int,
    horizon_step: int,
    lag_max: int,
    lag_default: int,
    controls_enabled: bool,
) -> None:
    st.subheader(title)
    start_date, end_date = date_bounds(data)
    min_date = start_date.to_pydatetime()
    max_date = end_date.to_pydatetime()

    col1, col2, col3 = st.columns(3)
    with col1:
        response_name = st.selectbox("Response", list(response_options), index=0, key=f"{key_prefix}_large_response")
        shock_name = st.selectbox("Shock", list(shock_options), index=0, key=f"{key_prefix}_large_shock")
    with col2:
        selected_control_labels: list[str] = []
        if controls_enabled:
            selected_control_labels = st.multiselect(
                "Controls",
                options=list(CONTROL_OPTIONS),
                default=[],
                key=f"{key_prefix}_large_controls",
            )
        ci_level = st.selectbox(
            "Confidence band",
            [0.68, 0.90, 0.95],
            index=1,
            format_func=lambda x: f"{int(x * 100)}%",
            key=f"{key_prefix}_large_ci",
        )
        date_range = st.slider(
            "Sample window",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
            key=f"{key_prefix}_large_date_range",
        )
    with col3:
        fx_path_label = st.selectbox(
            "FX path",
            ["Standard", "Layered 1% move", "Forward shocks"],
            index=0,
            key=f"{key_prefix}_large_fx_path",
        )
        appreciation_percentile = st.slider(
            "Large threshold",
            min_value=50,
            max_value=99,
            value=90,
            step=1,
            format="p%d",
            key=f"{key_prefix}_large_percentile",
        )
        horizons = st.slider(
            "Horizon",
            min_value=horizon_min,
            max_value=horizon_max,
            value=horizon_default,
            step=horizon_step,
            key=f"{key_prefix}_large_horizon",
        )
        lags = st.slider("Lags", min_value=1, max_value=lag_max, value=lag_default, step=1, key=f"{key_prefix}_large_lags")

    response_settings = response_options[response_name]
    shock_settings = shock_options[shock_name]
    include_forward_shocks = fx_path_label == "Forward shocks"
    use_layered_shocks = fx_path_label == "Layered 1% move"
    selected_start, selected_end = date_range
    sample = data.loc[(data["date"] >= pd.Timestamp(selected_start)) & (data["date"] <= pd.Timestamp(selected_end))].copy()

    if controls_enabled:
        selected_controls, missing_controls = selected_valid_controls(
            sample,
            selected_control_labels,
            [response_settings["column"], shock_settings["column"]],
        )
        if missing_controls:
            st.warning(f"Unavailable controls omitted: {', '.join(missing_controls)}")
    else:
        selected_controls = []

    required_columns = [
        response_settings["column"],
        shock_settings["column"],
        shock_settings["level_column"],
        *selected_controls,
    ]
    available_required = [column for column in required_columns if column in sample.columns]
    lp_sample = sample.loc[:, ["date", *available_required]].dropna().copy()
    if len(available_required) < len(required_columns):
        missing = sorted(set(required_columns).difference(sample.columns))
        st.warning(f"Unavailable required columns: {', '.join(missing)}")
    if len(lp_sample) <= (3 + 2 * lags + len(selected_controls) * lags + horizons + 5):
        st.warning("The selected large-appreciation model has a thin usable sample. Consider fewer lags, fewer controls, or a wider date range.")
    if lp_sample.empty:
        st.info("No usable observations for the selected large-appreciation LP setup.")
        return

    results, fx_results, threshold, large_count, positive_count = run_large_appreciation_lp(
        data=lp_sample,
        response_column=response_settings["column"],
        response_label=response_settings["label"],
        shock_column=shock_settings["column"],
        shock_label=shock_settings["label"],
        shock_level_column=shock_settings["level_column"],
        shock_level_label=shock_settings["level_label"],
        controls=selected_controls,
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        cumulative_response=bool(response_settings["cumulative"]),
        include_forward_shocks=include_forward_shocks,
        appreciation_percentile=float(appreciation_percentile),
    )
    symmetric_results = run_headline_lp(
        data=lp_sample,
        response_column=response_settings["column"],
        response_label=response_settings["label"],
        shock_column=shock_settings["column"],
        shock_label=shock_settings["label"],
        controls=selected_controls,
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        cumulative_response=bool(response_settings["cumulative"]),
        include_forward_shocks=include_forward_shocks,
    )
    symmetric_fx_results = run_headline_lp(
        data=lp_sample,
        response_column=shock_settings["level_column"],
        response_label=shock_settings["level_label"],
        shock_column=shock_settings["column"],
        shock_label=shock_settings["label"],
        controls=selected_controls,
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        cumulative_response=True,
        include_forward_shocks=include_forward_shocks,
    )
    if use_layered_shocks and not fx_results.empty:
        results = layered_split_response_from_irfs(results, fx_results, ci_level)
        symmetric_results = maintained_response_from_irfs(
            symmetric_results,
            symmetric_fx_results,
            ci_level=ci_level,
            label=f"{response_settings['label']} under maintained 1% CHF appreciation",
        )
        fx_results = layered_split_response_from_irfs(fx_results, fx_results, ci_level)
        symmetric_fx_results = maintained_response_from_irfs(
            symmetric_fx_results,
            symmetric_fx_results,
            ci_level=ci_level,
            label=f"{shock_settings['level_label']} under maintained 1% CHF appreciation",
        )

    if results.empty:
        st.info("No usable large-appreciation LP estimates are available for the selected setup.")
        return

    metric_cols = st.columns(4)
    valid_nobs = results["nobs"].replace(0, np.nan) if "nobs" in results else pd.Series(dtype=float)
    metric_cols[0].metric("Usable rows", f"{len(lp_sample):,}")
    metric_cols[1].metric("Positive appreciation periods", f"{positive_count:,}")
    metric_cols[2].metric("Large appreciation periods", f"{large_count:,}")
    metric_cols[3].metric("Threshold", "n/a" if np.isnan(threshold) else f"{threshold:.2f} pp", delta=fx_path_label)

    st.caption(
        f"Large appreciations are positive CHF moves at or above p{int(appreciation_percentile)} among positive appreciation periods "
        f"in the selected sample. Controls: {', '.join(selected_controls) if selected_controls else 'none'}."
    )
    if fx_path_label == "Standard":
        st.caption("FX path: one initial split FX move followed by the estimated FX path for that split component.")
    elif fx_path_label == "Layered 1% move":
        st.caption(
            "FX path: split one-shock IRFs are layered so each split component's CHF appreciation path is maintained at 1%. "
            "Confidence bands are approximate and ignore uncertainty in the estimated FX-path weights."
        )
    else:
        st.caption("FX path: future split FX moves enter the regression, isolating the initial split move conditional on later FX changes.")

    large_appreciation_definition_chart(
        lp_sample,
        shock_settings["level_column"],
        shock_settings["column"],
        shock_settings["label"],
        threshold,
        "Quarterly" if horizon_unit == "quarters" else "Monthly",
    )

    regimes = ["Large appreciations", "All other FX moves"]
    st.subheader("Response")
    split_lp_chart(results, symmetric_results, response_settings["unit"], regimes, horizon_unit=horizon_unit)
    display_columns = [
        column
        for column in ["regime", "horizon", "beta", "std_error", "ci_lower", "ci_upper", "nobs", "r_squared", "num_forward_shocks"]
        if column in results.columns
    ]
    st.dataframe(results.loc[:, display_columns], use_container_width=True, hide_index=True)

    if not fx_results.empty:
        st.subheader("FX Reaction")
        split_lp_chart(fx_results, symmetric_fx_results, "percent log points", regimes, horizon_unit=horizon_unit, reference_value=1.0)
        with st.expander("FX reaction estimates", expanded=False):
            fx_columns = [
                column
                for column in [
                    "regime",
                    "horizon",
                    "beta",
                    "std_error",
                    "ci_lower",
                    "ci_upper",
                    "nobs",
                    "r_squared",
                    "maintenance_shock",
                    "maintained_exchange_rate_path",
                    "num_forward_shocks",
                ]
                if column in fx_results.columns
            ]
            st.dataframe(fx_results.loc[:, fx_columns], use_container_width=True, hide_index=True)
    else:
        st.info("No usable FX reaction estimates are available for the selected setup.")


def render_monthly_labor_lp_page(monthly_data: pd.DataFrame) -> None:
    st.title("Unemployment LP")
    st.caption("Monthly labour-market responses to CHF appreciation shocks. Labour-market data are from the SNB data portal cube amarbma, sourced from SECO.")
    if monthly_data.empty:
        st.warning("No monthly labour-market data file found yet. Run `python scripts/fetch_data.py` from this project.")
        return
    standard_tab, asymmetric_tab, large_appreciation_tab = st.tabs(["Standard IRFs", "Asymmetric IRFs", "Large Appreciations"])
    with standard_tab:
        render_monthly_labor_standard_section(monthly_data)
    with asymmetric_tab:
        render_monthly_labor_asymmetric_section(monthly_data)
    with large_appreciation_tab:
        render_large_appreciation_lp_section(
            data=monthly_data,
            response_options=MONTHLY_LABOR_RESPONSE_OPTIONS,
            shock_options=MONTHLY_SHOCK_OPTIONS,
            title="Large-Appreciation Monthly IRFs",
            key_prefix="labor",
            horizon_unit="months",
            horizon_min=12,
            horizon_max=60,
            horizon_default=36,
            horizon_step=6,
            lag_max=24,
            lag_default=6,
            controls_enabled=False,
        )


def render_monthly_labor_standard_section(monthly_data: pd.DataFrame) -> None:
    start_date, end_date = date_bounds(monthly_data)
    min_date = start_date.to_pydatetime()
    max_date = end_date.to_pydatetime()
    col1, col2, col3 = st.columns(3)
    with col1:
        response_name = st.selectbox("Response", list(MONTHLY_LABOR_RESPONSE_OPTIONS), index=2, key="labor_response")
        shock_name = st.selectbox("Shock", list(MONTHLY_SHOCK_OPTIONS), index=0, key="labor_shock")
    with col2:
        ci_level = st.selectbox(
            "Confidence band",
            [0.68, 0.90, 0.95],
            index=1,
            format_func=lambda x: f"{int(x * 100)}%",
            key="labor_ci",
        )
        date_range = st.slider(
            "Sample window",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
            key="labor_date_range",
        )
    with col3:
        fx_path_label = st.selectbox(
            "FX path",
            ["Standard", "Layered 1% move", "Forward shocks"],
            index=0,
            key="labor_fx_path",
        )
        horizons = st.slider("Horizon", min_value=12, max_value=60, value=36, step=6, key="labor_horizon")
        lags = st.slider("Lags", min_value=1, max_value=24, value=6, step=1, key="labor_lags")

    response_settings = MONTHLY_LABOR_RESPONSE_OPTIONS[response_name]
    shock_settings = MONTHLY_SHOCK_OPTIONS[shock_name]
    include_forward_shocks = fx_path_label == "Forward shocks"
    use_layered_shocks = fx_path_label == "Layered 1% move"
    selected_start, selected_end = date_range
    sample = monthly_data.loc[
        (monthly_data["date"] >= pd.Timestamp(selected_start)) & (monthly_data["date"] <= pd.Timestamp(selected_end))
    ].copy()

    required_columns = [response_settings["column"], shock_settings["column"]]
    lp_sample = sample.loc[:, ["date", *required_columns]].dropna().copy()
    if len(lp_sample) <= (2 * lags + horizons + 5):
        st.warning("The selected monthly model has a thin usable sample. Consider fewer lags or a wider date range.")
    if lp_sample.empty:
        st.info("No usable observations for the selected monthly LP setup.")
        return

    results = run_headline_lp(
        data=lp_sample,
        response_column=response_settings["column"],
        response_label=response_settings["label"],
        shock_column=shock_settings["column"],
        shock_label=shock_settings["label"],
        controls=[],
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        cumulative_response=bool(response_settings["cumulative"]),
        include_forward_shocks=include_forward_shocks,
    )
    fx_results = pd.DataFrame()
    shock_level_column = shock_settings["level_column"]
    if shock_level_column in sample.columns:
        fx_sample = sample.loc[:, ["date", shock_level_column, shock_settings["column"]]].dropna().copy()
        if not fx_sample.empty:
            fx_results = run_headline_lp(
                data=fx_sample,
                response_column=shock_level_column,
                response_label=shock_settings["level_label"],
                shock_column=shock_settings["column"],
                shock_label=shock_settings["label"],
                controls=[],
                horizons=horizons,
                lags=lags,
                ci_level=ci_level,
                cumulative_response=True,
                include_forward_shocks=include_forward_shocks,
            )
    if use_layered_shocks:
        standard_results = run_headline_lp(
            data=lp_sample,
            response_column=response_settings["column"],
            response_label=response_settings["label"],
            shock_column=shock_settings["column"],
            shock_label=shock_settings["label"],
            controls=[],
            horizons=horizons,
            lags=lags,
            ci_level=ci_level,
            cumulative_response=bool(response_settings["cumulative"]),
            include_forward_shocks=False,
        )
        standard_fx_results = pd.DataFrame()
        if shock_level_column in sample.columns:
            fx_sample = sample.loc[:, ["date", shock_level_column, shock_settings["column"]]].dropna().copy()
            if not fx_sample.empty:
                standard_fx_results = run_headline_lp(
                    data=fx_sample,
                    response_column=shock_level_column,
                    response_label=shock_settings["level_label"],
                    shock_column=shock_settings["column"],
                    shock_label=shock_settings["label"],
                    controls=[],
                    horizons=horizons,
                    lags=lags,
                    ci_level=ci_level,
                    cumulative_response=True,
                    include_forward_shocks=False,
                )
        if not standard_fx_results.empty:
            results = maintained_response_from_irfs(
                standard_results,
                standard_fx_results,
                ci_level=ci_level,
                label=f"{response_settings['label']} under maintained 1% CHF appreciation",
            )
            fx_results = maintained_response_from_irfs(
                standard_fx_results,
                standard_fx_results,
                ci_level=ci_level,
                label=f"{shock_settings['level_label']} under maintained 1% CHF appreciation",
            )
    if results.empty:
        st.info("No usable monthly LP estimates are available for the selected setup.")
        return

    metric_cols = st.columns(4)
    valid_nobs = results["nobs"].replace(0, np.nan)
    metric_cols[0].metric("Usable rows", f"{len(lp_sample):,}")
    metric_cols[1].metric("First month", pd.Timestamp(lp_sample["date"].min()).strftime("%Y-%m"))
    metric_cols[2].metric("Last month", pd.Timestamp(lp_sample["date"].max()).strftime("%Y-%m"))
    metric_cols[3].metric("Minimum nobs", f"{int(valid_nobs.min()) if valid_nobs.notna().any() else 0:,}")

    st.caption(
        f"Response to a {shock_settings['unit']}. Count targets use cumulative log changes; "
        "rate targets use cumulative percentage-point changes from the pre-shock month."
    )
    if fx_path_label == "Standard":
        st.caption("FX path: one initial monthly CHF move followed by the estimated CHF path.")
    elif fx_path_label == "Layered 1% move":
        st.caption(
            "FX path: one-shock monthly IRFs are layered so the CHF appreciation path is maintained at 1%. "
            "Confidence bands are approximate and ignore uncertainty in the estimated FX-path weights."
        )
    else:
        st.caption("FX path: future monthly CHF moves enter the regression, isolating the initial move conditional on later CHF changes.")
    if response_settings["column"] == "job_seeker_rate":
        st.caption("Job seeker rate is calculated as registered job seekers divided by labour force, multiplied by 100.")
    lp_chart(results, response_settings["unit"], horizon_unit="months")
    st.dataframe(
        results.loc[:, ["horizon", "beta", "std_error", "ci_lower", "ci_upper", "nobs", "r_squared"]],
        use_container_width=True,
        hide_index=True,
    )

    if not fx_results.empty:
        st.subheader("FX Reaction")
        st.caption(
            f"Estimated path of {shock_settings['level_label']} after the selected "
            f"{shock_settings['unit']}."
        )
        lp_chart(fx_results, "percent log points", horizon_unit="months", reference_value=1.0)
        with st.expander("FX reaction estimates", expanded=False):
            display_columns = [
                column
                for column in [
                    "horizon",
                    "beta",
                    "std_error",
                    "ci_lower",
                    "ci_upper",
                    "nobs",
                    "r_squared",
                    "maintenance_shock",
                    "maintained_exchange_rate_path",
                ]
                if column in fx_results.columns
            ]
            st.dataframe(fx_results.loc[:, display_columns], use_container_width=True, hide_index=True)
    else:
        st.info("No usable FX reaction estimates are available for the selected monthly setup.")

    st.subheader("Monthly Inputs")
    input_cols = [
        column
        for column in [
            "registered_unemployed_sa",
            "registered_job_seekers",
            "unemployment_rate_sa",
            "job_seeker_rate",
            "labour_force",
            shock_settings["column"],
        ]
        if column in sample.columns
    ]
    if input_cols:
        line_chart(sample, input_cols, "Selected monthly labour-market and CHF inputs", "Level / percent / percentage points")
        st.dataframe(sample.loc[:, ["date", *input_cols]].tail(24), use_container_width=True, hide_index=True)


def render_monthly_labor_asymmetric_section(monthly_data: pd.DataFrame) -> None:
    st.subheader("Asymmetric Monthly IRFs")
    st.caption("Sign-split monthly labour-market responses to CHF appreciations and depreciations.")

    start_date, end_date = date_bounds(monthly_data)
    min_date = start_date.to_pydatetime()
    max_date = end_date.to_pydatetime()

    col1, col2, col3 = st.columns(3)
    with col1:
        response_name = st.selectbox(
            "Response",
            list(MONTHLY_LABOR_RESPONSE_OPTIONS),
            index=2,
            key="labor_asym_response",
        )
        shock_name = st.selectbox("Shock", list(MONTHLY_SHOCK_OPTIONS), index=0, key="labor_asym_shock")
    with col2:
        ci_level = st.selectbox(
            "Confidence band",
            [0.68, 0.90, 0.95],
            index=1,
            format_func=lambda x: f"{int(x * 100)}%",
            key="labor_asym_ci",
        )
        date_range = st.slider(
            "Sample window",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
            key="labor_asym_date_range",
        )
    with col3:
        dynamic_mode = st.selectbox(
            "Dynamics",
            ["Standard", "Layered 1% move", "Forward shocks"],
            index=0,
            key="labor_asym_dynamics",
        )
        horizons = st.slider("Horizon", min_value=12, max_value=60, value=36, step=6, key="labor_asym_horizon")
        lags = st.slider("Lags", min_value=1, max_value=24, value=6, step=1, key="labor_asym_lags")

    response_settings = MONTHLY_LABOR_RESPONSE_OPTIONS[response_name]
    shock_settings = MONTHLY_SHOCK_OPTIONS[shock_name]
    include_forward_shocks = dynamic_mode == "Forward shocks"
    use_layered_shocks = dynamic_mode == "Layered 1% move"
    selected_start, selected_end = date_range
    sample = monthly_data.loc[
        (monthly_data["date"] >= pd.Timestamp(selected_start)) & (monthly_data["date"] <= pd.Timestamp(selected_end))
    ].copy()

    required_columns = [response_settings["column"], shock_settings["column"]]
    lp_sample = sample.loc[:, ["date", *required_columns]].dropna().copy()
    if len(lp_sample) <= (3 + 2 * lags + horizons + 5):
        st.warning("The selected asymmetric monthly model has a thin usable sample. Consider fewer lags or a wider date range.")
    if lp_sample.empty:
        st.info("No usable observations for the selected asymmetric monthly LP setup.")
        return

    results = run_asymmetric_lp(
        data=lp_sample,
        response_column=response_settings["column"],
        response_label=response_settings["label"],
        shock_column=shock_settings["column"],
        shock_label=shock_settings["label"],
        controls=[],
        horizons=horizons,
        lags=lags,
        ci_level=ci_level,
        cumulative_response=bool(response_settings["cumulative"]),
        include_forward_shocks=include_forward_shocks,
    )
    fx_results = pd.DataFrame()
    shock_level_column = shock_settings["level_column"]
    fx_sample = (
        sample.loc[:, ["date", shock_level_column, shock_settings["column"]]].dropna().copy()
        if shock_level_column in sample.columns
        else pd.DataFrame()
    )
    if not fx_sample.empty:
        fx_results = run_asymmetric_lp(
            data=fx_sample,
            response_column=shock_level_column,
            response_label=shock_settings["level_label"],
            shock_column=shock_settings["column"],
            shock_label=shock_settings["label"],
            controls=[],
            horizons=horizons,
            lags=lags,
            ci_level=ci_level,
            cumulative_response=True,
            include_forward_shocks=include_forward_shocks,
        )
    if use_layered_shocks:
        standard_fx_results = pd.DataFrame()
        if not fx_sample.empty:
            standard_fx_results = run_asymmetric_lp(
                data=fx_sample,
                response_column=shock_level_column,
                response_label=shock_settings["level_label"],
                shock_column=shock_settings["column"],
                shock_label=shock_settings["label"],
                controls=[],
                horizons=horizons,
                lags=lags,
                ci_level=ci_level,
                cumulative_response=True,
                include_forward_shocks=False,
            )
            fx_results = layered_asymmetric_response_from_irfs(
                standard_fx_results,
                standard_fx_results,
                ci_level=ci_level,
                label=f"{shock_settings['level_label']} under maintained 1% CHF appreciation/depreciation",
            )
        if not standard_fx_results.empty:
            results = layered_asymmetric_response_from_irfs(
                results,
                standard_fx_results,
                ci_level=ci_level,
                label=f"{response_settings['label']} under maintained 1% CHF appreciation/depreciation",
            )
    if results.empty:
        st.info("No usable asymmetric monthly LP estimates are available for the selected setup.")
        return

    shown_regimes = st.multiselect(
        "IRFs",
        options=["Appreciation (+1 pp)", "Depreciation (-1 pp)", "Appreciation minus depreciation"],
        default=["Appreciation (+1 pp)", "Depreciation (-1 pp)"],
        key="labor_asym_regimes",
    )

    metric_cols = st.columns(4)
    valid_nobs = results["nobs"].replace(0, np.nan)
    metric_cols[0].metric("Usable rows", f"{len(lp_sample):,}")
    metric_cols[1].metric("First month", pd.Timestamp(lp_sample["date"].min()).strftime("%Y-%m"))
    metric_cols[2].metric("Last month", pd.Timestamp(lp_sample["date"].max()).strftime("%Y-%m"))
    metric_cols[3].metric("Minimum nobs", f"{int(valid_nobs.min()) if valid_nobs.notna().any() else 0:,}")

    st.caption(
        f"The monthly shock is split into positive CHF appreciations and negative CHF depreciations in one regression. "
        f"The depreciation line is reported as the response to a 1 percentage-point CHF depreciation; "
        f"the raw negative-shock coefficient is multiplied by -1."
    )
    if dynamic_mode == "Forward shocks":
        st.caption("Dynamics: future monthly CHF moves enter the regression, so the sign-split effect is conditional on later CHF changes.")
    elif dynamic_mode == "Layered 1% move":
        st.caption(
            "Dynamics: sign-specific one-shock monthly IRFs are layered so the appreciation and depreciation paths are maintained at 1%. "
            "Confidence bands are approximate and ignore uncertainty in the estimated FX-path weights."
        )
    else:
        st.caption("Dynamics: standard sign-split monthly local projection with one initial CHF move.")

    if shown_regimes:
        if not fx_results.empty:
            st.subheader("FX Reaction")
            st.caption(
                f"Sign-specific estimated path of {shock_settings['level_label']} after the selected "
                f"{shock_settings['unit']}."
            )
            asymmetric_lp_chart(
                fx_results,
                "percent log points",
                shown_regimes,
                horizon_unit="months",
                reference_value=1.0,
            )
            with st.expander("FX reaction estimates", expanded=False):
                fx_display = fx_results.loc[fx_results["regime"].isin(shown_regimes), :]
                fx_display_columns = [
                    column
                    for column in [
                        "regime",
                        "horizon",
                        "beta",
                        "raw_coefficient",
                        "display_multiplier",
                        "std_error",
                        "ci_lower",
                        "ci_upper",
                        "nobs",
                        "r_squared",
                        "maintenance_shock",
                        "maintained_exchange_rate_path",
                    ]
                    if column in fx_display.columns
                ]
                st.dataframe(fx_display.loc[:, fx_display_columns], use_container_width=True, hide_index=True)
        else:
            st.info("No usable FX reaction estimates are available for the selected asymmetric monthly setup.")

        st.subheader("Labour-Market Response")
        asymmetric_lp_chart(results, response_settings["unit"], shown_regimes, horizon_unit="months")
        display = results.loc[results["regime"].isin(shown_regimes), :]
        display_columns = [
            column
            for column in [
                "regime",
                "horizon",
                "beta",
                "raw_coefficient",
                "display_multiplier",
                "std_error",
                "ci_lower",
                "ci_upper",
                "nobs",
                "r_squared",
            ]
            if column in display.columns
        ]
        st.dataframe(display.loc[:, display_columns], use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one asymmetric monthly IRF.")


def render_output_gap_tab(sample: pd.DataFrame) -> None:
    st.subheader("SNB Output Gap")
    output_gap_cols = [
        column
        for column in ["snb_output_gap", "snb_output_gap_hp", "snb_output_gap_multivariate"]
        if column in sample.columns
    ]
    if output_gap_cols:
        labels = {
            "snb_output_gap": "Production function",
            "snb_output_gap_hp": "Hodrick-Prescott filter",
            "snb_output_gap_multivariate": "Multivariate filter",
        }
        selected = st.multiselect(
            "Series",
            options=output_gap_cols,
            default=output_gap_cols,
            format_func=lambda column: labels.get(column, column),
            key="output_gap_series",
        )
        if selected:
            line_chart(sample, selected, "SNB output gap estimates", "Percent of potential output")
            latest = sample.loc[sample[selected].notna().any(axis=1), ["date", *selected]].tail(12)
            st.dataframe(latest, use_container_width=True, hide_index=True)
        else:
            st.info("Select at least one output-gap estimate.")
    else:
        st.info("SNB output-gap columns are not available in the current data file.")
    st.caption(
        "Source: SNB data portal chart snbprodluch. The baseline column uses the production-function estimate."
    )


def render_controls_tab(sample: pd.DataFrame) -> None:
    st.subheader("Exchange Rates and Macro Controls")
    fx_cols = [
        column
        for column in [
            "chf_neer_appreciation_qoq",
            "snb_chf_neer_appreciation_qoq",
            "eur_chf_appreciation_qoq",
            "usd_chf_appreciation_qoq",
        ]
        if column in sample.columns
    ]
    if fx_cols:
        line_chart(sample, fx_cols, "Quarterly CHF appreciation measures", "Percent log points")
    macro_cols = [
        column
        for column in [
            "ea_real_gdp_growth_qoq",
            "brent_oil_change_qoq",
            "snb_output_gap",
            "snb_output_gap_hp",
            "snb_output_gap_multivariate",
        ]
        if column in sample.columns
    ]
    if macro_cols:
        line_chart(sample, macro_cols, "Baseline control variables", "Percent / percentage points")
    st.dataframe(sample.loc[:, ["date", *fx_cols, *macro_cols]].tail(20), use_container_width=True)


def render_audit_tab(data: pd.DataFrame, metadata: pd.DataFrame, sources: pd.DataFrame) -> None:
    st.subheader("Sources")
    if sources.empty:
        st.info("Source metadata file is not available yet.")
    else:
        st.dataframe(sources, use_container_width=True, hide_index=True)

    st.subheader("SECO Component Coverage")
    if metadata.empty:
        st.info("Component metadata file is not available yet.")
    else:
        st.dataframe(coverage_table(data, metadata), use_container_width=True, hide_index=True)

    st.subheader("Missing Values")
    missing = (
        data.drop(columns=["date"])
        .isna()
        .mean()
        .mul(100)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_pct", ascending=False)
    )
    st.dataframe(missing, use_container_width=True, hide_index=True)


def render_data_audit_page(data: pd.DataFrame, sample: pd.DataFrame, metadata: pd.DataFrame, sources: pd.DataFrame) -> None:
    st.title("Real Indicator Data Audit")
    st.caption("Quarterly real GDP components, CHF appreciation measures, output gap estimates, and macro controls.")
    headline, expenditure, value_added, output_gap, controls, audit = st.tabs(
        ["Headline GDP", "Expenditure", "Value Added", "Output Gap", "Controls", "Audit"]
    )
    with headline:
        render_headline_data_tab(sample)
    with expenditure:
        component_panel(sample, metadata, "expenditure", "Expenditure-Side Components")
    with value_added:
        component_panel(sample, metadata, "value_added", "Value-Added Components")
    with output_gap:
        render_output_gap_tab(sample)
    with controls:
        render_controls_tab(sample)
    with audit:
        render_audit_tab(data, metadata, sources)


def render_lp_page(sample: pd.DataFrame) -> None:
    st.title("GDP LP")
    st.caption("First headline real GDP LP layer. Component LPs can be added once the headline mechanics look right.")
    headline_lp, asymmetric_irfs, large_appreciations, diagnostics = st.tabs(
        ["Headline LP", "Asymmetric IRFs", "Large Appreciations", "LP Sample"]
    )
    with headline_lp:
        render_headline_lp_tab(sample)
    with asymmetric_irfs:
        render_asymmetric_irf_tab(sample)
    with large_appreciations:
        render_large_appreciation_lp_section(
            data=sample,
            response_options=LP_RESPONSE_OPTIONS,
            shock_options=SHOCK_OPTIONS,
            title="Large-Appreciation GDP IRFs",
            key_prefix="gdp",
            horizon_unit="quarters",
            horizon_min=4,
            horizon_max=16,
            horizon_default=12,
            horizon_step=1,
            lag_max=8,
            lag_default=4,
            controls_enabled=True,
        )
    with diagnostics:
        st.subheader("Available LP Inputs")
        lp_columns = [
            "gdp_log_level",
            "gdp_growth_qoq",
            "chf_neer_appreciation_qoq",
            "eur_chf_appreciation_qoq",
            "ea_real_gdp_growth_qoq",
            "brent_oil_change_qoq",
            "snb_output_gap",
        ]
        available = [column for column in lp_columns if column in sample.columns]
        if available:
            line_chart(sample, available, "LP inputs in the selected sample", "Percent / percentage points")
            st.dataframe(sample.loc[:, ["date", *available]].tail(20), use_container_width=True, hide_index=True)
        else:
            st.info("No LP input columns are available in the current sample.")


def main() -> None:
    data = safe_load_data(DATA_PATH)
    monthly_labor = safe_load_data(MONTHLY_LABOR_PATH)
    metadata = safe_load_metadata(COMPONENT_METADATA_PATH)
    sources = safe_load_metadata(SOURCE_METADATA_PATH)

    st.sidebar.title("GDP LP Dashboard")
    page = st.sidebar.radio("Page", ["GDP LP", "Unemployment LP", "Real Indicator Data Audit"], index=0)

    if page == "Unemployment LP":
        render_monthly_labor_lp_page(monthly_labor)
        return

    if data.empty:
        st.warning("No merged data file found yet. Run `python scripts/fetch_data.py` from this project.")
        st.stop()

    start_date, end_date = date_bounds(data)
    min_date = start_date.to_pydatetime()
    max_date = end_date.to_pydatetime()
    with st.sidebar:
        st.header("Sample")
        selected_range = st.slider(
            "Sample window",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
        )
        st.metric("Rows", f"{len(data):,}")
        st.metric("Latest quarter", quarter_label(end_date))

    selected_start, selected_end = selected_range

    sample = data.loc[
        (data["date"] >= pd.Timestamp(selected_start)) & (data["date"] <= pd.Timestamp(selected_end))
    ].copy()

    if page == "GDP LP":
        render_lp_page(sample)
    else:
        render_data_audit_page(data, sample, metadata, sources)


if __name__ == "__main__":
    main()
