from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpi_lp_chf.local_projections import (
    LPConfig,
    estimate_asymmetric_dashboard_lp,
    estimate_dashboard_lp,
    estimate_split_shock_dashboard_lp,
)

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "swiss_macro_real.csv"
REAL_INDICATOR_PROJECT_ROOT = PROJECT_ROOT / "real_indicators"
REAL_INDICATOR_APP_PATH = REAL_INDICATOR_PROJECT_ROOT / "app.py"

SOURCE_ROWS = [
    {
        "Variable": "Swiss headline CPI",
        "Columns": "cpi, cpi_sa",
        "Source": "Swiss Federal Statistical Office LIK application data",
        "URL": "https://www.bfs.admin.ch/bfs/en/home/statistics/prices/consumer-price-index.html",
        "Notes": "Seasonally adjusted with STL in the project pipeline.",
    },
    {
        "Variable": "CHF NEER",
        "Columns": "chf_neer",
        "Source": "BIS effective exchange rates",
        "URL": "https://data.bis.org/topics/EER",
        "Notes": "Nominal effective exchange rate. Higher values mean CHF appreciation.",
    },
    {
        "Variable": "SNB CHF NEER",
        "Columns": "snb_chf_neer",
        "Source": "SNB data portal, devwkieffim, code {N,G,I}",
        "URL": "https://data.snb.ch/en/topics/ziredev/cube/devwkieffim",
        "Notes": "Nominal overall effective exchange rate index, December 2000 = 100, monthly average. Higher values mean CHF appreciation.",
    },
    {
        "Variable": "EURCHF",
        "Columns": "eur_chf",
        "Source": "ECB Data Portal, EXR.D.CHF.EUR.SP00.A",
        "URL": "https://data.ecb.europa.eu/",
        "Notes": "CHF per EUR, monthly average of daily observations. Dashboard shock is transformed so higher means CHF appreciation.",
    },
    {
        "Variable": "USDCHF",
        "Columns": "usd_chf",
        "Source": "FRED daily series DEXSZUS",
        "URL": "https://fred.stlouisfed.org/series/DEXSZUS",
        "Notes": "CHF per USD, monthly average of daily observations. Dashboard shock is transformed so higher means CHF appreciation.",
    },
    {
        "Variable": "Swiss core CPI 1",
        "Columns": "core_cpi_1, core_cpi_1_sa",
        "Source": "SNB data portal, plkoprex, code K1",
        "URL": "https://data.snb.ch/",
        "Notes": "Seasonally adjusted with STL in the project pipeline.",
    },
    {
        "Variable": "Swiss core CPI 2",
        "Columns": "core_cpi_2, core_cpi_2_sa",
        "Source": "SNB data portal, plkoprex, code K2",
        "URL": "https://data.snb.ch/",
        "Notes": "Seasonally adjusted with STL in the project pipeline.",
    },
    {
        "Variable": "Swiss fresh and seasonal products CPI",
        "Columns": "fresh_seasonal, fresh_seasonal_sa",
        "Source": "SNB data portal, plkoprex, code FP",
        "URL": "https://data.snb.ch/en/topics/uvo/cube/plkoprex?dimSel=D0(K1,FP,ET)",
        "Notes": "Seasonally adjusted with STL in the project pipeline.",
    },
    {
        "Variable": "Swiss energy and fuels CPI",
        "Columns": "energy_fuel, energy_fuel_sa",
        "Source": "SNB data portal, plkoprex, code ET",
        "URL": "https://data.snb.ch/en/topics/uvo/cube/plkoprex?dimSel=D0(K1,FP,ET)",
        "Notes": "Seasonally adjusted with STL in the project pipeline.",
    },
    {
        "Variable": "Euro area core HICP",
        "Columns": "ea_core_hicp, ea_core_inflation",
        "Source": "ECB Data Portal, HICP.M.U2.Y.XEF000.4F0.INX",
        "URL": "https://data.ecb.europa.eu/",
        "Notes": "Inflation is the 12-month log change times 100.",
    },
    {
        "Variable": "Brent oil",
        "Columns": "brent_oil, brent_oil_inflation",
        "Source": "FRED daily Brent series DCOILBRENTEU",
        "URL": "https://fred.stlouisfed.org/series/DCOILBRENTEU",
        "Notes": "Daily prices are averaged to monthly frequency.",
    },
    {
        "Variable": "Swiss unemployment",
        "Columns": "ch_unemployment_rate",
        "Source": "SNB data portal, amarbma, code S1",
        "URL": "https://data.snb.ch/",
        "Notes": "Domestic slack control.",
    },
    {
        "Variable": "Swiss CPI major groups",
        "Columns": "major_* and major_*_sa",
        "Source": "SNB data portal, plkoprgru",
        "URL": "https://data.snb.ch/en/topics/uvo/cube/plkoprgru",
        "Notes": "14 major grouping indexes. Seasonally adjusted with STL in the project pipeline.",
    },
    {
        "Variable": "Swiss CPI major-group weights",
        "Columns": "weight_pct metadata",
        "Source": "Swiss Federal Statistical Office, LIK basket and weights 2026",
        "URL": "https://www.bfs.admin.ch/bfs/de/home/statistiken/preise/erhebungen/lik/warenkorb.html",
        "Notes": "Annual basket shares in percent. Used as metadata for the major-group dashboard panels.",
    },
    {
        "Variable": "Swiss CPI focus weights",
        "Columns": "weight_pct metadata",
        "Source": "SNB data portal, plkoprex, 2026 index aggregation identity",
        "URL": "https://data.snb.ch/en/topics/uvo/cube/plkoprex?dimSel=D0(K1,FP,ET)",
        "Notes": "Recovered from the 2026 headline identity: headline = Core 1 + fresh/seasonal + energy/fuels weighted aggregate.",
    },
]

MAJOR_GROUPS = [
    {"code": "NG", "column": "major_ng", "label": "Food and non-alcoholic beverages", "weight_pct": 10.307},
    {"code": "AGT", "column": "major_agt", "label": "Alcoholic beverages and tobacco", "weight_pct": 3.468},
    {"code": "BS", "column": "major_bs", "label": "Clothing and footwear", "weight_pct": 2.420},
    {"code": "WE", "column": "major_we", "label": "Housing and energy", "weight_pct": 25.595},
    {"code": "HH", "column": "major_hh", "label": "Household goods and services", "weight_pct": 3.302},
    {"code": "G", "column": "major_g", "label": "Health", "weight_pct": 17.379},
    {"code": "V", "column": "major_v", "label": "Transport", "weight_pct": 10.715},
    {"code": "N", "column": "major_n", "label": "Information and communication", "weight_pct": 3.279},
    {"code": "FK", "column": "major_fk", "label": "Recreation, sport and culture", "weight_pct": 7.501},
    {"code": "EU", "column": "major_eu", "label": "Education", "weight_pct": 0.851},
    {"code": "RH", "column": "major_rh", "label": "Restaurants and hotels", "weight_pct": 9.568},
    {"code": "VF", "column": "major_vf", "label": "Insurance and financial services", "weight_pct": 2.252},
    {"code": "SWD", "column": "major_swd", "label": "Other goods and services", "weight_pct": 3.363},
    {"code": "T", "column": "major_t", "label": "Total CPI", "weight_pct": 100.000},
]

FOCUS_GROUPS = [
    {"code": "HEADLINE", "column": "cpi", "label": "Headline CPI", "weight_pct": 100.000},
    {"code": "K1", "column": "core_cpi_1", "label": "Core CPI 1", "weight_pct": 84.205},
    {"code": "FP", "column": "fresh_seasonal", "label": "Fresh and seasonal products", "weight_pct": 10.717},
    {"code": "ET", "column": "energy_fuel", "label": "Energy and fuels", "weight_pct": 5.078},
]

CORE_COLUMNS = [
    "cpi",
    "cpi_sa",
    "chf_neer",
    "snb_chf_neer",
    "eur_chf",
    "usd_chf",
    "core_cpi_1",
    "core_cpi_1_sa",
    "core_cpi_2",
    "core_cpi_2_sa",
    "fresh_seasonal",
    "fresh_seasonal_sa",
    "energy_fuel",
    "energy_fuel_sa",
    "ea_core_hicp",
    "ea_core_inflation",
    "brent_oil",
    "brent_oil_inflation",
    "ch_unemployment_rate",
    *[group["column"] for group in MAJOR_GROUPS],
    *[f"{group['column']}_sa" for group in MAJOR_GROUPS],
]

PRICE_SERIES = {
    "Headline CPI": ("cpi", "cpi_sa"),
    "Core CPI 1": ("core_cpi_1", "core_cpi_1_sa"),
    "Core CPI 2": ("core_cpi_2", "core_cpi_2_sa"),
    "Fresh and seasonal products": ("fresh_seasonal", "fresh_seasonal_sa"),
    "Energy and fuels CPI": ("energy_fuel", "energy_fuel_sa"),
}

CONTROL_CHARTS = {
    "BIS vs SNB CHF NEER level": (["chf_neer", "snb_chf_neer"], "BIS vs SNB CHF NEER", "Index, Dec 2019 = 100"),
    "BIS vs SNB CHF NEER monthly change": (["chf_neer_change", "snb_chf_neer_change"], "BIS vs SNB CHF NEER Monthly Log Change", "Percent log points"),
    "EURCHF level": (["eur_chf"], "EURCHF", "CHF per EUR"),
    "EURCHF monthly change": (["eur_chf_appreciation_change"], "EURCHF Monthly Move", "CHF appreciation, percent log points"),
    "USDCHF level": (["usd_chf"], "USDCHF", "CHF per USD"),
    "USDCHF monthly change": (["usd_chf_appreciation_change"], "USDCHF Monthly Move", "CHF appreciation, percent log points"),
    "Euro area core HICP": (["ea_core_hicp"], "Euro Area Core HICP", "Index"),
    "Euro area core inflation": (["ea_core_inflation"], "Euro Area Core Inflation", "12-month log change, percent"),
    "Brent oil price": (["brent_oil"], "Brent Oil Price", "USD per barrel"),
    "Brent oil inflation": (["brent_oil_inflation"], "Brent Oil Monthly Inflation", "Monthly log change, percent"),
    "Swiss unemployment": (["ch_unemployment_rate"], "Swiss Unemployment Rate", "Percent"),
}

EXCHANGE_RATE_SHOCKS = {
    "BIS CHF NEER": {
        "level": "chf_neer",
        "log_level": "log_chf_neer_pct",
        "change": "chf_neer_change",
        "display": "BIS CHF NEER",
        "level_label": "BIS CHF NEER level",
        "held_label": "BIS CHF NEER held at +1",
        "path_label": "Estimated BIS CHF NEER path",
        "positive_text": "CHF appreciation",
    },
    "SNB CHF NEER": {
        "level": "snb_chf_neer",
        "log_level": "log_snb_chf_neer_pct",
        "change": "snb_chf_neer_change",
        "display": "SNB CHF NEER",
        "level_label": "SNB CHF NEER level",
        "held_label": "SNB CHF NEER held at +1",
        "path_label": "Estimated SNB CHF NEER path",
        "positive_text": "CHF appreciation",
    },
    "EURCHF": {
        "level": "eur_chf",
        "log_level": "log_eur_chf_appreciation_pct",
        "change": "eur_chf_appreciation_change",
        "display": "EURCHF",
        "level_label": "EURCHF, CHF-appreciation-positive level",
        "held_label": "EURCHF shock held at +1",
        "path_label": "Estimated EURCHF-implied CHF path",
        "positive_text": "CHF appreciation against EUR",
    },
    "USDCHF": {
        "level": "usd_chf",
        "log_level": "log_usd_chf_appreciation_pct",
        "change": "usd_chf_appreciation_change",
        "display": "USDCHF",
        "level_label": "USDCHF, CHF-appreciation-positive level",
        "held_label": "USDCHF shock held at +1",
        "path_label": "Estimated USDCHF-implied CHF path",
        "positive_text": "CHF appreciation against USD",
    },
}

LP_RESPONSES = {
    "Headline CPI": {"nsa": "cpi", "sa": "cpi_sa"},
    "Core CPI 1": {"nsa": "core_cpi_1", "sa": "core_cpi_1_sa"},
    "Core CPI 2": {"nsa": "core_cpi_2", "sa": "core_cpi_2_sa"},
    "Fresh and seasonal products": {"nsa": "fresh_seasonal", "sa": "fresh_seasonal_sa"},
    "Energy and fuels CPI": {"nsa": "energy_fuel", "sa": "energy_fuel_sa"},
    "Goods CPI": {"nsa": "goods", "sa": "goods_sa"},
    "Services CPI": {"nsa": "services", "sa": "services_sa"},
    "Domestic CPI": {"nsa": "domestic", "sa": "domestic_sa"},
    "Imported CPI": {"nsa": "imported", "sa": "imported_sa"},
}

BASELINE_LP_RESPONSES = [
    "Headline CPI",
    "Core CPI 1",
    "Core CPI 2",
    "Fresh and seasonal products",
    "Energy and fuels CPI",
]

LP_TRANSFORMS = {
    "y/y inflation": {
        "suffix": "yoy",
        "chart_label": "y/y Inflation Response",
        "equation_label": r"\pi^{yy}",
        "cumulative": False,
    },
    "m/m inflation": {
        "suffix": "mom",
        "equation_label": r"\pi^{mm}",
        "cumulative": False,
    },
    "Cumulative price difference": {
        "suffix": "log_level",
        "equation_label": r"p",
        "cumulative": True,
    },
}

LP_CONTROL_SETS = {
    "None": [],
    "External": ["ea_core_inflation", "brent_oil_inflation"],
    "Full": ["ea_core_inflation", "brent_oil_inflation", "ch_unemployment_rate"],
}

EVENT_PRESETS = {
    "Pre-floor and floor introduction, censored": {
        "start": "2011-07-01",
        "end": "2012-06-01",
        "action": "Censor",
    },
    "Floor removal year, censored": {
        "start": "2015-01-01",
        "end": "2015-12-01",
        "action": "Censor",
    },
}


st.set_page_config(
    page_title="CPI Local Projections to CHF Moves",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(path: str, file_mtime: float | None = None) -> pd.DataFrame:
    _ = file_mtime
    data = pd.read_csv(path, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)
    return add_transforms(data)


@st.cache_resource(show_spinner=False)
def load_real_indicator_app():
    if not REAL_INDICATOR_APP_PATH.exists():
        return None
    original_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        spec = importlib.util.spec_from_file_location("real_indicator_dashboard_app", REAL_INDICATOR_APP_PATH)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["real_indicator_dashboard_app"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        st.set_page_config = original_set_page_config


def real_indicator_module_or_stop():
    module = load_real_indicator_app()
    if module is None:
        st.error(f"Real indicator dashboard app not found: {REAL_INDICATOR_APP_PATH}")
        st.stop()
    return module


def add_transforms(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    derived = {}
    derived["log_chf_neer_pct"] = 100 * np.log(out["chf_neer"])
    derived["chf_neer_change"] = 100 * (np.log(out["chf_neer"]) - np.log(out["chf_neer"]).shift(1))
    if "snb_chf_neer" in out:
        derived["log_snb_chf_neer_pct"] = 100 * np.log(out["snb_chf_neer"])
        derived["snb_chf_neer_change"] = 100 * (np.log(out["snb_chf_neer"]) - np.log(out["snb_chf_neer"]).shift(1))
    if "eur_chf" in out:
        derived["log_eur_chf_appreciation_pct"] = -100 * np.log(out["eur_chf"])
        derived["eur_chf_appreciation_change"] = derived["log_eur_chf_appreciation_pct"] - derived[
            "log_eur_chf_appreciation_pct"
        ].shift(1)
    if "usd_chf" in out:
        derived["log_usd_chf_appreciation_pct"] = -100 * np.log(out["usd_chf"])
        derived["usd_chf_appreciation_change"] = derived["log_usd_chf_appreciation_pct"] - derived[
            "log_usd_chf_appreciation_pct"
        ].shift(1)

    price_sources = {
        "headline_nsa": "cpi",
        "headline_sa": "cpi_sa",
        "core_1_nsa": "core_cpi_1",
        "core_1_sa": "core_cpi_1_sa",
        "core_2_nsa": "core_cpi_2",
        "core_2_sa": "core_cpi_2_sa",
        "fresh_seasonal_nsa": "fresh_seasonal",
        "fresh_seasonal_sa": "fresh_seasonal_sa",
        "energy_fuel_nsa": "energy_fuel",
        "energy_fuel_sa": "energy_fuel_sa",
        "goods_nsa": "goods",
        "goods_sa": "goods_sa",
        "services_nsa": "services",
        "services_sa": "services_sa",
        "domestic_nsa": "domestic",
        "domestic_sa": "domestic_sa",
        "imported_nsa": "imported",
        "imported_sa": "imported_sa",
    }
    for group in MAJOR_GROUPS:
        source = group["column"]
        price_sources[f"{source}_nsa"] = source
        price_sources[f"{source}_sa"] = f"{source}_sa"
    for prefix, source in price_sources.items():
        if source in out:
            log_price = 100 * np.log(out[source])
            derived[f"{prefix}_log_level"] = log_price
            derived[f"{prefix}_inflation_mom"] = log_price - log_price.shift(1)
            derived[f"{prefix}_inflation_yoy"] = log_price - log_price.shift(12)

    if derived:
        out = out.drop(columns=[column for column in derived if column in out.columns])
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
    return out


def response_column(base_column: str, transform_label: str) -> str:
    prefix_by_base = {
        "cpi": "headline_nsa",
        "cpi_sa": "headline_sa",
        "core_cpi_1": "core_1_nsa",
        "core_cpi_1_sa": "core_1_sa",
        "core_cpi_2": "core_2_nsa",
        "core_cpi_2_sa": "core_2_sa",
        "fresh_seasonal": "fresh_seasonal_nsa",
        "fresh_seasonal_sa": "fresh_seasonal_sa",
        "energy_fuel": "energy_fuel_nsa",
        "energy_fuel_sa": "energy_fuel_sa",
        "goods": "goods_nsa",
        "goods_sa": "goods_sa",
        "services": "services_nsa",
        "services_sa": "services_sa",
        "domestic": "domestic_nsa",
        "domestic_sa": "domestic_sa",
        "imported": "imported_nsa",
        "imported_sa": "imported_sa",
    }
    prefix = prefix_by_base[base_column]
    suffix = LP_TRANSFORMS[transform_label]["suffix"]
    if suffix == "mom":
        return f"{prefix}_inflation_mom"
    if suffix == "yoy":
        return f"{prefix}_inflation_yoy"
    return f"{prefix}_log_level"


def major_group_response_column(group_column: str, seasonal_adjustment: str) -> str:
    suffix = seasonal_adjustment.lower()
    return f"{group_column}_{suffix}_inflation_yoy"


def slugify_name(name: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "event"


def add_event_dummies(data: pd.DataFrame, event_windows: list[dict[str, pd.Timestamp]]) -> tuple[pd.DataFrame, list[str]]:
    out = data.copy()
    dummy_columns = []
    used_names: set[str] = set()

    def unique_name(base_name: str) -> str:
        name = base_name
        counter = 2
        while name in used_names:
            name = f"{base_name}_{counter}"
            counter += 1
        used_names.add(name)
        return name

    for window in event_windows:
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        if end < start:
            start, end = end, start
        base_name = f"event_{slugify_name(str(window['name']))}"
        if window.get("monthly_dummies", False):
            for period in pd.period_range(start=start, end=end, freq="M"):
                month = period.to_timestamp()
                name = unique_name(f"{base_name}_{month:%Y_%m}")
                out[name] = out["date"].dt.to_period("M").eq(period).astype(float)
                dummy_columns.append(name)
        else:
            name = unique_name(base_name)
            out[name] = ((out["date"] >= start) & (out["date"] <= end)).astype(float)
            dummy_columns.append(name)
    return out, dummy_columns


def apply_censor_windows(data: pd.DataFrame, censor_windows: list[dict[str, pd.Timestamp]]) -> pd.DataFrame:
    if not censor_windows:
        return data
    keep = pd.Series(True, index=data.index)
    for window in censor_windows:
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        if end < start:
            start, end = end, start
        keep &= ~((data["date"] >= start) & (data["date"] <= end))
    return data.loc[keep].copy()


def mask_censored_values(
    data: pd.DataFrame,
    censor_windows: list[dict[str, pd.Timestamp]],
    columns: list[str],
) -> pd.DataFrame:
    if not censor_windows:
        return data
    out = data.copy()
    columns = list(dict.fromkeys(column for column in columns if column in out.columns))
    for window in censor_windows:
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        if end < start:
            start, end = end, start
        mask = (out["date"] >= start) & (out["date"] <= end)
        out.loc[mask, columns] = np.nan
    return out


def to_presented_yoy_response(results: pd.DataFrame, transform_label: str) -> pd.DataFrame:
    return convert_response(results, transform_label, "y/y inflation")


def available_exchange_rate_shocks(data: pd.DataFrame) -> list[str]:
    return [
        label
        for label, settings in EXCHANGE_RATE_SHOCKS.items()
        if settings["level"] in data.columns and data[settings["level"]].notna().any()
    ]


def _with_ci(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    z = 1.6448536269514722
    out["ci_lower"] = out["beta"] - z * out["std_error"]
    out["ci_upper"] = out["beta"] + z * out["std_error"]
    return out


def convert_response(results: pd.DataFrame, from_transform: str, to_transform: str) -> pd.DataFrame | None:
    """Convert LP coefficients across price transformations when algebraically possible.

    Standard errors use a diagonal delta-method approximation across horizons.
    """
    out = results.copy()
    if from_transform == to_transform:
        out["display_response"] = out["response"]
        out["display_method"] = "direct"
        return out

    beta = out["beta"].to_numpy(dtype=float)
    se = out["std_error"].to_numpy(dtype=float)
    var = se**2

    if from_transform == "m/m inflation":
        cumulative_beta = np.cumsum(beta)
        cumulative_var = np.cumsum(var)
    elif from_transform == "Cumulative price difference":
        cumulative_beta = beta
        cumulative_var = var
    else:
        return None

    if to_transform == "Cumulative price difference":
        out["beta"] = cumulative_beta
        out["std_error"] = np.sqrt(cumulative_var)
    elif to_transform == "m/m inflation":
        lag_beta = np.r_[0.0, cumulative_beta[:-1]]
        lag_var = np.r_[0.0, cumulative_var[:-1]]
        out["beta"] = cumulative_beta - lag_beta
        out["std_error"] = np.sqrt(cumulative_var + lag_var)
    elif to_transform == "y/y inflation":
        n_horizons = len(cumulative_beta)
        if n_horizons > 12:
            lag_beta = np.r_[np.zeros(12), cumulative_beta[:-12]]
            lag_var = np.r_[np.zeros(12), cumulative_var[:-12]]
        else:
            lag_beta = np.zeros(n_horizons)
            lag_var = np.zeros(n_horizons)
        out["beta"] = cumulative_beta - lag_beta
        out["std_error"] = np.sqrt(cumulative_var + lag_var)
    else:
        return None

    out = _with_ci(out)
    out["display_response"] = f"Implied {to_transform} from {from_transform}"
    out["display_method"] = "converted"
    return out


def convert_response_by_group(
    results: pd.DataFrame,
    from_transform: str,
    to_transform: str,
    group_column: str,
) -> pd.DataFrame | None:
    converted = []
    for _, group in results.groupby(group_column, sort=False):
        converted_group = convert_response(group.sort_values("horizon"), from_transform, to_transform)
        if converted_group is None:
            return None
        converted.append(converted_group)
    return pd.concat(converted, ignore_index=True)


def implied_reer_response(
    price_results: pd.DataFrame,
    chf_results: pd.DataFrame,
    include_forward_shocks: bool,
) -> pd.DataFrame:
    """Construct implied REER response assuming foreign prices do not react."""
    price = price_results.loc[:, ["horizon", "beta", "std_error"]].rename(
        columns={"beta": "price_beta", "std_error": "price_std_error"}
    )
    if include_forward_shocks:
        neer = price.loc[:, ["horizon"]].copy()
        neer["neer_beta"] = 1.0
        neer["neer_std_error"] = 0.0
        neer["neer_path"] = "held_fixed_at_initial_shock"
    else:
        neer = chf_results.loc[:, ["horizon", "beta", "std_error"]].rename(
            columns={"beta": "neer_beta", "std_error": "neer_std_error"}
        )
        neer["neer_path"] = "estimated_persistence"

    out = price.merge(neer, on="horizon", how="inner")
    out["beta"] = out["neer_beta"] + out["price_beta"]
    out["std_error"] = np.sqrt(out["neer_std_error"].fillna(0.0) ** 2 + out["price_std_error"].fillna(0.0) ** 2)
    out = _with_ci(out)
    out["response"] = "Implied REER"
    out["display_response"] = "Implied REER, foreign prices fixed"
    return out


@st.cache_data(show_spinner=False)
def run_lp(
    data: pd.DataFrame,
    response: str,
    display_response: str,
    price_response: str,
    response_label: str,
    transform_label: str,
    display_transform_label: str,
    controls: list[str],
    shock_name: str,
    shock_level_response: str,
    shock_response_label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: int,
    lags: int,
    include_forward_shocks: bool,
    event_windows: list[dict[str, pd.Timestamp]],
    censor_windows: list[dict[str, pd.Timestamp]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = add_transforms(data)
    data, dummy_controls = add_event_dummies(data, event_windows)
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    sample = mask_censored_values(
        sample,
        censor_windows or [],
        [response, display_response, price_response, shock_name, shock_level_response, *controls],
    )
    transform = LP_TRANSFORMS[transform_label]
    display_transform = LP_TRANSFORMS[display_transform_label]
    cpi_config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=0.9,
        include_forward_shocks=include_forward_shocks,
    )
    chf_config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=0.9,
        include_forward_shocks=False,
    )
    cpi_results = estimate_dashboard_lp(
        data=sample,
        response=response,
        shock_name=shock_name,
        controls=controls,
        config=cpi_config,
        response_label=f"{response_label} {transform_label}",
        cumulative_response=bool(transform["cumulative"]),
        dummy_controls=dummy_controls,
    )
    display_results = convert_response(cpi_results, transform_label, display_transform_label)
    if display_results is None:
        display_results = estimate_dashboard_lp(
            data=sample,
            response=display_response,
            shock_name=shock_name,
            controls=controls,
            config=cpi_config,
            response_label=f"{response_label} {display_transform_label}",
            cumulative_response=bool(display_transform["cumulative"]),
            dummy_controls=dummy_controls,
        )
        display_results["display_response"] = display_results["response"]
        display_results["display_method"] = "direct_display_estimate"
    price_results = convert_response(cpi_results, transform_label, "Cumulative price difference")
    if price_results is None:
        price_results = estimate_dashboard_lp(
            data=sample,
            response=price_response,
            shock_name=shock_name,
            controls=controls,
            config=cpi_config,
            response_label=f"{response_label} Cumulative price difference",
            cumulative_response=True,
            dummy_controls=dummy_controls,
        )
        price_results["display_response"] = price_results["response"]
        price_results["display_method"] = "direct_price_estimate"
    chf_results = estimate_dashboard_lp(
        data=sample,
        response=shock_level_response,
        shock_name=shock_name,
        controls=controls,
        config=chf_config,
        response_label=shock_response_label,
        cumulative_response=True,
        dummy_controls=dummy_controls,
    )
    reer_results = implied_reer_response(price_results, chf_results, include_forward_shocks)
    return display_results, chf_results, price_results, reer_results


@st.cache_data(show_spinner=False)
def run_asymmetric_lp(
    data: pd.DataFrame,
    response: str,
    display_response: str,
    response_label: str,
    transform_label: str,
    display_transform_label: str,
    controls: list[str],
    shock_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: int,
    lags: int,
    include_forward_shocks: bool,
    event_windows: list[dict[str, pd.Timestamp]],
    censor_windows: list[dict[str, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    data = add_transforms(data)
    data, dummy_controls = add_event_dummies(data, event_windows)
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    sample = mask_censored_values(
        sample,
        censor_windows or [],
        [response, display_response, shock_name, *controls],
    )
    transform = LP_TRANSFORMS[transform_label]
    display_transform = LP_TRANSFORMS[display_transform_label]
    config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=0.9,
        include_forward_shocks=include_forward_shocks,
    )
    asym_results = estimate_asymmetric_dashboard_lp(
        data=sample,
        response=response,
        shock_name=shock_name,
        controls=controls,
        config=config,
        response_label=f"{response_label} {transform_label}",
        cumulative_response=bool(transform["cumulative"]),
        dummy_controls=dummy_controls,
    )
    display_results = convert_response_by_group(
        asym_results,
        transform_label,
        display_transform_label,
        "shock_component",
    )
    if display_results is None:
        display_results = estimate_asymmetric_dashboard_lp(
            data=sample,
            response=display_response,
            shock_name=shock_name,
            controls=controls,
            config=config,
            response_label=f"{response_label} {display_transform_label}",
            cumulative_response=bool(display_transform["cumulative"]),
            dummy_controls=dummy_controls,
        )
        display_results["display_response"] = display_results["response"]
        display_results["display_method"] = "direct_display_estimate"
    display_results["shock_component"] = display_results["shock_component"].replace(
        {
            "Positive CHF shock": "+1pp CHF appreciation",
            "Negative CHF shock": "-1 x 1pp CHF depreciation",
        }
    )
    display_results["raw_beta"] = display_results["beta"]
    display_results["raw_ci_lower"] = display_results["ci_lower"]
    display_results["raw_ci_upper"] = display_results["ci_upper"]
    depreciation_mask = display_results["shock_component"].eq("-1 x 1pp CHF depreciation")
    for column in ["beta", "ci_lower", "ci_upper"]:
        if column in display_results:
            display_results.loc[depreciation_mask, column] = -display_results.loc[depreciation_mask, column]
    lower = display_results.loc[depreciation_mask, "ci_lower"].copy()
    upper = display_results.loc[depreciation_mask, "ci_upper"].copy()
    display_results.loc[depreciation_mask, "ci_lower"] = upper
    display_results.loc[depreciation_mask, "ci_upper"] = lower
    return display_results


@st.cache_data(show_spinner=False)
def run_large_appreciation_lp(
    data: pd.DataFrame,
    response: str,
    display_response: str,
    shock_level_response: str,
    response_label: str,
    transform_label: str,
    display_transform_label: str,
    controls: list[str],
    shock_name: str,
    shock_response_label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: int,
    lags: int,
    include_forward_shocks: bool,
    appreciation_percentile: float,
    event_windows: list[dict[str, pd.Timestamp]],
    censor_windows: list[dict[str, pd.Timestamp]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, int, int]:
    data = add_transforms(data)
    data, dummy_controls = add_event_dummies(data, event_windows)
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    sample = mask_censored_values(
        sample,
        censor_windows or [],
        [response, display_response, shock_level_response, shock_name, *controls],
    )
    positive_moves = sample.loc[sample[shock_name].gt(0), shock_name].dropna()
    if positive_moves.empty:
        empty = pd.DataFrame(
            columns=[
                "response",
                "shock_component",
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
    large_component = f"{shock_name}_large_appreciation"
    other_component = f"{shock_name}_all_other_moves"
    large_mask = sample[shock_name].gt(0) & sample[shock_name].ge(threshold)
    sample[large_component] = sample[shock_name].where(large_mask, 0.0)
    sample[other_component] = sample[shock_name].where(~large_mask, 0.0)
    shock_components = {
        large_component: "Large appreciations",
        other_component: "All other FX moves",
    }

    transform = LP_TRANSFORMS[transform_label]
    display_transform = LP_TRANSFORMS[display_transform_label]
    config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=0.9,
        include_forward_shocks=include_forward_shocks,
    )
    split_results = estimate_split_shock_dashboard_lp(
        data=sample,
        response=response,
        shock_name=shock_name,
        shock_components=shock_components,
        controls=controls,
        config=config,
        response_label=f"{response_label} {transform_label}",
        cumulative_response=bool(transform["cumulative"]),
        dummy_controls=dummy_controls,
    )
    display_results = convert_response_by_group(
        split_results,
        transform_label,
        display_transform_label,
        "shock_component",
    )
    if display_results is None:
        display_results = estimate_split_shock_dashboard_lp(
            data=sample,
            response=display_response,
            shock_name=shock_name,
            shock_components=shock_components,
            controls=controls,
            config=config,
            response_label=f"{response_label} {display_transform_label}",
            cumulative_response=bool(display_transform["cumulative"]),
            dummy_controls=dummy_controls,
        )
        display_results["display_response"] = display_results["response"]
        display_results["display_method"] = "direct_display_estimate"
    fx_results = estimate_split_shock_dashboard_lp(
        data=sample,
        response=shock_level_response,
        shock_name=shock_name,
        shock_components=shock_components,
        controls=controls,
        config=config,
        response_label=shock_response_label,
        cumulative_response=True,
        dummy_controls=dummy_controls,
    )
    return display_results, fx_results, threshold, int(large_mask.sum()), int(positive_moves.size)


@st.cache_data(show_spinner=False)
def run_major_group_yoy_lps(
    data: pd.DataFrame,
    seasonal_adjustment: str,
    controls: list[str],
    shock_name: str,
    shock_level_response: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: int,
    lags: int,
    use_layered_shocks: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_transforms(data)
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=0.9,
        include_forward_shocks=False,
    )
    headline_base = LP_RESPONSES["Headline CPI"][seasonal_adjustment.lower()]
    headline_results = estimate_dashboard_lp(
        data=sample,
        response=response_column(headline_base, "y/y inflation"),
        shock_name=shock_name,
        controls=controls,
        config=config,
        response_label="Headline CPI y/y inflation",
        cumulative_response=False,
    )
    headline_results["group_code"] = "HEADLINE"
    headline_results["group_label"] = "Headline CPI"

    shock_path_results = estimate_dashboard_lp(
        data=sample,
        response=shock_level_response,
        shock_name=shock_name,
        controls=controls,
        config=config,
        response_label="CHF path",
        cumulative_response=True,
    )

    group_results = []
    for group in MAJOR_GROUPS:
        response_name = major_group_response_column(group["column"], seasonal_adjustment)
        if response_name not in sample.columns:
            continue
        result = estimate_dashboard_lp(
            data=sample,
            response=response_name,
            shock_name=shock_name,
            controls=controls,
            config=config,
            response_label=f"{group['label']} y/y inflation",
            cumulative_response=False,
        )
        result["group_code"] = group["code"]
        result["group_label"] = group["label"]
        result["group_column"] = group["column"]
        result["weight_pct"] = group["weight_pct"]
        group_results.append(result)

    if use_layered_shocks:
        headline_results = maintained_response_from_irfs(
            headline_results,
            shock_path_results,
            "Headline CPI maintained symmetric IRF",
        )
        headline_results["group_code"] = "HEADLINE"
        headline_results["group_label"] = "Headline CPI"

        layered_groups = []
        for result, group in zip(group_results, [group for group in MAJOR_GROUPS if major_group_response_column(group["column"], seasonal_adjustment) in sample.columns]):
            layered = maintained_response_from_irfs(
                result,
                shock_path_results,
                f"{group['label']} maintained symmetric IRF",
            )
            layered["group_code"] = group["code"]
            layered["group_label"] = group["label"]
            layered["group_column"] = group["column"]
            layered["weight_pct"] = group["weight_pct"]
            layered_groups.append(layered)
        groups = pd.concat(layered_groups, ignore_index=True) if layered_groups else pd.DataFrame()
    elif group_results:
        groups = pd.concat(group_results, ignore_index=True)
    else:
        groups = pd.DataFrame()
    return groups, headline_results


@st.cache_data(show_spinner=False)
def run_major_group_asymmetry_yoy_lps(
    data: pd.DataFrame,
    seasonal_adjustment: str,
    controls: list[str],
    shock_name: str,
    shock_level_response: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: int,
    lags: int,
    use_layered_shocks: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = add_transforms(data)
    shock_path_results = run_asymmetric_lp(
        data=data,
        response=shock_level_response,
        display_response=shock_level_response,
        response_label="CHF path",
        transform_label="Cumulative price difference",
        display_transform_label="Cumulative price difference",
        controls=controls,
        shock_name=shock_name,
        start=start,
        end=end,
        horizons=horizons,
        lags=lags,
        include_forward_shocks=False,
        event_windows=[],
    )

    headline_base = LP_RESPONSES["Headline CPI"][seasonal_adjustment.lower()]
    headline_response = response_column(headline_base, "y/y inflation")
    symmetric_headline, symmetric_chf, _, _ = run_lp(
        data=data,
        response=headline_response,
        display_response=headline_response,
        price_response=response_column(headline_base, "Cumulative price difference"),
        response_label="Headline CPI",
        transform_label="y/y inflation",
        display_transform_label="y/y inflation",
        controls=controls,
        shock_name=shock_name,
        shock_level_response=shock_level_response,
        shock_response_label="CHF path",
        start=start,
        end=end,
        horizons=horizons,
        lags=lags,
        include_forward_shocks=False,
        event_windows=[],
    )
    if use_layered_shocks:
        headline_reference = maintained_response_from_irfs(
            symmetric_headline,
            symmetric_chf,
            "Headline CPI maintained symmetric IRF",
        )
    else:
        headline_reference = symmetric_headline.copy()
    headline_reference["group_code"] = "HEADLINE"
    headline_reference["group_label"] = "Headline CPI"

    maintained_groups = []
    one_shock_groups = []
    for group in MAJOR_GROUPS:
        response_name = major_group_response_column(group["column"], seasonal_adjustment)
        if response_name not in data.columns:
            continue
        one_shock = run_asymmetric_lp(
            data=data,
            response=response_name,
            display_response=response_name,
            response_label=group["label"],
            transform_label="y/y inflation",
            display_transform_label="y/y inflation",
            controls=controls,
            shock_name=shock_name,
            start=start,
            end=end,
            horizons=horizons,
            lags=lags,
            include_forward_shocks=False,
            event_windows=[],
        )
        one_shock["group_code"] = group["code"]
        one_shock["group_label"] = group["label"]
        one_shock["group_column"] = group["column"]
        one_shock["weight_pct"] = group["weight_pct"]
        one_shock_groups.append(one_shock)

        if use_layered_shocks:
            displayed = build_maintained_asymmetry_results(one_shock, shock_path_results)
        else:
            displayed = one_shock.copy()
        displayed["group_code"] = group["code"]
        displayed["group_label"] = group["label"]
        displayed["group_column"] = group["column"]
        displayed["weight_pct"] = group["weight_pct"]
        maintained_groups.append(displayed)

    groups = pd.concat(maintained_groups, ignore_index=True) if maintained_groups else pd.DataFrame()
    one_shock = pd.concat(one_shock_groups, ignore_index=True) if one_shock_groups else pd.DataFrame()
    return groups, headline_reference, one_shock, shock_path_results


@st.cache_data(show_spinner=False)
def run_focus_group_yoy_lps(
    data: pd.DataFrame,
    seasonal_adjustment: str,
    controls: list[str],
    shock_name: str,
    shock_level_response: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: int,
    lags: int,
    use_layered_shocks: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_transforms(data)
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    config = LPConfig(
        horizons=horizons,
        lags=lags,
        ci_level=0.9,
        include_forward_shocks=False,
    )
    headline_base = LP_RESPONSES["Headline CPI"][seasonal_adjustment.lower()]
    headline_response = response_column(headline_base, "y/y inflation")
    headline_results = estimate_dashboard_lp(
        data=sample,
        response=headline_response,
        shock_name=shock_name,
        controls=controls,
        config=config,
        response_label="Headline CPI y/y inflation",
        cumulative_response=False,
    )
    headline_results["group_code"] = "HEADLINE"
    headline_results["group_label"] = "Headline CPI"

    shock_path_results = estimate_dashboard_lp(
        data=sample,
        response=shock_level_response,
        shock_name=shock_name,
        controls=controls,
        config=config,
        response_label="CHF path",
        cumulative_response=True,
    )

    focus_results = []
    for group in FOCUS_GROUPS:
        base_column = group["column"] if seasonal_adjustment == "NSA" else f"{group['column']}_sa"
        if base_column not in sample.columns:
            continue
        response_name = response_column(base_column, "y/y inflation")
        if response_name not in sample.columns:
            continue
        result = estimate_dashboard_lp(
            data=sample,
            response=response_name,
            shock_name=shock_name,
            controls=controls,
            config=config,
            response_label=f"{group['label']} y/y inflation",
            cumulative_response=False,
        )
        if use_layered_shocks:
            result = maintained_response_from_irfs(
                result,
                shock_path_results,
                f"{group['label']} maintained symmetric IRF",
            )
        result["group_code"] = group["code"]
        result["group_label"] = group["label"]
        result["group_column"] = group["column"]
        result["weight_pct"] = group["weight_pct"]
        focus_results.append(result)

    groups = pd.concat(focus_results, ignore_index=True) if focus_results else pd.DataFrame()
    if use_layered_shocks:
        headline_results = maintained_response_from_irfs(
            headline_results,
            shock_path_results,
            "Headline CPI maintained symmetric IRF",
        )
        headline_results["group_code"] = "HEADLINE"
        headline_results["group_label"] = "Headline CPI"
    return groups, headline_results


@st.cache_data(show_spinner=False)
def run_focus_group_asymmetry_yoy_lps(
    data: pd.DataFrame,
    seasonal_adjustment: str,
    controls: list[str],
    shock_name: str,
    shock_level_response: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: int,
    lags: int,
    use_layered_shocks: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = add_transforms(data)
    shock_path_results = run_asymmetric_lp(
        data=data,
        response=shock_level_response,
        display_response=shock_level_response,
        response_label="CHF path",
        transform_label="Cumulative price difference",
        display_transform_label="Cumulative price difference",
        controls=controls,
        shock_name=shock_name,
        start=start,
        end=end,
        horizons=horizons,
        lags=lags,
        include_forward_shocks=False,
        event_windows=[],
    )

    headline_base = LP_RESPONSES["Headline CPI"][seasonal_adjustment.lower()]
    headline_response = response_column(headline_base, "y/y inflation")
    symmetric_headline, symmetric_chf, _, _ = run_lp(
        data=data,
        response=headline_response,
        display_response=headline_response,
        price_response=response_column(headline_base, "Cumulative price difference"),
        response_label="Headline CPI",
        transform_label="y/y inflation",
        display_transform_label="y/y inflation",
        controls=controls,
        shock_name=shock_name,
        shock_level_response=shock_level_response,
        shock_response_label="CHF path",
        start=start,
        end=end,
        horizons=horizons,
        lags=lags,
        include_forward_shocks=False,
        event_windows=[],
    )
    if use_layered_shocks:
        headline_reference = maintained_response_from_irfs(
            symmetric_headline,
            symmetric_chf,
            "Headline CPI maintained symmetric IRF",
        )
    else:
        headline_reference = symmetric_headline.copy()
    headline_reference["group_code"] = "HEADLINE"
    headline_reference["group_label"] = "Headline CPI"
    headline_reference["weight_pct"] = 100.0

    displayed_groups = []
    one_shock_groups = []
    for group in FOCUS_GROUPS:
        base_column = group["column"] if seasonal_adjustment == "NSA" else f"{group['column']}_sa"
        if base_column not in data.columns:
            continue
        response_name = response_column(base_column, "y/y inflation")
        if response_name not in data.columns:
            continue
        one_shock = run_asymmetric_lp(
            data=data,
            response=response_name,
            display_response=response_name,
            response_label=group["label"],
            transform_label="y/y inflation",
            display_transform_label="y/y inflation",
            controls=controls,
            shock_name=shock_name,
            start=start,
            end=end,
            horizons=horizons,
            lags=lags,
            include_forward_shocks=False,
            event_windows=[],
        )
        one_shock["group_code"] = group["code"]
        one_shock["group_label"] = group["label"]
        one_shock["group_column"] = group["column"]
        one_shock["weight_pct"] = group["weight_pct"]
        one_shock_groups.append(one_shock)

        if use_layered_shocks:
            displayed = build_maintained_asymmetry_results(one_shock, shock_path_results)
        else:
            displayed = one_shock.copy()
        displayed["group_code"] = group["code"]
        displayed["group_label"] = group["label"]
        displayed["group_column"] = group["column"]
        displayed["weight_pct"] = group["weight_pct"]
        displayed_groups.append(displayed)

    groups = pd.concat(displayed_groups, ignore_index=True) if displayed_groups else pd.DataFrame()
    one_shock = pd.concat(one_shock_groups, ignore_index=True) if one_shock_groups else pd.DataFrame()
    return groups, headline_reference, one_shock, shock_path_results


def extract_asymmetric_component(results: pd.DataFrame, component: str) -> pd.DataFrame:
    out = results.loc[results["shock_component"].eq(component)].sort_values("horizon").copy()
    if "raw_beta" not in out:
        out["raw_beta"] = out["beta"]
    if "raw_ci_lower" not in out:
        out["raw_ci_lower"] = out["ci_lower"]
    if "raw_ci_upper" not in out:
        out["raw_ci_upper"] = out["ci_upper"]
    if "std_error" not in out:
        out["std_error"] = (out["raw_ci_upper"] - out["raw_ci_lower"]) / (2 * 1.6448536269514722)
    return out


def maintained_response_from_irfs(
    response_irf: pd.DataFrame,
    shock_path_irf: pd.DataFrame,
    label: str,
    flip_display: bool = False,
) -> pd.DataFrame:
    """Layer one-shock IRFs to keep the exchange-rate path at 1 percent."""
    response = response_irf.sort_values("horizon").reset_index(drop=True)
    path = shock_path_irf.sort_values("horizon").reset_index(drop=True)
    if "raw_beta" not in response:
        response["raw_beta"] = response["beta"]
    if "raw_beta" not in path:
        path["raw_beta"] = path["beta"]
    n = min(len(response), len(path))
    response = response.iloc[:n]
    path = path.iloc[:n]

    path_beta = path["raw_beta"].to_numpy(dtype=float)
    response_beta = response["raw_beta"].to_numpy(dtype=float)
    response_se = response["std_error"].to_numpy(dtype=float)

    if n == 0 or not np.isfinite(path_beta[0]) or np.isclose(path_beta[0], 0.0):
        return pd.DataFrame(
            columns=[
                "shock_component",
                "horizon",
                "beta",
                "std_error",
                "ci_lower",
                "ci_upper",
                "maintenance_shock",
                "maintained_exchange_rate_path",
            ]
        )

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

    sign = -1.0 if flip_display else 1.0
    beta = sign * layered_beta
    std_error = np.sqrt(layered_var)
    z = 1.6448536269514722
    out = pd.DataFrame(
        {
            "shock_component": label,
            "horizon": response["horizon"].to_numpy(),
            "beta": beta,
            "std_error": std_error,
            "ci_lower": beta - z * std_error,
            "ci_upper": beta + z * std_error,
            "raw_beta": layered_beta,
            "maintenance_shock": shocks,
            "maintained_exchange_rate_path": maintained_path,
        }
    )
    return out


def build_maintained_split_results(
    response_results: pd.DataFrame,
    shock_path_results: pd.DataFrame,
) -> pd.DataFrame:
    empty_columns = [
        "response",
        "shock_component",
        "horizon",
        "beta",
        "std_error",
        "ci_lower",
        "ci_upper",
        "raw_beta",
        "maintenance_shock",
        "maintained_exchange_rate_path",
        "display_method",
    ]
    layered_results = []
    for component, response_component in response_results.groupby("shock_component", sort=False):
        path_component = shock_path_results.loc[shock_path_results["shock_component"].eq(component)].copy()
        if path_component.empty:
            continue
        layered = maintained_response_from_irfs(
            response_component,
            path_component,
            component,
        )
        if not layered.empty:
            layered["response"] = response_component["response"].iloc[0] if "response" in response_component else component
            layered["display_method"] = "layered"
            layered_results.append(layered)
    return pd.concat(layered_results, ignore_index=True) if layered_results else pd.DataFrame(columns=empty_columns)


def build_maintained_asymmetry_results(
    response_results: pd.DataFrame,
    shock_path_results: pd.DataFrame,
) -> pd.DataFrame:
    appreciation_response = extract_asymmetric_component(response_results, "+1pp CHF appreciation")
    appreciation_path = extract_asymmetric_component(shock_path_results, "+1pp CHF appreciation")
    depreciation_response = extract_asymmetric_component(response_results, "-1 x 1pp CHF depreciation")
    depreciation_path = extract_asymmetric_component(shock_path_results, "-1 x 1pp CHF depreciation")

    depreciation_path = depreciation_path.copy()
    for column in ["raw_beta", "raw_ci_lower", "raw_ci_upper", "beta", "ci_lower", "ci_upper"]:
        if column in depreciation_path:
            depreciation_path[column] = -depreciation_path[column]
    lower = depreciation_path["raw_ci_lower"].copy()
    upper = depreciation_path["raw_ci_upper"].copy()
    depreciation_path["raw_ci_lower"] = upper
    depreciation_path["raw_ci_upper"] = lower

    maintained = [
        maintained_response_from_irfs(
            appreciation_response,
            appreciation_path,
            "Maintained +1% CHF appreciation",
            flip_display=False,
        ),
        maintained_response_from_irfs(
            depreciation_response,
            depreciation_path,
            "-1 x maintained 1% CHF depreciation",
            flip_display=True,
        ),
    ]
    return pd.concat(maintained, ignore_index=True)


def coverage_table(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in data.columns:
            continue
        first = data[column].first_valid_index()
        last = data[column].last_valid_index()
        rows.append(
            {
                "column": column,
                "start": data.loc[first, "date"].date() if first is not None else None,
                "end": data.loc[last, "date"].date() if last is not None else None,
                "non_missing": int(data[column].notna().sum()),
                "missing": int(data[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def draw_line_chart(
    data: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str = "",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> None:
    columns = [column for column in columns if column in data.columns]
    if not columns:
        st.info("No available columns for this chart.")
        return

    plot_data = data.loc[:, ["date", *columns]].dropna(how="all", subset=columns)
    if start is not None:
        plot_data = plot_data.loc[plot_data["date"] >= start]
    if end is not None:
        plot_data = plot_data.loc[plot_data["date"] <= end]
    if plot_data.empty:
        st.info("No observations in the selected sample window.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for column in columns:
        if column in plot_data:
            ax.plot(plot_data["date"], plot_data[column], linewidth=1.8, label=column)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=min(len(columns), 3))
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_normalized_line_chart(
    data: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    base_date: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    labels: dict[str, str] | None = None,
) -> None:
    columns = [column for column in columns if column in data.columns]
    if not columns:
        st.info("No available columns for this chart.")
        return

    base_ts = pd.Timestamp(base_date)
    full = data.loc[:, ["date", *columns]].copy()
    base_row = full.loc[full["date"].eq(base_ts), columns]
    if base_row.empty:
        st.info(f"Base month {base_ts:%Y-%m} is not available for normalization.")
        return
    base_values = base_row.iloc[0].replace(0, np.nan)
    normalized = full.copy()
    normalized.loc[:, columns] = 100 * normalized.loc[:, columns] / base_values

    plot_data = normalized.dropna(how="all", subset=columns)
    if start is not None:
        plot_data = plot_data.loc[plot_data["date"] >= start]
    if end is not None:
        plot_data = plot_data.loc[plot_data["date"] <= end]
    if plot_data.empty:
        st.info("No observations in the selected sample window.")
        return

    labels = labels or {}
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for column in columns:
        ax.plot(plot_data["date"], plot_data[column], linewidth=1.8, label=labels.get(column, column))
    ax.axhline(100, color="#4B5563", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=min(len(columns), 3))
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_sample_preview_chart(
    data: pd.DataFrame,
    response_column_name: str,
    shock_column_name: str,
    event_windows: list[dict[str, pd.Timestamp]],
) -> None:
    columns = [column for column in [response_column_name, shock_column_name] if column in data.columns]
    if not columns:
        return

    fig, axes = plt.subplots(len(columns), 1, figsize=(10.5, 2.6 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        ax.plot(data["date"], data[column], linewidth=1.5, color="#0F766E")
        ax.axhline(0, color="#4B5563", linewidth=0.8)
        for window in event_windows:
            ax.axvspan(pd.Timestamp(window["start"]), pd.Timestamp(window["end"]), color="#F59E0B", alpha=0.16)
        ax.set_title(column, fontsize=10, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.22)
    axes[-1].set_xlabel("")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_large_appreciation_definition_chart(
    data: pd.DataFrame,
    shock_level_column: str,
    shock_change_column: str,
    shock_label: str,
    threshold: float,
) -> None:
    columns = [column for column in [shock_level_column, shock_change_column] if column in data.columns]
    if len(columns) < 2 or np.isnan(threshold):
        st.info("Not enough exchange-rate data to show the large-appreciation definition chart.")
        return

    plot_data = data.loc[:, ["date", shock_level_column, shock_change_column]].dropna(subset=[shock_change_column]).copy()
    if plot_data.empty:
        st.info("No exchange-rate observations are available in the selected sample.")
        return

    large_mask = plot_data[shock_change_column].gt(0) & plot_data[shock_change_column].ge(threshold)
    large_dates = plot_data.loc[large_mask, "date"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    for ax in axes:
        for date in large_dates:
            ax.axvspan(date, date + pd.offsets.MonthEnd(1), color="#0F766E", alpha=0.14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.22)

    axes[0].plot(plot_data["date"], plot_data[shock_level_column], color="#2563EB", linewidth=1.7)
    axes[0].set_title(f"{shock_label}: Appreciation-Positive FX Level", fontsize=11, pad=8)
    axes[0].set_ylabel("Log level")

    axes[1].plot(plot_data["date"], plot_data[shock_change_column], color="#4B5563", linewidth=1.2, label="Monthly FX move")
    axes[1].scatter(
        plot_data.loc[large_mask, "date"],
        plot_data.loc[large_mask, shock_change_column],
        color="#0F766E",
        s=28,
        zorder=3,
        label="Large appreciation month",
    )
    axes[1].axhline(0, color="#4B5563", linewidth=0.8)
    axes[1].axhline(threshold, color="#0F766E", linewidth=1.4, linestyle="--", label=f"Threshold: {threshold:.2f} pp")
    axes[1].set_title("Monthly FX Move Used for the Large-Appreciation Split", fontsize=11, pad=8)
    axes[1].set_ylabel("Percent log points")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_missingness_heatmap(data: pd.DataFrame, columns: list[str]) -> None:
    columns = [column for column in columns if column in data.columns]
    if not columns:
        st.info("No columns available for the missingness heatmap.")
        return

    matrix = data.set_index("date")[columns].isna().T
    fig_height = max(3.2, 0.34 * len(columns))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=9)
    ax.set_xticks([])
    ax.set_title("Missing Values by Series", fontsize=12, pad=12)
    ax.set_xlabel("Monthly sample, left to right")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_irf_chart(results: pd.DataFrame, title: str, ylabel: str) -> None:
    plot_data = results.dropna(subset=["beta"])
    if plot_data.empty:
        st.info("Not enough observations for this specification.")
        return

    x = plot_data["horizon"].to_numpy()
    beta = plot_data["beta"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axhline(0, color="#4B5563", linewidth=1)
    ax.plot(x, beta, color="#0F766E", linewidth=2.2, label="Estimate")
    if plot_data[["ci_lower", "ci_upper"]].notna().all(axis=None):
        lower = plot_data["ci_lower"].to_numpy()
        upper = plot_data["ci_upper"].to_numpy()
        ax.fill_between(x, lower, upper, color="#0F766E", alpha=0.18, label="90% CI")
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Months after CHF shock")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_neer_reer_chart(
    chf_results: pd.DataFrame,
    reer_results: pd.DataFrame,
    include_forward_shocks: bool,
    shock_settings: dict[str, str],
) -> None:
    chf_plot = chf_results.dropna(subset=["beta"])
    reer_plot = reer_results.dropna(subset=["beta"])
    if chf_plot.empty or reer_plot.empty:
        st.info("Not enough observations for the CHF/REER companion chart.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axhline(0, color="#4B5563", linewidth=1)
    if include_forward_shocks:
        x = reer_plot["horizon"].to_numpy()
        ax.plot(x, np.ones_like(x), color="#2563EB", linewidth=2, label=shock_settings["held_label"])
    else:
        ax.plot(
            chf_plot["horizon"],
            chf_plot["beta"],
            color="#2563EB",
            linewidth=2,
            label=shock_settings["path_label"],
        )
    ax.plot(
        reer_plot["horizon"],
        reer_plot["beta"],
        color="#0F766E",
        linewidth=2.2,
        label="Implied REER, foreign prices fixed",
    )
    if reer_plot[["ci_lower", "ci_upper"]].notna().all(axis=None):
        ax.fill_between(
            reer_plot["horizon"].to_numpy(),
            reer_plot["ci_lower"].to_numpy(),
            reer_plot["ci_upper"].to_numpy(),
            color="#0F766E",
            alpha=0.14,
            label="REER 90% CI",
        )
    ax.set_title(f"{shock_settings['display']} and Implied REER Response", fontsize=12, pad=12)
    ax.set_xlabel("Months after CHF shock")
    ax.set_ylabel("Percent log points")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_comparison_chart(specs: list[dict], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(0, color="#4B5563", linewidth=1)
    for spec in specs:
        results = spec["results"].dropna(subset=["beta"])
        if results.empty:
            continue
        ax.plot(
            results["horizon"],
            results["beta"],
            linewidth=2,
            label=spec["label"],
        )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Months after CHF shock")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_fx_comparison_chart(specs: list[dict], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(0, color="#4B5563", linewidth=1)
    colors = {
        "+1pp CHF appreciation": "#0F766E",
        "-1 x 1pp CHF depreciation": "#B45309",
        "Maintained +1% CHF appreciation": "#0F766E",
        "-1 x maintained 1% CHF depreciation": "#B45309",
        "Large appreciations": "#0F766E",
        "All other FX moves": "#B45309",
    }
    for spec in specs:
        results = spec["results"].dropna(subset=["beta"]).copy()
        if results.empty:
            continue
        if "shock_component" not in results:
            ax.plot(results["horizon"], results["beta"], linewidth=2, label=spec["label"])
            continue
        for component, group in results.groupby("shock_component", sort=False):
            color = colors.get(component)
            ax.plot(
                group["horizon"],
                group["beta"],
                linewidth=2,
                color=color,
                label=f"{spec['label']}: {component}",
            )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Months after CHF shock")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_asymmetric_irf_chart(
    asymmetric_results: pd.DataFrame,
    symmetric_results: pd.DataFrame,
    title: str,
    ylabel: str,
) -> None:
    asym_plot = asymmetric_results.dropna(subset=["beta"])
    sym_plot = symmetric_results.dropna(subset=["beta"])
    if asym_plot.empty and sym_plot.empty:
        st.info("Not enough observations for the asymmetric-shock chart.")
        return

    colors = {
        "+1pp CHF appreciation": "#0F766E",
        "-1 x 1pp CHF depreciation": "#B45309",
        "Large appreciations": "#0F766E",
        "All other FX moves": "#B45309",
    }
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(0, color="#4B5563", linewidth=1)

    for component, group in asym_plot.groupby("shock_component", sort=False):
        x = group["horizon"].to_numpy()
        beta = group["beta"].to_numpy()
        color = colors.get(component, "#2563EB")
        ax.plot(x, beta, linewidth=2.2, color=color, label=component)
        if group[["ci_lower", "ci_upper"]].notna().all(axis=None):
            ax.fill_between(
                x,
                group["ci_lower"].to_numpy(),
                group["ci_upper"].to_numpy(),
                color=color,
                alpha=0.13,
            )

    if not sym_plot.empty:
        ax.plot(
            sym_plot["horizon"],
            sym_plot["beta"],
            color="#111827",
            linewidth=2,
            linestyle="--",
            label="Symmetric IRF",
        )

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Months after CHF shock")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_maintained_asymmetry_chart(results: pd.DataFrame, title: str, ylabel: str) -> None:
    plot_data = results.dropna(subset=["beta"])
    if plot_data.empty:
        st.info("Not enough observations for the maintained-shock asymmetry chart.")
        return

    colors = {
        "Maintained +1% CHF appreciation": "#0F766E",
        "-1 x maintained 1% CHF depreciation": "#B45309",
        "Maintained symmetric IRF": "#111827",
    }
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(0, color="#4B5563", linewidth=1)
    for component, group in plot_data.groupby("shock_component", sort=False):
        x = group["horizon"].to_numpy()
        beta = group["beta"].to_numpy()
        color = colors.get(component, "#2563EB")
        is_symmetric = component == "Maintained symmetric IRF"
        ax.plot(
            x,
            beta,
            linewidth=2.3 if not is_symmetric else 2,
            color=color,
            linestyle="--" if is_symmetric else "-",
            label=component,
        )
        if not is_symmetric and group[["ci_lower", "ci_upper"]].notna().all(axis=None):
            ax.fill_between(
                x,
                group["ci_lower"].to_numpy(),
                group["ci_upper"].to_numpy(),
                color=color,
                alpha=0.13,
            )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Months after initial CHF shock")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_major_group_grid(group_results: pd.DataFrame, headline_results: pd.DataFrame, title: str) -> None:
    plot_data = group_results.dropna(subset=["beta"])
    headline_plot = headline_results.dropna(subset=["beta"])
    if plot_data.empty:
        st.info("Not enough observations for the selected subgroup specification.")
        return

    groups = [group for group in MAJOR_GROUPS if group["code"] in set(plot_data["group_code"])]
    ncols = 2
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, group in zip(axes, groups):
        current = plot_data.loc[plot_data["group_code"].eq(group["code"])].sort_values("horizon")
        x = current["horizon"].to_numpy()
        ax.axhline(0, color="#4B5563", linewidth=0.8)
        ax.plot(x, current["beta"].to_numpy(), color="#0F766E", linewidth=1.9, label=group["label"])
        if current[["ci_lower", "ci_upper"]].notna().all(axis=None):
            ax.fill_between(
                x,
                current["ci_lower"].to_numpy(),
                current["ci_upper"].to_numpy(),
                color="#0F766E",
                alpha=0.14,
            )
        if not headline_plot.empty:
            ax.plot(
                headline_plot["horizon"].to_numpy(),
                headline_plot["beta"].to_numpy(),
                color="#111827",
                linewidth=1.6,
                linestyle="--",
                label="Headline CPI",
            )
        ax.set_title(f"{group['code']}: {group['label']} ({group['weight_pct']:.1f}%)", fontsize=10, pad=8)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="best")

    for ax in axes[len(groups) :]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Months after CHF shock")
    for row in range(nrows):
        axes[row * ncols].set_ylabel("Percentage points")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_major_group_asymmetry_grid(group_results: pd.DataFrame, headline_results: pd.DataFrame, title: str) -> None:
    plot_data = group_results.dropna(subset=["beta"])
    headline_plot = headline_results.dropna(subset=["beta"])
    if plot_data.empty:
        st.info("Not enough observations for the selected subgroup asymmetry specification.")
        return

    groups = [group for group in MAJOR_GROUPS if group["code"] in set(plot_data["group_code"])]
    ncols = 2
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    colors = {
        "Maintained +1% CHF appreciation": "#0F766E",
        "-1 x maintained 1% CHF depreciation": "#B45309",
        "+1pp CHF appreciation": "#0F766E",
        "-1 x 1pp CHF depreciation": "#B45309",
    }
    for ax, group in zip(axes, groups):
        current = plot_data.loc[plot_data["group_code"].eq(group["code"])].sort_values(["shock_component", "horizon"])
        ax.axhline(0, color="#4B5563", linewidth=0.8)
        for component, component_data in current.groupby("shock_component", sort=False):
            x = component_data["horizon"].to_numpy()
            color = colors.get(component, "#2563EB")
            ax.plot(x, component_data["beta"].to_numpy(), color=color, linewidth=1.9, label=component)
            if component_data[["ci_lower", "ci_upper"]].notna().all(axis=None):
                ax.fill_between(
                    x,
                    component_data["ci_lower"].to_numpy(),
                    component_data["ci_upper"].to_numpy(),
                    color=color,
                    alpha=0.11,
                )
        if not headline_plot.empty:
            ax.plot(
                headline_plot["horizon"].to_numpy(),
                headline_plot["beta"].to_numpy(),
                color="#111827",
                linewidth=1.6,
                linestyle="--",
                label="Headline CPI",
            )
        ax.set_title(f"{group['code']}: {group['label']} ({group['weight_pct']:.1f}%)", fontsize=10, pad=8)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="best")

    for ax in axes[len(groups) :]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Months after initial CHF shock")
    for row in range(nrows):
        axes[row * ncols].set_ylabel("Percentage points")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_focus_group_grid(group_results: pd.DataFrame, headline_results: pd.DataFrame, title: str) -> None:
    plot_data = group_results.dropna(subset=["beta"])
    headline_plot = headline_results.dropna(subset=["beta"])
    if plot_data.empty:
        st.info("Not enough observations for the selected focus-group specification.")
        return

    groups = [group for group in FOCUS_GROUPS if group["code"] in set(plot_data["group_code"])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, group in zip(axes, groups):
        current = plot_data.loc[plot_data["group_code"].eq(group["code"])].sort_values("horizon")
        x = current["horizon"].to_numpy()
        ax.axhline(0, color="#4B5563", linewidth=0.8)
        ax.plot(x, current["beta"].to_numpy(), color="#0F766E", linewidth=2.1, label=group["label"])
        if current[["ci_lower", "ci_upper"]].notna().all(axis=None):
            ax.fill_between(
                x,
                current["ci_lower"].to_numpy(),
                current["ci_upper"].to_numpy(),
                color="#0F766E",
                alpha=0.14,
                label="90% CI",
            )
        if group["code"] != "HEADLINE" and not headline_plot.empty:
            ax.plot(
                headline_plot["horizon"].to_numpy(),
                headline_plot["beta"].to_numpy(),
                color="#111827",
                linewidth=1.8,
                linestyle="--",
                label="Headline CPI",
            )
        if "weight_pct" in current and current["weight_pct"].notna().any():
            weight = float(current["weight_pct"].dropna().iloc[0])
            title_label = f"{group['code']}: {group['label']} ({weight:.1f}%)"
        else:
            title_label = f"{group['code']}: {group['label']}"
        ax.set_title(title_label, fontsize=11, pad=8)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")

    for ax in axes[len(groups) :]:
        ax.axis("off")
    for ax in axes[-2:]:
        ax.set_xlabel("Months after CHF shock")
    axes[0].set_ylabel("Percentage points")
    axes[2].set_ylabel("Percentage points")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_focus_group_asymmetry_grid(group_results: pd.DataFrame, headline_results: pd.DataFrame, title: str) -> None:
    plot_data = group_results.dropna(subset=["beta"])
    headline_plot = headline_results.dropna(subset=["beta"])
    if plot_data.empty:
        st.info("Not enough observations for the selected focus-group asymmetry specification.")
        return

    groups = [group for group in FOCUS_GROUPS if group["code"] in set(plot_data["group_code"])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    colors = {
        "Maintained +1% CHF appreciation": "#0F766E",
        "-1 x maintained 1% CHF depreciation": "#B45309",
        "+1pp CHF appreciation": "#0F766E",
        "-1 x 1pp CHF depreciation": "#B45309",
    }

    for ax, group in zip(axes, groups):
        current = plot_data.loc[plot_data["group_code"].eq(group["code"])].sort_values(["shock_component", "horizon"])
        ax.axhline(0, color="#4B5563", linewidth=0.8)
        for component, component_data in current.groupby("shock_component", sort=False):
            x = component_data["horizon"].to_numpy()
            color = colors.get(component, "#2563EB")
            ax.plot(x, component_data["beta"].to_numpy(), color=color, linewidth=1.9, label=component)
            if component_data[["ci_lower", "ci_upper"]].notna().all(axis=None):
                ax.fill_between(
                    x,
                    component_data["ci_lower"].to_numpy(),
                    component_data["ci_upper"].to_numpy(),
                    color=color,
                    alpha=0.11,
                )
        if group["code"] != "HEADLINE" and not headline_plot.empty:
            ax.plot(
                headline_plot["horizon"].to_numpy(),
                headline_plot["beta"].to_numpy(),
                color="#111827",
                linewidth=1.6,
                linestyle="--",
                label="Headline symmetric IRF",
            )
        weight = group.get("weight_pct")
        title_label = f"{group['code']}: {group['label']} ({weight:.1f}%)" if weight is not None else f"{group['code']}: {group['label']}"
        ax.set_title(title_label, fontsize=11, pad=8)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="best")

    for ax in axes[len(groups) :]:
        ax.axis("off")
    for ax in axes[-2:]:
        ax.set_xlabel("Months after initial CHF shock")
    axes[0].set_ylabel("Percentage points")
    axes[2].set_ylabel("Percentage points")

    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def focus_bottom_up_contributions(group_results: pd.DataFrame, headline_results: pd.DataFrame) -> pd.DataFrame:
    required_codes = {"K1", "FP", "ET"}
    pieces = group_results.loc[group_results["group_code"].isin(required_codes)].dropna(subset=["beta"]).copy()
    if pieces.empty or not required_codes.issubset(set(pieces["group_code"])):
        return pd.DataFrame()

    pieces["contribution"] = pieces["beta"] * pieces["weight_pct"] / 100
    pieces["contribution_std_error"] = pieces["std_error"] * pieces["weight_pct"] / 100
    bottom_up = (
        pieces.groupby("horizon", as_index=False)
        .agg(
            bottom_up_beta=("contribution", "sum"),
            bottom_up_var=("contribution_std_error", lambda x: float(np.sum(np.square(x)))),
        )
        .sort_values("horizon")
    )
    bottom_up["bottom_up_std_error"] = np.sqrt(bottom_up["bottom_up_var"])
    z = 1.6448536269514722
    bottom_up["bottom_up_ci_lower"] = bottom_up["bottom_up_beta"] - z * bottom_up["bottom_up_std_error"]
    bottom_up["bottom_up_ci_upper"] = bottom_up["bottom_up_beta"] + z * bottom_up["bottom_up_std_error"]

    headline = headline_results.loc[:, ["horizon", "beta", "std_error", "ci_lower", "ci_upper"]].rename(
        columns={
            "beta": "top_down_beta",
            "std_error": "top_down_std_error",
            "ci_lower": "top_down_ci_lower",
            "ci_upper": "top_down_ci_upper",
        }
    )
    out = pieces.merge(bottom_up, on="horizon", how="left").merge(headline, on="horizon", how="left")
    out["bottom_up_gap"] = out["bottom_up_beta"] - out["top_down_beta"]
    return out


def draw_focus_contribution_chart(
    group_results: pd.DataFrame,
    headline_results: pd.DataFrame,
    component: str | None = None,
    title_suffix: str = "",
) -> pd.DataFrame:
    chart_results = group_results
    if component is not None and "shock_component" in chart_results:
        chart_results = chart_results.loc[chart_results["shock_component"].eq(component)].copy()
    headline_for_chart = headline_results
    if "group_code" in chart_results:
        headline_component = chart_results.loc[chart_results["group_code"].eq("HEADLINE")].copy()
        if not headline_component.empty:
            headline_for_chart = headline_component
    contribution_data = focus_bottom_up_contributions(chart_results, headline_for_chart)
    if contribution_data.empty:
        st.info("Not enough estimates to build the bottom-up focus contribution chart.")
        return contribution_data

    colors = {
        "K1": "#0F766E",
        "FP": "#2563EB",
        "ET": "#B45309",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)

    for code, group in contribution_data.groupby("group_code", sort=False):
        label = group["group_label"].iloc[0]
        weight = float(group["weight_pct"].iloc[0])
        axes[0].plot(
            group["horizon"],
            group["contribution"],
            linewidth=2,
            color=colors.get(code, "#4B5563"),
            label=f"{label} ({weight:.1f}%)",
        )
    axes[0].axhline(0, color="#4B5563", linewidth=0.9)
    suffix = f": {title_suffix}" if title_suffix else ""
    axes[0].set_title(f"Weighted Contributions to Headline CPI IRF{suffix}", fontsize=12, pad=10)
    axes[0].set_ylabel("Percentage points")
    axes[0].grid(True, alpha=0.24)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncols=3)

    comparison = (
        contribution_data.loc[
            :,
            [
                "horizon",
                "bottom_up_beta",
                "bottom_up_ci_lower",
                "bottom_up_ci_upper",
                "top_down_beta",
                "top_down_ci_lower",
                "top_down_ci_upper",
                "bottom_up_gap",
            ],
        ]
        .drop_duplicates()
        .sort_values("horizon")
    )
    component_order = ["K1", "FP", "ET"]
    component_labels = {
        code: contribution_data.loc[contribution_data["group_code"].eq(code), "group_label"].iloc[0]
        for code in component_order
        if code in set(contribution_data["group_code"])
    }
    stacked = (
        contribution_data.pivot_table(
            index="horizon",
            columns="group_code",
            values="contribution",
            aggfunc="first",
        )
        .reindex(comparison["horizon"])
        .fillna(0.0)
    )
    x = comparison["horizon"].to_numpy()
    axes[1].axhline(0, color="#4B5563", linewidth=0.9)
    positive_base = np.zeros(len(stacked))
    negative_base = np.zeros(len(stacked))
    for code in component_order:
        if code not in stacked:
            continue
        values = stacked[code].to_numpy(dtype=float)
        positive = np.where(values > 0, values, 0.0)
        negative = np.where(values < 0, values, 0.0)
        color = colors.get(code, "#4B5563")
        label = f"{component_labels.get(code, code)} contribution"
        if np.any(positive):
            axes[1].fill_between(
                x,
                positive_base,
                positive_base + positive,
                color=color,
                alpha=0.22,
                linewidth=0,
                label=label,
            )
            positive_base = positive_base + positive
        if np.any(negative):
            axes[1].fill_between(
                x,
                negative_base,
                negative_base + negative,
                color=color,
                alpha=0.22,
                linewidth=0,
                label=label if not np.any(positive) else None,
            )
            negative_base = negative_base + negative
    axes[1].plot(
        comparison["horizon"],
        comparison["bottom_up_beta"],
        color="#0F766E",
        linewidth=2.3,
        label="Bottom-up: K1 + FP + ET contributions",
    )
    axes[1].plot(
        comparison["horizon"],
        comparison["top_down_beta"],
        color="#111827",
        linewidth=2,
        linestyle="--",
        label="Top-down headline CPI IRF",
    )
    axes[1].set_title(f"Bottom-Up Focus Decomposition vs Top-Down Headline IRF{suffix}", fontsize=12, pad=10)
    axes[1].set_xlabel("Months after CHF shock")
    axes[1].set_ylabel("Percentage points")
    axes[1].grid(True, alpha=0.24)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    return contribution_data


def draw_major_group_horizon_bars(
    group_results: pd.DataFrame,
    headline_results: pd.DataFrame,
    component: str | None = None,
) -> None:
    plot_data = group_results.dropna(subset=["beta"]).copy()
    if plot_data.empty:
        st.info("Not enough observations for the horizon ranking.")
        return
    if component is not None and "shock_component" in plot_data:
        plot_data = plot_data.loc[plot_data["shock_component"].eq(component)].copy()
    plot_data = plot_data.loc[plot_data["group_code"].ne("T")].copy()
    if plot_data.empty:
        st.info("Not enough non-total group observations for the horizon ranking.")
        return

    plot_data["weighted_contribution"] = plot_data["beta"] * plot_data["weight_pct"] / 100
    headline_plot = headline_results.dropna(subset=["beta"]).copy()
    horizons = [(12, "1 year"), (24, "2 years"), (36, "3 years"), ("level36", "Implied price level after 36 months")]

    def implied_level_at_36(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, group in frame.groupby(["group_code", "group_label", "weight_pct"], sort=False):
            by_horizon = group.set_index("horizon")["beta"].to_dict()
            if 36 not in by_horizon:
                continue
            level_effect = by_horizon.get(36, 0.0) + by_horizon.get(24, 0.0) + by_horizon.get(12, 0.0)
            group_code, group_label, weight_pct = keys
            rows.append(
                {
                    "group_code": group_code,
                    "group_label": group_label,
                    "weight_pct": weight_pct,
                    "beta": level_effect,
                    "weighted_contribution": level_effect * weight_pct / 100,
                }
            )
        return pd.DataFrame(rows)

    def headline_level_at_36(frame: pd.DataFrame) -> float:
        by_horizon = frame.set_index("horizon")["beta"].to_dict()
        if 36 not in by_horizon:
            return np.nan
        return by_horizon.get(36, 0.0) + by_horizon.get(24, 0.0) + by_horizon.get(12, 0.0)

    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 19), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, (horizon, label) in zip(axes, horizons):
        if horizon == "level36":
            current = implied_level_at_36(plot_data)
            headline_value = headline_level_at_36(headline_plot)
            x_label = "Percent log points"
            title_prefix = "Implied cumulative price-level effect after 36 months"
        else:
            current = plot_data.loc[plot_data["horizon"].eq(horizon)].copy()
            headline_at_horizon = headline_plot.loc[headline_plot["horizon"].eq(horizon), "beta"]
            headline_value = float(headline_at_horizon.iloc[0]) if not headline_at_horizon.empty else np.nan
            x_label = "Percentage points"
            title_prefix = f"After {label}"
        if current.empty:
            ax.text(0.5, 0.5, f"No horizon {horizon} estimate", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        current["bar_label"] = current["group_code"] + "  " + current["group_label"]
        current = current.sort_values("beta", ascending=True)
        y = np.arange(len(current))
        height = 0.36
        ax.axvline(0, color="#4B5563", linewidth=0.9)
        ax.barh(
            y + height / 2,
            current["beta"].to_numpy(),
            height=height,
            color="#0F766E",
            alpha=0.86,
            label="Subgroup IRF",
        )
        ax.barh(
            y - height / 2,
            current["weighted_contribution"].to_numpy(),
            height=height,
            color="#B45309",
            alpha=0.78,
            label="Weighted contribution",
        )
        if np.isfinite(headline_value):
            ax.axvline(
                float(headline_value),
                color="#111827",
                linestyle="--",
                linewidth=1.4,
                label="Headline CPI IRF",
            )
        contribution_sum = current["weighted_contribution"].sum()
        ax.set_yticks(y)
        ax.set_yticklabels(current["bar_label"], fontsize=8)
        ax.set_title(
            f"{title_prefix}: sorted by subgroup effect; contribution sum = {contribution_sum:.3f}",
            fontsize=11,
            pad=8,
        )
        ax.set_xlabel(x_label)
        ax.grid(True, axis="x", alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_maintenance_chart(results: pd.DataFrame) -> None:
    plot_data = results.dropna(subset=["maintenance_shock"])
    if plot_data.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    colors = {
        "Maintained +1% CHF appreciation": "#0F766E",
        "-1 x maintained 1% CHF depreciation": "#B45309",
    }
    for component, group in plot_data.groupby("shock_component", sort=False):
        color = colors.get(component, "#2563EB")
        axes[0].plot(
            group["horizon"],
            group["maintenance_shock"],
            linewidth=2,
            color=color,
            label=component,
        )
        axes[1].plot(
            group["horizon"],
            group["maintained_exchange_rate_path"],
            linewidth=2,
            color=color,
            label=component,
        )
    axes[0].axhline(0, color="#4B5563", linewidth=1)
    axes[0].set_title("Layered Shock Sequence", fontsize=11, pad=8)
    axes[0].set_ylabel("Percent log points")
    axes[1].axhline(1, color="#4B5563", linewidth=1, linestyle="--")
    axes[1].set_title("Maintained Exchange-Rate Path", fontsize=11, pad=8)
    axes[1].set_ylabel("Percent log points")
    axes[1].set_xlabel("Months after initial CHF shock")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def default_spec_label(
    response_label: str,
    shock_label: str,
    transform_label: str,
    display_transform_label: str,
    controls_label: str,
    include_forward_shocks: bool,
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_windows_for_estimation: list[dict[str, pd.Timestamp]],
    event_windows_for_censoring: list[dict[str, pd.Timestamp]] | None = None,
) -> str:
    event_part = f", {len(event_windows_for_estimation)}D" if event_windows_for_estimation else ""
    censor_part = f", {len(event_windows_for_censoring or [])}C" if event_windows_for_censoring else ""
    fwd = "fwd" if include_forward_shocks else "no fwd"
    return (
        f"{response_label}, {shock_label}, est {transform_label}, disp {display_transform_label}, "
        f"{controls_label}, {fwd}{event_part}{censor_part}, {start:%Y-%m}-{end:%Y-%m}"
    )


def comparison_store() -> list[dict]:
    if "lp_comparison_specs" not in st.session_state:
        st.session_state.lp_comparison_specs = []
    return st.session_state.lp_comparison_specs


def named_comparison_store(name: str) -> list[dict]:
    key = f"{name}_comparison_specs"
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]


def equation_text(
    response_label: str,
    transform_label: str,
    controls: list[str],
    lags: int,
    include_forward_shocks: bool,
    include_dummies: bool,
) -> str:
    control_term = r" + \theta_h' Z_t" if controls else ""
    lagged_control_term = r" + \sum_{j=1}^{p} \eta_{h,j}' Z_{t-j}" if controls else ""
    forward_term = r" + \sum_{f=1}^{h} \phi_{h,f} \Delta s_{t+f}" if include_forward_shocks else ""
    dummy_term = r" + \kappa_h' D_t" if include_dummies else ""
    lhs = rf"\Delta_h {response_label}_{{t+h}}" if LP_TRANSFORMS[transform_label]["cumulative"] else rf"{response_label}_{{t+h}}"
    return (
        rf"{lhs} = \alpha_h + \beta_h \Delta s_t"
        f"{control_term}{dummy_term}{forward_term}"
        rf" + \sum_{{j=1}}^{{p}} \gamma_{{h,j}} {response_label}_{{t-j}}"
        rf" + \sum_{{j=1}}^{{p}} \delta_{{h,j}} \Delta s_{{t-j}}"
        rf"{lagged_control_term} + u_{{t+h}}, \quad p = {lags}"
    )


def monthly_options(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [period.strftime("%Y-%m") for period in pd.period_range(start=start, end=end, freq="M")]


def event_window_label(window: dict) -> str:
    suffix = " monthly" if window.get("monthly_dummies", False) else ""
    return f"{window['name']}{suffix}"


WINDOW_ACTIONS = ["Period dummy", "Dummy for each month", "Censor"]


def apply_window_action(window: dict, action: str) -> dict:
    out = window.copy()
    out["treatment"] = "Censor" if action == "Censor" else "Dummy"
    out["monthly_dummies"] = action == "Dummy for each month"
    out["action"] = action
    return out


def default_window_action(window: dict) -> str:
    if window.get("treatment") == "Censor":
        return "Censor"
    if window.get("monthly_dummies", False):
        return "Dummy for each month"
    return "Period dummy"


def event_window_type(window: dict) -> str:
    treatment = window.get("treatment", "Dummy")
    if treatment == "Censor":
        return "Censored observations"
    return "Separate monthly dummies" if window.get("monthly_dummies", False) else "Block dummy"


def event_dummy_column_count(window: dict) -> int:
    if window.get("treatment", "Dummy") != "Dummy":
        return 0
    if window.get("monthly_dummies", False):
        return len(pd.period_range(window["start"], window["end"], freq="M"))
    return 1


def dummy_windows(event_windows: list[dict]) -> list[dict]:
    return [window for window in event_windows if window.get("treatment", "Dummy") == "Dummy"]


def censor_windows(event_windows: list[dict]) -> list[dict]:
    return [window for window in event_windows if window.get("treatment") == "Censor"]


def event_preset_window(name: str) -> dict:
    preset = EVENT_PRESETS[name]
    if isinstance(preset, dict):
        action = preset.get(
            "action",
            "Dummy for each month" if preset.get("monthly_dummies", False) else "Period dummy",
        )
        return {
            "name": name,
            "start": pd.Timestamp(preset["start"]),
            "end": pd.Timestamp(preset["end"]),
            "monthly_dummies": action == "Dummy for each month",
            "treatment": "Censor" if action == "Censor" else "Dummy",
            "action": action,
        }
    return {
        "name": name,
        "start": pd.Timestamp(preset[0]),
        "end": pd.Timestamp(preset[1]),
        "monthly_dummies": False,
        "treatment": "Dummy",
        "action": "Period dummy",
    }


def sidebar() -> tuple[str, Path]:
    st.sidebar.title("CPI LP Dashboard")
    pages = [
        "FX IRF",
        "Baseline LP",
        "Asymmetry LP",
        "Large Appreciations",
        "Major Groups",
        "Focus Groups",
        "Data",
        "Methodology",
        "GDP LP",
        "Unemployment LP",
        "Real Indicator Data Audit",
    ]
    if "sidebar_page" not in st.session_state or st.session_state.sidebar_page not in pages:
        st.session_state.sidebar_page = "FX IRF"

    def nav_button(label: str) -> None:
        button_type = "primary" if st.session_state.sidebar_page == label else "secondary"
        if st.sidebar.button(label, type=button_type, width="stretch", key=f"nav_{slugify_name(label)}"):
            st.session_state.sidebar_page = label

    nav_button("FX IRF")
    st.sidebar.markdown("### CPI")
    for label in ["Baseline LP", "Asymmetry LP", "Large Appreciations", "Major Groups", "Focus Groups", "Data", "Methodology"]:
        nav_button(label)
    st.sidebar.markdown("### Real Variables")
    for label in ["GDP LP", "Unemployment LP", "Real Indicator Data Audit"]:
        nav_button(label)

    page = st.session_state.sidebar_page
    data_path_text = st.sidebar.text_input("Data file", str(DEFAULT_DATA_PATH))
    return page, Path(data_path_text).expanduser()


def render_methodology_page(data: pd.DataFrame) -> None:
    st.title("Methodology")
    st.caption(
        "A compact guide to the estimands, FX-path choices, uncertainty bands, and interpretation limits used in the dashboard."
    )

    st.subheader("Interpretation")
    st.markdown(
        """
The dashboard estimates conditional local-projection responses of Swiss CPI inflation to observed CHF moves. The coefficient on the exchange-rate variable should be read as a pass-through or dynamic-correlation object under the selected controls, not as a fully structural exchange-rate shock.

All exchange-rate choices use the same sign convention: a positive CHF move means CHF appreciation. For EURCHF and USDCHF, the bilateral rates are transformed so that a fall in CHF per foreign currency is recorded as a positive CHF appreciation.
"""
    )

    st.subheader("Baseline LP")
    st.latex(
        r"""
        y_{t+h} =
        \alpha_h + \beta_h \Delta s_t
        + \theta_h' Z_t
        + \sum_{j=1}^{p} \gamma_{h,j} y_{t-j}
        + \sum_{j=1}^{p} \delta_{h,j} \Delta s_{t-j}
        + \sum_{j=1}^{p} \eta_{h,j}' Z_{t-j}
        + u_{t+h}
        """
    )
    st.markdown(
        """
Here `y` is the selected CPI transformation, `Delta s` is the monthly CHF move, and `Z` contains the selected controls. The displayed response can differ from the estimated dependent variable; when possible, responses are converted algebraically between m/m inflation, cumulative price-level changes, and y/y inflation.
"""
    )

    st.subheader("FX Path Options")
    st.markdown(
        """
- `Standard`: estimates the response to one initial CHF move, while the exchange rate subsequently follows its estimated IRF.
- `Layered 1% move`: first estimates the one-move system, then layers future CHF moves so the nominal CHF path is held at a maintained 1% appreciation. This is a constructed counterfactual path under linear superposition.
- `Forward shocks`: includes future CHF moves in the CPI equation, isolating the initial move conditional on later observed FX changes.
"""
    )

    st.subheader("Asymmetry")
    st.latex(r"\Delta s_t^+ = \max(\Delta s_t, 0), \qquad \Delta s_t^- = -\min(\Delta s_t, 0)")
    st.markdown(
        """
The asymmetry page estimates separate appreciation and depreciation coefficients. Depreciation responses are multiplied by `-1` in the displayed chart, so appreciation and depreciation pass-through can be compared in the same direction. The black dashed line is the corresponding symmetric response.
"""
    )

    st.subheader("Major Groups")
    st.markdown(
        """
The major-group page estimates direct y/y LPs for each SNB CPI major group and overlays the headline CPI response as a black dashed reference. These subgroup IRFs are not CPI-weighted contributions to headline inflation. The `T: Total CPI` panel is the SNB total CPI major-group series; the dashed line is the headline CPI series used elsewhere in the dashboard.

Panel titles show the 2026 BFS LIK basket weights in percent. These weights are metadata for interpretation; they do not rescale the plotted IRFs.
"""
    )

    st.subheader("Uncertainty")
    st.markdown(
        """
Direct LP charts show 90% HAC confidence intervals. Converted-response intervals use a diagonal delta-method approximation across horizons. Layered-path intervals are more approximate: they ignore uncertainty in the estimated FX path used to construct the maintained 1% path and do not account for full cross-horizon covariance.
"""
    )

    st.subheader("Seasonal Adjustment")
    st.markdown(
        """
Seasonally adjusted Swiss CPI indexes are produced in the project pipeline with full-sample STL. This is appropriate for ex-post exploratory analysis, but it is not a real-time seasonal-adjustment procedure.
"""
    )

    with st.expander("Current Data Window", expanded=False):
        st.write(
            {
                "first_month": data["date"].min().strftime("%Y-%m"),
                "last_month": data["date"].max().strftime("%Y-%m"),
                "monthly_rows": int(len(data)),
            }
        )


def render_data_page(data: pd.DataFrame) -> None:
    st.title("Data")
    st.caption("Monthly Swiss CPI, CHF exchange-rate, and external-control panel for the local-projections workflow.")

    min_date = data["date"].min().to_pydatetime()
    max_date = data["date"].max().to_pydatetime()
    selected_dates = st.slider(
        "Sample window",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM",
    )
    start = pd.Timestamp(selected_dates[0])
    end = pd.Timestamp(selected_dates[1])

    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    available_columns = [column for column in CORE_COLUMNS if column in data.columns]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", f"{len(sample):,}")
    col_b.metric("Start", sample["date"].min().strftime("%Y-%m"))
    col_c.metric("End", sample["date"].max().strftime("%Y-%m"))
    col_d.metric("Active columns", f"{len(available_columns)}/{len(CORE_COLUMNS)}")

    tabs = st.tabs(["Sources", "Coverage", "Headline and Focus", "Major Groups", "FX and Controls", "Panel"])

    with tabs[0]:
        st.subheader("Source Documentation")
        st.dataframe(
            pd.DataFrame(SOURCE_ROWS),
            width="stretch",
            hide_index=True,
            column_config={"URL": st.column_config.LinkColumn("Link")},
        )

    with tabs[1]:
        st.subheader("Coverage and Missing Values")
        st.dataframe(coverage_table(sample, available_columns), width="stretch", hide_index=True)
        draw_missingness_heatmap(sample, available_columns)

    with tabs[2]:
        st.subheader("Headline and Focus CPI Series")
        control_a, control_b, control_c = st.columns([1.5, 1, 1])
        price_choice = control_a.selectbox("Series", list(PRICE_SERIES), index=0)
        show_nsa = control_b.toggle("NSA", value=True)
        show_sa = control_c.toggle("SA", value=True)
        nsa_column, sa_column = PRICE_SERIES[price_choice]
        selected_price_columns = []
        if show_nsa:
            selected_price_columns.append(nsa_column)
        if show_sa:
            selected_price_columns.append(sa_column)
        draw_line_chart(sample, selected_price_columns, f"Swiss {price_choice}: NSA vs SA", "Index")

        st.divider()
        draw_line_chart(
            sample,
            [
                "headline_sa_inflation_yoy",
                "core_1_sa_inflation_yoy",
                "core_2_sa_inflation_yoy",
                "fresh_seasonal_sa_inflation_yoy",
                "energy_fuel_sa_inflation_yoy",
            ],
            "Headline, Core, Fresh/Seasonal, and Energy/Fuels Inflation",
            "12-month log change, percent",
        )

    with tabs[3]:
        st.subheader("SNB CPI Major Groups")
        control_a, control_b, control_c = st.columns([1.6, 1, 1])
        group_options = [f"{group['code']}: {group['label']}" for group in MAJOR_GROUPS]
        group_choice = control_a.selectbox(
            "Major group",
            group_options,
            index=0,
        )
        selected_group = MAJOR_GROUPS[group_options.index(group_choice)]
        control_a.caption(f"2026 CPI basket weight: {selected_group['weight_pct']:.3f}%")
        show_group_nsa = control_b.toggle("NSA", value=True, key="major_group_nsa")
        show_group_sa = control_c.toggle("SA", value=True, key="major_group_sa")
        group_columns = []
        if show_group_nsa:
            group_columns.append(selected_group["column"])
        if show_group_sa:
            group_columns.append(f"{selected_group['column']}_sa")
        draw_line_chart(sample, group_columns, f"{selected_group['label']}: NSA vs SA", "Index")

        st.divider()
        group_yoy_columns = [
            column
            for column in [
                f"{selected_group['column']}_nsa_inflation_yoy",
                f"{selected_group['column']}_sa_inflation_yoy",
            ]
            if column in sample.columns
        ]
        draw_line_chart(sample, group_yoy_columns, f"{selected_group['label']}: y/y Inflation", "12-month log change, percent")

        st.divider()
        st.dataframe(
            pd.DataFrame(
                {
                    "code": [group["code"] for group in MAJOR_GROUPS],
                    "label": [group["label"] for group in MAJOR_GROUPS],
                    "weight_pct": [group["weight_pct"] for group in MAJOR_GROUPS],
                    "nsa_column": [group["column"] for group in MAJOR_GROUPS],
                    "sa_column": [f"{group['column']}_sa" for group in MAJOR_GROUPS],
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with tabs[4]:
        st.subheader("Exchange Rate and External Controls")
        chart_choices = st.multiselect(
            "Charts",
            options=list(CONTROL_CHARTS),
            default=[
                "BIS vs SNB CHF NEER level",
                "BIS vs SNB CHF NEER monthly change",
                "Euro area core inflation",
                "Brent oil inflation",
            ],
        )
        for chart_choice in chart_choices:
            columns, title, ylabel = CONTROL_CHARTS[chart_choice]
            if chart_choice == "BIS vs SNB CHF NEER level":
                draw_normalized_line_chart(
                    data,
                    columns,
                    title,
                    ylabel,
                    base_date="2019-12-01",
                    start=start,
                    end=end,
                    labels={"chf_neer": "BIS CHF NEER", "snb_chf_neer": "SNB CHF NEER"},
                )
            else:
                draw_line_chart(sample, columns, title, ylabel)

    with tabs[5]:
        st.subheader("Merged Monthly Panel")
        show_all_columns = st.toggle("Show all columns in CSV", value=False)
        panel_options = [column for column in (data.columns if show_all_columns else available_columns) if column != "date"]
        selected_columns = st.multiselect(
            "Columns",
            options=panel_options,
            default=available_columns,
        )
        st.dataframe(sample.loc[:, ["date", *selected_columns]], width="stretch", hide_index=True)
        st.download_button(
            "Download selected panel",
            data=sample.loc[:, ["date", *selected_columns]].to_csv(index=False).encode("utf-8"),
            file_name="cpi_lp_chf_selected_panel.csv",
            mime="text/csv",
        )


def render_baseline_lp_page(data: pd.DataFrame) -> None:
    st.title("Baseline LP")
    st.caption(
        "Displayed response to a 1 percentage point CHF appreciation shock. "
        "The estimated dependent variable, displayed transformation, and index adjustment can be selected separately."
    )

    min_date = data["date"].min().to_pydatetime()
    max_date = data["date"].max().to_pydatetime()

    with st.container(border=True):
        col_a, col_b, col_c, col_d, col_e, col_f = st.columns([1.2, 0.9, 1.05, 1.15, 1.15, 1.1])
        response_label = col_a.selectbox("Response", BASELINE_LP_RESPONSES, index=0)
        seasonal_adjustment = col_b.selectbox("Index", ["SA", "NSA"], index=0)
        available_shocks = available_exchange_rate_shocks(data)
        shock_label = col_c.selectbox("CHF shock", available_shocks, index=0)
        transform_label = col_d.selectbox("Estimated dependent variable", list(LP_TRANSFORMS), index=0)
        display_transform_label = col_e.selectbox("Displayed response", list(LP_TRANSFORMS), index=0)
        fx_path_label = col_f.selectbox("FX path", ["Standard", "Layered 1% move", "Forward shocks"], index=0)

        col_d, col_e, col_f, col_g = st.columns([1.4, 1, 1, 1])
        selected_dates = col_d.slider(
            "Estimation range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
        )
        controls_label = col_e.selectbox("Controls", list(LP_CONTROL_SETS), index=1)
        horizons = col_f.number_input("Horizon", min_value=1, max_value=60, value=36, step=1)
        lags = col_g.number_input("Lags", min_value=1, max_value=24, value=12, step=1)

    with st.expander("Period Dummies and Event Windows", expanded=False):
        month_options = monthly_options(pd.Timestamp(min_date), pd.Timestamp(max_date))
        selected_presets = st.multiselect(
            "Preset windows",
            options=list(EVENT_PRESETS),
            default=[],
        )
        preset_windows = []
        for name in selected_presets:
            preset_window = event_preset_window(name)
            action = st.selectbox(
                f"{name} action",
                WINDOW_ACTIONS,
                index=WINDOW_ACTIONS.index(default_window_action(preset_window)),
                key=f"preset_action_{slugify_name(name)}",
            )
            preset_windows.append(apply_window_action(preset_window, action))
        custom_count = st.number_input("Custom windows", min_value=0, max_value=3, value=0, step=1)
        custom_windows = []
        for idx in range(int(custom_count)):
            col_a, col_b, col_c, col_d = st.columns([1.4, 1, 1, 1.2])
            custom_name = col_a.text_input("Name", value=f"Custom window {idx + 1}", key=f"custom_event_name_{idx}")
            custom_start = col_b.selectbox(
                "Start",
                options=month_options,
                index=0,
                key=f"custom_event_start_{idx}",
            )
            custom_end = col_c.selectbox(
                "End",
                options=month_options,
                index=0,
                key=f"custom_event_end_{idx}",
            )
            action = col_d.selectbox(
                "Action",
                WINDOW_ACTIONS,
                index=0,
                key=f"custom_event_action_{idx}",
            )
            custom_windows.append(
                apply_window_action(
                    {
                        "name": custom_name,
                        "start": pd.Timestamp(custom_start),
                        "end": pd.Timestamp(custom_end),
                    },
                    action,
                )
            )
        st.caption(
            "`Period dummy` adds one D_t control for the full range. `Dummy for each month` adds one D_t control per selected month. "
            "`Censor` drops the selected months from estimation."
        )

    event_windows = preset_windows
    event_windows.extend(custom_windows)
    event_windows_for_estimation = dummy_windows(event_windows)
    event_windows_for_censoring = censor_windows(event_windows)

    start = pd.Timestamp(selected_dates[0])
    end = pd.Timestamp(selected_dates[1])
    shock_settings = EXCHANGE_RATE_SHOCKS[shock_label]
    shock_name = shock_settings["change"]
    shock_level_response = shock_settings["log_level"]
    include_forward_shocks = fx_path_label == "Forward shocks"
    use_layered_shocks = fx_path_label == "Layered 1% move"
    base_response = LP_RESPONSES[response_label][seasonal_adjustment.lower()]
    response = response_column(base_response, transform_label)
    display_response = response_column(base_response, display_transform_label)
    price_response = response_column(base_response, "Cumulative price difference")
    controls = [control for control in LP_CONTROL_SETS[controls_label] if control in data.columns]
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    sample_after_censoring = apply_censor_windows(sample, event_windows_for_censoring)
    if sample_after_censoring.empty:
        st.error("The selected censor windows remove the entire estimation sample. Shorten the censor range or choose a dummy action.")
        st.stop()

    st.subheader("Estimated Equation")
    equation_symbol = LP_TRANSFORMS[transform_label]["equation_label"]
    st.latex(
        equation_text(
            equation_symbol,
            transform_label,
            controls,
            int(lags),
            include_forward_shocks,
            include_dummies=bool(event_windows_for_estimation),
        )
    )
    st.caption(
        f"Here {equation_symbol} is the selected {response_label} transformation ({transform_label}), "
        f"using the {seasonal_adjustment} index. Delta s is the monthly {shock_label} log move, "
        f"and a positive Delta s means {shock_settings['positive_text']}."
    )
    if fx_path_label == "Standard":
        st.caption("FX path: one initial CHF move followed by the estimated exchange-rate IRF.")
    elif fx_path_label == "Layered 1% move":
        st.caption(
            "FX path: the LP is estimated without forward-shock controls, then future CHF moves are layered so the nominal CHF path stays at 1%."
        )
    else:
        st.caption("FX path: forward CHF moves enter the CPI equation, isolating the initial CHF move conditional on later FX changes.")
    st.caption("Confidence intervals shown in the charts are 90% HAC intervals.")
    if transform_label != display_transform_label:
        if transform_label == "y/y inflation":
            st.caption(
                f"The requested {display_transform_label} display cannot be recovered from y/y estimates, "
                "so it is estimated directly with the same controls and sample."
            )
        else:
            st.caption(
                f"The estimated {transform_label} response is converted to a {display_transform_label} response for display. "
                "Confidence intervals use a diagonal delta-method approximation across horizons."
            )
    if use_layered_shocks:
        st.caption("Layered-response confidence bands are approximate and ignore uncertainty in the estimated FX path used for layering.")
    if controls:
        st.caption(f"Controls in Z: {', '.join(controls)}.")
    else:
        st.caption("Controls in Z: none.")
    if event_windows_for_estimation:
        st.caption("Period dummies in D: " + ", ".join(event_window_label(window) for window in event_windows_for_estimation) + ".")
    if event_windows_for_censoring:
        st.caption("Censored windows are excluded from estimation: " + ", ".join(event_window_label(window) for window in event_windows_for_censoring) + ".")
    if event_windows and not event_windows_for_estimation and not event_windows_for_censoring:
        st.caption("Selected period windows are shaded but not included as controls.")

    cpi_results, chf_results, price_results, reer_results = run_lp(
        data=data,
        response=response,
        display_response=display_response,
        price_response=price_response,
        response_label=response_label,
        transform_label=transform_label,
        display_transform_label=display_transform_label,
        controls=controls,
        shock_name=shock_name,
        shock_level_response=shock_level_response,
        shock_response_label=shock_settings["level_label"],
        start=start,
        end=end,
        horizons=int(horizons),
        lags=int(lags),
        include_forward_shocks=include_forward_shocks,
        event_windows=event_windows_for_estimation,
        censor_windows=event_windows_for_censoring,
    )
    if use_layered_shocks:
        cpi_results = maintained_response_from_irfs(
            cpi_results,
            chf_results,
            "Layered 1% CHF move",
        )
        price_results = maintained_response_from_irfs(
            price_results,
            chf_results,
            "Layered cumulative price response",
        )
        reer_results = implied_reer_response(price_results, chf_results, include_forward_shocks=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows after censoring", f"{len(sample_after_censoring):,}", delta=f"{len(sample_after_censoring) - len(sample):,}")
    col_b.metric("First month", sample_after_censoring["date"].min().strftime("%Y-%m"))
    col_c.metric("Last month", sample_after_censoring["date"].max().strftime("%Y-%m"))
    col_d.metric("FX path", fx_path_label)

    if event_windows:
        st.subheader("Sample Preview")
        draw_sample_preview_chart(sample, response, shock_name, event_windows)
        st.dataframe(
            pd.DataFrame(
                {
                    "window": [event_window_label(window) for window in event_windows],
                    "start": [pd.Timestamp(window["start"]).date() for window in event_windows],
                    "end": [pd.Timestamp(window["end"]).date() for window in event_windows],
                    "treatment": [event_window_type(window) for window in event_windows],
                    "dummy_columns": [event_dummy_column_count(window) for window in event_windows],
                    "censored_months": [
                        len(pd.period_range(window["start"], window["end"], freq="M"))
                        if window.get("treatment") == "Censor"
                        else 0
                        for window in event_windows
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Shaded ranges show selected windows. `Block dummy` and `Separate monthly dummies` keep observations and add controls. "
            "`Censored observations` are dropped from the estimation sample."
        )

    chart_a, chart_b = st.columns(2)
    with chart_a:
        display_method = cpi_results["display_method"].iloc[0] if "display_method" in cpi_results else "direct"
        draw_irf_chart(
            cpi_results,
            f"{response_label} {display_transform_label} Response",
            "Percentage points" if display_transform_label != "Cumulative price difference" else "Percent log points",
        )
        if display_method == "converted":
            st.caption("Converted-response confidence bands use a diagonal delta-method approximation across horizons.")
        elif display_method == "direct_display_estimate":
            st.caption("Displayed response is estimated directly because it cannot be recovered from the selected background estimate.")
    with chart_b:
        draw_neer_reer_chart(chf_results, reer_results, include_forward_shocks or use_layered_shocks, shock_settings)
        if use_layered_shocks:
            st.caption(
                "The nominal CHF path is held at +1 by layering the estimated exchange-rate IRF. "
                "Foreign prices are assumed not to react, so implied REER equals the held-fixed nominal CHF path plus the layered Swiss cumulative price response."
            )
        elif include_forward_shocks:
            st.caption(
                "For this REER calculation, the nominal CHF path is held at +1 because forward CHF shocks are "
                "included in the CPI equation. Foreign prices are assumed not to react, so implied REER equals "
                "the held-fixed nominal CHF path plus the Swiss cumulative price response."
            )
        else:
            st.caption(
                "The nominal CHF path is the estimated CHF persistence response without forward CHF-shock controls. "
                "Foreign prices are assumed not to react, so implied REER equals this nominal CHF path plus the Swiss cumulative price response."
            )

    st.subheader("Specification Comparison")
    store = comparison_store()
    default_label = default_spec_label(
        response_label=response_label,
        shock_label=shock_label,
        transform_label=transform_label,
        display_transform_label=display_transform_label,
        controls_label=controls_label,
        include_forward_shocks=include_forward_shocks,
        start=start,
        end=end,
        event_windows_for_estimation=event_windows_for_estimation,
        event_windows_for_censoring=event_windows_for_censoring,
    )
    default_label = f"{default_label}, {seasonal_adjustment}, {fx_path_label}"
    col_add, col_clear = st.columns([3, 1])
    spec_label = col_add.text_input("Current specification label", value=default_label)
    if col_add.button("Add current specification", type="primary"):
        stored_results = cpi_results.copy()
        stored_results["spec_label"] = spec_label
        store.append(
            {
                "label": spec_label,
                "results": stored_results,
                "response": response_label,
                "estimated": transform_label,
                "displayed": display_transform_label,
                "index": seasonal_adjustment,
                "shock": shock_label,
                "fx_path": fx_path_label,
                "controls": controls_label,
                "forward_shocks": include_forward_shocks,
                "sample": f"{start:%Y-%m} to {end:%Y-%m}",
                "dummies": ", ".join(event_window_label(window) for window in event_windows_for_estimation) or "none",
                "censored": ", ".join(event_window_label(window) for window in event_windows_for_censoring) or "none",
            }
        )
        st.success(f"Added: {spec_label}")
    if col_clear.button("Clear saved specs"):
        store.clear()
        st.success("Cleared saved specifications.")

    if store:
        labels = [spec["label"] for spec in store]
        selected_labels = st.multiselect("Compare saved specifications", labels, default=labels[-min(4, len(labels)):])
        selected_specs = [spec for spec in store if spec["label"] in selected_labels]
        if selected_specs:
            draw_comparison_chart(
                selected_specs,
                f"Displayed CPI {display_transform_label} Response: Specification Comparison",
                "Percentage points" if display_transform_label != "Cumulative price difference" else "Percent log points",
            )
            summary = pd.DataFrame(
                [
                    {
                        "label": spec["label"],
                        "response": spec["response"],
                        "estimated": spec["estimated"],
                        "displayed": spec["displayed"],
                        "index": spec.get("index", ""),
                        "shock": spec.get("shock", "BIS CHF NEER"),
                        "fx_path": spec.get("fx_path", "Forward shocks" if spec.get("forward_shocks", False) else "Standard"),
                        "controls": spec["controls"],
                        "sample": spec["sample"],
                        "dummies": spec["dummies"],
                        "censored": spec.get("censored", "none"),
                    }
                    for spec in selected_specs
                ]
            )
            st.dataframe(summary, width="stretch", hide_index=True)
            combined = pd.concat(
                [spec["results"].assign(spec_label=spec["label"]) for spec in selected_specs],
                ignore_index=True,
            )
            st.download_button(
                "Download comparison results",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="lp_specification_comparison.csv",
                mime="text/csv",
            )
    else:
        st.caption("Add specifications here to compare several displayed CPI responses in one chart.")

    st.subheader("Results")
    table_choice = st.radio(
        "Table",
        [f"Displayed CPI {display_transform_label} response", shock_settings["display"], "Implied REER"],
        horizontal=True,
    )
    if table_choice == f"Displayed CPI {display_transform_label} response":
        table = cpi_results
    elif table_choice == shock_settings["display"]:
        table = chf_results
    else:
        table = reer_results
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download selected results",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{table_choice.lower().replace(' ', '_')}_lp_results.csv",
        mime="text/csv",
    )


def render_large_appreciations_page(data: pd.DataFrame) -> None:
    st.title("Large Appreciations")
    st.caption(
        "Split the selected CHF move into large appreciations and all other FX moves. "
        "The threshold is a percentile among positive CHF appreciation months in the selected estimation sample."
    )

    min_date = data["date"].min().to_pydatetime()
    max_date = data["date"].max().to_pydatetime()

    with st.container(border=True):
        col_a, col_b, col_c, col_d, col_e, col_f = st.columns([1.2, 0.9, 1.05, 1.15, 1.15, 1.05])
        response_label = col_a.selectbox("Response", BASELINE_LP_RESPONSES, index=0, key="large_response")
        seasonal_adjustment = col_b.selectbox("Index", ["SA", "NSA"], index=0, key="large_index")
        available_shocks = available_exchange_rate_shocks(data)
        shock_label = col_c.selectbox("CHF shock", available_shocks, index=0, key="large_shock")
        transform_label = col_d.selectbox("Estimated dependent variable", list(LP_TRANSFORMS), index=0, key="large_transform")
        display_transform_label = col_e.selectbox("Displayed response", list(LP_TRANSFORMS), index=0, key="large_display_transform")
        appreciation_percentile = col_f.slider(
            "Large threshold",
            min_value=50,
            max_value=99,
            value=90,
            step=1,
            format="p%d",
        )

        col_d, col_e, col_f, col_g, col_h = st.columns([1.4, 1, 1, 1, 1.1])
        selected_dates = col_d.slider(
            "Estimation range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
            key="large_dates",
        )
        controls_label = col_e.selectbox("Controls", list(LP_CONTROL_SETS), index=1, key="large_controls")
        horizons = col_f.number_input("Horizon", min_value=1, max_value=60, value=36, step=1, key="large_horizons")
        lags = col_g.number_input("Lags", min_value=1, max_value=24, value=12, step=1, key="large_lags")
        fx_path_label = col_h.selectbox(
            "FX path",
            ["Standard", "Layered 1% move", "Forward shocks"],
            index=0,
            key="large_fx_path",
        )

    with st.expander("Period Dummies and Event Windows", expanded=False):
        month_options = monthly_options(pd.Timestamp(min_date), pd.Timestamp(max_date))
        selected_presets = st.multiselect(
            "Preset windows",
            options=list(EVENT_PRESETS),
            default=[],
            key="large_presets",
        )
        preset_windows = []
        for name in selected_presets:
            preset_window = event_preset_window(name)
            action = st.selectbox(
                f"{name} action",
                WINDOW_ACTIONS,
                index=WINDOW_ACTIONS.index(default_window_action(preset_window)),
                key=f"large_preset_action_{slugify_name(name)}",
            )
            preset_windows.append(apply_window_action(preset_window, action))
        custom_count = st.number_input("Custom windows", min_value=0, max_value=3, value=0, step=1, key="large_custom_count")
        custom_windows = []
        for idx in range(int(custom_count)):
            col_a, col_b, col_c, col_d = st.columns([1.4, 1, 1, 1.2])
            custom_name = col_a.text_input("Name", value=f"Custom window {idx + 1}", key=f"large_custom_event_name_{idx}")
            custom_start = col_b.selectbox(
                "Start",
                options=month_options,
                index=0,
                key=f"large_custom_event_start_{idx}",
            )
            custom_end = col_c.selectbox(
                "End",
                options=month_options,
                index=0,
                key=f"large_custom_event_end_{idx}",
            )
            action = col_d.selectbox(
                "Action",
                WINDOW_ACTIONS,
                index=0,
                key=f"large_custom_event_action_{idx}",
            )
            custom_windows.append(
                apply_window_action(
                    {
                        "name": custom_name,
                        "start": pd.Timestamp(custom_start),
                        "end": pd.Timestamp(custom_end),
                    },
                    action,
                )
            )
        st.caption(
            "`Period dummy` adds one D_t control for the full range. `Dummy for each month` adds one D_t control per selected month. "
            "`Censor` drops the selected months from estimation."
        )

    event_windows = preset_windows
    event_windows.extend(custom_windows)
    event_windows_for_estimation = dummy_windows(event_windows)
    event_windows_for_censoring = censor_windows(event_windows)

    start = pd.Timestamp(selected_dates[0])
    end = pd.Timestamp(selected_dates[1])
    shock_settings = EXCHANGE_RATE_SHOCKS[shock_label]
    shock_name = shock_settings["change"]
    shock_level_response = shock_settings["log_level"]
    include_forward_shocks = fx_path_label == "Forward shocks"
    use_layered_shocks = fx_path_label == "Layered 1% move"
    base_response = LP_RESPONSES[response_label][seasonal_adjustment.lower()]
    response = response_column(base_response, transform_label)
    display_response = response_column(base_response, display_transform_label)
    price_response = response_column(base_response, "Cumulative price difference")
    controls = [control for control in LP_CONTROL_SETS[controls_label] if control in data.columns]
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    sample_after_censoring = apply_censor_windows(sample, event_windows_for_censoring)
    if sample_after_censoring.empty:
        st.error("The selected censor windows remove the entire estimation sample. Shorten the censor range or choose a dummy action.")
        st.stop()

    st.subheader("Estimated Equation")
    equation_symbol = LP_TRANSFORMS[transform_label]["equation_label"]
    st.latex(
        rf"""
        {equation_symbol}_{{t+h}}
        = \alpha_h + \beta^L_h \Delta s^L_t + \beta^O_h \Delta s^O_t
        + \Gamma_h Z_t + \sum_{{j=1}}^{{{int(lags)}}} A_{{h,j}} X_{{t-j}} + D_t'\Theta_h + \varepsilon_{{t+h}}
        """
    )
    st.latex(
        rf"""
        \Delta s^L_t = \Delta s_t \mathbf{{1}}\{{\Delta s_t \ge q_{{{int(appreciation_percentile)}}}(\Delta s_t \mid \Delta s_t>0)\}},
        \quad
        \Delta s^O_t = \Delta s_t - \Delta s^L_t
        """
    )
    st.caption(
        f"Here {equation_symbol} is the selected {response_label} transformation ({transform_label}), "
        f"using the {seasonal_adjustment} index. Delta s is the monthly {shock_label} log move, "
        f"and a positive Delta s means {shock_settings['positive_text']}."
    )
    if fx_path_label == "Standard":
        st.caption("FX path: one initial split FX move followed by the estimated exchange-rate IRF for that split component.")
    elif fx_path_label == "Layered 1% move":
        st.caption(
            "FX path: each split component is first estimated without forward-shock controls, then future FX moves are layered so that component's FX path stays at 1%."
        )
    else:
        st.caption("FX path: forward split FX moves enter the CPI and FX equations, isolating the initial split move conditional on later FX changes.")
    st.caption("Confidence intervals shown in the charts are 90% HAC intervals.")
    if controls:
        st.caption(f"Controls in Z: {', '.join(controls)}.")
    else:
        st.caption("Controls in Z: none.")
    if event_windows_for_estimation:
        st.caption("Period dummies in D: " + ", ".join(event_window_label(window) for window in event_windows_for_estimation) + ".")
    if event_windows_for_censoring:
        st.caption("Censored windows are excluded from estimation: " + ", ".join(event_window_label(window) for window in event_windows_for_censoring) + ".")

    results, fx_results, threshold, large_count, positive_count = run_large_appreciation_lp(
        data=data,
        response=response,
        display_response=display_response,
        shock_level_response=shock_level_response,
        response_label=response_label,
        transform_label=transform_label,
        display_transform_label=display_transform_label,
        controls=controls,
        shock_name=shock_name,
        shock_response_label=shock_settings["level_label"],
        start=start,
        end=end,
        horizons=int(horizons),
        lags=int(lags),
        include_forward_shocks=include_forward_shocks,
        appreciation_percentile=float(appreciation_percentile),
        event_windows=event_windows_for_estimation,
        censor_windows=event_windows_for_censoring,
    )
    symmetric_results, symmetric_fx_results, _, _ = run_lp(
        data=data,
        response=response,
        display_response=display_response,
        price_response=price_response,
        response_label=response_label,
        transform_label=transform_label,
        display_transform_label=display_transform_label,
        controls=controls,
        shock_name=shock_name,
        shock_level_response=shock_level_response,
        shock_response_label=shock_settings["level_label"],
        start=start,
        end=end,
        horizons=int(horizons),
        lags=int(lags),
        include_forward_shocks=include_forward_shocks,
        event_windows=event_windows_for_estimation,
        censor_windows=event_windows_for_censoring,
    )
    if use_layered_shocks:
        results = build_maintained_split_results(results, fx_results)
        symmetric_results = maintained_response_from_irfs(
            symmetric_results,
            symmetric_fx_results,
            "Symmetric IRF",
        )
        fx_results = build_maintained_split_results(fx_results, fx_results)
        symmetric_fx_results = maintained_response_from_irfs(
            symmetric_fx_results,
            symmetric_fx_results,
            "Symmetric IRF",
        )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows after censoring", f"{len(sample_after_censoring):,}", delta=f"{len(sample_after_censoring) - len(sample):,}")
    col_b.metric("Positive appreciation months", f"{positive_count:,}")
    col_c.metric("Large appreciation months", f"{large_count:,}")
    col_d.metric("Threshold", "n/a" if np.isnan(threshold) else f"{threshold:.2f} pp", delta=fx_path_label)

    if results.empty:
        st.error("No positive appreciation months are available in the selected sample.")
        st.stop()

    st.subheader("Large-Appreciation Definition")
    draw_large_appreciation_definition_chart(
        sample_after_censoring,
        shock_level_response,
        shock_name,
        shock_settings["display"],
        threshold,
    )
    st.caption(
        "Green shading marks months classified as large appreciations: positive CHF moves at or above the selected percentile "
        "among all positive appreciation months in the estimation sample after censoring."
    )

    if event_windows:
        st.subheader("Sample Preview")
        draw_sample_preview_chart(sample, response, shock_name, event_windows)
        st.dataframe(
            pd.DataFrame(
                {
                    "window": [event_window_label(window) for window in event_windows],
                    "start": [pd.Timestamp(window["start"]).date() for window in event_windows],
                    "end": [pd.Timestamp(window["end"]).date() for window in event_windows],
                    "treatment": [event_window_type(window) for window in event_windows],
                    "dummy_columns": [event_dummy_column_count(window) for window in event_windows],
                    "censored_months": [
                        len(pd.period_range(window["start"], window["end"], freq="M"))
                        if window.get("treatment") == "Censor"
                        else 0
                        for window in event_windows
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )

    display_method = results["display_method"].iloc[0] if "display_method" in results else "direct"
    draw_asymmetric_irf_chart(
        results,
        symmetric_results,
        f"{response_label} {display_transform_label}: Large Appreciations vs All Else",
        "Percentage points" if display_transform_label != "Cumulative price difference" else "Percent log points",
    )
    if display_method == "converted":
        st.caption("Converted-response confidence bands use a diagonal delta-method approximation across horizons.")
    elif display_method == "direct_display_estimate":
        st.caption("Displayed response is estimated directly because it cannot be recovered from the selected background estimate.")
    elif display_method == "layered":
        st.caption("Layered-response confidence bands are approximate and ignore uncertainty in the estimated FX path used for layering.")

    st.subheader("FX Response")
    draw_asymmetric_irf_chart(
        fx_results,
        symmetric_fx_results,
        f"{shock_settings['display']} Response by Split FX Shock",
        "Percent log points",
    )
    if use_layered_shocks:
        st.caption("Layered FX path is held at +1 for each split component.")
    elif include_forward_shocks:
        st.caption("FX response is conditional on future split FX moves included through the selected horizon.")
    else:
        st.caption("FX response is the estimated persistence of each split shock component.")

    st.subheader("Specification Comparison")
    store = named_comparison_store("large_appreciations")
    default_label = default_spec_label(
        response_label=response_label,
        shock_label=shock_label,
        transform_label=transform_label,
        display_transform_label=display_transform_label,
        controls_label=controls_label,
        include_forward_shocks=False,
        start=start,
        end=end,
        event_windows_for_estimation=event_windows_for_estimation,
        event_windows_for_censoring=event_windows_for_censoring,
    )
    default_label = f"{default_label}, {seasonal_adjustment}, p{int(appreciation_percentile)}"
    default_label = f"{default_label}, {fx_path_label}"
    col_add, col_clear = st.columns([3, 1])
    spec_label = col_add.text_input("Current specification label", value=default_label, key="large_spec_label")
    if col_add.button("Add current specification", type="primary", key="large_add_spec"):
        stored_results = results.copy()
        stored_results["spec_label"] = spec_label
        store.append(
            {
                "label": spec_label,
                "results": stored_results,
                "response": response_label,
                "estimated": transform_label,
                "displayed": display_transform_label,
                "index": seasonal_adjustment,
                "shock": shock_label,
                "threshold": f"p{int(appreciation_percentile)}",
                "fx_path": fx_path_label,
                "controls": controls_label,
                "sample": f"{start:%Y-%m} to {end:%Y-%m}",
                "dummies": ", ".join(event_window_label(window) for window in event_windows_for_estimation) or "none",
                "censored": ", ".join(event_window_label(window) for window in event_windows_for_censoring) or "none",
            }
        )
        st.success(f"Added: {spec_label}")
    if col_clear.button("Clear saved specs", key="large_clear_specs"):
        store.clear()
        st.success("Cleared saved specifications.")

    if store:
        labels = [spec["label"] for spec in store]
        selected_labels = st.multiselect("Compare saved specifications", labels, default=labels[-min(4, len(labels)):], key="large_compare_specs")
        selected_specs = [spec for spec in store if spec["label"] in selected_labels]
        if selected_specs:
            draw_fx_comparison_chart(
                selected_specs,
                f"{display_transform_label} Response: Large-Appreciation Specification Comparison",
                "Percentage points" if display_transform_label != "Cumulative price difference" else "Percent log points",
            )
            summary = pd.DataFrame(
                [
                    {
                        "label": spec["label"],
                        "response": spec["response"],
                        "estimated": spec["estimated"],
                        "displayed": spec["displayed"],
                        "index": spec.get("index", ""),
                        "shock": spec.get("shock", "BIS CHF NEER"),
                        "threshold": spec.get("threshold", ""),
                        "fx_path": spec.get("fx_path", ""),
                        "controls": spec["controls"],
                        "sample": spec["sample"],
                        "dummies": spec["dummies"],
                        "censored": spec.get("censored", "none"),
                    }
                    for spec in selected_specs
                ]
            )
            st.dataframe(summary, width="stretch", hide_index=True)
    else:
        st.caption("Add specifications here to compare several large-appreciation splits in one chart.")

    st.subheader("Results")
    table_choice = st.radio(
        "Table",
        [f"Displayed CPI {display_transform_label} response", shock_settings["display"]],
        horizontal=True,
        key="large_table_choice",
    )
    table = results if table_choice == f"Displayed CPI {display_transform_label} response" else fx_results
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download results",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="large_appreciations_lp_results.csv",
        mime="text/csv",
    )


def render_asymmetry_lp_page(data: pd.DataFrame) -> None:
    st.title("Asymmetry LP")
    st.caption(
        "Separate appreciation and depreciation responses. Forward CHF shocks are not included; "
        "the displayed response is layered afterward so the selected CHF move is maintained at 1%."
    )

    min_date = data["date"].min().to_pydatetime()
    max_date = data["date"].max().to_pydatetime()

    with st.container(border=True):
        col_a, col_b, col_c, col_d, col_e = st.columns([1.2, 0.9, 1.05, 1.15, 1.15])
        response_label = col_a.selectbox("Response", BASELINE_LP_RESPONSES, index=0)
        seasonal_adjustment = col_b.selectbox("Index", ["SA", "NSA"], index=0)
        shock_label = col_c.selectbox("CHF shock", available_exchange_rate_shocks(data), index=0)
        transform_label = col_d.selectbox("Estimated dependent variable", list(LP_TRANSFORMS), index=0)
        display_transform_label = col_e.selectbox("Displayed response", list(LP_TRANSFORMS), index=0)

        col_f, col_g, col_h, col_i = st.columns([1.4, 1, 1, 1])
        selected_dates = col_f.slider(
            "Estimation range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
        )
        controls_label = col_g.selectbox("Controls", list(LP_CONTROL_SETS), index=1)
        horizons = col_h.number_input("Horizon", min_value=1, max_value=60, value=36, step=1)
        lags = col_i.number_input("Lags", min_value=1, max_value=24, value=6, step=1)

    with st.expander("Period Dummies and Event Windows", expanded=False):
        month_options = monthly_options(pd.Timestamp(min_date), pd.Timestamp(max_date))
        selected_presets = st.multiselect(
            "Preset windows",
            options=list(EVENT_PRESETS),
            default=[],
            key="asym_event_presets",
        )
        preset_windows = []
        for name in selected_presets:
            preset_window = event_preset_window(name)
            action = st.selectbox(
                f"{name} action",
                WINDOW_ACTIONS,
                index=WINDOW_ACTIONS.index(default_window_action(preset_window)),
                key=f"asym_preset_action_{slugify_name(name)}",
            )
            preset_windows.append(apply_window_action(preset_window, action))
        custom_count = st.number_input("Custom windows", min_value=0, max_value=3, value=0, step=1, key="asym_custom_count")
        custom_windows = []
        for idx in range(int(custom_count)):
            col_a, col_b, col_c, col_d = st.columns([1.4, 1, 1, 1.2])
            custom_name = col_a.text_input(
                "Name",
                value=f"Custom window {idx + 1}",
                key=f"asym_custom_event_name_{idx}",
            )
            custom_start = col_b.selectbox(
                "Start",
                options=month_options,
                index=0,
                key=f"asym_custom_event_start_{idx}",
            )
            custom_end = col_c.selectbox(
                "End",
                options=month_options,
                index=0,
                key=f"asym_custom_event_end_{idx}",
            )
            action = col_d.selectbox(
                "Action",
                WINDOW_ACTIONS,
                index=0,
                key=f"asym_custom_event_action_{idx}",
            )
            custom_windows.append(
                apply_window_action(
                    {
                        "name": custom_name,
                        "start": pd.Timestamp(custom_start),
                        "end": pd.Timestamp(custom_end),
                    },
                    action,
                )
            )
        st.caption(
            "`Period dummy` adds one D_t control for the full range. `Dummy for each month` adds one D_t control per selected month. "
            "`Censor` drops the selected months from estimation."
        )

    event_windows = preset_windows
    event_windows.extend(custom_windows)
    event_windows_for_estimation = dummy_windows(event_windows)
    event_windows_for_censoring = censor_windows(event_windows)

    start = pd.Timestamp(selected_dates[0])
    end = pd.Timestamp(selected_dates[1])
    shock_settings = EXCHANGE_RATE_SHOCKS[shock_label]
    shock_name = shock_settings["change"]
    shock_level_response = shock_settings["log_level"]
    base_response = LP_RESPONSES[response_label][seasonal_adjustment.lower()]
    response = response_column(base_response, transform_label)
    display_response = response_column(base_response, display_transform_label)
    price_response = response_column(base_response, "Cumulative price difference")
    controls = [control for control in LP_CONTROL_SETS[controls_label] if control in data.columns]
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    sample_after_censoring = apply_censor_windows(sample, event_windows_for_censoring)
    if sample_after_censoring.empty:
        st.error("The selected censor windows remove the entire estimation sample. Shorten the censor range or choose a dummy action.")
        st.stop()

    st.subheader("Estimated Equation")
    equation_symbol = LP_TRANSFORMS[transform_label]["equation_label"]
    dummy_term = r" + \kappa_h' D_t" if event_windows_for_estimation else ""
    control_term = r" + \theta_h' Z_t" if controls else ""
    lagged_control_term = r" + \sum_{j=1}^{p} \eta_{h,j}' Z_{t-j}" if controls else ""
    lhs = rf"\Delta_h {equation_symbol}_{{t+h}}" if LP_TRANSFORMS[transform_label]["cumulative"] else rf"{equation_symbol}_{{t+h}}"
    st.latex(
        rf"{lhs} = \alpha_h + \beta_h^+ \Delta s_t^+ + \beta_h^- \Delta s_t^-"
        f"{control_term}{dummy_term}"
        rf" + \sum_{{j=1}}^{{p}} \gamma_{{h,j}} {equation_symbol}_{{t-j}}"
        rf" + \sum_{{j=1}}^{{p}} \left(\delta_{{h,j}}^+ \Delta s_{{t-j}}^+ + \delta_{{h,j}}^- \Delta s_{{t-j}}^-\right)"
        rf"{lagged_control_term} + u_{{t+h}}, \quad p = {int(lags)}"
    )
    st.latex(r"\Delta s_t^+ = \max(\Delta s_t, 0), \qquad \Delta s_t^- = -\min(\Delta s_t, 0)")
    st.caption(
        f"Delta s is the monthly {shock_label} log move, with positive values meaning {shock_settings['positive_text']}. "
        "Forward CHF shocks are excluded on this page. The maintained-path chart layers the estimated one-shock IRFs so the nominal CHF path stays at 1%."
    )
    st.caption("Confidence intervals are approximate 90% HAC intervals; maintained-path bands ignore uncertainty in the estimated exchange-rate layering weights.")
    if event_windows_for_censoring:
        st.caption("Censored windows are excluded from estimation: " + ", ".join(event_window_label(window) for window in event_windows_for_censoring) + ".")

    response_results = run_asymmetric_lp(
        data=data,
        response=response,
        display_response=display_response,
        response_label=response_label,
        transform_label=transform_label,
        display_transform_label=display_transform_label,
        controls=controls,
        shock_name=shock_name,
        start=start,
        end=end,
        horizons=int(horizons),
        lags=int(lags),
        include_forward_shocks=False,
        event_windows=event_windows_for_estimation,
        censor_windows=event_windows_for_censoring,
    )
    shock_path_results = run_asymmetric_lp(
        data=data,
        response=shock_level_response,
        display_response=shock_level_response,
        response_label=shock_settings["level_label"],
        transform_label="Cumulative price difference",
        display_transform_label="Cumulative price difference",
        controls=controls,
        shock_name=shock_name,
        start=start,
        end=end,
        horizons=int(horizons),
        lags=int(lags),
        include_forward_shocks=False,
        event_windows=event_windows_for_estimation,
        censor_windows=event_windows_for_censoring,
    )
    maintained_results = build_maintained_asymmetry_results(response_results, shock_path_results)
    symmetric_results, symmetric_chf_results, _, _ = run_lp(
        data=data,
        response=response,
        display_response=display_response,
        price_response=price_response,
        response_label=response_label,
        transform_label=transform_label,
        display_transform_label=display_transform_label,
        controls=controls,
        shock_name=shock_name,
        shock_level_response=shock_level_response,
        shock_response_label=shock_settings["level_label"],
        start=start,
        end=end,
        horizons=int(horizons),
        lags=int(lags),
        include_forward_shocks=False,
        event_windows=event_windows_for_estimation,
        censor_windows=event_windows_for_censoring,
    )
    symmetric_maintained_results = maintained_response_from_irfs(
        symmetric_results,
        symmetric_chf_results,
        "Maintained symmetric IRF",
    )
    displayed_maintained_results = pd.concat(
        [maintained_results, symmetric_maintained_results],
        ignore_index=True,
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows after censoring", f"{len(sample_after_censoring):,}", delta=f"{len(sample_after_censoring) - len(sample):,}")
    col_b.metric("First month", sample_after_censoring["date"].min().strftime("%Y-%m"))
    col_c.metric("Last month", sample_after_censoring["date"].max().strftime("%Y-%m"))
    col_d.metric("Forward shocks", "No")

    if event_windows:
        st.subheader("Sample Preview")
        draw_sample_preview_chart(sample, response, shock_name, event_windows)
        st.dataframe(
            pd.DataFrame(
                {
                    "window": [event_window_label(window) for window in event_windows],
                    "start": [pd.Timestamp(window["start"]).date() for window in event_windows],
                    "end": [pd.Timestamp(window["end"]).date() for window in event_windows],
                    "treatment": [event_window_type(window) for window in event_windows],
                    "dummy_columns": [event_dummy_column_count(window) for window in event_windows],
                    "censored_months": [
                        len(pd.period_range(window["start"], window["end"], freq="M"))
                        if window.get("treatment") == "Censor"
                        else 0
                        for window in event_windows
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Shaded ranges show selected windows. `Block dummy` and `Separate monthly dummies` keep observations and add controls. "
            "`Censored observations` are dropped from the estimation sample."
        )

    st.subheader("Maintained 1% CHF Move")
    draw_maintained_asymmetry_chart(
        displayed_maintained_results,
        f"{response_label} {display_transform_label}: Maintained Appreciation vs Depreciation",
        "Percentage points" if display_transform_label != "Cumulative price difference" else "Percent log points",
    )
    st.caption(
        "The depreciation line is multiplied by -1 for visual comparison. "
        "The black dashed line is the maintained symmetric IRF estimated on the same sample and controls."
    )

    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.subheader("Layering Weights")
        draw_maintenance_chart(maintained_results)
    with chart_b:
        st.subheader("FX IRFs")
        draw_asymmetric_irf_chart(
            shock_path_results,
            symmetric_chf_results,
            f"{shock_settings['display']}: One-Shock Appreciation and Depreciation Paths",
            "Percent log points",
        )
        st.caption(
            "These are the estimated nominal CHF paths used to build the maintained 1% appreciation/depreciation responses."
        )
        st.subheader("One-Shock IRFs")
        draw_asymmetric_irf_chart(
            response_results,
            pd.DataFrame(columns=["horizon", "beta"]),
            f"{response_label} {display_transform_label}: One-Shock Asymmetric Responses",
            "Percentage points" if display_transform_label != "Cumulative price difference" else "Percent log points",
        )

    st.subheader("Results")
    table_choice = st.radio(
        "Table",
        ["Maintained response", "One-shock response", "Symmetric response", f"{shock_settings['display']} path", "Layering weights"],
        horizontal=True,
    )
    if table_choice == "Maintained response":
        table = displayed_maintained_results
    elif table_choice == "One-shock response":
        table = response_results
    elif table_choice == "Symmetric response":
        table = symmetric_results
    elif table_choice == f"{shock_settings['display']} path":
        table = shock_path_results
    else:
        table = maintained_results.loc[
            :,
            ["shock_component", "horizon", "maintenance_shock", "maintained_exchange_rate_path"],
        ]
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download selected asymmetry results",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{table_choice.lower().replace(' ', '_')}_asymmetry_results.csv",
        mime="text/csv",
    )


def render_fx_irf_page(data: pd.DataFrame) -> None:
    st.title("FX IRF")
    st.caption(
        "Exchange-rate response to the selected CHF shock. The page follows the baseline LP controls "
        "and estimates either a symmetric or sign-split one-shock FX persistence path."
    )

    min_date = data["date"].min().to_pydatetime()
    max_date = data["date"].max().to_pydatetime()

    with st.container(border=True):
        col_a, col_b, col_c = st.columns([1.05, 1.05, 1.15])
        available_shocks = available_exchange_rate_shocks(data)
        shock_label = col_a.selectbox("CHF shock", available_shocks, index=0, key="fx_irf_shock")
        mode_label = col_b.selectbox("Mode", ["Symmetric", "Asymmetry"], index=0, key="fx_irf_mode")
        controls_label = col_c.selectbox("Controls", list(LP_CONTROL_SETS), index=1, key="fx_irf_controls")

        col_e, col_f, col_g = st.columns([1.6, 1, 1])
        selected_dates = col_e.slider(
            "Estimation range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
            key="fx_irf_range",
        )
        horizons = col_f.number_input("Horizon", min_value=1, max_value=60, value=36, step=1, key="fx_irf_horizon")
        lags = col_g.number_input("Lags", min_value=1, max_value=24, value=12, step=1, key="fx_irf_lags")

    with st.expander("Period Dummies and Event Windows", expanded=False):
        month_options = monthly_options(pd.Timestamp(min_date), pd.Timestamp(max_date))
        selected_presets = st.multiselect(
            "Preset windows",
            options=list(EVENT_PRESETS),
            default=[],
            key="fx_irf_event_presets",
        )
        preset_windows = []
        for name in selected_presets:
            preset_window = event_preset_window(name)
            action = st.selectbox(
                f"{name} action",
                WINDOW_ACTIONS,
                index=WINDOW_ACTIONS.index(default_window_action(preset_window)),
                key=f"fx_irf_preset_action_{slugify_name(name)}",
            )
            preset_windows.append(apply_window_action(preset_window, action))
        custom_count = st.number_input("Custom windows", min_value=0, max_value=3, value=0, step=1, key="fx_irf_custom_count")
        custom_windows = []
        for idx in range(int(custom_count)):
            col_a, col_b, col_c, col_d = st.columns([1.4, 1, 1, 1.2])
            custom_name = col_a.text_input(
                "Name",
                value=f"Custom window {idx + 1}",
                key=f"fx_irf_custom_event_name_{idx}",
            )
            custom_start = col_b.selectbox("Start", options=month_options, index=0, key=f"fx_irf_custom_event_start_{idx}")
            custom_end = col_c.selectbox("End", options=month_options, index=0, key=f"fx_irf_custom_event_end_{idx}")
            action = col_d.selectbox("Action", WINDOW_ACTIONS, index=0, key=f"fx_irf_custom_event_action_{idx}")
            custom_windows.append(
                apply_window_action(
                    {"name": custom_name, "start": pd.Timestamp(custom_start), "end": pd.Timestamp(custom_end)},
                    action,
                )
            )
        st.caption(
            "`Period dummy` adds one D_t control for the full range. `Dummy for each month` adds one D_t control per selected month. "
            "`Censor` drops the selected months from estimation."
        )

    event_windows = preset_windows
    event_windows.extend(custom_windows)
    event_windows_for_estimation = dummy_windows(event_windows)
    event_windows_for_censoring = censor_windows(event_windows)

    start = pd.Timestamp(selected_dates[0])
    end = pd.Timestamp(selected_dates[1])
    shock_settings = EXCHANGE_RATE_SHOCKS[shock_label]
    shock_name = shock_settings["change"]
    shock_level_response = shock_settings["log_level"]
    controls = [control for control in LP_CONTROL_SETS[controls_label] if control in data.columns]
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()
    sample_after_censoring = apply_censor_windows(sample, event_windows_for_censoring)
    if sample_after_censoring.empty:
        st.error("The selected censor windows remove the entire estimation sample. Shorten the censor range or choose a dummy action.")
        st.stop()

    st.subheader("Estimated Equation")
    dummy_term = r" + \kappa_h' D_t" if event_windows_for_estimation else ""
    control_term = r" + \theta_h' Z_t" if controls else ""
    lagged_control_term = r" + \sum_{j=1}^{p} \eta_{h,j}' Z_{t-j}" if controls else ""
    if mode_label == "Symmetric":
        st.latex(
            rf"\Delta_h s_{{t+h}} = \alpha_h + \beta_h \Delta s_t"
            f"{control_term}{dummy_term}"
            rf" + \sum_{{j=1}}^{{p}} \gamma_{{h,j}}\Delta_h s_{{t-j}}"
            rf" + \sum_{{j=1}}^{{p}} \delta_{{h,j}}\Delta s_{{t-j}}"
            rf"{lagged_control_term} + u_{{t+h}}, \quad p = {int(lags)}"
        )
    else:
        st.latex(
            rf"\Delta_h s_{{t+h}} = \alpha_h + \beta_h^+ \Delta s_t^+ + \beta_h^- \Delta s_t^-"
            f"{control_term}{dummy_term}"
            rf" + \sum_{{j=1}}^{{p}} \gamma_{{h,j}}\Delta_h s_{{t-j}}"
            rf" + \sum_{{j=1}}^{{p}} \left(\delta_{{h,j}}^+ \Delta s_{{t-j}}^+ + \delta_{{h,j}}^- \Delta s_{{t-j}}^-\right)"
            rf"{lagged_control_term} + u_{{t+h}}, \quad p = {int(lags)}"
        )
        st.latex(r"\Delta s_t^+ = \max(\Delta s_t, 0), \qquad \Delta s_t^- = -\min(\Delta s_t, 0)")
    st.caption(
        f"Delta s is the monthly {shock_label} log move, with positive values meaning {shock_settings['positive_text']}. "
        "The dependent variable is the cumulative log level of the same selected exchange-rate measure."
    )
    st.caption("The chart shows the estimated one-shock exchange-rate persistence path; no maintained-path construction is imposed.")
    if controls:
        st.caption(f"Controls in Z: {', '.join(controls)}.")
    else:
        st.caption("Controls in Z: none.")
    if event_windows_for_estimation:
        st.caption("Period dummies in D: " + ", ".join(event_window_label(window) for window in event_windows_for_estimation) + ".")
    if event_windows_for_censoring:
        st.caption("Censored windows are excluded from estimation: " + ", ".join(event_window_label(window) for window in event_windows_for_censoring) + ".")

    if mode_label == "Symmetric":
        fx_results, _, _, _ = run_lp(
            data=data,
            response=shock_level_response,
            display_response=shock_level_response,
            price_response=shock_level_response,
            response_label=shock_settings["level_label"],
            transform_label="Cumulative price difference",
            display_transform_label="Cumulative price difference",
            controls=controls,
            shock_name=shock_name,
            shock_level_response=shock_level_response,
            shock_response_label=shock_settings["level_label"],
            start=start,
            end=end,
            horizons=int(horizons),
            lags=int(lags),
            include_forward_shocks=False,
            event_windows=event_windows_for_estimation,
            censor_windows=event_windows_for_censoring,
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Rows after censoring", f"{len(sample_after_censoring):,}", delta=f"{len(sample_after_censoring) - len(sample):,}")
        col_b.metric("First month", sample_after_censoring["date"].min().strftime("%Y-%m"))
        col_c.metric("Last month", sample_after_censoring["date"].max().strftime("%Y-%m"))
        col_d.metric("Mode", mode_label)

        draw_irf_chart(fx_results, f"{shock_settings['display']}: Symmetric FX IRF", "Percent log points")
        table = fx_results
    else:
        one_shock_results = run_asymmetric_lp(
            data=data,
            response=shock_level_response,
            display_response=shock_level_response,
            response_label=shock_settings["level_label"],
            transform_label="Cumulative price difference",
            display_transform_label="Cumulative price difference",
            controls=controls,
            shock_name=shock_name,
            start=start,
            end=end,
            horizons=int(horizons),
            lags=int(lags),
            include_forward_shocks=False,
            event_windows=event_windows_for_estimation,
            censor_windows=event_windows_for_censoring,
        )
        symmetric_results, _, _, _ = run_lp(
            data=data,
            response=shock_level_response,
            display_response=shock_level_response,
            price_response=shock_level_response,
            response_label=shock_settings["level_label"],
            transform_label="Cumulative price difference",
            display_transform_label="Cumulative price difference",
            controls=controls,
            shock_name=shock_name,
            shock_level_response=shock_level_response,
            shock_response_label=shock_settings["level_label"],
            start=start,
            end=end,
            horizons=int(horizons),
            lags=int(lags),
            include_forward_shocks=False,
            event_windows=event_windows_for_estimation,
            censor_windows=event_windows_for_censoring,
        )
        displayed_results = one_shock_results

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Rows after censoring", f"{len(sample_after_censoring):,}", delta=f"{len(sample_after_censoring) - len(sample):,}")
        col_b.metric("First month", sample_after_censoring["date"].min().strftime("%Y-%m"))
        col_c.metric("Last month", sample_after_censoring["date"].max().strftime("%Y-%m"))
        col_d.metric("Mode", mode_label)

        draw_asymmetric_irf_chart(
            displayed_results,
            symmetric_results,
            f"{shock_settings['display']}: Appreciation and Depreciation FX IRFs",
            "Percent log points",
        )
        st.caption("The depreciation line is multiplied by -1 for visual comparison. The black dashed line is the symmetric FX IRF.")
        table = displayed_results

    if event_windows:
        st.subheader("Sample Preview")
        draw_sample_preview_chart(sample, shock_level_response, shock_name, event_windows)

    st.subheader("Specification Comparison")
    store = named_comparison_store("fx_irf")
    default_label = (
        f"{shock_label}, {mode_label}, {controls_label}, "
        f"p={int(lags)}, h={int(horizons)}, {start:%Y-%m} to {end:%Y-%m}"
    )
    if event_windows_for_estimation:
        default_label += ", dummies: " + ", ".join(event_window_label(window) for window in event_windows_for_estimation)
    if event_windows_for_censoring:
        default_label += ", censored: " + ", ".join(event_window_label(window) for window in event_windows_for_censoring)

    col_add, col_clear = st.columns([3, 1])
    spec_label = col_add.text_input("Current FX specification label", value=default_label, key="fx_irf_spec_label")
    if col_add.button("Add current FX specification", type="primary", key="fx_irf_add_spec"):
        stored_results = table.copy()
        stored_results["spec_label"] = spec_label
        store.append(
            {
                "label": spec_label,
                "results": stored_results,
                "shock": shock_label,
                "mode": mode_label,
                "controls": controls_label,
                "sample": f"{start:%Y-%m} to {end:%Y-%m}",
                "lags": int(lags),
                "horizon": int(horizons),
                "dummies": ", ".join(event_window_label(window) for window in event_windows_for_estimation) or "none",
                "censored": ", ".join(event_window_label(window) for window in event_windows_for_censoring) or "none",
            }
        )
        st.success(f"Added: {spec_label}")
    if col_clear.button("Clear saved FX specs", key="fx_irf_clear_specs"):
        store.clear()
        st.success("Cleared saved FX specifications.")

    if store:
        labels = [spec["label"] for spec in store]
        selected_labels = st.multiselect(
            "Compare saved FX specifications",
            labels,
            default=labels[-min(4, len(labels)):],
            key="fx_irf_compare_specs",
        )
        selected_specs = [spec for spec in store if spec["label"] in selected_labels]
        if selected_specs:
            draw_fx_comparison_chart(
                selected_specs,
                "FX IRF Specification Comparison",
                "Percent log points",
            )
            summary = pd.DataFrame(
                [
                    {
                        "label": spec["label"],
                        "shock": spec["shock"],
                        "mode": spec["mode"],
                        "controls": spec["controls"],
                        "sample": spec["sample"],
                        "lags": spec["lags"],
                        "horizon": spec["horizon"],
                        "dummies": spec["dummies"],
                        "censored": spec["censored"],
                    }
                    for spec in selected_specs
                ]
            )
            st.dataframe(summary, width="stretch", hide_index=True)
            combined = pd.concat(
                [spec["results"].assign(spec_label=spec["label"]) for spec in selected_specs],
                ignore_index=True,
            )
            st.download_button(
                "Download FX comparison results",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="fx_irf_specification_comparison.csv",
                mime="text/csv",
            )
    else:
        st.caption("Add FX specifications here to compare several exchange-rate paths in one chart.")

    st.subheader("Results")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download FX IRF results",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="fx_irf_results.csv",
        mime="text/csv",
    )


def render_real_indicator_gdp_page(data: pd.DataFrame) -> None:
    module = real_indicator_module_or_stop()
    real_data = module.safe_load_data(module.DATA_PATH)
    if real_data.empty:
        st.warning("No merged real-indicator data file found yet. Run the GDP dashboard data fetcher.")
        return

    start_date, end_date = module.date_bounds(real_data)
    selected_range = st.slider(
        "Sample window",
        min_value=start_date.to_pydatetime(),
        max_value=end_date.to_pydatetime(),
        value=(start_date.to_pydatetime(), end_date.to_pydatetime()),
        format="YYYY-Q",
        key="real_indicator_gdp_sample",
    )
    start, end = [pd.Timestamp(value).to_period("Q").to_timestamp() for value in selected_range]
    sample = real_data.loc[(real_data["date"] >= start) & (real_data["date"] <= end)].copy()
    module.render_lp_page(sample)


def render_real_indicator_unemployment_page(data: pd.DataFrame) -> None:
    module = real_indicator_module_or_stop()
    monthly_labor = module.safe_load_data(module.MONTHLY_LABOR_PATH)
    module.render_monthly_labor_lp_page(monthly_labor)


def render_real_indicator_data_audit_page(data: pd.DataFrame) -> None:
    module = real_indicator_module_or_stop()
    real_data = module.safe_load_data(module.DATA_PATH)
    metadata = module.safe_load_metadata(module.COMPONENT_METADATA_PATH)
    sources = module.safe_load_metadata(module.SOURCE_METADATA_PATH)
    if real_data.empty:
        st.warning("No merged real-indicator data file found yet. Run the GDP dashboard data fetcher.")
        return

    start_date, end_date = module.date_bounds(real_data)
    selected_range = st.slider(
        "Sample window",
        min_value=start_date.to_pydatetime(),
        max_value=end_date.to_pydatetime(),
        value=(start_date.to_pydatetime(), end_date.to_pydatetime()),
        format="YYYY-Q",
        key="real_indicator_audit_sample",
    )
    start, end = [pd.Timestamp(value).to_period("Q").to_timestamp() for value in selected_range]
    sample = real_data.loc[(real_data["date"] >= start) & (real_data["date"] <= end)].copy()
    module.render_data_audit_page(real_data, sample, metadata, sources)


def render_focus_groups_page(data: pd.DataFrame) -> None:
    st.title("Focus Groups")
    st.caption(
        "Direct y/y inflation LPs for headline CPI, core CPI 1, fresh and seasonal products, and energy and fuels. "
        "Non-headline panels include the headline CPI response as a black dashed reference."
    )

    min_date = data["date"].min().to_pydatetime()
    max_date = data["date"].max().to_pydatetime()

    with st.container(border=True):
        col_a, col_b, col_c, col_d = st.columns([1, 0.9, 1.15, 1.05])
        mode_label = col_a.selectbox("Mode", ["Symmetric", "Asymmetry"], index=0, key="focus_mode")
        seasonal_adjustment = col_b.selectbox("Index", ["SA", "NSA"], index=0, key="focus_index")
        shock_path_label = col_c.selectbox("FX path", ["Normal FX IRF", "Layered 1% move"], index=0, key="focus_fx_path")
        shock_label = col_d.selectbox("CHF shock", available_exchange_rate_shocks(data), index=0, key="focus_shock")
        col_d, col_e, col_f, col_g = st.columns([1.4, 1, 1, 1])
        selected_dates = col_d.slider(
            "Estimation range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
            key="focus_range",
        )
        controls_label = col_e.selectbox("Controls", list(LP_CONTROL_SETS), index=1, key="focus_controls")
        horizons = col_f.number_input("Horizon", min_value=1, max_value=60, value=36, step=1, key="focus_horizon")
        lags = col_g.number_input(
            "Lags",
            min_value=1,
            max_value=24,
            value=12 if mode_label == "Symmetric" else 6,
            step=1,
            key="focus_lags",
        )

    start = pd.Timestamp(selected_dates[0])
    end = pd.Timestamp(selected_dates[1])
    shock_settings = EXCHANGE_RATE_SHOCKS[shock_label]
    shock_name = shock_settings["change"]
    shock_level_response = shock_settings["log_level"]
    use_layered_shocks = shock_path_label == "Layered 1% move"
    controls = [control for control in LP_CONTROL_SETS[controls_label] if control in data.columns]
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()

    st.subheader("Estimated Equation")
    if mode_label == "Symmetric":
        st.latex(
            equation_text(
                r"\pi^{yy}",
                "y/y inflation",
                controls,
                int(lags),
                include_forward_shocks=False,
                include_dummies=False,
            )
        )
    else:
        control_term = r" + \theta_h' Z_t" if controls else ""
        lagged_control_term = r" + \sum_{j=1}^{p} \eta_{h,j}' Z_{t-j}" if controls else ""
        st.latex(
            rf"\pi^{{yy}}_{{g,t+h}} = \alpha_{{g,h}} + \beta_{{g,h}}^+ \Delta s_t^+ + \beta_{{g,h}}^- \Delta s_t^-"
            f"{control_term}"
            rf" + \sum_{{j=1}}^{{p}} \gamma_{{g,h,j}} \pi^{{yy}}_{{g,t-j}}"
            rf" + \sum_{{j=1}}^{{p}} \left(\delta_{{g,h,j}}^+ \Delta s_{{t-j}}^+ + \delta_{{g,h,j}}^- \Delta s_{{t-j}}^-\right)"
            rf"{lagged_control_term} + u_{{g,t+h}}, \quad p = {int(lags)}"
        )
        st.latex(r"\Delta s_t^+ = \max(\Delta s_t, 0), \qquad \Delta s_t^- = -\min(\Delta s_t, 0)")
    st.caption(
        f"The same direct y/y specification is estimated for each focus CPI index using the {seasonal_adjustment} series. "
        f"Delta s is the monthly {shock_label} log move, with positive values meaning {shock_settings['positive_text']}. "
        "Confidence intervals are 90% HAC intervals."
    )
    st.caption(
        "Focus weights are 2026 shares recovered from the SNB plkoprex aggregation identity: "
        "Core 1 84.205%, fresh/seasonal products 10.717%, energy/fuels 5.078%."
    )
    st.caption(
        "Forward CHF shocks are excluded on this page. "
        + (
            "FX path: one initial shock followed by the estimated exchange-rate IRF."
            if not use_layered_shocks
            else "FX path: layered future shocks keep the selected CHF move at 1%."
        )
    )
    if use_layered_shocks:
        st.caption("Layered-response confidence bands are approximate and ignore uncertainty in the estimated FX path used for layering.")
    if mode_label == "Asymmetry":
        st.caption("Asymmetry mode splits appreciation and depreciation shocks. Depreciation responses are multiplied by -1 for visual comparison.")

    missing_groups = [
        group["label"]
        for group in FOCUS_GROUPS
        if (group["column"] if seasonal_adjustment == "NSA" else f"{group['column']}_sa") not in data.columns
    ]
    if missing_groups:
        st.warning(
            "Some focus columns are missing from the data file. Refresh the data pipeline to show all panels: "
            + ", ".join(missing_groups)
        )

    if mode_label == "Symmetric":
        group_results, headline_results = run_focus_group_yoy_lps(
            data=data,
            seasonal_adjustment=seasonal_adjustment,
            controls=controls,
            shock_name=shock_name,
            shock_level_response=shock_level_response,
            start=start,
            end=end,
            horizons=int(horizons),
            lags=int(lags),
            use_layered_shocks=use_layered_shocks,
        )
        one_shock_results = pd.DataFrame()
        shock_path_results = estimate_dashboard_lp(
            data=add_transforms(data).loc[(data["date"] >= start) & (data["date"] <= end)].copy(),
            response=shock_level_response,
            shock_name=shock_name,
            controls=controls,
            config=LPConfig(
                horizons=int(horizons),
                lags=int(lags),
                ci_level=0.9,
                include_forward_shocks=False,
            ),
            response_label=shock_settings["level_label"],
            cumulative_response=True,
        )
    else:
        group_results, headline_results, one_shock_results, shock_path_results = run_focus_group_asymmetry_yoy_lps(
            data=data,
            seasonal_adjustment=seasonal_adjustment,
            controls=controls,
            shock_name=shock_name,
            shock_level_response=shock_level_response,
            start=start,
            end=end,
            horizons=int(horizons),
            lags=int(lags),
            use_layered_shocks=use_layered_shocks,
        )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows in sample", f"{len(sample):,}")
    col_b.metric("First month", sample["date"].min().strftime("%Y-%m"))
    col_c.metric("Last month", sample["date"].max().strftime("%Y-%m"))
    col_d.metric("Panels", f"{group_results['group_code'].nunique() if not group_results.empty else 0}/4")

    st.subheader("Bottom-Up Contributions")
    if mode_label == "Symmetric":
        contribution_data = draw_focus_contribution_chart(group_results, headline_results)
    else:
        appreciation_component = "Maintained +1% CHF appreciation" if use_layered_shocks else "+1pp CHF appreciation"
        depreciation_component = "-1 x maintained 1% CHF depreciation" if use_layered_shocks else "-1 x 1pp CHF depreciation"
        contribution_data = pd.concat(
            [
                draw_focus_contribution_chart(
                    group_results,
                    headline_results,
                    component=appreciation_component,
                    title_suffix="Appreciation",
                ),
                draw_focus_contribution_chart(
                    group_results,
                    headline_results,
                    component=depreciation_component,
                    title_suffix="Depreciation x -1",
                ),
            ],
            ignore_index=True,
        )
    st.caption(
        "Bottom-up IRF is the weighted sum of Core 1, fresh/seasonal products, and energy/fuels IRFs. "
        "Contribution uncertainty uses a diagonal approximation and ignores covariance across component estimates."
    )

    st.subheader("Focus IRFs")
    if mode_label == "Symmetric":
        draw_focus_group_grid(
            group_results,
            headline_results,
            f"Focus CPI Groups: y/y Inflation Response to {shock_label} ({shock_path_label})",
        )
    else:
        draw_focus_group_asymmetry_grid(
            group_results,
            headline_results,
            f"Focus CPI Groups: Asymmetric y/y Responses to {shock_label} ({shock_path_label})",
        )

    st.subheader("FX IRF")
    if mode_label == "Symmetric":
        draw_irf_chart(
            shock_path_results,
            f"{shock_settings['display']} Response",
            "Percent log points",
        )
    else:
        draw_asymmetric_irf_chart(
            shock_path_results,
            pd.DataFrame(columns=["horizon", "beta"]),
            f"{shock_settings['display']}: One-Shock Appreciation and Depreciation Paths",
            "Percent log points",
        )
        st.caption(
            "These are the estimated nominal CHF paths used to build the asymmetric focus-group responses."
        )

    st.subheader("Results")
    table_choice = st.radio(
        "Table",
        (
            ["Focus groups", "Contribution decomposition", "Headline reference"]
            if mode_label == "Symmetric"
            else ["Maintained focus groups", "Contribution decomposition", "One-shock focus groups", "Headline reference", f"{shock_settings['display']} path"]
        ),
        horizontal=True,
        key="focus_table",
    )
    if table_choice in ["Focus groups", "Maintained focus groups"]:
        table = group_results
    elif table_choice == "Contribution decomposition":
        table = contribution_data
    elif table_choice == "One-shock focus groups":
        table = one_shock_results
    elif table_choice == f"{shock_settings['display']} path":
        table = shock_path_results
    else:
        table = headline_results
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download selected focus-group results",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{table_choice.lower().replace(' ', '_')}_focus_group_lp_results.csv",
        mime="text/csv",
    )


def render_major_groups_page(data: pd.DataFrame) -> None:
    st.title("Major Groups")
    st.caption(
        "Direct y/y inflation LPs for the SNB CPI major-group indexes. "
        "Each panel shows the subgroup response, with the headline CPI response as a black dashed reference."
    )

    min_date = data["date"].min().to_pydatetime()
    max_date = data["date"].max().to_pydatetime()

    with st.container(border=True):
        col_a, col_b, col_c, col_d = st.columns([1, 1.15, 0.9, 1.05])
        mode_label = col_a.selectbox("Mode", ["Symmetric", "Asymmetry"], index=0)
        shock_path_label = col_b.selectbox("FX path", ["Normal FX IRF", "Layered 1% move"], index=0)
        seasonal_adjustment = col_c.selectbox("Index", ["SA", "NSA"], index=0)
        shock_label = col_d.selectbox("CHF shock", available_exchange_rate_shocks(data), index=0)
        col_e, col_f, col_g, col_h = st.columns([1.4, 1, 1, 1])
        selected_dates = col_e.slider(
            "Estimation range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
        )
        controls_label = col_f.selectbox("Controls", list(LP_CONTROL_SETS), index=1)
        horizons = col_g.number_input("Horizon", min_value=1, max_value=60, value=36, step=1)
        lags = col_h.number_input("Lags", min_value=1, max_value=24, value=12 if mode_label == "Symmetric" else 6, step=1)

    start = pd.Timestamp(selected_dates[0])
    end = pd.Timestamp(selected_dates[1])
    shock_settings = EXCHANGE_RATE_SHOCKS[shock_label]
    shock_name = shock_settings["change"]
    shock_level_response = shock_settings["log_level"]
    use_layered_shocks = shock_path_label == "Layered 1% move"
    controls = [control for control in LP_CONTROL_SETS[controls_label] if control in data.columns]
    sample = data.loc[(data["date"] >= start) & (data["date"] <= end)].copy()

    st.subheader("Estimated Equation")
    if mode_label == "Symmetric":
        st.latex(
            equation_text(
                r"\pi^{yy}",
                "y/y inflation",
                controls,
                int(lags),
                include_forward_shocks=False,
                include_dummies=False,
            )
        )
    else:
        control_term = r" + \theta_h' Z_t" if controls else ""
        lagged_control_term = r" + \sum_{j=1}^{p} \eta_{h,j}' Z_{t-j}" if controls else ""
        st.latex(
            rf"\pi^{{yy}}_{{g,t+h}} = \alpha_{{g,h}} + \beta_{{g,h}}^+ \Delta s_t^+ + \beta_{{g,h}}^- \Delta s_t^-"
            f"{control_term}"
            rf" + \sum_{{j=1}}^{{p}} \gamma_{{g,h,j}} \pi^{{yy}}_{{g,t-j}}"
            rf" + \sum_{{j=1}}^{{p}} \left(\delta_{{g,h,j}}^+ \Delta s_{{t-j}}^+ + \delta_{{g,h,j}}^- \Delta s_{{t-j}}^-\right)"
            rf"{lagged_control_term} + u_{{g,t+h}}, \quad p = {int(lags)}"
        )
        st.latex(r"\Delta s_t^+ = \max(\Delta s_t, 0), \qquad \Delta s_t^- = -\min(\Delta s_t, 0)")
    st.caption(
        f"The same direct y/y specification is estimated for each SNB major group using the {seasonal_adjustment} index. "
        f"Delta s is the monthly {shock_label} log move, with positive values meaning {shock_settings['positive_text']}. "
        "Confidence intervals are 90% HAC intervals."
    )
    st.caption(
        "Forward CHF shocks are excluded on this page. "
        + (
            "FX path: one initial shock followed by the estimated exchange-rate IRF."
            if not use_layered_shocks
            else "FX path: layered future shocks keep the selected CHF move at 1%."
        )
    )
    if use_layered_shocks:
        st.caption("Layered-response confidence bands are approximate and ignore uncertainty in the estimated FX path used for layering.")
    if mode_label == "Asymmetry":
        st.caption(
            "Asymmetry mode splits appreciation and depreciation shocks. Depreciation responses are multiplied by -1 for visual comparison."
        )

    missing_groups = [
        group["label"]
        for group in MAJOR_GROUPS
        if (group["column"] if seasonal_adjustment == "NSA" else f"{group['column']}_sa") not in data.columns
    ]
    if missing_groups:
        st.warning(
            "Some major-group columns are missing from the data file. Refresh the data pipeline to show all panels: "
            + ", ".join(missing_groups)
        )

    if mode_label == "Symmetric":
        group_results, headline_results = run_major_group_yoy_lps(
            data=data,
            seasonal_adjustment=seasonal_adjustment,
            controls=controls,
            shock_name=shock_name,
            shock_level_response=shock_level_response,
            start=start,
            end=end,
            horizons=int(horizons),
            lags=int(lags),
            use_layered_shocks=use_layered_shocks,
        )
        one_shock_results = pd.DataFrame()
        shock_path_results = pd.DataFrame()
    else:
        group_results, headline_results, one_shock_results, shock_path_results = run_major_group_asymmetry_yoy_lps(
            data=data,
            seasonal_adjustment=seasonal_adjustment,
            controls=controls,
            shock_name=shock_name,
            shock_level_response=shock_level_response,
            start=start,
            end=end,
            horizons=int(horizons),
            lags=int(lags),
            use_layered_shocks=use_layered_shocks,
        )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows in sample", f"{len(sample):,}")
    col_b.metric("First month", sample["date"].min().strftime("%Y-%m"))
    col_c.metric("Last month", sample["date"].max().strftime("%Y-%m"))
    col_d.metric("Panels", f"{group_results['group_code'].nunique() if not group_results.empty else 0}/14")

    st.subheader("Horizon Ranking and Contributions")
    st.caption(
        "Bars compare each subgroup y/y IRF with its CPI-weighted contribution "
        "(2026 BFS LIK weight times subgroup IRF). Point-in-time contributions are percentage points. "
        "The fourth chart shows the implied 36-month price-level effect recovered from y/y responses as h36 + h24 + h12. "
        "All contribution charts exclude the total-CPI panel."
    )
    ranking_component = None
    if "shock_component" in group_results.columns:
        components = list(group_results["shock_component"].dropna().unique())
        if components:
            ranking_component = st.selectbox("Rank component", components, index=0)
    draw_major_group_horizon_bars(group_results, headline_results, component=ranking_component)

    st.subheader("Subgroup IRFs")
    st.caption("Each subgroup panel uses its own y-axis scale.")
    if mode_label == "Symmetric":
        draw_major_group_grid(
            group_results,
            headline_results,
            f"SNB CPI Major Groups: y/y Inflation Response to {shock_label} ({shock_path_label})",
        )
    else:
        draw_major_group_asymmetry_grid(
            group_results,
            headline_results,
            f"SNB CPI Major Groups: Asymmetric y/y Responses to {shock_label} ({shock_path_label})",
        )

    st.subheader("Results")
    if mode_label == "Symmetric":
        table_options = ["Major groups", "Headline reference"]
    else:
        table_options = ["Maintained major groups", "One-shock major groups", "Headline reference", f"{shock_settings['display']} path"]
    table_choice = st.radio("Table", table_options, horizontal=True)
    if table_choice in ["Major groups", "Maintained major groups"]:
        table = group_results
    elif table_choice == "One-shock major groups":
        table = one_shock_results
    elif table_choice == f"{shock_settings['display']} path":
        table = shock_path_results
    else:
        table = headline_results
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download selected major-group results",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{table_choice.lower().replace(' ', '_')}_major_group_lp_results.csv",
        mime="text/csv",
    )


def main() -> None:
    page, data_path = sidebar()
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.stop()

    data = load_data(str(data_path), data_path.stat().st_mtime)
    if page == "Data":
        render_data_page(data)
    elif page == "Methodology":
        render_methodology_page(data)
    elif page == "Baseline LP":
        render_baseline_lp_page(data)
    elif page == "Asymmetry LP":
        render_asymmetry_lp_page(data)
    elif page == "Large Appreciations":
        render_large_appreciations_page(data)
    elif page == "FX IRF":
        render_fx_irf_page(data)
    elif page == "GDP LP":
        render_real_indicator_gdp_page(data)
    elif page == "Unemployment LP":
        render_real_indicator_unemployment_page(data)
    elif page == "Real Indicator Data Audit":
        render_real_indicator_data_audit_page(data)
    elif page == "Major Groups":
        render_major_groups_page(data)
    elif page == "Focus Groups":
        render_focus_groups_page(data)


if __name__ == "__main__":
    main()
