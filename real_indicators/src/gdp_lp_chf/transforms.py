from __future__ import annotations

import numpy as np
import pandas as pd


def quarter_start(date: pd.Series) -> pd.Series:
    return pd.to_datetime(date).dt.to_period("Q").dt.to_timestamp()


def monthly_average_to_quarterly(data: pd.DataFrame, value_column: str) -> pd.DataFrame:
    out = data.copy()
    out["date"] = quarter_start(out["date"])
    return out.groupby("date", as_index=False)[value_column].mean().sort_values("date")


def add_log_level_and_growth(
    data: pd.DataFrame,
    columns: list[str],
    prefix: str = "",
) -> pd.DataFrame:
    out = data.copy()
    derived: dict[str, pd.Series] = {}
    for column in columns:
        if column not in out:
            continue
        log_column = f"{prefix}{column}_log_level"
        growth_column = f"{prefix}{column}_growth_qoq"
        yoy_column = f"{prefix}{column}_growth_yoy"
        log_level = 100 * np.log(out[column].astype(float))
        derived[log_column] = log_level
        derived[growth_column] = log_level - log_level.shift(1)
        derived[yoy_column] = log_level - log_level.shift(4)
    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
    return out


def add_exchange_rate_transforms(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    derived: dict[str, pd.Series] = {}
    if "chf_neer" in out:
        chf_neer_log = 100 * np.log(out["chf_neer"].astype(float))
        derived["log_chf_neer_pct"] = chf_neer_log
        derived["chf_neer_appreciation_qoq"] = chf_neer_log - chf_neer_log.shift(1)
    if "snb_chf_neer" in out:
        snb_chf_neer_log = 100 * np.log(out["snb_chf_neer"].astype(float))
        derived["log_snb_chf_neer_pct"] = snb_chf_neer_log
        derived["snb_chf_neer_appreciation_qoq"] = snb_chf_neer_log - snb_chf_neer_log.shift(1)
    if "eur_chf" in out:
        eur_app_log = -100 * np.log(out["eur_chf"].astype(float))
        derived["log_eur_chf_appreciation_pct"] = eur_app_log
        derived["eur_chf_appreciation_qoq"] = eur_app_log - eur_app_log.shift(1)
    if "usd_chf" in out:
        usd_app_log = -100 * np.log(out["usd_chf"].astype(float))
        derived["log_usd_chf_appreciation_pct"] = usd_app_log
        derived["usd_chf_appreciation_qoq"] = usd_app_log - usd_app_log.shift(1)
    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
    return out


def add_monthly_exchange_rate_transforms(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    derived: dict[str, pd.Series] = {}
    if "chf_neer" in out:
        chf_neer_log = 100 * np.log(out["chf_neer"].astype(float))
        derived["log_chf_neer_pct"] = chf_neer_log
        derived["chf_neer_appreciation_mom"] = chf_neer_log - chf_neer_log.shift(1)
    if "snb_chf_neer" in out:
        snb_chf_neer_log = 100 * np.log(out["snb_chf_neer"].astype(float))
        derived["log_snb_chf_neer_pct"] = snb_chf_neer_log
        derived["snb_chf_neer_appreciation_mom"] = snb_chf_neer_log - snb_chf_neer_log.shift(1)
    if "eur_chf" in out:
        eur_app_log = -100 * np.log(out["eur_chf"].astype(float))
        derived["log_eur_chf_appreciation_pct"] = eur_app_log
        derived["eur_chf_appreciation_mom"] = eur_app_log - eur_app_log.shift(1)
    if "usd_chf" in out:
        usd_app_log = -100 * np.log(out["usd_chf"].astype(float))
        derived["log_usd_chf_appreciation_pct"] = usd_app_log
        derived["usd_chf_appreciation_mom"] = usd_app_log - usd_app_log.shift(1)
    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)
    return out
