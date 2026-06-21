from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_monthly_panel(path: str | Path) -> pd.DataFrame:
    """Load and minimally validate a monthly CPI/CHF panel."""
    panel = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    required = {"date", "cpi", "chf_neer"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in ("cpi", "chf_neer"):
        if (panel[column] <= 0).any():
            raise ValueError(f"Column '{column}' must be strictly positive to take logs.")

    return panel


def add_baseline_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """Add headline CPI inflation and CHF appreciation measures."""
    out = panel.copy()
    out["log_cpi"] = np.log(out["cpi"])
    out["log_chf_neer"] = np.log(out["chf_neer"])
    out["inflation_mom"] = 100 * out["log_cpi"].diff()
    out["inflation_yoy"] = 100 * (out["log_cpi"] - out["log_cpi"].shift(12))
    out["chf_move"] = 100 * out["log_chf_neer"].diff()
    return out


def prepare_baseline_lp_dataset(
    panel: pd.DataFrame,
    response: str = "inflation_mom",
    max_lags: int = 12,
    max_horizon: int = 36,
) -> pd.DataFrame:
    """Create lead and lag columns for a first local-projection pass."""
    if response not in panel.columns:
        raise ValueError(f"Response column '{response}' not found.")
    if "chf_move" not in panel.columns:
        raise ValueError("Column 'chf_move' not found. Run add_baseline_transforms first.")

    derived: dict[str, pd.Series] = {}
    for horizon in range(max_horizon + 1):
        derived[f"{response}_lead_{horizon}"] = panel[response].shift(-horizon)

    for lag in range(1, max_lags + 1):
        derived[f"{response}_lag_{lag}"] = panel[response].shift(lag)
        derived[f"chf_move_lag_{lag}"] = panel["chf_move"].shift(lag)

    return pd.concat([panel.copy(), pd.DataFrame(derived, index=panel.index)], axis=1)
