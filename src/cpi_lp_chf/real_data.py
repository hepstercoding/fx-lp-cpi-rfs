from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


BFS_LIK_APP_STATE_URL = (
    "https://dam-api.bfs.admin.ch/hub/api/dam/assets/"
    "orderNr:ds-q-05.02-lik-app-state/master"
)
BIS_EER_ZIP_URL = "https://data.bis.org/static/bulk/WS_EER_csv_flat.zip"
BIS_EER_CACHE_NAME = "WS_EER_csv_flat.zip"
FRED_BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
FRED_USDCHF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSZUS"
ECB_DATA_API_URL = "https://data-api.ecb.europa.eu/service/data"
ECB_EA_CORE_SERIES_KEY = "M.U2.Y.XEF000.4F0.INX"
ECB_EURCHF_SERIES_KEY = "D.CHF.EUR.SP00.A"
SNB_APP_PROPERTIES_URL = "https://data.snb.ch/json/application/properties"
SNB_CUBE_EXPORT_URL = "https://data.snb.ch/json/file/cube"
SNB_UNEMPLOYMENT_CUBE_ID = "amarbma"
SNB_UNEMPLOYMENT_SERIES_CODE = "S1"
SNB_CPI_CUBE_ID = "plkoprex"
SNB_EFFECTIVE_EXCHANGE_RATE_CUBE_ID = "devwkieffim"
SNB_PRODUCT_TYPE_ORIGIN_CUBE_ID = "plkoprart"
SNB_MAJOR_GROUP_CUBE_ID = "plkoprgru"
SNB_CORE1_SERIES_CODE = "K1"
SNB_CORE2_SERIES_CODE = "K2"
SNB_FRESH_SEASONAL_SERIES_CODE = "FP"
SNB_ENERGY_FUEL_SERIES_CODE = "ET"
SNB_GOODS_SERIES_CODE = "T0"
SNB_SERVICES_SERIES_CODE = "T1"
SNB_DOMESTIC_SERIES_CODE = "I"
SNB_IMPORTED_SERIES_CODE = "A"
SNB_MAJOR_GROUPS = [
    ("NG", "major_ng"),
    ("AGT", "major_agt"),
    ("BS", "major_bs"),
    ("WE", "major_we"),
    ("HH", "major_hh"),
    ("G", "major_g"),
    ("V", "major_v"),
    ("N", "major_n"),
    ("FK", "major_fk"),
    ("EU", "major_eu"),
    ("RH", "major_rh"),
    ("VF", "major_vf"),
    ("SWD", "major_swd"),
    ("T", "major_t"),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def seasonally_adjust_index(df: pd.DataFrame, source_column: str, target_column: str) -> pd.DataFrame:
    adjusted = df.copy()
    source = adjusted[source_column].astype(float)
    if source.notna().sum() < 24:
        adjusted[target_column] = np.nan
        return adjusted
    filled = source.interpolate(limit_direction="both")
    result = STL(
        filled,
        period=12,
        seasonal=13,
        trend=25,
        robust=True,
    ).fit()
    adjusted[target_column] = source - result.seasonal
    adjusted.loc[source.isna(), target_column] = np.nan
    return adjusted


def fetch_bfs_cpi_series(base_name: str = "Ewige Reihe") -> pd.DataFrame:
    with urlopen(BFS_LIK_APP_STATE_URL) as response:
        payload = json.load(response)

    series = next((item for item in payload["monthlySeries"] if item["basis"] == base_name), None)
    if series is None:
        available = sorted(item["basis"] for item in payload["monthlySeries"])
        raise ValueError(f"Unknown BFS CPI base '{base_name}'. Available bases: {available}")

    cpi = pd.DataFrame(series["values"])
    cpi["date"] = pd.to_datetime(cpi["indexDate"].astype(str), format="%Y%m%d")
    cpi = cpi.rename(columns={"indexValue": "cpi"})
    return cpi.loc[:, ["date", "cpi"]].sort_values("date").reset_index(drop=True)


def fetch_bis_neer_series(
    basket: str = "B",
    ref_area: str = "CH: Switzerland",
    value_name: str = "chf_neer",
) -> pd.DataFrame:
    cache_path = project_root() / "data" / "raw" / BIS_EER_CACHE_NAME
    if cache_path.exists():
        zipped = io.BytesIO(cache_path.read_bytes())
    else:
        try:
            with urlopen(BIS_EER_ZIP_URL) as response:
                payload = response.read()
        except URLError as exc:
            raise RuntimeError(
                "Could not fetch the BIS NEER bulk file and no local cache was found at "
                f"{cache_path}."
            ) from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
        zipped = io.BytesIO(payload)

    with zipfile.ZipFile(zipped) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as raw_file:
            eer = pd.read_csv(raw_file, low_memory=False)

    mask = (
        (eer["FREQ:Frequency"] == "M: Monthly")
        & (eer["EER_TYPE:Type"] == "N: Nominal")
        & (eer["EER_BASKET:Basket"].str.startswith(f"{basket}:"))
        & (eer["REF_AREA:Reference area"] == ref_area)
    )
    neer = eer.loc[mask, ["TIME_PERIOD:Time period or range", "OBS_VALUE:Observation Value"]].copy()
    neer["date"] = pd.to_datetime(neer["TIME_PERIOD:Time period or range"], format="%Y-%m")
    neer[value_name] = pd.to_numeric(neer["OBS_VALUE:Observation Value"], errors="coerce")
    return neer.loc[:, ["date", value_name]].dropna().sort_values("date").reset_index(drop=True)


def fetch_brent_monthly() -> pd.DataFrame:
    brent = pd.read_csv(FRED_BRENT_URL, parse_dates=["observation_date"])
    brent = brent.rename(columns={"observation_date": "date", "DCOILBRENTEU": "brent_oil"})
    brent["brent_oil"] = pd.to_numeric(brent["brent_oil"], errors="coerce")
    brent = brent.dropna(subset=["brent_oil"])
    monthly = (
        brent.assign(date=lambda df: df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("date", as_index=False)["brent_oil"]
        .mean()
    )
    monthly["brent_oil_inflation"] = 100 * np.log(monthly["brent_oil"] / monthly["brent_oil"].shift(1))
    return monthly.sort_values("date").reset_index(drop=True)


def fetch_usdchf_monthly() -> pd.DataFrame:
    usdchf = pd.read_csv(FRED_USDCHF_URL, parse_dates=["observation_date"])
    usdchf = usdchf.rename(columns={"observation_date": "date", "DEXSZUS": "usd_chf"})
    usdchf["usd_chf"] = pd.to_numeric(usdchf["usd_chf"], errors="coerce")
    usdchf = usdchf.dropna(subset=["usd_chf"])
    monthly = (
        usdchf.assign(date=lambda df: df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("date", as_index=False)["usd_chf"]
        .mean()
    )
    return monthly.sort_values("date").reset_index(drop=True)


def fetch_eurchf_monthly() -> pd.DataFrame:
    url = f"{ECB_DATA_API_URL}/EXR/{ECB_EURCHF_SERIES_KEY}?format=csvdata"
    eurchf = pd.read_csv(url)
    eurchf = eurchf.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "eur_chf"})
    eurchf["date"] = pd.to_datetime(eurchf["date"])
    eurchf["eur_chf"] = pd.to_numeric(eurchf["eur_chf"], errors="coerce")
    eurchf = eurchf.dropna(subset=["eur_chf"])
    monthly = (
        eurchf.assign(date=lambda df: df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("date", as_index=False)["eur_chf"]
        .mean()
    )
    return monthly.sort_values("date").reset_index(drop=True)


def fetch_ea_core_inflation() -> pd.DataFrame:
    url = f"{ECB_DATA_API_URL}/HICP/{ECB_EA_CORE_SERIES_KEY}?format=csvdata"
    core = pd.read_csv(url)
    core = core.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "ea_core_hicp"})
    core["date"] = pd.to_datetime(core["date"], format="%Y-%m")
    core["ea_core_hicp"] = pd.to_numeric(core["ea_core_hicp"], errors="coerce")
    core = core.loc[:, ["date", "ea_core_hicp"]].dropna(subset=["ea_core_hicp"])
    core["ea_core_inflation"] = 100 * np.log(core["ea_core_hicp"] / core["ea_core_hicp"].shift(12))
    return core.sort_values("date").reset_index(drop=True)


def fetch_snb_application_properties() -> dict[str, str]:
    with urlopen(SNB_APP_PROPERTIES_URL) as response:
        payload = json.load(response)
    return {
        "pageViewTime": payload["pageViewTime"],
        "applicationId": payload["applicationId"],
        "environmentId": payload["environmentId"],
        "userName": payload["userName"],
    }


def fetch_snb_cube_series(cube_id: str, series_code: str, value_name: str) -> pd.DataFrame:
    query = {
        "fileType": "CSV",
        "lang": "en",
        "isWarehouse": "false",
        "cubeId": cube_id,
        **fetch_snb_application_properties(),
    }
    request = Request(
        f"{SNB_CUBE_EXPORT_URL}?{urlencode(query)}",
        data=json.dumps({"getAllData": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        raw_csv = response.read().decode("utf-8-sig")

    df = pd.read_csv(io.StringIO(raw_csv), sep=";", quotechar='"', skiprows=3)
    df.columns = [column.strip('"') for column in df.columns]
    df = df.rename(columns={"Date": "date", "D0": "series_code", "Value": value_name})
    df = df.loc[df["series_code"] == series_code, ["date", value_name]].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df.dropna(subset=[value_name]).sort_values("date").reset_index(drop=True)


def fetch_snb_cube_series_map(cube_id: str, series_map: list[tuple[str, str]]) -> pd.DataFrame:
    query = {
        "fileType": "CSV",
        "lang": "en",
        "isWarehouse": "false",
        "cubeId": cube_id,
        **fetch_snb_application_properties(),
    }
    request = Request(
        f"{SNB_CUBE_EXPORT_URL}?{urlencode(query)}",
        data=json.dumps({"getAllData": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        raw_csv = response.read().decode("utf-8-sig")

    df = pd.read_csv(io.StringIO(raw_csv), sep=";", quotechar='"', skiprows=3)
    df.columns = [column.strip('"') for column in df.columns]
    df = df.rename(columns={"Date": "date", "D0": "series_code", "Value": "value"})
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    frames = []
    for series_code, value_name in series_map:
        current = df.loc[df["series_code"] == series_code, ["date", "value"]].rename(columns={"value": value_name})
        current = current.dropna(subset=[value_name]).sort_values("date").reset_index(drop=True)
        frames.append(current)

    merged = None
    for current in frames:
        merged = current if merged is None else merged.merge(current, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def fetch_snb_effective_exchange_rate_index(
    d0: str = "N",
    d1: str = "G",
    d2: str = "I",
    value_name: str = "snb_chf_neer",
) -> pd.DataFrame:
    query = {
        "fileType": "CSV",
        "lang": "en",
        "isWarehouse": "false",
        "cubeId": SNB_EFFECTIVE_EXCHANGE_RATE_CUBE_ID,
        **fetch_snb_application_properties(),
    }
    request = Request(
        f"{SNB_CUBE_EXPORT_URL}?{urlencode(query)}",
        data=json.dumps({"getAllData": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        raw_csv = response.read().decode("utf-8-sig")

    df = pd.read_csv(io.StringIO(raw_csv), sep=";", quotechar='"', skiprows=3)
    df.columns = [column.strip('"') for column in df.columns]
    df = df.rename(columns={"Date": "date", "Value": value_name})
    mask = df["D0"].eq(d0) & df["D1"].eq(d1) & df["D2"].eq(d2)
    out = df.loc[mask, ["date", value_name]].copy()
    out["date"] = pd.to_datetime(out["date"], format="%Y-%m")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out.dropna(subset=[value_name]).sort_values("date").reset_index(drop=True)


def fetch_ch_unemployment_monthly() -> pd.DataFrame:
    return fetch_snb_cube_series(
        cube_id=SNB_UNEMPLOYMENT_CUBE_ID,
        series_code=SNB_UNEMPLOYMENT_SERIES_CODE,
        value_name="ch_unemployment_rate",
    )


def fetch_ch_core_cpi_indices() -> pd.DataFrame:
    core1 = fetch_snb_cube_series(SNB_CPI_CUBE_ID, SNB_CORE1_SERIES_CODE, "core_cpi_1")
    core1 = seasonally_adjust_index(core1, "core_cpi_1", "core_cpi_1_sa")

    core2 = fetch_snb_cube_series(SNB_CPI_CUBE_ID, SNB_CORE2_SERIES_CODE, "core_cpi_2")
    core2 = seasonally_adjust_index(core2, "core_cpi_2", "core_cpi_2_sa")

    return core1.merge(core2, on="date", how="outer").sort_values("date").reset_index(drop=True)


def fetch_ch_energy_fuel_index() -> pd.DataFrame:
    energy = fetch_snb_cube_series(SNB_CPI_CUBE_ID, SNB_ENERGY_FUEL_SERIES_CODE, "energy_fuel")
    return seasonally_adjust_index(energy, "energy_fuel", "energy_fuel_sa")


def fetch_ch_fresh_seasonal_index() -> pd.DataFrame:
    fresh = fetch_snb_cube_series(SNB_CPI_CUBE_ID, SNB_FRESH_SEASONAL_SERIES_CODE, "fresh_seasonal")
    return seasonally_adjust_index(fresh, "fresh_seasonal", "fresh_seasonal_sa")


def fetch_ch_product_type_origin_indices() -> pd.DataFrame:
    series = [
        (SNB_GOODS_SERIES_CODE, "goods"),
        (SNB_SERVICES_SERIES_CODE, "services"),
        (SNB_DOMESTIC_SERIES_CODE, "domestic"),
        (SNB_IMPORTED_SERIES_CODE, "imported"),
    ]
    merged = None
    for code, value_name in series:
        current = fetch_snb_cube_series(SNB_PRODUCT_TYPE_ORIGIN_CUBE_ID, code, value_name)
        current = seasonally_adjust_index(current, value_name, f"{value_name}_sa")
        merged = current if merged is None else merged.merge(current, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def fetch_ch_major_group_indices() -> pd.DataFrame:
    groups = fetch_snb_cube_series_map(SNB_MAJOR_GROUP_CUBE_ID, SNB_MAJOR_GROUPS)
    adjusted = groups.copy()
    for _, value_name in SNB_MAJOR_GROUPS:
        adjusted = seasonally_adjust_index(adjusted, value_name, f"{value_name}_sa")
    return adjusted.sort_values("date").reset_index(drop=True)


def build_real_dataset(
    output_path: str | Path,
    cpi_base_name: str = "Ewige Reihe",
    neer_basket: str = "B",
) -> pd.DataFrame:
    cpi = fetch_bfs_cpi_series(base_name=cpi_base_name)
    cpi = seasonally_adjust_index(cpi, "cpi", "cpi_sa")
    chf_neer = fetch_bis_neer_series(basket=neer_basket)
    snb_chf_neer = fetch_snb_effective_exchange_rate_index()
    eurchf = fetch_eurchf_monthly()
    usdchf = fetch_usdchf_monthly()
    brent = fetch_brent_monthly()
    ea_core = fetch_ea_core_inflation()
    unemployment = fetch_ch_unemployment_monthly()
    core_cpi = fetch_ch_core_cpi_indices()
    fresh_seasonal = fetch_ch_fresh_seasonal_index()
    energy = fetch_ch_energy_fuel_index()
    product_type_origin = fetch_ch_product_type_origin_indices()
    major_groups = fetch_ch_major_group_indices()

    merged = cpi.merge(chf_neer, on="date", how="inner")
    for add_on in (
        snb_chf_neer,
        eurchf,
        usdchf,
        brent,
        ea_core,
        unemployment,
        core_cpi,
        fresh_seasonal,
        energy,
        product_type_origin,
        major_groups,
    ):
        merged = merged.merge(add_on, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged
