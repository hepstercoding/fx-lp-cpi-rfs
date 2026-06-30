from __future__ import annotations

import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from gdp_lp_chf.transforms import (
    add_exchange_rate_transforms,
    add_log_level_and_growth,
    add_monthly_exchange_rate_transforms,
    monthly_average_to_quarterly,
)


SECO_GDP_CSV_URL = "https://scheduler.swissdatas.ch/scheduled/ch-seco-gdp.csv"
SECO_GDP_JSON_URL = "https://scheduler.swissdatas.ch/scheduled/ch-seco-gdp.json"
BIS_EER_ZIP_URL = "https://data.bis.org/static/bulk/WS_EER_csv_flat.zip"
BIS_EER_CACHE_NAME = "WS_EER_csv_flat.zip"
ECB_DATA_API_URL = "https://data-api.ecb.europa.eu/service/data"
ECB_EURCHF_SERIES_KEY = "D.CHF.EUR.SP00.A"
FRED_USDCHF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSZUS"
FRED_BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
FRED_EA_REAL_GDP_SERIES = "CLV10MNACB1GQSCAEA20Q"
FRED_EA_REAL_GDP_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_EA_REAL_GDP_SERIES}"
SNB_APP_PROPERTIES_URL = "https://data.snb.ch/json/application/properties"
SNB_CHART_FILE_URL = "https://data.snb.ch/json/file/chart"
SNB_CUBE_FILE_URL = "https://data.snb.ch/json/file/cube"
SNB_OUTPUT_GAP_CHART_ID = "snbprodluch"
SNB_OUTPUT_GAP_CHART_URL = f"https://data.snb.ch/en/topics/snb/chart/{SNB_OUTPUT_GAP_CHART_ID}"
SNB_LABOUR_MARKET_CUBE_ID = "amarbma"
SNB_LABOUR_MARKET_CUBE_URL = (
    "https://data.snb.ch/en/topics/uvo/cube/amarbma?"
    "fromDate=1953-05&dimSel=D0(S0,S1,RS,E)&dimChartSel=D0(S0,S1,RS)"
)
SNB_EFFECTIVE_EXCHANGE_RATE_CUBE_ID = "devwkieffim"
SNB_EFFECTIVE_EXCHANGE_RATE_CUBE_URL = "https://data.snb.ch/en/topics/ziredev/cube/devwkieffim"


PROJECT_ROOT = Path(__file__).resolve().parents[2]


SECO_COMPONENT_GROUPS = {
    "headline": ["gdp"],
    "expenditure": [
        "cons",
        "cons_priv",
        "cons_gov",
        "inv",
        "inv_fixed",
        "inv_gfcf",
        "inv_constr",
        "exp",
        "exp_good",
        "exp_serv",
        "imp",
        "imp_good",
        "imp_serv",
        "stocks",
    ],
    "value_added": [
        "gva",
        "agric",
        "mining",
        "manu",
        "chem_pharm",
        "manu_other",
        "energy",
        "constr",
        "trade",
        "trade_retail",
        "transp",
        "hotel",
        "com",
        "finance",
        "insur",
        "findl",
        "admin",
        "edu",
        "health",
        "entertain",
        "other_serv",
        "hh_prod",
    ],
}


def _read_csv_url(url: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(url, **kwargs)


def fetch_seco_gdp_metadata() -> dict:
    with urlopen(SECO_GDP_JSON_URL) as response:
        return json.load(response)


def _label_map(metadata: dict, key: str, language: str = "en") -> dict[str, str]:
    labels = metadata.get("labels", {}).get(key, {})
    return {code: values.get(language, values.get("de", code)) for code, values in labels.items()}


def fetch_seco_real_cssa_components() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _read_csv_url(SECO_GDP_CSV_URL, parse_dates=["date"])
    metadata = fetch_seco_gdp_metadata()
    structure_labels = _label_map(metadata, "structure")
    type_labels = _label_map(metadata, "type")
    seas_adj_labels = _label_map(metadata, "seas_adj")

    wanted = sorted({item for values in SECO_COMPONENT_GROUPS.values() for item in values})
    filtered = raw.loc[
        (raw["type"] == "real") & (raw["seas_adj"] == "cssa") & (raw["structure"].isin(wanted))
    ].copy()
    filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")

    available = sorted(filtered["structure"].dropna().unique())
    wide = (
        filtered.pivot_table(index="date", columns="structure", values="value", aggfunc="first")
        .reset_index()
        .sort_values("date")
    )
    wide.columns.name = None
    wide = add_log_level_and_growth(wide, [column for column in wide.columns if column != "date"])

    info_rows = []
    for group, structures in SECO_COMPONENT_GROUPS.items():
        for structure in structures:
            info_rows.append(
                {
                    "group": group,
                    "structure": structure,
                    "label": structure_labels.get(structure, structure),
                    "type": "real",
                    "type_label": type_labels.get("real", "real"),
                    "seas_adj": "cssa",
                    "seas_adj_label": seas_adj_labels.get(
                        "cssa", "seasonally, calendar and sports event adjusted"
                    ),
                    "available": structure in available,
                    "source": "SECO machine-readable quarterly national accounts",
                    "url": SECO_GDP_CSV_URL,
                }
            )
    info = pd.DataFrame(info_rows)
    return wide, info


def fetch_bis_neer_monthly(
    basket: str = "B",
    ref_area: str = "CH: Switzerland",
    value_name: str = "chf_neer",
) -> pd.DataFrame:
    cache_path = PROJECT_ROOT / "data" / "raw" / BIS_EER_CACHE_NAME
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


def fetch_eurchf_monthly() -> pd.DataFrame:
    url = f"{ECB_DATA_API_URL}/EXR/{ECB_EURCHF_SERIES_KEY}?format=csvdata"
    eurchf = _read_csv_url(url)
    eurchf = eurchf.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "eur_chf"})
    eurchf["date"] = pd.to_datetime(eurchf["date"])
    eurchf["eur_chf"] = pd.to_numeric(eurchf["eur_chf"], errors="coerce")
    monthly = (
        eurchf.dropna(subset=["eur_chf"])
        .assign(date=lambda df: df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("date", as_index=False)["eur_chf"]
        .mean()
    )
    return monthly.sort_values("date").reset_index(drop=True)


def fetch_usdchf_monthly() -> pd.DataFrame:
    usdchf = _read_csv_url(FRED_USDCHF_URL, parse_dates=["observation_date"])
    usdchf = usdchf.rename(columns={"observation_date": "date", "DEXSZUS": "usd_chf"})
    usdchf["usd_chf"] = pd.to_numeric(usdchf["usd_chf"], errors="coerce")
    monthly = (
        usdchf.dropna(subset=["usd_chf"])
        .assign(date=lambda df: df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("date", as_index=False)["usd_chf"]
        .mean()
    )
    return monthly.sort_values("date").reset_index(drop=True)


def fetch_brent_quarterly() -> pd.DataFrame:
    brent = _read_csv_url(FRED_BRENT_URL, parse_dates=["observation_date"])
    brent = brent.rename(columns={"observation_date": "date", "DCOILBRENTEU": "brent_oil"})
    brent["brent_oil"] = pd.to_numeric(brent["brent_oil"], errors="coerce")
    quarterly = monthly_average_to_quarterly(brent.dropna(subset=["brent_oil"]), "brent_oil")
    quarterly["brent_oil_change_qoq"] = 100 * np.log(quarterly["brent_oil"] / quarterly["brent_oil"].shift(1))
    return quarterly


def fetch_ea_real_gdp_quarterly() -> pd.DataFrame:
    ea_gdp = _read_csv_url(FRED_EA_REAL_GDP_URL, parse_dates=["observation_date"])
    ea_gdp = ea_gdp.rename(columns={"observation_date": "date", FRED_EA_REAL_GDP_SERIES: "ea_real_gdp"})
    ea_gdp["date"] = pd.to_datetime(ea_gdp["date"]).dt.to_period("Q").dt.to_timestamp()
    ea_gdp["ea_real_gdp"] = pd.to_numeric(ea_gdp["ea_real_gdp"], errors="coerce")
    ea_gdp = ea_gdp.loc[:, ["date", "ea_real_gdp"]].dropna().sort_values("date")
    ea_gdp["ea_real_gdp_growth_qoq"] = 100 * np.log(ea_gdp["ea_real_gdp"] / ea_gdp["ea_real_gdp"].shift(1))
    return ea_gdp.reset_index(drop=True)


def fetch_snb_application_properties() -> dict[str, str]:
    with urlopen(SNB_APP_PROPERTIES_URL) as response:
        payload = json.load(response)
    return {
        "pageViewTime": payload["pageViewTime"],
        "applicationId": payload["applicationId"],
        "environmentId": payload["environmentId"],
        "userName": payload["userName"],
    }


def _xlsx_rows(payload: bytes) -> list[list[str]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", ns):
                texts = [node.text or "" for node in item.findall(".//x:t", ns)]
                shared_strings.append("".join(texts))

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in sheet.findall(".//x:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("x:c", ns):
                ref = cell.attrib.get("r", "")
                column_letters = "".join(char for char in ref if char.isalpha())
                column_index = 0
                for char in column_letters:
                    column_index = column_index * 26 + (ord(char.upper()) - ord("A") + 1)
                value_node = cell.find("x:v", ns)
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                if column_index:
                    values[column_index - 1] = value
            if values:
                rows.append([values.get(index, "") for index in range(max(values) + 1)])
        return rows


def fetch_snb_output_gap_quarterly() -> pd.DataFrame:
    query = {
        "lang": "en",
        "fileType": "xlsx",
        **fetch_snb_application_properties(),
    }
    request = Request(
        f"{SNB_CHART_FILE_URL}?{urlencode(query)}",
        data=json.dumps({"chartId": SNB_OUTPUT_GAP_CHART_ID}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        rows = _xlsx_rows(response.read())

    header_index = next(
        index
        for index, row in enumerate(rows)
        if len(row) >= 4 and row[1] == "Production function" and row[2] == "Hodrick-Prescott filter"
    )
    records = []
    for row in rows[header_index + 1 :]:
        if len(row) < 4 or "-Q" not in row[0]:
            continue
        year, quarter = row[0].split("-Q")
        records.append(
            {
                "date": pd.Period(f"{year}Q{quarter}", freq="Q").to_timestamp(),
                "snb_output_gap": pd.to_numeric(row[1], errors="coerce"),
                "snb_output_gap_hp": pd.to_numeric(row[2], errors="coerce"),
                "snb_output_gap_multivariate": pd.to_numeric(row[3], errors="coerce"),
            }
        )
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


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
        f"{SNB_CUBE_FILE_URL}?{urlencode(query)}",
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


def fetch_snb_labour_market_monthly() -> pd.DataFrame:
    query = {
        "lang": "en",
        "fileType": "CSV",
        "isWarehouse": "false",
        "cubeId": SNB_LABOUR_MARKET_CUBE_ID,
        **fetch_snb_application_properties(),
    }
    request = Request(
        f"{SNB_CUBE_FILE_URL}?{urlencode(query)}",
        data=json.dumps({"getAllData": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        raw = response.read().decode("utf-8-sig")

    records = []
    for line in raw.splitlines():
        parts = [part.strip().strip('"') for part in line.split(";")]
        if len(parts) < 3 or not parts[0][:4].isdigit():
            continue
        records.append({"date": parts[0], "code": parts[1], "value": parts[2]})

    long = pd.DataFrame(records)
    if long.empty:
        return pd.DataFrame()
    long["date"] = pd.to_datetime(long["date"], format="%Y-%m")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    selected = long.loc[long["code"].isin(["S0", "S1", "RS", "E"])].copy()
    wide = selected.pivot_table(index="date", columns="code", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "S0": "registered_unemployed_sa",
            "S1": "unemployment_rate_sa",
            "RS": "registered_job_seekers",
            "E": "labour_force",
        }
    ).sort_values("date")

    if {"registered_job_seekers", "labour_force"}.issubset(wide.columns):
        wide["job_seeker_rate"] = 100 * wide["registered_job_seekers"] / wide["labour_force"]

    derived: dict[str, pd.Series] = {}
    if "registered_unemployed_sa" in wide:
        log_unemp = 100 * np.log(wide["registered_unemployed_sa"].where(wide["registered_unemployed_sa"] > 0))
        derived["log_registered_unemployed_sa_pct"] = log_unemp
        derived["registered_unemployed_sa_change_mom"] = log_unemp - log_unemp.shift(1)
    if "registered_job_seekers" in wide:
        log_seekers = 100 * np.log(wide["registered_job_seekers"].where(wide["registered_job_seekers"] > 0))
        derived["log_registered_job_seekers_pct"] = log_seekers
        derived["registered_job_seekers_change_mom"] = log_seekers - log_seekers.shift(1)
    if derived:
        wide = pd.concat([wide, pd.DataFrame(derived, index=wide.index)], axis=1)
    return wide.reset_index(drop=True)


def build_monthly_labor_dataset(
    output_path: str | Path,
    neer_basket: str = "B",
) -> pd.DataFrame:
    labour = fetch_snb_labour_market_monthly()
    chf_neer = fetch_bis_neer_monthly(basket=neer_basket)
    snb_chf_neer = fetch_snb_effective_exchange_rate_index()
    eurchf = fetch_eurchf_monthly()
    usdchf = fetch_usdchf_monthly()

    merged = labour.merge(chf_neer, on="date", how="left")
    for add_on in (snb_chf_neer, eurchf, usdchf):
        merged = merged.merge(add_on, on="date", how="left")
    merged = add_monthly_exchange_rate_transforms(merged).sort_values("date").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged




def build_real_dataset(
    output_path: str | Path,
    metadata_output_path: str | Path | None = None,
    neer_basket: str = "B",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seco, metadata = fetch_seco_real_cssa_components()
    chf_neer_q = monthly_average_to_quarterly(fetch_bis_neer_monthly(basket=neer_basket), "chf_neer")
    snb_chf_neer_q = monthly_average_to_quarterly(fetch_snb_effective_exchange_rate_index(), "snb_chf_neer")
    eurchf_q = monthly_average_to_quarterly(fetch_eurchf_monthly(), "eur_chf")
    usdchf_q = monthly_average_to_quarterly(fetch_usdchf_monthly(), "usd_chf")
    brent_q = fetch_brent_quarterly()
    ea_gdp_q = fetch_ea_real_gdp_quarterly()
    output_gap = fetch_snb_output_gap_quarterly()

    merged = seco.merge(chf_neer_q, on="date", how="left")
    for add_on in (snb_chf_neer_q, eurchf_q, usdchf_q, brent_q, ea_gdp_q, output_gap):
        merged = merged.merge(add_on, on="date", how="left")
    merged = add_exchange_rate_transforms(merged).sort_values("date").reset_index(drop=True)

    source_rows = [
        {
            "variable": "Swiss real GDP and components",
            "columns": "SECO structure codes plus *_log_level and *_growth_* transforms",
            "source": "SECO machine-readable quarterly national accounts",
            "url": SECO_GDP_CSV_URL,
            "notes": "Filtered to type=real and seas_adj=cssa: seasonally, calendar and sports event adjusted.",
        },
        {
            "variable": "CHF NEER",
            "columns": "chf_neer, chf_neer_appreciation_qoq",
            "source": "BIS effective exchange rates",
            "url": "https://data.bis.org/topics/EER",
            "notes": "Monthly nominal broad basket averaged to quarters; higher values mean CHF appreciation.",
        },
        {
            "variable": "EURCHF",
            "columns": "eur_chf, eur_chf_appreciation_qoq",
            "source": "ECB Data Portal, EXR.D.CHF.EUR.SP00.A",
            "url": "https://data.ecb.europa.eu/",
            "notes": "Monthly average of daily CHF per EUR; transformed so positive change means CHF appreciation.",
        },
        {
            "variable": "USDCHF",
            "columns": "usd_chf, usd_chf_appreciation_qoq",
            "source": "FRED daily USDCHF series DEXSZUS",
            "url": "https://fred.stlouisfed.org/series/DEXSZUS",
            "notes": "Daily CHF per USD averaged to months, then quarters; transformed so positive change means CHF appreciation.",
        },
        {
            "variable": "SNB CHF NEER",
            "columns": "snb_chf_neer, snb_chf_neer_appreciation_qoq",
            "source": "SNB data portal cube devwkieffim, code {N,G,I}",
            "url": SNB_EFFECTIVE_EXCHANGE_RATE_CUBE_URL,
            "notes": "Nominal overall effective exchange-rate index, December 2000 = 100, monthly average; higher values mean CHF appreciation.",
        },
        {
            "variable": "Euro area real GDP",
            "columns": "ea_real_gdp, ea_real_gdp_growth_qoq",
            "source": f"FRED, Eurostat euro-area real GDP series {FRED_EA_REAL_GDP_SERIES}",
            "url": f"https://fred.stlouisfed.org/series/{FRED_EA_REAL_GDP_SERIES}",
            "notes": "Quarterly chain-linked euro area GDP, used as foreign-demand control.",
        },
        {
            "variable": "Brent oil",
            "columns": "brent_oil_change_qoq",
            "source": "FRED daily Brent series DCOILBRENTEU",
            "url": "https://fred.stlouisfed.org/series/DCOILBRENTEU",
            "notes": "Daily prices averaged to quarters, then log-differenced.",
        },
        {
            "variable": "SNB output gap",
            "columns": "snb_output_gap, snb_output_gap_hp, snb_output_gap_multivariate",
            "source": "SNB data portal chart snbprodluch",
            "url": SNB_OUTPUT_GAP_CHART_URL,
            "notes": "Baseline column uses the production-function estimate; HP and multivariate variants are retained.",
        },
    ]
    source_metadata = pd.DataFrame(source_rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    if metadata_output_path is not None:
        metadata_path = Path(metadata_output_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path, index=False)
        source_metadata.to_csv(metadata_path.with_name("source_metadata.csv"), index=False)

    return merged, metadata
