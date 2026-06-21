# CPI Local Projections to CHF Moves

This project estimates local projections for Swiss CPI inflation around movements in the Swiss franc. The first version is intentionally small: get the data into a clean monthly panel, define CHF shocks consistently, and estimate headline CPI responses before adding richer robustness checks.

## Research Question

How does Swiss consumer price inflation respond over monthly horizons after CHF appreciation or depreciation moves?

Initial response variables:

- headline CPI month-over-month inflation
- headline CPI year-over-year inflation
- cumulative CPI price-level response

Initial CHF move measures:

- CHF nominal effective exchange rate changes, where a positive change means CHF appreciation
- optional bilateral CHF moves against EUR and USD once the baseline is stable

## Baseline Specification

For horizon `h`, the starting monthly local projection is:

```text
pi_{t+h} = alpha_h
          + beta_h * chf_move_t
          + lagged inflation controls
          + lagged CHF move controls
          + macro controls
          + error_{t+h}
```

The first pass will keep inference simple with OLS and HAC standard errors. Identification and robustness checks can then be layered in deliberately.

## Project Layout

- `data/raw/`: original downloaded data
- `data/processed/`: cleaned monthly analysis panels
- `scripts/`: command-line workflows for fetching data and running analysis
- `src/cpi_lp_chf/`: reusable data, estimation, and plotting code
- `reports/`: notebooks, markdown notes, and exported reports
- `outputs/`: generated tables and figures
- `tests/`: focused tests for data transforms and estimation helpers

## Step Plan

1. Scaffold the project and define the baseline question.
2. Reuse or adapt the Swiss CPI and CHF data pipeline.
3. Build a first monthly analysis panel.
4. Estimate baseline local projections.
5. Produce first charts and a short interpretation note.
6. Add robustness checks: core CPI, asymmetry, threshold moves, bilateral rates, and subperiods.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Analysis scripts will be added as the data pipeline is built.

## Dashboard

The first dashboard page presents the merged data panel, source documentation, coverage checks, and visual data audit:

```bash
streamlit run app.py
```

For public hosting, see `DEPLOYMENT.md`.

## Refresh Data

Once dependencies are installed, refresh the merged data panel with:

```bash
python scripts/fetch_data.py
```

The script writes `data/raw/swiss_macro_real.csv`, which is the file used by the dashboard.
