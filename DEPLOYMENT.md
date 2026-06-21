# Deployment

This dashboard is designed to run locally and on Streamlit Community Cloud.

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Select the branch to deploy.
5. Set the main file path to:

```text
app.py
```

6. Deploy the app.

The dashboard reads the committed file:

```text
data/raw/swiss_macro_real.csv
```

To refresh the public dashboard, update the CSV locally with `scripts/fetch_data.py`, commit the changed data file, and push to GitHub.

## Repository Checklist

- `app.py` is at the repository root.
- `requirements.txt` includes all Python dependencies.
- `runtime.txt` pins the hosted Python version.
- `.streamlit/config.toml` is committed.
- `data/raw/swiss_macro_real.csv` is committed.
- `.venv/`, `.cache/`, and generated Python caches are ignored.
