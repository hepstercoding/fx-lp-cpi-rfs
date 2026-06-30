from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gdp_lp_chf.real_data import build_monthly_labor_dataset, build_real_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and merge the Swiss GDP/CHF quarterly data panel.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "swiss_gdp_macro_real.csv"),
        help="Path for the merged quarterly CSV.",
    )
    parser.add_argument(
        "--metadata-output",
        default=str(PROJECT_ROOT / "data" / "raw" / "seco_component_metadata.csv"),
        help="Path for the SECO component metadata CSV.",
    )
    parser.add_argument(
        "--neer-basket",
        default="B",
        choices=["B", "N"],
        help="BIS nominal effective exchange-rate basket: B for broad, N for narrow.",
    )
    parser.add_argument(
        "--monthly-labor-output",
        default=str(PROJECT_ROOT / "data" / "raw" / "swiss_labor_market_monthly.csv"),
        help="Path for the merged monthly labour-market CSV.",
    )
    args = parser.parse_args()

    data, metadata = build_real_dataset(
        output_path=args.output,
        metadata_output_path=args.metadata_output,
        neer_basket=args.neer_basket,
    )
    print(f"Saved {len(data):,} quarterly rows to {args.output}")
    print(f"Sample: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Saved {len(metadata):,} SECO component metadata rows to {args.metadata_output}")
    print("")
    print(data.tail(8).to_string(index=False))

    labor = build_monthly_labor_dataset(
        output_path=args.monthly_labor_output,
        neer_basket=args.neer_basket,
    )
    print("")
    print(f"Saved {len(labor):,} monthly labour-market rows to {args.monthly_labor_output}")
    print(f"Monthly sample: {labor['date'].min().date()} to {labor['date'].max().date()}")
    print(labor.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
