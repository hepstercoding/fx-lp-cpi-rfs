from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpi_lp_chf.real_data import build_real_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and merge the CPI/CHF dashboard data panel.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "swiss_macro_real.csv"),
        help="Path for the merged monthly CSV.",
    )
    parser.add_argument(
        "--cpi-base",
        default="Ewige Reihe",
        help="BFS CPI base to use, for example '12.2020=100' or 'Ewige Reihe'.",
    )
    parser.add_argument(
        "--neer-basket",
        default="B",
        choices=["B", "N"],
        help="BIS nominal effective exchange-rate basket: B for broad, N for narrow.",
    )
    args = parser.parse_args()

    data = build_real_dataset(
        output_path=args.output,
        cpi_base_name=args.cpi_base,
        neer_basket=args.neer_basket,
    )

    print(f"Saved {len(data):,} monthly rows to {args.output}")
    print(f"Sample: {data['date'].min().date()} to {data['date'].max().date()}")
    print("")
    print(data.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
