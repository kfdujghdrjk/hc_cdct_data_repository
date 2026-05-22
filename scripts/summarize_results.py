from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Summarize pruning experiment results.")
    parser.add_argument(
        "--csv",
        default="results/tables/pruning_results.csv",
        help="Path to result CSV.",
    )
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"Result CSV not found: {path}")

    df = pd.read_csv(path)
    columns = [
        "pruning_amount",
        "dct_ber_vote",
        "hc_space_ber_vote",
        "dct_distortion_var",
        "hc_space_distortion_var",
    ]
    print(df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
