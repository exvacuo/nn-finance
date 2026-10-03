"""Load M3C's monthly FINANCE series and preserve the competition holdout."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "raw" / "m3_monthly.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "monthly_finance.csv"
METADATA_COLUMNS = [
    "Series", "N", "NF", "Category", "Starting Year", "Starting Month"
]


def prepare_monthly_finance(sheet: pd.DataFrame) -> pd.DataFrame:
    """Convert the M3Month sheet to one row per monthly observation.

    N counts ALL observations, including the NF withheld test observations.
    Values retain their original scale; fit any later preprocessing on train only.
    """
    sheet = sheet.copy()
    sheet.columns = [str(column).strip() for column in sheet.columns]
    missing = set(METADATA_COLUMNS) - set(sheet.columns)
    if missing:
        raise ValueError(f"Missing M3Month columns: {sorted(missing)}")

    finance = sheet.loc[
        sheet["Category"].astype(str).str.strip().str.upper() == "FINANCE"
    ]
    if finance.empty:
        raise ValueError("No FINANCE series found in the monthly data.")
    if finance["Series"].duplicated().any():
        raise ValueError("Duplicate series IDs in the monthly data.")

    frames = []
    for _, row in finance.iterrows():
        series_id = str(row["Series"]).strip()
        n, horizon = int(row["N"]), int(row["NF"])
        if horizon != 18 or n <= horizon:
            raise ValueError(f"{series_id}: expected 18 test months and training data.")
        value_columns = [str(month) for month in range(1, n + 1)]
        if not set(value_columns).issubset(sheet.columns):
            raise ValueError(f"{series_id}: missing observation columns.")
        values = pd.to_numeric(row[value_columns], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{series_id}: missing or non-finite observations.")

        start = pd.Timestamp(
            year=int(row["Starting Year"]), month=int(row["Starting Month"]), day=1
        )
        train_length = n - horizon
        frames.append(pd.DataFrame({
            "series_id": series_id,
            "date": pd.date_range(start, periods=n, freq="MS"),
            "month_index": np.arange(1, n + 1),
            "value": values,
            "split": ["train"] * train_length + ["test"] * horizon,
            "category": "FINANCE",
        }))

    return pd.concat(frames, ignore_index=True)


def load_monthly_finance(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Read either the original workbook or a CSV export of its M3Month sheet."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Input not found: {path}. Pass --input /path/to/M3C.xls."
        )
    if path.suffix.lower() == ".xls":
        try:
            sheet = pd.read_excel(path, sheet_name="M3Month", engine="xlrd")
        except ImportError as error:
            raise ImportError(
                "Reading .xls requires xlrd. Run: python3 -m pip install -r requirements.txt"
            ) from error
    elif path.suffix.lower() == ".csv":
        sheet = pd.read_csv(path)
    else:
        raise ValueError("Input must be M3C.xls or a CSV export of M3Month.")
    return prepare_monthly_finance(sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        data = load_monthly_finance(args.input)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, date_format="%Y-%m-%d")
    print(f"Loaded {data.series_id.nunique()} monthly Finance series ({len(data):,} observations).")
    print(f"Train: {(data.split == 'train').sum():,}; test: {(data.split == 'test').sum():,}")
    print(f"Saved {output}")
    print(f"Example series IDs: {', '.join(data.series_id.unique()[:6])}")


if __name__ == "__main__":
    main()
