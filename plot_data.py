"""Plot M3 monthly Finance series, highlighting the withheld test months."""

import argparse
import os
from pathlib import Path

from load_data import DEFAULT_OUTPUT, ROOT

# Keep Matplotlib's cache inside the project, including on restricted machines.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
import pandas as pd


def plot_series(data: pd.DataFrame, series_ids: list[str]):
    """Give each series its own axis because scales and dates can differ."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(series_ids), 1, figsize=(11, 3.3 * len(series_ids)), squeeze=False,
        layout="constrained",
    )
    for axis, series_id in zip(axes[:, 0], series_ids):
        series = data.loc[data["series_id"] == series_id].sort_values("month_index")
        train = series.loc[series["split"] == "train"]
        test = series.loc[series["split"] == "test"]
        axis.plot(train["date"], train["value"], color="#2563eb", label="Training data")
        if not test.empty:
            # Join the final training point to the first test point.
            connected = pd.concat([train.tail(1), test])
            axis.plot(connected["date"], connected["value"], color="#ea580c", label="Test actuals")
            axis.axvspan(test["date"].iloc[0], test["date"].iloc[-1], color="#ea580c", alpha=0.09)
            axis.axvline(test["date"].iloc[0], color="#64748b", linestyle="--", linewidth=1)
        axis.set_title(f"{series_id} · M3 monthly Finance")
        axis.set_xlabel("Month")
        axis.set_ylabel("Value (original scale)")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT, help="CSV created by load_data.py")
    parser.add_argument("--series", nargs="+", help="IDs to plot, e.g. N2522 N2523")
    parser.add_argument("--count", type=int, default=3, help="Number of series if --series is omitted")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "monthly_finance.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive plot window")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1.")
    if not args.input.expanduser().is_file():
        parser.error("Processed data not found. Run python3 load_data.py first.")

    data = pd.read_csv(args.input.expanduser(), parse_dates=["date"])
    available = data["series_id"].unique().tolist()
    series_ids = list(dict.fromkeys(args.series)) if args.series else available[:args.count]
    unknown = set(series_ids) - set(available)
    if unknown:
        parser.error(f"Unknown series IDs: {sorted(unknown)}")
    if not series_ids:
        parser.error("No series found in the input.")

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plot_series(data, series_ids)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
