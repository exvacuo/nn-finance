"""Plot holdout results for the report: sMAPE per forecast step and example forecasts.

Reads the tables written by utilities.evaluate_predictions, so run that first.
"""

import argparse
import os
from pathlib import Path

from utilities.load_data import DEFAULT_OUTPUT, ROOT

# Keep Matplotlib's cache inside the project, including on restricted machines.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order (validated palette slots 1-4); Naive2 is a neutral reference.
# Distinct markers keep identity readable where yellow has low contrast and in greyscale.
STYLES = {
    "lstm": dict(label="LSTM", color="#2a78d6", marker="o"),
    "mlp": dict(label="MLP", color="#eb6834", marker="s"),
    "ridge_recursive": dict(label="Ridge", color="#1baf7a", marker="^"),
    "arima": dict(label="ARIMA", color="#eda100", marker="D"),
    "naive2": dict(label="Naive2", color="#6b6a66", marker=None, linestyle="--"),
}
INK, MUTED = "#0b0b0b", "#52514e"


def style_axis(axis):
    axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=8)


def plot_horizon(per_horizon):
    """sMAPE at each of the 18 forecast steps, one line per model."""
    figure, axis = plt.subplots(figsize=(6.2, 3.0), layout="constrained")
    for model, style in STYLES.items():
        rows = per_horizon.loc[per_horizon["model"].eq(model)].sort_values("step")
        axis.plot(rows["step"], rows["smape"], linewidth=1.6, markersize=3.5,
                  linestyle=style.get("linestyle", "-"), marker=style["marker"],
                  color=style["color"], label=style["label"])
    axis.set_xticks(range(1, 19))
    axis.set_xlim(0.5, 18.5)
    axis.set_xlabel("Forecast step (months ahead)", color=INK, fontsize=9)
    axis.set_ylabel("sMAPE (%)", color=INK, fontsize=9)
    style_axis(axis)
    axis.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    return figure


def representative_series(per_series, model="lstm"):
    """Series whose sMAPE for the model sits nearest its 25th, 50th and 75th percentiles."""
    scores = per_series.loc[per_series["model"].eq(model)].set_index("series_id")["smape"]
    chosen = []
    for quantile in (0.25, 0.5, 0.75):
        chosen.append(((scores - scores.quantile(quantile)).abs()).idxmin())
    return chosen, scores


def plot_examples(observations, per_forecast, series_ids, scores, context=36):
    """Last `context` training months, the 18 test months and each model's forecast."""
    figure, axes = plt.subplots(len(series_ids), 1, figsize=(6.2, 2.2 * len(series_ids)),
                                layout="constrained", squeeze=False)
    for axis, series_id, label in zip(axes[:, 0], series_ids, ("Lower quartile", "Median",
                                                                "Upper quartile")):
        series = observations.loc[observations["series_id"].eq(series_id)].sort_values("month_index")
        train = series.loc[series["split"].eq("train")].tail(context)
        test = series.loc[series["split"].eq("test")]
        axis.axvspan(test["date"].iloc[0], test["date"].iloc[-1], color="#f1f0ec", zorder=0)
        axis.plot(pd.concat([train, test])["date"], pd.concat([train, test])["value"],
                  color=INK, linewidth=1.4, label="Actual")
        for model, style in STYLES.items():
            rows = per_forecast.loc[per_forecast["model"].eq(model)
                                    & per_forecast["series_id"].eq(series_id)].sort_values("step")
            axis.plot(rows["date"], rows["prediction"], linewidth=1.2, markersize=2.5,
                      linestyle=style.get("linestyle", "-"), marker=style["marker"],
                      color=style["color"], label=style["label"])
        axis.set_title(f"{series_id} ({label.lower()} of LSTM error, sMAPE "
                       f"{scores[series_id]:.1f}%)", fontsize=9, color=INK, loc="left")
        axis.set_ylabel("Value", color=INK, fontsize=8)
        style_axis(axis)
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=6, loc="upper left",
                      bbox_to_anchor=(0, 1.32))
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "data" / "evaluation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    try:
        per_horizon = pd.read_csv(args.evaluation_dir / "per_horizon.csv")
        per_series = pd.read_csv(args.evaluation_dir / "per_series.csv")
        per_forecast = pd.read_csv(args.evaluation_dir / "per_forecast.csv", parse_dates=["date"])
        observations = pd.read_csv(args.input, parse_dates=["date"])
    except FileNotFoundError as error:
        parser.exit(1, f"Error: {error}. Run python -m utilities.evaluate_predictions first.\n")
    missing = set(STYLES) - set(per_horizon["model"])
    if missing:
        parser.exit(1, f"Error: evaluation lacks models {sorted(missing)}.\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_horizon(per_horizon).savefig(args.output_dir / "smape_by_horizon.png", dpi=300)
    series_ids, scores = representative_series(per_series)
    plot_examples(observations, per_forecast, series_ids, scores).savefig(
        args.output_dir / "example_forecasts.png", dpi=300)
    print(f"Representative series: {', '.join(series_ids)}")
    print(f"Saved smape_by_horizon.png and example_forecasts.png to {args.output_dir}")


if __name__ == "__main__":
    main()
