"""Evaluate saved 18-month forecasts using the five M3 Appendix A measures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.preprocessing import PERIOD, seasonal_indices
from utilities.load_data import DEFAULT_OUTPUT, HORIZON, ROOT

METRICS = ("smape", "median_sape", "average_rank", "percentage_better", "median_rae")
GROUPS = {"1-18": (1, 18), "1-3": (1, 3), "4-12": (4, 12),
          "13-18": (13, 18), "1-6": (1, 6), "1-12": (1, 12)}


def require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}.")
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{label}: required columns contain missing values.")


def integer_column(frame, column, label):
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise ValueError(f"{label}: {column} must contain finite integers.")
    frame[column] = values.astype(np.int64)


def load_observations(path):
    """Validate the competition split and return histories and ordered test rows."""
    data = pd.read_csv(path, dtype={"series_id": "string"})
    require_columns(data, ("series_id", "month_index", "date", "value", "split"), str(path))
    if data.empty or not data["split"].isin(["train", "test"]).all():
        raise ValueError("Observations must contain train and test rows only.")
    integer_column(data, "month_index", str(path))
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if not np.isfinite(data["value"]).all():
        raise ValueError("Observations contain non-finite values.")
    if data.duplicated(["series_id", "month_index"]).any():
        raise ValueError("Observations contain duplicate series/month keys.")
    histories, expected = {}, []
    for series_id, rows in data.groupby("series_id", sort=True):
        rows = rows.sort_values("month_index")
        train = rows.loc[rows["split"].eq("train")]
        test = rows.loc[rows["split"].eq("test")].copy()
        if len(train) < 2 or len(test) != HORIZON:
            raise ValueError(f"{series_id}: need training history and exactly {HORIZON} test rows.")
        if not np.array_equal(rows["month_index"], np.arange(1, len(rows) + 1)):
            raise ValueError(f"{series_id}: month indices must be consecutive, starting at 1.")
        if not np.array_equal(test["month_index"], len(train) + np.arange(1, HORIZON + 1)):
            raise ValueError(f"{series_id}: test months must follow all training months.")
        dates = pd.date_range(rows["date"].iloc[0], periods=len(rows), freq="MS")
        if not np.array_equal(rows["date"].to_numpy(), dates.to_numpy()):
            raise ValueError(f"{series_id}: dates must be consecutive month starts.")
        histories[str(series_id)] = train["value"].to_numpy(dtype=float)
        test["step"] = np.arange(1, HORIZON + 1)
        expected.append(test)
    return histories, pd.concat(expected, ignore_index=True)


def load_forecasts(path, expected):
    """Join on series/month; require every test observation exactly once."""
    forecast = pd.read_csv(path, dtype={"series_id": "string"})
    require_columns(forecast, ("series_id", "month_index", "step", "prediction"), str(path))
    for column in ("month_index", "step"):
        integer_column(forecast, column, str(path))
    forecast["prediction"] = pd.to_numeric(forecast["prediction"], errors="raise")
    if not np.isfinite(forecast["prediction"]).all():
        raise ValueError(f"{path}: predictions must be finite.")
    keys = ["series_id", "month_index"]
    if forecast.duplicated(keys).any():
        raise ValueError(f"{path}: duplicate series/month predictions.")
    columns = [*keys, "step", "prediction"] + (["date"] if "date" in forecast else [])
    forecast = forecast[columns]
    aligned = expected.merge(forecast, on=keys, how="outer", validate="one_to_one",
                             suffixes=("_actual", "_forecast"), indicator=True, sort=False)
    if not aligned["_merge"].eq("both").all():
        counts = aligned["_merge"].value_counts()
        raise ValueError(f"{path}: forecasts must cover all test months exactly; "
                         f"missing={counts.get('left_only', 0)}, unexpected={counts.get('right_only', 0)}. "
                         "Training-month predictions cannot be scored as competition forecasts.")
    if not aligned["step_actual"].eq(aligned["step_forecast"]).all():
        raise ValueError(f"{path}: steps must be 1-18 relative to the training end.")
    if "date_forecast" in aligned:
        dates = pd.to_datetime(aligned["date_forecast"], errors="raise")
        if not dates.eq(aligned["date_actual"]).all():
            raise ValueError(f"{path}: forecast dates disagree with observation dates.")
    return (aligned.sort_values(["series_id", "month_index"])["prediction"]
            .to_numpy(dtype=float).reshape(-1, HORIZON))


def naive2_forecasts(histories):
    """Classical multiplicative Naive2 approximation, fitted on train only.

    Short (<24 points) or nonpositive histories use last-value naive instead.
    This is our implementation, not the original M3 competition submission.
    """
    forecasts, fallbacks = [], []
    for series_id, values in histories.items():
        indices = np.ones(PERIOD)
        if len(values) >= 2 * PERIOD and np.all(values > 0):
            indices = seasonal_indices(values)
        else:
            fallbacks.append(series_id)
        if not np.isfinite(indices).all() or np.any(indices <= 0):
            raise ValueError(f"{series_id}: invalid Naive2 seasonal indices.")
        level = values[-1] / indices[(len(values) - 1) % PERIOD]
        forecasts.append(level * indices[np.arange(len(values), len(values) + HORIZON) % PERIOD])
    return np.stack(forecasts), fallbacks


def symmetric_ape(actual, forecast, convention):
    denominator = actual + forecast if convention == "m3" else np.abs(actual) + np.abs(forecast)
    if np.any((denominator == 0) & (actual != forecast)):
        raise ValueError("M3 sAPE is undefined for opposing values with a zero sum; "
                         "use --smape-convention absolute.")
    return np.divide(200 * np.abs(actual - forecast), denominator,
                     out=np.zeros_like(actual), where=denominator != 0)


def median_with_undefined(values):
    defined = values[~np.isnan(values)]
    return float(np.median(defined)) if len(defined) else float("nan")


def aggregate(rows):
    """One population of individual forecasts, for a horizon or a series."""
    return pd.Series({
        "forecasts": len(rows), "smape": rows["sape"].mean(),
        "median_sape": rows["sape"].median(), "average_rank": rows["rank"].mean(),
        "percentage_better": 100 * rows["better"].mean(),
        "median_rae": median_with_undefined(rows["rae"].to_numpy()),
        "rae_defined": rows["rae"].notna().sum(),
        "rae_infinite": np.isinf(rows["rae"]).sum(),
        "reference_zero_errors": rows["reference_absolute_error"].eq(0).sum(),
    })


def evaluate(expected, forecasts, benchmark, convention):
    """Ranks compare all supplied models plus the Naive2 reference at each point."""
    actual = expected["value"].to_numpy(dtype=float).reshape(-1, HORIZON)
    sapes = {name: symmetric_ape(actual, values, convention) for name, values in forecasts.items()}
    ranks = pd.DataFrame({name: values.ravel() for name, values in sapes.items()}).rank(
        axis=1, method="average")
    reference_error = np.abs(actual - forecasts["naive2"])
    benchmark_error = np.abs(actual - forecasts[benchmark])
    point_tables = []
    for name, values in forecasts.items():
        error = np.abs(actual - values)
        relative = np.divide(error, reference_error, out=np.full_like(error, np.nan),
                             where=reference_error != 0)
        relative[(reference_error == 0) & (error > 0)] = np.inf
        points = expected[["series_id", "month_index", "date", "step"]].copy()
        points.insert(0, "model", name)
        points["actual"] = actual.ravel()
        points["prediction"] = values.ravel()
        points["absolute_error"] = error.ravel()
        points["sape"] = sapes[name].ravel()
        points["rank"] = ranks[name].to_numpy()
        points["better"] = (error < benchmark_error).ravel()
        points["reference_absolute_error"] = reference_error.ravel()
        points["rae"] = relative.ravel()
        point_tables.append(points)
    points = pd.concat(point_tables, ignore_index=True)
    horizon_rows, series_rows, summary_rows = [], [], []
    for (name, step), rows in points.groupby(["model", "step"], sort=True):
        horizon_rows.append({"model": name, "step": step, **aggregate(rows).to_dict()})
    for (name, series_id), rows in points.groupby(["model", "series_id"], sort=True):
        series_rows.append({"model": name, "series_id": series_id, **aggregate(rows).to_dict()})
    per_horizon = pd.DataFrame(horizon_rows)
    per_series = pd.DataFrame(series_rows)
    for table in (per_horizon, per_series):
        for column in ("forecasts", "rae_defined", "rae_infinite", "reference_zero_errors"):
            table[column] = table[column].astype(int)
    for name, rows in per_horizon.groupby("model", sort=True):
        for group, (start, end) in GROUPS.items():
            selected = rows.loc[rows["step"].between(start, end)]
            # Appendix A's medians are calculated across series at each horizon.
            # Group summaries average those horizon statistics, not series means.
            summary_rows.append({"model": name, "horizons": group,
                                 "series": expected["series_id"].nunique(),
                                 "forecasts": int(selected["forecasts"].sum()),
                                 **{key: selected[key].mean() for key in METRICS},
                                 **{key: int(selected[key].sum()) for key in
                                    ("rae_defined", "rae_infinite", "reference_zero_errors")}})
    return {"summary": pd.DataFrame(summary_rows), "per_horizon": per_horizon,
            "per_series": per_series, "per_forecast": points}


def fingerprint(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT, help="Processed observations CSV")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "data" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "evaluation")
    parser.add_argument("--benchmark", default="naive2",
                        help="Model filename stem for percentage better (default: Naive2)")
    parser.add_argument("--smape-convention", choices=["m3", "absolute"], default="m3",
                        help="M3 uses actual+forecast; absolute uses |actual|+|forecast|")
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.predictions_dir = args.predictions_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    try:
        if not args.predictions_dir.is_dir():
            raise ValueError(f"Predictions directory does not exist: {args.predictions_dir}")
        paths = sorted(args.predictions_dir.glob("*.csv"))
        if not paths:
            raise ValueError(f"No prediction CSVs in {args.predictions_dir}; run a model first.")
        if args.output_dir.resolve() == args.predictions_dir.resolve():
            raise ValueError("Output directory must differ from predictions directory.")
        histories, expected = load_observations(args.input)
        forecasts = {path.stem: load_forecasts(path, expected) for path in paths}
        fallbacks = []
        generated_reference = "naive2" not in forecasts
        if generated_reference:
            forecasts["naive2"], fallbacks = naive2_forecasts(histories)
        if args.benchmark not in forecasts:
            raise ValueError(f"Benchmark {args.benchmark!r} is not an available model: {sorted(forecasts)}")
        tables = evaluate(expected, forecasts, args.benchmark, args.smape_convention)
        metadata = {
            "evaluation_split": "test", "horizon": HORIZON,
            "observations": fingerprint(args.input), "prediction_files": [fingerprint(p) for p in paths],
            "models_ranked": sorted(forecasts), "percentage_better_benchmark": args.benchmark,
            "relative_absolute_error_reference": "naive2", "smape_convention": args.smape_convention,
            "rank_ties": "average rank", "percentage_better_ties": "not wins",
            "zero_pair_sape": "0 when actual and forecast are both zero",
            "zero_reference_rae": "0/0 undefined (excluded from median); positive/0 infinite (retained)",
            "summary_aggregation": "arithmetic mean of per-horizon statistics; medians are across series",
            "naive2_source": "training-only classical multiplicative approximation" if generated_reference
                             else "supplied naive2.csv",
            "naive2_short_or_nonpositive_fallback_series": fallbacks,
            "metric_source": "Makridakis & Hibon (2000), Appendix A, pp. 461-462",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(args.output_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except (ValueError, OSError, pd.errors.ParserError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(tables["summary"].loc[tables["summary"]["horizons"].eq("1-18"),
                                ["model", *METRICS]].to_string(index=False))
    print(f"Scored {len(paths)} prediction files against test observations; saved metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
