"""Validation folds, sMAPE scoring and forecast export."""

import numpy as np
import pandas as pd

from utilities.load_data import HORIZON

FOLDS = 3        # validation forecast origins
FOLD_STEP = 6    # months between origins


def make_folds(series):
    """Expand training prefixes over time; forecast the next 18 points in each fold.

    Origins are relative to each series' training end. With a step below 18,
    validation blocks overlap, so fold scores are correlated.
    """
    result = []
    for offset in reversed(range(FOLDS)):
        histories, actual = [], []
        for values in series.values():
            origin = len(values) - HORIZON - offset * FOLD_STEP
            if origin < 2:
                raise ValueError("Not enough history for the requested folds.")
            histories.append(values[:origin])
            actual.append(values[origin:origin + HORIZON])
        result.append((histories, np.stack(actual)))
    return result


def smape_by_step(actual, forecast):
    """sMAPE (%) for each forecast step, averaged over series (M3 Appendix A).

    The denominator uses |actual| + |forecast|; this equals M3's (actual + forecast)
    for positive values. A 0/0 pair counts as zero error.
    """
    if not (np.isfinite(actual).all() and np.isfinite(forecast).all()):
        raise ValueError("sMAPE received non-finite actuals or forecasts.")
    denominator = np.abs(actual) + np.abs(forecast)
    ratios = np.divide(2 * np.abs(forecast - actual), denominator,
                       out=np.zeros_like(actual, dtype=float), where=denominator > 0)
    return 100 * ratios.mean(axis=0)


def smape(actual, forecast):
    return float(smape_by_step(actual, forecast).mean())


def step_scores(per_step):
    """Average step curves into the M3 summary columns plus one column per step."""
    per_step = np.mean(per_step, axis=0)
    return {"smape_1_6": float(per_step[:6].mean()), "smape_1_12": float(per_step[:12].mean()),
            **{f"smape_step_{step}": float(value) for step, value in enumerate(per_step, 1)}}


def save_predictions(path, input_path, ids, forecast):
    """Write 18 predictions per series with dates and month indices for later evaluation."""
    metadata = pd.read_csv(input_path, usecols=["series_id", "date", "month_index", "split"])
    last = (metadata.loc[metadata["split"].eq("train")].sort_values("month_index")
            .groupby("series_id", sort=False).tail(1).set_index("series_id").loc[ids])
    steps = np.tile(np.arange(1, HORIZON + 1), len(ids))
    predictions = pd.DataFrame({
        "series_id": np.repeat(ids, HORIZON), "step": steps,
        "month_index": np.repeat(last["month_index"].to_numpy(), HORIZON) + steps,
        "date": np.concatenate([pd.date_range(pd.Timestamp(date) + pd.offsets.MonthBegin(1),
                                              periods=HORIZON, freq="MS").to_numpy()
                                for date in last["date"]]),
        "prediction": forecast.ravel(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(predictions):,} predictions to {path}")
