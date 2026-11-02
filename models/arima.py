"""Fit one AutoARIMA per series to the residuals, score it on the shared folds, export forecasts.

Design choices and reasons are logged in the "ARIMA: design decisions" doc.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA

from models.evaluation import FOLDS, make_folds, save_predictions, smape_by_step, step_scores
from models.preprocessing import PERIOD, decompose
from utilities.load_data import DEFAULT_OUTPUT, HORIZON, ROOT, load_training_data

ORDER_COLUMNS = ["p", "d", "q", "P", "D", "Q", "period"]


def fit_series(residual, seasonal, steps=HORIZON):
    """AutoARIMA (AICc, stepwise) on one residual series; seasonal terms only when asked.

    Returns the residual forecast and the chosen order, or zeros and None if fitting fails;
    a zero residual forecast restores to the decomposition's own trend and seasonal pattern.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoARIMA(seasonal=seasonal, season_length=PERIOD if seasonal else 1)
            forecast = model.fit(residual).predict(h=steps)["mean"]
        if not np.isfinite(forecast).all():
            raise ValueError("non-finite forecast")
    except Exception:  # Any fitting failure takes the documented fallback.
        return np.zeros(steps), None
    p, q, P, Q, period, d, D = model.model_["arma"]
    return forecast, dict(p=p, d=d, q=q, P=P, D=D, Q=Q, period=period)


def predict(decompositions, seasonal):
    """Forecast 18 residuals per series and restore trend and seasonality; no clipping."""
    forecasts, orders = [], []
    for part in decompositions:
        residual, order = fit_series(part.residual, seasonal)
        forecasts.append(part.restore(residual))
        orders.append(order)
    return np.stack(forecasts), orders


def cross_validate(series, seasonality=True, trend=True):
    """Score per-series AutoARIMA on the shared folds; ARIMA has no grid, so this only reports."""
    # Seasonal ARIMA terms stand in for seasonal adjustment when it is switched off.
    seasonal = not seasonality
    print(f"AutoARIMA per series, seasonal terms={seasonal}; {FOLDS} folds, "
          f"seasonality={seasonality}, trend={trend}, {len(series)} series", flush=True)
    per_step, failures = [], 0
    for fold, (histories, actual) in enumerate(make_folds(series), 1):
        forecast, orders = predict(decompose(histories, seasonality, trend), seasonal)
        per_step.append(smape_by_step(actual, forecast))
        failures += orders.count(None)
        print(f"fold {fold}: CV sMAPE={per_step[-1].mean():.3f}%, "
              f"fallbacks={orders.count(None)}", flush=True)
    scores = np.mean(per_step, axis=1)
    return dict(seasonal_terms=seasonal, validation_smape=float(scores.mean()),
                validation_smape_std=float(scores.std()), runs=len(scores),
                fallbacks=failures, **step_scores(per_step))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-seasonality", dest="seasonality", action="store_false",
                        help="Skip multiplicative deseasonalisation; AutoARIMA then may use seasonal terms")
    parser.add_argument("--no-trend", dest="trend", action="store_false",
                        help="Skip linear detrending")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data" / "predictions" / "arima.csv")
    parser.add_argument("--cv-results", type=Path,
                        help="Optional one-row CSV of CV sMAPE, with sMAPE per forecast step")
    parser.add_argument("--orders", type=Path,
                        help="Optional CSV of each series' order in the final fit")
    args = parser.parse_args()
    try:
        series = load_training_data(args.input)
        cv = cross_validate(series, args.seasonality, args.trend)
        # Final fit uses all non-holdout observations; each series re-selects its order.
        ids, histories = list(series), list(series.values())
        forecast, orders = predict(decompose(histories, args.seasonality, args.trend),
                                   not args.seasonality)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"CV sMAPE {cv['validation_smape']:.3f}% (SD {cv['validation_smape_std']:.3f}; "
          f"steps 1-6 {cv['smape_1_6']:.3f}%, 1-12 {cv['smape_1_12']:.3f}%); "
          f"CV fallbacks {cv['fallbacks']}")
    table = pd.DataFrame([{"series_id": key, **(order or dict.fromkeys(ORDER_COLUMNS)),
                           "fallback": order is None} for key, order in zip(ids, orders)])
    counts = table.loc[~table["fallback"]].groupby(["p", "d", "q"]).size().sort_values(ascending=False)
    print(f"Final fit: {int(table['fallback'].sum())} fallbacks; most common (p,d,q): "
          + ", ".join(f"{order} x{count}" for order, count in counts.head(5).items()))
    save_predictions(args.predictions, args.input, ids, forecast)
    if args.cv_results:
        args.cv_results.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([cv]).to_csv(args.cv_results, index=False)
    if args.orders:
        args.orders.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.orders, index=False)


if __name__ == "__main__":
    main()
