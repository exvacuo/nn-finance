"""Ridge regression baseline on deseasonalised, detrended residuals.

Design choices and reasons are logged in the "Ridge baseline: design decisions" doc.
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.evaluation import FOLDS, make_folds, save_predictions, smape_by_step, step_scores
from models.preprocessing import decompose
from utilities.load_data import DEFAULT_OUTPUT, HORIZON, ROOT, load_training_data

WINDOW_SIZES = (3, 6, 12, 18)
ALPHAS = tuple(10.0 ** power for power in range(-4, 4))


def prepare_windows(residuals, window_size, strategy):
    """Make window_size past residuals as inputs, with one next-value target (recursive) or the next 18 (direct).

    Direct mode needs window_size + 18 points per example; shorter histories contribute none.
    """
    width = window_size + (1 if strategy == "recursive" else HORIZON)
    examples = [np.lib.stride_tricks.sliding_window_view(values, width)
                for values in residuals if len(values) >= width]
    if not examples:
        raise ValueError(f"No history supplies {width} points for window size {window_size}.")
    examples = np.concatenate(examples)
    return examples[:, :window_size], examples[:, window_size:]


def fit_ridge(residuals, window_size, alpha, strategy):
    """Unpenalised intercept; direct mode is one multi-output fit with a shared alpha."""
    x, y = prepare_windows(residuals, window_size, strategy)
    return Ridge(alpha=alpha, fit_intercept=True).fit(x, y.ravel() if strategy == "recursive" else y)


def predict(model, decompositions, window_size, strategy):
    """Forecast 18 residuals per series and restore trend and seasonality; no clipping."""
    windows = np.stack([part.residual[-window_size:] for part in decompositions])
    if strategy == "direct":
        residuals = model.predict(windows)
    else:
        steps = []
        for _ in range(HORIZON):
            value = model.predict(windows)[:, None]
            steps.append(value)
            windows = np.concatenate((windows[:, 1:], value), axis=1)
        residuals = np.concatenate(steps, axis=1)
    return np.stack([part.restore(values) for part, values in zip(decompositions, residuals)])


def search_hyperparameters(series, window_sizes=WINDOW_SIZES, alphas=ALPHAS, strategy="recursive",
                           seasonality=True, trend=True):
    """Rank (window_size, alpha) by mean 18-step sMAPE over the shared validation folds."""
    validation_folds = make_folds(series)
    maximum = min(len(values) for values in validation_folds[0][0]) - 1
    if not window_sizes or min(window_sizes) < 1 or max(window_sizes) > maximum:
        raise ValueError(f"Every window size must be between 1 and {maximum} for these folds.")
    if not alphas or min(alphas) <= 0:
        raise ValueError("Provide positive alphas.")
    print(f"Searching window sizes {list(window_sizes)} x {len(alphas)} alphas; strategy={strategy}, "
          f"seasonality={seasonality}, trend={trend}, {FOLDS} folds, {len(series)} series",
          flush=True)
    # Decomposition depends only on the fold, so fit it once per fold.
    parts = [decompose(histories, seasonality, trend) for histories, _ in validation_folds]
    rows = []
    for window_size, alpha in product(window_sizes, alphas):
        per_step = []
        for fold_parts, (_, actual) in zip(parts, validation_folds):
            model = fit_ridge([part.residual for part in fold_parts], window_size, alpha, strategy)
            per_step.append(smape_by_step(actual, predict(model, fold_parts, window_size, strategy)))
        scores = np.mean(per_step, axis=1)
        rows.append(dict(window_size=window_size, alpha=alpha, validation_smape=float(scores.mean()),
                         validation_smape_std=float(scores.std()), runs=len(scores),
                         **step_scores(per_step)))
        print(f"window_size={window_size}, alpha={alpha:g}: CV sMAPE={rows[-1]['validation_smape']:.3f}% "
              f"(SD {rows[-1]['validation_smape_std']:.3f})", flush=True)
    results = pd.DataFrame(rows)
    return results.loc[results.validation_smape.idxmin()].to_dict(), results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strategy", choices=["recursive", "direct"], default="recursive")
    parser.add_argument("--no-seasonality", dest="seasonality", action="store_false",
                        help="Skip multiplicative deseasonalisation")
    parser.add_argument("--no-trend", dest="trend", action="store_false",
                        help="Skip linear detrending")
    parser.add_argument("--window-sizes", type=int, nargs="+", default=list(WINDOW_SIZES))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--predictions", type=Path,
                        help="Default: data/predictions/ridge_<strategy>.csv")
    parser.add_argument("--search-results", type=Path,
                        help="Optional CSV of every configuration, with sMAPE per forecast step")
    args = parser.parse_args()
    predictions_path = args.predictions or ROOT / "data" / "predictions" / f"ridge_{args.strategy}.csv"
    try:
        series = load_training_data(args.input)
        best, results = search_hyperparameters(
            series, args.window_sizes, args.alphas, args.strategy, args.seasonality, args.trend)
        # Final fit uses all non-holdout observations with the selected configuration.
        ids, histories = list(series), list(series.values())
        window_size = int(best["window_size"])
        parts = decompose(histories, args.seasonality, args.trend)
        model = fit_ridge([part.residual for part in parts], window_size, best["alpha"], args.strategy)
        forecast = predict(model, parts, window_size, args.strategy)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Selected window_size={window_size}, alpha={best['alpha']:g} "
          f"(CV sMAPE {best['validation_smape']:.3f}%; steps 1-6 {best['smape_1_6']:.3f}%, "
          f"1-12 {best['smape_1_12']:.3f}%)")
    save_predictions(predictions_path, args.input, ids, forecast)
    if args.search_results:
        args.search_results.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.search_results, index=False)


if __name__ == "__main__":
    main()
