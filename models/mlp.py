"""Select and train a small one-step MLP on residuals, then export recursive forecasts to CSV.

Design choices and reasons are logged in the "MLP: design decisions" doc.
"""

import argparse
import copy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.evaluation import (FOLDS, make_folds, save_predictions, smape, smape_by_step,
                               step_scores)
from models.preprocessing import decompose
from utilities.load_data import DEFAULT_OUTPUT, HORIZON, ROOT, load_training_data

WINDOW_SIZES = (3, 6, 12, 18)
HIDDEN_SIZES = (1, 2, 3, 4, 5)
WEIGHT_DECAYS = (0.0, 1e-4, 1e-3, 1e-2)
LEARNING_RATES = (3e-4, 1e-3)


def prepare_windows(residuals, window):
    """Make n consecutive residuals as inputs and the next residual as target."""
    if window < 1 or not residuals or any(len(values) <= window for values in residuals):
        raise ValueError("Every history must supply a full input window and a target.")
    examples = np.concatenate([
        np.lib.stride_tricks.sliding_window_view(values, window + 1)
        for values in residuals
    ]).astype(np.float32)
    return examples[:, :-1], examples[:, -1:]


class MLP(nn.Module):
    """One tanh hidden layer; output is always one next-month residual."""

    def __init__(self, window, hidden=3):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(nn.Linear(window, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, decompositions, steps=HORIZON):
    """Predict a residual, feed it back, repeat; then restore trend and seasonality."""
    if steps < 1 or not decompositions or any(
            part.length < model.window for part in decompositions):
        raise ValueError("Steps must be positive and histories must supply a full input window.")
    windows = np.stack([part.residual[-model.window:] for part in decompositions]).astype(np.float32)
    model.eval()
    residuals = []
    for _ in range(steps):
        value = model(torch.from_numpy(windows)).numpy()
        residuals.append(value)
        windows = np.concatenate((windows[:, 1:], value), axis=1)
    residuals = np.concatenate(residuals, axis=1)
    return np.stack([part.restore(values) for part, values in zip(decompositions, residuals)])


def train_mlp(arrays, window, epochs=150, seed=42, hidden=3, lr=1e-3, weight_decay=0.0,
              validation=None):
    """Fit one-step targets; choose epochs using recursive 18-step validation sMAPE.

    validation is (fold decompositions, future validation values). With None,
    train fresh weights for exactly epochs passes without early stopping.
    Adam's weight_decay adds an L2 penalty on all weights and biases.
    """
    if epochs < 1:
        raise ValueError("Epochs must be positive.")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = MLP(window, hidden)
    x_train, y_train = [torch.from_numpy(a) for a in arrays]
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_score, stale, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(x_train)).split(512):
            optimiser.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item() * len(batch)
        train_loss = total / len(x_train)
        if not np.isfinite(train_loss):
            raise ValueError("Training produced a non-finite loss.")
        score = float("nan")
        if validation is not None:
            decompositions, actual = validation
            score = smape(actual, predict(model, decompositions))
            if not np.isfinite(score):
                raise ValueError("Validation produced a non-finite score.")
        history.append((epoch, train_loss, score))
        if validation is None:
            continue
        if score < best_score:
            best_score, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 15:
                break
    if validation is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history, columns=["epoch", "train_loss", "validation_smape"])


def search_hyperparameters(series, windows=WINDOW_SIZES, hidden_sizes=HIDDEN_SIZES,
                           weight_decays=WEIGHT_DECAYS, learning_rates=LEARNING_RATES,
                           epochs=150, seeds=(42,), seasonality=True, trend=True):
    """Rank configurations by mean recursive validation sMAPE across folds and seeds."""
    validation_folds = make_folds(series)
    maximum = min(len(values) for values in validation_folds[0][0]) - 1
    if not windows or min(windows) < 1 or max(windows) > maximum:
        raise ValueError(f"Every window must be between 1 and {maximum} for these folds.")
    if not seeds or not hidden_sizes or not learning_rates or not weight_decays:
        raise ValueError("Provide a nonempty search grid and at least one seed.")
    if min(hidden_sizes) < 1 or min(learning_rates) <= 0 or min(weight_decays) < 0:
        raise ValueError("Hidden sizes and learning rates must be positive; weight decays nonnegative.")
    print(f"Searching windows {list(windows)}, hidden {list(hidden_sizes)}, weight decays "
          f"{list(weight_decays)}, learning rates {list(learning_rates)}; {FOLDS} folds, "
          f"{len(seeds)} seeds, seasonality={seasonality}, trend={trend}, {len(series)} series",
          flush=True)
    # Decomposition depends only on the fold, so fit it once per fold.
    parts = [decompose(histories, seasonality, trend) for histories, _ in validation_folds]
    rows, best, best_history = [], None, None
    for window in windows:
        fitting_arrays = [prepare_windows([part.residual for part in fold_parts], window)
                          for fold_parts in parts]
        for hidden, weight_decay, lr in product(hidden_sizes, weight_decays, learning_rates):
            per_step, best_epochs, histories = [], [], []
            for fold, (arrays, fold_parts, (_, actual)) in enumerate(
                    zip(fitting_arrays, parts, validation_folds), 1):
                for seed in seeds:
                    model, history = train_mlp(arrays, window, epochs, seed, hidden, lr,
                                               weight_decay, (fold_parts, actual))
                    # The returned model holds the best checkpoint, so these are its step scores.
                    per_step.append(smape_by_step(actual, predict(model, fold_parts)))
                    best_epochs.append(int(history.loc[history.validation_smape.idxmin(), "epoch"]))
                    histories.append(history.assign(fold=fold, seed=seed))
            scores = np.mean(per_step, axis=1)
            row = dict(window=window, hidden=hidden, weight_decay=weight_decay, lr=lr,
                       best_epoch=max(1, int(np.median(best_epochs))),
                       validation_smape=float(scores.mean()),
                       validation_smape_std=float(scores.std()), runs=len(scores),
                       **step_scores(per_step))
            rows.append(row)
            print(f"window={window}, hidden={hidden}, weight_decay={weight_decay:g}, lr={lr:g}: "
                  f"recursive CV sMAPE={row['validation_smape']:.3f}% "
                  f"(SD {row['validation_smape_std']:.3f}), epochs={row['best_epoch']}", flush=True)
            if best is None or row["validation_smape"] < best["validation_smape"]:
                best, best_history = row, pd.concat(histories, ignore_index=True)
    return best, pd.DataFrame(rows), best_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", "--window", type=int, nargs="+", default=list(WINDOW_SIZES))
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(HIDDEN_SIZES),
                        help="Units in the single tanh hidden layer")
    parser.add_argument("--weight-decays", type=float, nargs="+", default=list(WEIGHT_DECAYS))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=list(LEARNING_RATES))
    parser.add_argument("--no-seasonality", dest="seasonality", action="store_false",
                        help="Skip multiplicative deseasonalisation")
    parser.add_argument("--no-trend", dest="trend", action="store_false",
                        help="Skip linear detrending")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds for CV")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for final retraining")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data" / "predictions" / "mlp.csv")
    parser.add_argument("--search-results", type=Path,
                        help="Optional CSV of every configuration, with sMAPE per forecast step")
    parser.add_argument("--history", type=Path, help="Optional winning configuration's fold/seed loss curves")
    args = parser.parse_args()
    try:
        series = load_training_data(args.input)
        best, results, history = search_hyperparameters(
            series, args.windows, args.hidden_sizes, args.weight_decays, args.learning_rates,
            args.epochs, args.seeds, args.seasonality, args.trend)
        # Fresh final training uses all non-holdout observations and the median CV epoch count.
        ids, histories = list(series), list(series.values())
        parts = decompose(histories, args.seasonality, args.trend)
        full_arrays = prepare_windows([part.residual for part in parts], best["window"])
        model, _ = train_mlp(full_arrays, best["window"], best["best_epoch"], args.seed,
                             best["hidden"], best["lr"], best["weight_decay"])
        forecast = predict(model, parts)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Selected window={best['window']}, hidden={best['hidden']}, "
          f"weight_decay={best['weight_decay']:g}, lr={best['lr']:g} "
          f"(CV sMAPE {best['validation_smape']:.3f}%; steps 1-6 {best['smape_1_6']:.3f}%, "
          f"1-12 {best['smape_1_12']:.3f}%)")
    print(f"Final model retrained on {len(full_arrays[0]):,} examples for {best['best_epoch']} epochs")
    save_predictions(args.predictions, args.input, ids, forecast)
    if args.search_results:
        args.search_results.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.search_results, index=False)
    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.history, index=False)


if __name__ == "__main__":
    main()
