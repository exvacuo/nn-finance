"""Select and train a one-step MLP, then export iterative forecasts to CSV."""

import argparse
import copy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from utilities.load_data import DEFAULT_OUTPUT, ROOT

HORIZON = 18


def load_data(path=DEFAULT_OUTPUT):
    """Read only training observations from the prepared CSV."""
    data = pd.read_csv(Path(path).expanduser())
    data = data.loc[data["split"].eq("train")].sort_values("month_index")
    return {key: rows["value"].to_numpy(dtype=float)
            for key, rows in data.groupby("series_id", sort=False)}


def normalise(x):
    """Scale each input window using only its own observations."""
    mean = x.mean(axis=1, keepdims=True)
    scale = np.maximum(x.std(axis=1, keepdims=True), np.maximum(abs(mean) * 0.01, 1e-6))
    return ((x - mean) / scale).astype(np.float32), mean, scale


def prepare_windows(histories, window):
    """Make n consecutive inputs and one next-value target within each history."""
    if window < 1 or not histories or any(len(values) <= window for values in histories):
        raise ValueError("Every history must supply a full input window and a target.")
    examples = np.concatenate([
        np.lib.stride_tricks.sliding_window_view(values, window + 1)
        for values in histories
    ])
    x, mean, scale = normalise(examples[:, :-1])
    y = ((examples[:, -1:] - mean) / scale).astype(np.float32)
    return x, y


def make_folds(series, folds=3, fold_step=6):
    """Expand training prefixes over time; forecast the next 18 points in each fold.

    Origins are relative to each series' training end. With a step below 18,
    validation blocks overlap, so fold scores are correlated.
    """
    if folds < 1 or fold_step < 1:
        raise ValueError("Fold count and step must be positive.")
    result = []
    for offset in reversed(range(folds)):
        histories, actual = [], []
        for values in series.values():
            origin = len(values) - HORIZON - offset * fold_step
            if origin < 2:
                raise ValueError("Not enough history for the requested folds.")
            histories.append(values[:origin])
            actual.append(values[origin:origin + HORIZON])
        result.append((histories, np.stack(actual)))
    return result


class MLP(nn.Module):
    """Configurable input length; output is always one next-month value."""

    def __init__(self, window, hidden=(64, 64)):
        super().__init__()
        self.window = window
        layers = []
        width = window
        for size in hidden:
            layers.extend((nn.Linear(width, size), nn.ReLU()))
            width = size
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, histories, steps=1):
    """Normalise each updated window, predict, undo scaling, and feed back the value."""
    if steps < 1 or not histories or any(len(values) < model.window for values in histories):
        raise ValueError("Steps must be positive and histories must supply a full input window.")
    windows = np.stack([values[-model.window:] for values in histories])
    model.eval()
    predictions = []
    for _ in range(steps):
        x, mean, scale = normalise(windows)
        value = model(torch.from_numpy(x)).numpy() * scale + mean
        predictions.append(value)
        windows = np.concatenate((windows[:, 1:], value), axis=1)
    return np.concatenate(predictions, axis=1)


def smape(actual, forecast):
    denominator = np.abs(actual) + np.abs(forecast)
    ratios = np.divide(2 * np.abs(forecast - actual), denominator,
                       out=np.zeros_like(actual, dtype=float), where=denominator > 0)
    return float(100 * ratios.mean())


def train_mlp(arrays, window, epochs=150, seed=0, hidden=(64, 64), lr=1e-3,
              validation=None):
    """Fit one-step targets; choose epochs using recursive 18-step validation sMAPE.

    validation is (observed histories, future validation values). With None,
    train fresh weights for exactly epochs passes without early stopping.
    """
    if epochs < 1:
        raise ValueError("Epochs must be positive.")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = MLP(window, hidden)
    x_train, y_train = [torch.from_numpy(a) for a in arrays]
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
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
            histories, actual = validation
            score = smape(actual, predict(model, histories, steps=HORIZON))
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


def search_hyperparameters(series, windows=None,
                           hidden_sizes=((32,), (64,), (64, 64)),
                           learning_rates=(3e-4, 1e-3), epochs=150,
                           seeds=(0, 1, 2), folds=3, fold_step=6):
    """Rank configurations by mean recursive validation sMAPE across folds and seeds."""
    validation_folds = make_folds(series, folds, fold_step)
    maximum = min(len(values) for values in validation_folds[0][0]) - 1
    if windows is None:
        windows = list(range(6, maximum + 1, 6))
    if not windows or min(windows) < 1 or max(windows) > maximum:
        raise ValueError(f"Every window must be between 1 and {maximum} for these folds.")
    if not seeds or not hidden_sizes or not learning_rates:
        raise ValueError("Provide a nonempty search grid and at least one seed.")
    if any(not sizes or min(sizes) < 1 for sizes in hidden_sizes) or min(learning_rates) <= 0:
        raise ValueError("Hidden sizes and learning rates must be positive.")
    print(f"Searching windows {list(windows)}; {folds} chronological folds, "
          f"{len(seeds)} seeds, {len(series)} series", flush=True)
    rows, best, best_history = [], None, None
    for window in windows:
        fitting_arrays = [prepare_windows(histories, window) for histories, _ in validation_folds]
        for hidden, lr in product(hidden_sizes, learning_rates):
            scores, best_epochs, histories = [], [], []
            for fold, (arrays, validation) in enumerate(zip(fitting_arrays, validation_folds), 1):
                for seed in seeds:
                    _, history = train_mlp(arrays, window, epochs, seed, hidden, lr, validation)
                    selected = history.loc[history.validation_smape.idxmin()]
                    scores.append(float(selected.validation_smape))
                    best_epochs.append(int(selected.epoch))
                    histories.append(history.assign(fold=fold, seed=seed))
            row = dict(window=window, hidden=tuple(hidden), lr=lr,
                       best_epoch=max(1, int(np.median(best_epochs))),
                       validation_smape=float(np.mean(scores)),
                       validation_smape_std=float(np.std(scores)), runs=len(scores))
            rows.append(row)
            print(f"window={window}, hidden={hidden}, lr={lr:g}: "
                  f"recursive CV sMAPE={row['validation_smape']:.3f}% "
                  f"(SD {row['validation_smape_std']:.3f}), epochs={row['best_epoch']}", flush=True)
            if best is None or row["validation_smape"] < best["validation_smape"]:
                best, best_history = row, pd.concat(histories, ignore_index=True)
    return best, pd.DataFrame(rows), best_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", "--window", type=int, nargs="+",
                        help="Default: multiples of 6 supported by every series in every fold")
    parser.add_argument("--hidden-sizes", nargs="+", default=["32", "64", "64,64"])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-4, 1e-3])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-step", type=int, default=6, help="Months between forecast origins")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds for CV")
    parser.add_argument("--seed", type=int, default=0, help="Fixed seed for final retraining")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--predictions", type=Path, default=ROOT / "data" / "predictions" / "mlp.csv")
    parser.add_argument("--search-results", type=Path, help="Optional CSV of candidate CV scores")
    parser.add_argument("--history", type=Path, help="Optional winning configuration's fold/seed loss curves")
    args = parser.parse_args()
    try:
        series = load_data(args.input)
        hidden_sizes = [tuple(int(size) for size in item.split(",")) for item in args.hidden_sizes]
        best, results, history = search_hyperparameters(
            series, args.windows, hidden_sizes, args.learning_rates, args.epochs,
            args.seeds, args.folds, args.fold_step)
        # Fresh final training uses all non-holdout observations and the median CV epoch count.
        ids, histories = list(series), list(series.values())
        full_arrays = prepare_windows(histories, best["window"])
        model, _ = train_mlp(full_arrays, best["window"], best["best_epoch"], args.seed,
                             best["hidden"], best["lr"])
        forecast = predict(model, histories, steps=HORIZON)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Selected {best}; using {len(ids)} series, output 1")
    print(f"Final model retrained on {len(full_arrays[0]):,} examples for {best['best_epoch']} epochs")
    # Dates and month indices identify predictions for later external evaluation.
    metadata = pd.read_csv(args.input, usecols=["series_id", "date", "month_index", "split"])
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
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(predictions):,} predictions to {args.predictions}")
    for output, frame in [(args.search_results, results), (args.history, history)]:
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)


if __name__ == "__main__":
    main()
