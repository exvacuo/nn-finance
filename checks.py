"""Self-checks for the windowing layer and the forecaster.

Each check is falsifiable and most carry a hard-coded expected value, so a silent
regression in slicing, normalisation or training shows up as a FAIL rather than as
a plausible-looking forecast.
"""

import argparse
import subprocess

import numpy as np

from models import (
    DirectMLP,
    TrainConfig,
    count_parameters,
    fit,
    predict,
    resolve_device,
    smape,
)
from series_data import (
    SCALER_MODES,
    fit_window_scaler,
    load_series_set,
    make_direct_windows,
    make_forecast_inputs,
)

NAIVE_SMAPE = (15.964, 10.258)     # last-value baseline on the holdout: mean, median
WINDOW_COUNTS = {3: (12528, 145), 12: (11223, 145), 24: (9483, 145), 36: (7756, 138)}


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, passed: bool, message: str) -> None:
        self.failures += not passed
        print(f"  {'PASS' if passed else 'FAIL'}  {message}")

    def note(self, message: str) -> None:
        print(f"        {message}")


def check_loading(report: Report, data) -> None:
    print("\n1. Loading")
    lengths = data.train_lengths()
    report.check(len(data) == 145, f"145 series (got {len(data)})")
    report.check(lengths.sum() == 15428, f"15,428 training months (got {lengths.sum():,})")
    report.check(data.test.shape == (145, 18), f"test block {data.test.shape}")
    summary = (int(lengths.min()), int(np.median(lengths)), int(lengths.max()))
    report.check(summary == (50, 116, 126), f"training length min/median/max {summary}")


def check_normalisation(report: Report, data) -> None:
    print("\n2. Normalisation round-trips")
    inputs = np.stack([values[:24] for values in data.train])
    stats = np.stack([[values.mean(), values.std()] for values in data.train])
    for mode in SCALER_MODES:
        scaler = fit_window_scaler(inputs, mode, series_stats=stats)
        restored = scaler.inverse(scaler.transform(inputs))
        error = np.abs(restored - inputs) / np.maximum(np.abs(inputs), 1e-12)
        report.check(error.max() < 1e-9, f"{mode:9s} max relative error {error.max():.1e}")

    scaled = fit_window_scaler(inputs, "zwin").transform(inputs)
    centred = abs(scaled.mean()) < 1e-10
    unit = abs(scaled.std(axis=1).mean() - 1) < 0.02
    report.check(centred and unit, f"zwin rows centred (sd {scaled.std(axis=1).mean():.4f})")


def check_windowing(report: Report, data) -> None:
    print("\n3. Windowing")
    for window, (expected_total, expected_series) in WINDOW_COUNTS.items():
        fit_batch, val_batch = make_direct_windows(data, window, origins=data.full_origins())
        total = len(fit_batch) + len(val_batch)
        series = len(np.unique(np.concatenate([fit_batch.series_index, val_batch.series_index])))
        report.check(
            (total, series) == (expected_total, expected_series),
            f"window {window:2d}: {total:,} windows from {series} series",
        )

    # A window must reconstruct the exact raw slice it was cut from. The fit half
    # holds the earliest windows of each series in order, so a row's position within
    # its series is also its start index in the raw array.
    window = 24
    fit_batch, _ = make_direct_windows(data, window, origins=data.full_origins())
    mismatches = 0
    for series_position in np.random.default_rng(0).choice(145, 20, replace=False):
        rows = np.flatnonzero(fit_batch.series_index == series_position)
        if not len(rows):
            continue
        for start, row in enumerate(rows[:3]):
            one = fit_batch.scaler.take(np.array([row]))
            raw_x = one.inverse(fit_batch.x[row][None, :])[0]
            raw_y = one.inverse(fit_batch.y[row][None, :])[0]
            series = data.train[series_position]
            mismatches += not np.allclose(raw_x, series[start : start + window])
            mismatches += not np.allclose(raw_y, series[start + window : start + window + 18])
    report.check(mismatches == 0, f"windows reconstruct their raw slice ({mismatches} mismatches)")


def check_leakage(report: Report, data) -> None:
    print("\n4. Leakage")
    # The firewall is series_data.py: nothing that builds training windows may read
    # the holdout. models.py touches it only inside main(), to score the demo run.
    grep = subprocess.run(
        ["grep", "-n", r"\.test\b", "series_data.py"], capture_output=True, text=True,
    )
    hits = grep.stdout.splitlines()
    report.check(not hits, f"the windowing layer never reads the test block ({len(hits)} refs)")

    source = open("models.py").read()
    before_main, _, in_main = source.partition("def main()")
    report.check(
        ".test" not in before_main,
        f"models.py reads the holdout only in main() ({in_main.count('.test')} refs there)",
    )

    lengths = data.train_lengths()
    deepest = []
    for window in WINDOW_COUNTS:
        fit_batch, val_batch = make_direct_windows(data, window, origins=data.full_origins())
        index = np.concatenate([fit_batch.series_index, val_batch.series_index])
        counts = np.bincount(index, minlength=145)
        deepest.append(((counts + window + 18 - 1)[counts > 0] <= lengths[counts > 0]).all())
    report.check(all(deepest), "every target block stays inside the training data")


def check_optimisation(report: Report, data, device) -> None:
    print("\n5. Optimisation (can the network fit at all?)")
    fit_batch, _ = make_direct_windows(data, 24, origins=data.full_origins())
    tiny = fit_batch.take(np.arange(64))
    # batch_size == len(tiny) means one optimiser step per epoch, so the budget has
    # to be counted in steps, not epochs.
    result = fit(
        DirectMLP(24, (64, 64)), tiny, None,
        TrainConfig(epochs=3000, batch_size=64, patience=10**9, early_stopping=False, seed=0),
        device,
    )
    loss = result.history.train_loss.iloc[-1]
    report.check(loss < 1e-3, f"overfits 64 windows in 3,000 steps (loss {loss:.1e})")


def check_linear_reference(report: Report, data, device) -> None:
    print("\n6. Linear model agrees with scikit-learn")
    from sklearn.linear_model import LinearRegression

    fit_batch, val_batch = make_direct_windows(data, 24, origins=data.full_origins())
    x, scaler, index = make_forecast_inputs(data, 24, data.full_origins())
    reference = LinearRegression().fit(fit_batch.x, fit_batch.y)
    sklearn_score = smape(data.test[index], scaler.inverse(reference.predict(x)))

    result = fit(DirectMLP(24, ()), fit_batch, val_batch,
                 TrainConfig(epochs=400, patience=40, seed=0), device)
    torch_score = smape(data.test[index], predict(result.model, x, scaler, device))
    gap = abs(sklearn_score.mean() - torch_score.mean())
    report.note(f"scikit-learn {sklearn_score.mean():.3f} / {np.median(sklearn_score):.3f}")
    report.note(f"torch        {torch_score.mean():.3f} / {np.median(torch_score):.3f}")
    report.check(gap < 0.5, f"0-hidden-layer MLP matches least squares (gap {gap:.3f} sMAPE)")


def check_forecasting(report: Report, data, device, seeds: int) -> None:
    print(f"\n7. Holdout forecasting ({seeds} seeds)")
    fit_batch, val_batch = make_direct_windows(data, 24, origins=data.full_origins())
    x, scaler, index = make_forecast_inputs(data, 24, data.full_origins())
    scores, epochs = [], []
    for seed in range(seeds):
        result = fit(DirectMLP(24, (64, 64)), fit_batch, val_batch, TrainConfig(seed=seed), device)
        scores.append(smape(data.test[index], predict(result.model, x, scaler, device)))
        epochs.append(result.best_epoch)
    means = [score.mean() for score in scores]
    medians = [np.median(score) for score in scores]

    report.note(f"parameters {count_parameters(DirectMLP(24, (64, 64))):,}; best epochs {epochs}")
    report.note(f"sMAPE mean   {np.mean(means):.3f} (sd {np.std(means):.3f})")
    report.note(f"sMAPE median {np.mean(medians):.3f} (sd {np.std(medians):.3f})")
    report.note(f"naive reference {NAIVE_SMAPE[0]} / {NAIVE_SMAPE[1]}")
    report.check(np.mean(means) < NAIVE_SMAPE[0], "beats the naive baseline on mean sMAPE")
    report.check(np.mean(medians) < NAIVE_SMAPE[1], "beats the naive baseline on median sMAPE")
    report.check(np.mean(means) > 6.0, "sMAPE is not implausibly low (would signal leakage)")

    if seeds > 1:
        repeat = fit(DirectMLP(24, (64, 64)), fit_batch, val_batch, TrainConfig(seed=0), device)
        again = smape(data.test[index], predict(repeat.model, x, scaler, device))
        report.check(np.array_equal(again, scores[0]), "the same seed reproduces the same scores")
        report.check(not np.array_equal(scores[0], scores[1]), "different seeds differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Skip the training checks")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1.")

    try:
        device = resolve_device(args.device)
        data = load_series_set()
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")

    report = Report()
    check_loading(report, data)
    check_normalisation(report, data)
    check_windowing(report, data)
    check_leakage(report, data)
    if not args.quick:
        check_optimisation(report, data, device)
        check_linear_reference(report, data, device)
        check_forecasting(report, data, device, args.seeds)

    print(f"\n{'All checks passed.' if not report.failures else f'{report.failures} check(s) FAILED.'}")
    raise SystemExit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
