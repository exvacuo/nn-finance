"""Turn the M3 monthly Finance series into normalised windows a global model can learn from.

The 145 series carry unrelated scales (values span 10 to 27,505), so a single model
trained on all of them only works if every training example is made locally
comparable first. Normalisation is therefore fitted per window, not per series.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import load_monthly_finance

HORIZON = 18          # months withheld by the competition
SEASONAL_PERIOD = 12
SCALER_MODES = ("zwin", "last", "loglast", "series_z")

_EPS = 1e-6
_RELATIVE_FLOOR = 0.01    # a window's scale never drops below 1% of its own level
_CACHE: dict[str, "SeriesSet"] = {}


@dataclass(frozen=True)
class SeriesSet:
    """Training histories and the withheld test block, kept in separate arrays.

    The separation is deliberate: windowing functions receive only ``train``, so
    a test observation cannot reach a training window by accident.
    """

    ids: list[str]
    train: list[np.ndarray]
    test: np.ndarray                  # (n_series, HORIZON)
    start_dates: list[pd.Timestamp]

    def __len__(self) -> int:
        return len(self.ids)

    def train_lengths(self) -> np.ndarray:
        return np.array([len(values) for values in self.train], dtype=int)

    def history(self, index: int, origin: int) -> np.ndarray:
        """Observations available to a forecaster standing at ``origin``."""
        if not 0 < origin <= len(self.train[index]):
            raise ValueError(
                f"{self.ids[index]}: origin {origin} outside 1..{len(self.train[index])}."
            )
        return self.train[index][:origin]

    def full_origins(self) -> np.ndarray:
        """Origins that use every training month, i.e. the final-evaluation setting."""
        return self.train_lengths()


def load_series_set(path: str | Path | None = None) -> SeriesSet:
    """Reshape the long-format loader output into per-series arrays (cached)."""
    key = str(path) if path is not None else "default"
    if key in _CACHE:
        return _CACHE[key]

    frame = load_monthly_finance(path) if path is not None else load_monthly_finance()
    ids, train, test, starts = [], [], [], []
    for series_id, series in frame.groupby("series_id", sort=False):
        series = series.sort_values("month_index")
        values = series["value"].to_numpy(dtype=float)
        is_test = (series["split"] == "test").to_numpy()
        if is_test.sum() != HORIZON:
            raise ValueError(f"{series_id}: expected {HORIZON} test months.")
        if is_test[:-HORIZON].any():
            raise ValueError(f"{series_id}: test months are not the final observations.")
        ids.append(str(series_id))
        train.append(values[:-HORIZON])
        test.append(values[-HORIZON:])
        starts.append(series["date"].iloc[0])

    data = SeriesSet(ids, train, np.asarray(test, dtype=float), starts)
    _CACHE[key] = data
    return data


@dataclass(frozen=True)
class WindowScaler:
    """A location/scale pair per window, frozen for that window's whole forecast.

    Keeping loc and scale fixed across all HORIZON steps means a recursive model
    can feed its own predictions back without ever changing reference frame.
    """

    loc: np.ndarray                   # (n_windows,)
    scale: np.ndarray                 # (n_windows,)
    mode: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.mode == "loglast":
            return (np.log(values) - self.loc[:, None]) / self.scale[:, None]
        return (values - self.loc[:, None]) / self.scale[:, None]

    def inverse(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        restored = values * self.scale[:, None] + self.loc[:, None]
        return np.exp(restored) if self.mode == "loglast" else restored

    def take(self, index: np.ndarray) -> "WindowScaler":
        return WindowScaler(self.loc[index], self.scale[index], self.mode)


def fit_window_scaler(
    windows: np.ndarray, mode: str = "zwin", series_stats: np.ndarray | None = None
) -> WindowScaler:
    """Derive per-window normalisation constants from input observations only.

    ``zwin`` is the default and the one the project uses: centre on the window's
    mean and divide by its standard deviation. The alternatives exist so the
    normalisation ablation can run without touching this module again.
    """
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 2:
        raise ValueError(f"Expected (n_windows, window) input, got {windows.shape}.")

    if mode == "zwin":
        loc = windows.mean(axis=1)
        spread, level = windows.std(axis=1), np.abs(loc)
    elif mode == "last":
        loc = np.zeros(len(windows))
        spread = level = np.abs(windows[:, -1])
    elif mode == "loglast":
        if (windows <= 0).any():
            raise ValueError("loglast needs strictly positive observations.")
        logged = np.log(windows)
        # Log differences are already scale-free, so the unit spread is deliberate
        # and must not be overridden by the floor below.
        loc, spread, level = logged[:, -1], np.ones(len(windows)), np.zeros(len(windows))
    elif mode == "series_z":
        if series_stats is None:
            raise ValueError("series_z needs series_stats of shape (n_windows, 2).")
        stats = np.asarray(series_stats, dtype=float)
        loc, spread, level = stats[:, 0], stats[:, 1], np.abs(stats[:, 0])
    else:
        raise ValueError(f"Unknown scaler mode {mode!r}; expected one of {SCALER_MODES}.")

    # A flat window would otherwise be amplified enormously by 1/scale and swamp
    # the loss, so the floor is relative to the window's own level.
    floor = np.maximum(_RELATIVE_FLOOR * level, _EPS)
    return WindowScaler(loc, np.maximum(spread, floor), mode)


@dataclass(frozen=True)
class WindowBatch:
    """Normalised supervised pairs pooled across series."""

    x: np.ndarray                     # (n_windows, window)
    y: np.ndarray                     # (n_windows, horizon)
    series_index: np.ndarray          # (n_windows,)
    scaler: WindowScaler

    def __len__(self) -> int:
        return len(self.x)

    @property
    def window(self) -> int:
        return self.x.shape[1]

    @property
    def horizon(self) -> int:
        return self.y.shape[1]

    def n_series(self) -> int:
        return len(np.unique(self.series_index))

    def as_tensors(self, device, recurrent: bool = False):
        """Float32 tensors; ``recurrent`` adds the feature axis an RNN expects."""
        import torch

        x = torch.as_tensor(self.x, dtype=torch.float32, device=device)
        y = torch.as_tensor(self.y, dtype=torch.float32, device=device)
        return (x.unsqueeze(-1) if recurrent else x), y

    def take(self, index: np.ndarray) -> "WindowBatch":
        return WindowBatch(
            self.x[index], self.y[index], self.series_index[index], self.scaler.take(index)
        )


def _series_window_stats(values: np.ndarray, count: int) -> np.ndarray:
    """Per-series mean/std repeated per window, for the series_z ablation."""
    spread = values.std()
    return np.tile([values.mean(), spread if spread > _EPS else 1.0], (count, 1))


def _slice_windows(
    data: SeriesSet, window: int, horizon: int, origins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect every (input, target) pair that fits strictly inside each origin."""
    inputs, targets, index, stats, ends = [], [], [], [], []
    for position, origin in enumerate(origins):
        available = data.history(position, int(origin))
        count = len(available) - window - horizon + 1
        if count < 1:
            continue
        starts = np.arange(count)
        offsets = np.arange(window)
        inputs.append(available[starts[:, None] + offsets[None, :]])
        targets.append(available[starts[:, None] + window + np.arange(horizon)[None, :]])
        index.append(np.full(count, position, dtype=int))
        stats.append(_series_window_stats(available, count))
        # Distance from the end of the available history, used for the time-ordered split.
        ends.append(len(available) - (starts + window + horizon))

    if not inputs:
        raise ValueError(f"No series is long enough for window={window}, horizon={horizon}.")
    return (
        np.concatenate(inputs),
        np.concatenate(targets),
        np.concatenate(index),
        np.concatenate(ends),
    ), np.concatenate(stats)


def make_direct_windows(
    data: SeriesSet,
    window: int,
    horizon: int = HORIZON,
    *,
    origins: np.ndarray,
    scaler_mode: str = "zwin",
    inner_val_months: int = HORIZON,
) -> tuple[WindowBatch, WindowBatch]:
    """Build pooled training windows, split chronologically for early stopping.

    The validation half holds the windows whose target block ends within the last
    ``inner_val_months`` of the available history. Splitting by time rather than at
    random matters because neighbouring windows share all but one observation; a
    random split would put near-duplicates on both sides and early stopping would
    never fire.
    """
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive.")
    origins = np.asarray(origins, dtype=int)
    lengths = data.train_lengths()
    if origins.shape != lengths.shape:
        raise ValueError(f"Expected {len(lengths)} origins, got {origins.shape}.")
    if (origins > lengths).any():
        offender = data.ids[int(np.argmax(origins > lengths))]
        raise ValueError(f"{offender}: origin runs past the training block.")

    (inputs, targets, index, ends), stats = _slice_windows(data, window, horizon, origins)
    scaler = fit_window_scaler(inputs, scaler_mode, series_stats=stats)
    batch = WindowBatch(
        scaler.transform(inputs).astype(np.float32),
        scaler.transform(targets).astype(np.float32),
        index,
        scaler,
    )

    is_validation = ends < inner_val_months
    if is_validation.all() or not is_validation.any():
        raise ValueError(
            f"inner_val_months={inner_val_months} leaves one side of the split empty."
        )
    return batch.take(np.flatnonzero(~is_validation)), batch.take(np.flatnonzero(is_validation))


def make_forecast_inputs(
    data: SeriesSet, window: int, origins: np.ndarray, *, scaler_mode: str = "zwin"
) -> tuple[np.ndarray, WindowScaler, np.ndarray]:
    """The last ``window`` observations before each origin, ready to forecast from.

    Series too short to supply a full window are skipped; ``series_index`` records
    which of them survived so forecasts can be matched back to their series.
    """
    origins = np.asarray(origins, dtype=int)
    inputs, index, stats = [], [], []
    for position, origin in enumerate(origins):
        available = data.history(position, int(origin))
        if len(available) < window:
            continue
        inputs.append(available[-window:])
        index.append(position)
        stats.append(_series_window_stats(available, 1)[0])

    if not inputs:
        raise ValueError(f"No series has {window} observations before its origin.")
    inputs = np.asarray(inputs, dtype=float)
    scaler = fit_window_scaler(inputs, scaler_mode, series_stats=np.asarray(stats))
    return scaler.transform(inputs).astype(np.float32), scaler, np.asarray(index, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="M3C.xls or a CSV export")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--scaler", choices=SCALER_MODES, default="zwin")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be at least 1.")

    try:
        data = load_series_set(args.input)
        fit, validation = make_direct_windows(
            data, args.window, origins=data.full_origins(), scaler_mode=args.scaler
        )
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")

    lengths = data.train_lengths()
    print(f"{len(data)} series; train months {lengths.min()}-{lengths.max()} (median {int(np.median(lengths))})")
    print(f"Window {args.window}, scaler {args.scaler}: {len(fit) + len(validation):,} windows total")
    print(f"Fit: {len(fit):,} from {fit.n_series()} series; validation: {len(validation):,}")
    print(f"Normalised inputs: mean {fit.x.mean():+.3f}, sd {fit.x.std():.3f}")


if __name__ == "__main__":
    main()
