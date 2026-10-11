"""Classical decomposition: seasonality -> linear trend -> per-series z-score."""

import numpy as np

PERIOD = 12


def seasonal_indices(values):
    """Classical multiplicative indices from a 2x12 centred moving average (as Naive2).

    Index k belongs to every position t with t % 12 == k, counted from the series start.
    Positions with no complete moving average get index 1 before renormalising to mean 1.
    """
    kernel = np.r_[0.5, np.ones(PERIOD - 1), 0.5] / PERIOD
    centred = np.convolve(values, kernel, mode="valid")
    positions = np.arange(PERIOD // 2, PERIOD // 2 + len(centred))
    ratios = values[positions] / centred
    indices = np.ones(PERIOD)
    for k in range(PERIOD):
        selected = ratios[positions % PERIOD == k]
        if len(selected):
            indices[k] = selected.mean()
    return indices / indices.mean()


class Decomposition:
    """Seasonality -> linear trend -> z-score, fitted on one history only."""

    def __init__(self, values, seasonality=True, trend=True):
        self.length = len(values)
        self.indices = seasonal_indices(values) if seasonality else np.ones(PERIOD)
        adjusted = values / self.season(np.arange(self.length))
        time = np.arange(self.length)
        self.slope, self.intercept = np.polyfit(time, adjusted, 1) if trend else (0.0, 0.0)
        remainder = adjusted - self.line(time)
        self.mean = remainder.mean()
        self.scale = max(remainder.std(), abs(adjusted.mean()) * 0.01, 1e-6)
        self.residual = (remainder - self.mean) / self.scale

    def season(self, time):
        return self.indices[time % PERIOD]

    def line(self, time):
        return self.intercept + self.slope * time

    def restore(self, residual):
        """Map future residuals (positions length, length+1, ...) to the original scale."""
        time = np.arange(self.length, self.length + len(residual))
        return (residual * self.scale + self.mean + self.line(time)) * self.season(time)


def decompose(histories, seasonality, trend):
    return [Decomposition(values, seasonality, trend) for values in histories]
