"""A global feed-forward forecaster: one network reads a lag window, emits 18 months.

Because every window arrives already normalised by its own level and spread, plain
MSE on the normalised scale is close to scale-free, so a single model can be fitted
across all 145 series without the largest ones dominating the gradient.
"""

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from series_data import (
    HORIZON,
    SCALER_MODES,
    WindowBatch,
    WindowScaler,
    load_series_set,
    make_direct_windows,
    make_forecast_inputs,
)

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


def resolve_device(name: str = "cpu") -> torch.device:
    """Pick a device, preferring CPU.

    Measured on this problem: 50 MLP epochs over ~9,500 windows take 0.99 s on CPU
    and 5.36 s on MPS, because kernel-launch overhead dominates at this model size.
    """
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this machine.")
    if name == "cpu":
        torch.set_num_threads(4)
    return torch.device(name)


class DirectMLP(nn.Module):
    """Map a window of ``window`` lags straight to all ``horizon`` future months.

    An empty ``hidden`` collapses the network to a single linear layer, which is the
    autoregressive linear model the deeper variants have to beat.
    """

    def __init__(
        self,
        window: int,
        hidden: tuple[int, ...] = (64, 64),
        horizon: int = HORIZON,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if window < 1 or horizon < 1:
            raise ValueError("window and horizon must be positive.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; expected {sorted(ACTIVATIONS)}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        self.window, self.horizon, self.hidden = window, horizon, tuple(hidden)
        layers: list[nn.Module] = []
        width = window
        for size in self.hidden:
            layers.append(nn.Linear(width, size))
            layers.append(ACTIVATIONS[activation]())
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, horizon))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.window:
            raise ValueError(f"Expected (batch, {self.window}) input, got {tuple(x.shape)}.")
        return self.net(x)


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 15          # epochs without validation improvement before stopping
    grad_clip: float = 1.0
    seed: int = 0
    early_stopping: bool = True
    reset_parameters: bool = True   # re-initialise weights so the seed alone fixes the run


@dataclass
class FitResult:
    model: nn.Module
    best_epoch: int
    best_val_loss: float
    seed: int
    history: pd.DataFrame = field(repr=False)


def _seeded(model: nn.Module, config: TrainConfig) -> torch.Generator:
    """Seed every source of randomness and hand back the shuffle generator.

    Weights are re-initialised here rather than at construction time: a model built
    earlier would otherwise carry whatever global RNG state happened to exist then,
    and two runs with the same seed would silently diverge.
    """
    torch.manual_seed(config.seed)
    if config.reset_parameters:
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return generator


def fit(
    model: nn.Module,
    fit_batch: WindowBatch,
    val_batch: WindowBatch | None,
    config: TrainConfig,
    device: torch.device,
    recurrent: bool = False,
) -> FitResult:
    """Train with Adam, early-stop on the chronological validation split.

    The weights returned are the ones from the best validation epoch, not the last.
    """
    generator = _seeded(model, config)
    model = model.to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    x_fit, y_fit = fit_batch.as_tensors(device, recurrent=recurrent)
    validation = val_batch.as_tensors(device, recurrent=recurrent) if val_batch else None

    history, best_state = [], copy.deepcopy(model.state_dict())
    best_val, best_epoch, stale = float("inf"), 0, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(x_fit), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad()
            loss = criterion(model(x_fit[batch]), y_fit[batch])
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            total += loss.item() * len(batch)
        train_loss = total / len(order)

        val_loss = float("nan")
        if validation is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(validation[0]), validation[1]).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        monitored = val_loss if validation is not None else train_loss
        if monitored < best_val - 1e-9:
            best_val, best_epoch, stale = monitored, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if config.early_stopping and stale >= config.patience:
                break

    model.load_state_dict(best_state)
    return FitResult(model, best_epoch, best_val, config.seed, pd.DataFrame(history))


@torch.no_grad()
def predict(
    model: nn.Module,
    x_norm: np.ndarray,
    scaler: WindowScaler,
    device: torch.device,
    recurrent: bool = False,
) -> np.ndarray:
    """Forecast on the original scale, undoing each window's own normalisation."""
    model = model.to(device).eval()
    x = torch.as_tensor(np.asarray(x_norm), dtype=torch.float32, device=device)
    if recurrent:
        x = x.unsqueeze(-1)
    return scaler.inverse(model(x).cpu().numpy())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def smape(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Symmetric MAPE per series, in percent, as tabulated by the M3 competition."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    denominator = np.abs(actual) + np.abs(forecast)
    ratio = np.where(denominator > 1e-12, 2.0 * np.abs(forecast - actual) / denominator, 0.0)
    return 100.0 * ratio.mean(axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="M3C.xls or a CSV export")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--hidden", type=int, nargs="*", default=[64, 64],
                        help="Hidden widths; pass none for the linear model")
    parser.add_argument("--scaler", choices=SCALER_MODES, default="zwin")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--history", type=Path, help="Optional CSV for the loss curves")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be at least 1.")
    if any(size < 1 for size in args.hidden):
        parser.error("--hidden widths must be positive.")

    try:
        device = resolve_device(args.device)
        data = load_series_set(args.input)
        origins = data.full_origins()
        fit_batch, val_batch = make_direct_windows(
            data, args.window, origins=origins, scaler_mode=args.scaler
        )
        model = DirectMLP(args.window, tuple(args.hidden))
        result = fit(model, fit_batch, val_batch, TrainConfig(epochs=args.epochs, seed=args.seed), device)
        x, scaler, series_index = make_forecast_inputs(
            data, args.window, origins, scaler_mode=args.scaler
        )
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")

    forecast = predict(result.model, x, scaler, device)
    scores = smape(data.test[series_index], forecast)

    print(f"DirectMLP(window={args.window}, hidden={tuple(args.hidden)}) "
          f"- {count_parameters(result.model):,} parameters on {device}")
    print(f"Fit on {len(fit_batch):,} windows, validated on {len(val_batch):,}; "
          f"best epoch {result.best_epoch} (val MSE {result.best_val_loss:.4f})")
    print(f"Holdout sMAPE over {len(scores)} series: "
          f"mean {scores.mean():.3f}, median {np.median(scores):.3f}")
    print(f"Reference - naive (last value): mean 15.964, median 10.258")

    if args.history:
        output = args.history.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.history.to_csv(output, index=False)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
