# M3 monthly financial forecasting

Starter scripts for the **145 monthly FINANCE series** in the M3 competition.
Source: [International Institute of Forecasters — M3 competition](https://forecasters.org/resources/time-series-data/m3-competition/).
The original workbook is [M3C.xls](https://forecasters.org/data/m3comp/M3C.xls).

## Run

Build the environment from a Python 3.12 interpreter — the default `python3` on some
machines is a bare 3.14 without pandas, and the imports below will fail there.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 load_data.py
python3 plot_data.py
python3 series_data.py          # window counts for the current settings
python3 models.py               # train the forecaster, score the holdout
python3 checks.py               # verify the pipeline end to end (~10 s)
```

The default input, `data/raw/m3_monthly.csv`, is an unmodified CSV export of the
`M3Month` sheet in your local `M3C.xls`. It contains all 1,428 monthly series;
the loader filters to FINANCE. This snapshot lets you run without downloading
the workbook. Original series IDs such as `N2522` are preserved.

You can also load the original workbook directly (requires `xlrd`):

```sh
python3 load_data.py --input /Users/marcos/Downloads/M3C.xls
python3 plot_data.py --series N2522 N2523 --show
python3 plot_data.py --series N2522 --output figures/N2522.pdf
```

`load_data.py` writes `data/processed/monthly_finance.csv`, with columns
`series_id`, `date`, `month_index`, `value`, `split`, and `category`.
`plot_data.py` saves `figures/monthly_finance.png` by default, showing three
series on separate axes. Training observations are blue; test actuals are orange.
Use `--count 6` to plot more series, or `--series` to choose specific IDs.

## Use in Python

```python
from load_data import load_monthly_finance

data = load_monthly_finance()
series = data[data.series_id == "N2522"]
train = series.loc[series.split == "train", "value"].to_numpy()
test = series.loc[series.split == "test", "value"].to_numpy()
```

## Forecasting

`series_data.py` turns the series into pooled lag windows and `models.py` fits one
network across all of them. A single global model is used because no individual
series has enough history to train on: training lengths run from 50 to 126 months,
but pooling gives 9,483 windows at a window size of 24.

Series scales are unrelated — values span 10 to 27,505 — so normalisation is fitted
**per window** rather than per series. Each window is centred on its own mean and
divided by its own standard deviation, and the same constants are applied to its
target block and reversed on the forecast. The network therefore only ever learns
the shape of a continuation, never an absolute level, which is what lets one model
serve all 145 series. The constants stay fixed across the whole 18-month forecast,
so a recursive model can later feed predictions back without changing reference
frame. `last`, `loglast` and `series_z` are also implemented, for comparison.

```python
from series_data import load_series_set, make_direct_windows, make_forecast_inputs
from models import DirectMLP, TrainConfig, fit, predict, resolve_device, smape

data = load_series_set()
origins = data.full_origins()                      # forecast from the end of training
device = resolve_device("cpu")                     # faster than MPS at this model size

fit_batch, val_batch = make_direct_windows(data, window=24, origins=origins)
result = fit(DirectMLP(window=24, hidden=(64, 64)), fit_batch, val_batch,
             TrainConfig(seed=0), device)

x, scaler, index = make_forecast_inputs(data, window=24, origins=origins)
forecast = predict(result.model, x, scaler, device)          # (n_series, 18)
print(smape(data.test[index], forecast).mean())
```

The early-stopping split is chronological, not random: neighbouring windows share
all but one observation, so a random split would place near-duplicates on both
sides and early stopping would never fire. `result.history` holds the per-epoch
train and validation losses.

Measured holdout sMAPE over 3 seeds, window 24, hidden (64, 64): **mean 14.41,
median 7.78**, against a last-value baseline of 15.96 / 10.26. A model with no
hidden layer — i.e. linear autoregression — scores 14.14 / 7.23, so depth buys
little here; this matches the M3 competition's own finding that simple methods are
hard to beat.

`checks.py` verifies the pipeline: window counts, normalisation round-trips, that
windows reconstruct their raw slices, that no training window can reach the test
block, that the network can overfit a small batch, that the zero-hidden-layer model
agrees with least squares, and that a seed reproduces its run exactly.

The workbook's `N` is the total series length, including the `NF=18` test
observations. The loader assigns the first `N - NF` months to training and the
last 18 to testing. Dates use the workbook's starting year/month and are represented
as month-start timestamps. Trailing spreadsheet padding is ignored; missing
observations inside a series cause an error instead of shifting the timeline.

Keep the competition test set for final evaluation. For model tuning, split off
validation months chronologically from training. Fit scaling and other learned
preprocessing only on the training portion. Values are kept on the supplied
competition scale; these are historical benchmark series, not current market feeds.

Dataset reference: Makridakis, S. and Hibon, M. (2000). *The M3-Competition:
results, conclusions and implications*. International Journal of Forecasting,
16(4), 451–476. [doi:10.1016/S0169-2070(00)00057-1](https://doi.org/10.1016/S0169-2070(00)00057-1).
