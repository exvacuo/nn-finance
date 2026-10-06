# M3 monthly financial forecasting

Prepare the 145 monthly FINANCE series from the M3 competition, then train a
shared MLP to predict one next-month value from a window of past observations.

## Run

Use Python 3.11 or newer with the dependencies in `requirements.txt`.
The local `.venv` uses Python 3.11 and PyTorch 2.10:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m utilities.load_data
python -m utilities.plot_data
python -m models.mlp
```

The loader reads `data/raw/m3_monthly.csv`, an unmodified export of the `M3Month`
sheet of M3C.xls. It filters FINANCE and writes
`data/processed/monthly_finance.csv` with columns `series_id`, `date`,
`month_index`, `value`, `split`, and `category`. Original IDs and month-start dates
are preserved. The workbook's `N` includes its `NF=18` test observations; only the
first `N - NF` observations are training data.

To prepare the workbook directly or select series to plot:

```sh
python -m utilities.load_data --input /path/to/M3C.xls
python -m utilities.plot_data --series N2522 N2523 --show
```

## MLP

`models/mlp.py` reads training observations from the **processed CSV**. It contains
data formatting, the one-step network, hyperparameter search, and forecast export.
Each example uses `n` consecutive observations from one series to predict its next
value. Examples never cross series boundaries. Each window and its target share
normalisation constants computed solely from that input window.

Hyperparameters are selected using expanding chronological folds. Defaults are
three forecast origins, six months apart, and seeds 0, 1, and 2. Each fold trains
on observations before its origin, then predicts the next 18 values recursively.
The validation blocks overlap, so scores across folds are correlated. All
candidates see the same series and forecast origins.

For a series with L training observations, default origins are L-30, L-24, and
L-18. The competition holdout is excluded from all folds. A previous fold's
validation observations can become observed history in a later fold, but each
fold's model is initialised and fitted from scratch.

The default window grid uses multiples of 6 up to the largest size that leaves at
least one fitting example in the earliest fold of every series. For this dataset
that is **6, 12, and 18**. The grid also searches hidden layers `(32,)`, `(64,)`,
and `(64, 64)`, with learning rates 0.0003 and 0.001: 18 configurations and 162
training runs across folds and seeds.

Training gradients use one-step normalised MSE. **Both checkpoint selection and
hyperparameter ranking use recursive 18-step sMAPE on the original scale.**
Early stopping returns each run's best recursive validation checkpoint. Candidate
scores average these best scores across folds and seeds; reported standard
deviations describe variation across those runs, not confidence intervals. These
are model-selection scores, not an independent estimate of final performance.

The winning configuration is retrained from fresh weights on all non-holdout
observations for the median selected epoch count across its CV runs. The final
seed is fixed in advance (default 0), not selected using validation scores.

During forecasting, each updated input window is normalised again. The predicted
value is converted back to the original scale, appended to the window, and used
for the next step. Validation and final prediction use this same procedure and
never feed in actual future values.

The command saves 18 predictions per series to `data/predictions/mlp.csv`.
Columns are `series_id`, `step` (1–18), `month_index`, `date`, and `prediction`.
Join this file to processed test observations on `series_id` and `month_index`
for later evaluation. The MLP script calculates no holdout scores.

```sh
python -m models.mlp --search-results search.csv --history loss.csv
python -m models.mlp --windows 6 12 18 --folds 3 --fold-step 6 --seeds 0 1 2
python -m models.mlp --folds 1 --windows 6 12 18 24 30
```

Use `--predictions` to choose the forecast CSV destination. `--search-results`
saves one row per configuration with mean CV score, standard deviation, median
epoch count, and run count. `--history` saves the winning configuration's epoch
curves, identified by fold and seed. `--seed` sets the final retraining seed;
`--seeds` sets the CV seeds. Larger explicit windows are rejected if any series
cannot provide a fitting example in the earliest fold.

For use in Python:

```python
from models.mlp import load_data, prepare_windows, train_mlp, search_hyperparameters, predict

series = load_data()
best, results, losses = search_hyperparameters(series)
histories = list(series.values())
arrays = prepare_windows(histories, window=best["window"])
model, _ = train_mlp(arrays, window=best["window"], epochs=best["best_epoch"],
                     hidden=best["hidden"], lr=best["lr"], seed=0)
forecast = predict(model, histories, steps=18)  # rows follow series.keys()
```

Keep the competition test set for final evaluation and tune using chronological
validation within training data. Values remain on the historical competition
scale; these are benchmark series rather than current market feeds.

Dataset: Makridakis, S. and Hibon, M. (2000). The M3-Competition: results,
conclusions and implications. International Journal of Forecasting, 16(4), 451–476.
[DOI](https://doi.org/10.1016/S0169-2070(00)00057-1).
