# M3 monthly financial forecasting

Prepare the 145 monthly FINANCE series from the M3 competition, remove seasonality
and trend, and forecast the 18 withheld months with a ridge regression baseline and
a small MLP.

## Run

Use Python 3.11 or newer with the dependencies in `requirements.txt`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m utilities.load_data
python -m utilities.plot_data
python -m models.ridge
python -m models.mlp
```

## Layout

| File | Role |
| --- | --- |
| `utilities/load_data.py` | Raw M3 sheet → processed CSV; `load_training_data` reads the training histories |
| `utilities/plot_data.py` | Plot series with the withheld test months highlighted |
| `models/preprocessing.py` | Classical decomposition: seasonality → linear trend → per-series z-score |
| `models/evaluation.py` | Validation folds, sMAPE (overall and per forecast step), forecast export |
| `models/ridge.py` | Ridge regression baseline |
| `models/mlp.py` | One-hidden-layer MLP |

## Data

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

## Preprocessing

Both models see the same residuals. For each history:

1. Divide by multiplicative seasonal indices from a 2×12 centred moving average (as
   M3's Naive2).
2. Subtract a least-squares line fitted to the deseasonalised series.
3. Z-score the remainder with that series' mean and standard deviation.

Every statistic is fitted only on the observations available at the forecast origin.
Forecast residuals are mapped back by undoing the three steps, with the line and
seasonal indices extended into the future. `--no-seasonality` and `--no-trend` switch
steps 1 and 2 off. Negative forecasts are not clipped.

## Validation

Both models are tuned on the same three expanding chronological folds
(`FOLDS = 3`, `FOLD_STEP = 6` in `models/evaluation.py`). For a series with L training
observations, the forecast origins are L-30, L-24, and L-18; each fold refits the
decomposition and model on observations before its origin and forecasts the next 18
recursively. The validation blocks overlap, so fold scores are correlated. The
competition holdout is excluded from all folds.

Configurations are ranked by mean 18-step sMAPE. `--search-results` saves every
configuration with its mean and standard deviation across runs, sMAPE averaged over
steps 1–6 and 1–12, and one column per forecast step (`smape_step_1` … `smape_step_18`).
These are model-selection scores, not an independent estimate of final performance.

## Ridge baseline

One global ridge regression maps `window_size` past residuals to the next one, with an
unpenalised intercept. It searches window sizes 3, 6, 12, 18 and α from 1e-4 to 1e3
(one per decade). `--strategy direct` instead fits one multi-output ridge that predicts
all 18 residuals at once.

```sh
python -m models.ridge --search-results ridge_search.csv
python -m models.ridge --strategy direct --no-trend
```

Forecasts are saved to `data/predictions/ridge_<strategy>.csv`.

## MLP

One global network with a single tanh hidden layer maps `window` past residuals to the
next one; forecasts are recursive. Training minimises one-step MSE with Adam, gradient
clipping at norm 1, batch size 512, and weight decay on all weights and biases.
Early stopping (patience 15) keeps the checkpoint with the best recursive 18-step
validation sMAPE; the same folds score the configuration, so CV scores are slightly
optimistic.

The default grid is windows 3, 6, 12, 18; hidden units 1–5; weight decay 0, 1e-4,
1e-3, 1e-2; learning rates 3e-4 and 1e-3: 160 configurations, each trained on three
folds. The winner is retrained from fresh weights on all training observations for
the median selected epoch count, with a fixed seed (default 42).

```sh
python -m models.mlp --search-results mlp_search.csv --history loss.csv
python -m models.mlp --windows 3 6 --hidden-sizes 2 4 --weight-decays 0 1e-3
```

`--history` saves the winning configuration's epoch curves by fold and seed;
`--seeds` sets the CV seeds and `--seed` the final retraining seed. Forecasts are
saved to `data/predictions/mlp.csv`.

## Output

Prediction files have columns `series_id`, `step` (1–18), `month_index`, `date`, and
`prediction`. Join them to the processed test observations on `series_id` and
`month_index` for evaluation. Neither script calculates holdout scores.

Dataset: Makridakis, S. and Hibon, M. (2000). The M3-Competition: results,
conclusions and implications. International Journal of Forecasting, 16(4), 451–476.
[DOI](https://doi.org/10.1016/S0169-2070(00)00057-1).
