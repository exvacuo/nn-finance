# M3 monthly financial forecasting

Starter scripts for the **145 monthly FINANCE series** in the M3 competition.
Source: [International Institute of Forecasters — M3 competition](https://forecasters.org/resources/time-series-data/m3-competition/).
The original workbook is [M3C.xls](https://forecasters.org/data/m3comp/M3C.xls).

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 load_data.py
python3 plot_data.py
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
# TODO: build lag windows, fit your model on train, and evaluate on test.
```

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
