# Predictive Analytics Using Historical Data

Build, evaluate and visualize models that forecast future trends from historical time-series data.

## Features
- **Cleaning & preprocessing**: duplicate removal, date parsing, regular monthly index, robust outlier detection (rolling median + MAD), linear interpolation of gaps
- **Models compared**
  - Seasonal Naive (baseline)
  - Linear Regression (trend + month-of-year dummies)
  - Random Forest (lag and rolling features, recursive multi-step forecast)
  - Holt-Winters additive exponential smoothing (implemented from scratch, parameters fitted with L-BFGS-B)
- **Evaluation**: hold-out test set scored with MAE, RMSE and MAPE
- **Forecasting**: best model (by RMSE) is refit on all data and projects the future with a 95% interval
- **Visualization**: 6 charts saved automatically

## Project structure
```
predictive-analytics-forecasting/
├── src/predictive_analytics.py   # full pipeline
├── data/historical_sales_raw.csv # demo dataset (synthetic, intentionally messy)
├── outputs/                      # charts + result CSVs
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start
```bash
git clone https://github.com/<your-username>/predictive-analytics-forecasting.git
cd predictive-analytics-forecasting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/predictive_analytics.py
```

### Use your own data
Provide a CSV with a date column and a numeric target column:
```bash
python src/predictive_analytics.py --data path/to/file.csv --date-col Date --target-col Value
```
Options: `--test-months 12` `--forecast-months 12` `--season 12`
(for daily data with weekly seasonality, use `--season 7`; note the script assumes a monthly-start index, so resample to monthly first or adapt the frequency).

## Results (demo data, 12-month hold-out)

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| Linear Regression | 4.42 | 5.10 | 0.95 |
| Holt-Winters | 5.66 | 6.13 | 1.22 |
| Random Forest | 5.99 | 6.58 | 1.29 |
| Seasonal Naive | 23.90 | 25.24 | 5.16 |

Linear Regression wins here because the demo series is a linear trend plus seasonality. On real data the ranking may differ, and the script selects the best model automatically.

![Forecast](outputs/5_future_forecast.png)
![Hold-out predictions](outputs/3_holdout_predictions.png)

## Outputs
| File | Description |
|---|---|
| `outputs/1_cleaning.png` | Raw vs cleaned series |
| `outputs/2_trend_seasonality.png` | Trend and monthly seasonal effect |
| `outputs/3_holdout_predictions.png` | All models vs actual on test set |
| `outputs/4_model_comparison.png` | RMSE comparison |
| `outputs/5_future_forecast.png` | Future forecast with 95% interval |
| `outputs/6_residuals.png` | Residual diagnostics |
| `outputs/model_comparison.csv`, `future_forecast.csv`, `cleaned_data.csv` | Numeric results |

## Limitations and next steps
- Prediction interval is an approximation from hold-out residuals, not a model-based interval.
- Possible extensions: SARIMA/Prophet, external regressors, cross-validation with expanding windows, hyperparameter tuning.

## License
MIT
