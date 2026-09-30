"""
Predictive Analytics Using Historical Data
==========================================
Pipeline: load -> clean/preprocess -> explore trend -> model -> evaluate -> forecast -> visualize

Models compared
  1. Seasonal Naive          (baseline)
  2. Linear Regression       (trend + month-of-year dummies)
  3. Random Forest           (lag / rolling features, recursive forecasting)
  4. Holt-Winters (additive) (classic time-series exponential smoothing)

Usage:
  python src/predictive_analytics.py                       # demo data
  python src/predictive_analytics.py --data my.csv --date-col Date --target-col Value
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- config
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description="Predictive analytics on historical time-series data")
parser.add_argument("--data", default=None, help="Path to CSV (default: generate demo data)")
parser.add_argument("--date-col", default="date")
parser.add_argument("--target-col", default="sales")
parser.add_argument("--test-months", type=int, default=12, help="Hold-out months for evaluation")
parser.add_argument("--forecast-months", type=int, default=12, help="Future horizon")
parser.add_argument("--season", type=int, default=12, help="Seasonal period (12 = monthly data)")
args = parser.parse_args()

DATA_PATH = args.data
DATE_COL, TARGET_COL = args.date_col, args.target_col
TEST_MONTHS, FORECAST_MONTHS, SEASON = args.test_months, args.forecast_months, args.season
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(42)


# ------------------------------------------------ 1. data (demo generator)
def make_demo_data(path=os.path.join(ROOT, "data", "historical_sales_raw.csv")):
    """Monthly sales 2015-2025 with trend + seasonality + noise,
    then deliberately corrupted (missing values, outliers, duplicates)."""
    dates = pd.date_range("2015-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(dates))
    season = 18 * np.sin(2 * np.pi * (dates.month - 3) / 12) + 8 * np.cos(4 * np.pi * dates.month / 12)
    y = 200 + 2.1 * t + season + rng.normal(0, 6, len(t))
    df = pd.DataFrame({DATE_COL: dates.strftime("%Y-%m-%d"), TARGET_COL: y.round(2)})
    df.loc[rng.choice(len(df), 6, replace=False), TARGET_COL] = np.nan          # missing
    df.loc[rng.choice(len(df), 3, replace=False), TARGET_COL] *= 3.5            # outliers
    df = pd.concat([df, df.sample(4, random_state=1)]).sample(frac=1, random_state=2)  # dups + shuffle
    df.to_csv(path, index=False)
    return path


if DATA_PATH is None:
    DATA_PATH = make_demo_data()

# ------------------------------------------------- 2. clean / preprocess
raw = pd.read_csv(DATA_PATH)
print(f"Raw shape: {raw.shape} | missing: {raw[TARGET_COL].isna().sum()} | "
      f"duplicates: {raw.duplicated().sum()}")

df = (raw.drop_duplicates()
         .assign(**{DATE_COL: lambda d: pd.to_datetime(d[DATE_COL], errors="coerce")})
         .dropna(subset=[DATE_COL])
         .sort_values(DATE_COL)
         .set_index(DATE_COL))
s = df[TARGET_COL].astype(float)
s = s.groupby(level=0).mean().asfreq("MS")          # regular monthly index

# outliers: flag values far from a rolling median (robust z-score), then treat as missing
med = s.rolling(7, center=True, min_periods=3).median()
resid = s - med
mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) or 1.0
outliers = (np.abs(resid) / (1.4826 * mad)) > 4
print(f"Outliers flagged: {int(outliers.sum())} | missing after reindex: {int(s.isna().sum())}")
s_clean = s.mask(outliers).interpolate(method="linear").bfill().ffill()
s_clean.name = TARGET_COL

# --------------------------------------------------------- 3. train/test
train, test = s_clean.iloc[:-TEST_MONTHS], s_clean.iloc[-TEST_MONTHS:]


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


# ------------------------------------------------------------ 4. models
def seasonal_naive(train_y, h):
    last = train_y.values[-SEASON:]
    return np.array([last[i % SEASON] for i in range(h)])


def trend_season_features(index, t0):
    t = np.arange(t0, t0 + len(index))
    X = pd.DataFrame({"t": t}, index=index)
    for m in range(2, 13):
        X[f"m{m}"] = (index.month == m).astype(int)
    return X


def fit_predict_linear(train_y, future_index):
    X = trend_season_features(train_y.index, 0)
    model = LinearRegression().fit(X, train_y.values)
    Xf = trend_season_features(future_index, len(train_y))
    return model.predict(Xf), model


def make_lag_frame(y):
    d = pd.DataFrame({"y": y})
    for lag in (1, 2, 3, 6, 12):
        d[f"lag{lag}"] = d["y"].shift(lag)
    d["roll3"] = d["y"].shift(1).rolling(3).mean()
    d["month"] = d.index.month
    d["t"] = np.arange(len(d))
    return d


def fit_predict_rf(train_y, future_index):
    d = make_lag_frame(train_y).dropna()
    feats = [c for c in d.columns if c != "y"]
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)
    # model the de-trended target so trees can extrapolate a rising trend
    lin = LinearRegression().fit(d[["t"]], d["y"])
    rf.fit(d[feats], d["y"] - lin.predict(d[["t"]]))
    hist = train_y.copy()
    preds = []
    for ts in future_index:                      # recursive multi-step forecast
        hist.loc[ts] = np.nan
        f = make_lag_frame(hist).iloc[[-1]]
        f = f.assign(t=len(hist) - 1)
        yhat = rf.predict(f[feats])[0] + lin.predict(f[["t"]])[0]
        hist.loc[ts] = yhat
        preds.append(yhat)
    return np.array(preds)


def holt_winters(y, h, alpha, beta, gamma, m=SEASON):
    """Additive Holt-Winters. Returns (fitted, forecast, SSE)."""
    y = np.asarray(y, float)
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m
    seas = list(y[:m] - level)
    fitted, sse = [], 0.0
    for i, obs in enumerate(y):
        s_i = seas[i] if i < m else seas[i - m]
        pred = level + trend + s_i
        fitted.append(pred)
        if i >= m:
            sse += (obs - pred) ** 2
        new_level = alpha * (obs - s_i) + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        new_seas = gamma * (obs - new_level) + (1 - gamma) * s_i
        level = new_level
        if i < m:
            seas[i % m] = new_seas
        else:
            seas.append(new_seas)
    fc = np.array([level + (k + 1) * trend + seas[len(y) - m + (k % m)] for k in range(h)])
    return np.array(fitted), fc, sse


def fit_predict_hw(train_y, h):
    obj = lambda p: holt_winters(train_y, 1, *p)[2]
    res = minimize(obj, x0=[0.3, 0.05, 0.3], bounds=[(0.01, 0.99)] * 3, method="L-BFGS-B")
    return holt_winters(train_y, h, *res.x)[1], res.x


lin_pred, _ = fit_predict_linear(train, test.index)
rf_pred = fit_predict_rf(train, test.index)
hw_pred, hw_params = fit_predict_hw(train, len(test))
naive_pred = seasonal_naive(train, len(test))

# ------------------------------------------------------- 5. evaluation
preds = {"Seasonal Naive": naive_pred, "Linear Regression": lin_pred,
         "Random Forest": rf_pred, "Holt-Winters": hw_pred}
results = pd.DataFrame({k: metrics(test.values, v) for k, v in preds.items()}).T.round(2)
results = results.sort_values("RMSE")
print("\nHold-out evaluation (last %d months):\n%s" % (TEST_MONTHS, results))
results.to_csv(f"{OUT_DIR}/model_comparison.csv")
best = results.index[0]
print(f"\nBest model by RMSE: {best}")

# ------------------------------------------- 6. refit best on all data
future_index = pd.date_range(s_clean.index[-1] + pd.offsets.MonthBegin(1), periods=FORECAST_MONTHS, freq="MS")
if best == "Linear Regression":
    future = fit_predict_linear(s_clean, future_index)[0]
elif best == "Random Forest":
    future = fit_predict_rf(s_clean, future_index)
elif best == "Holt-Winters":
    future = fit_predict_hw(s_clean, FORECAST_MONTHS)[0]
else:
    future = seasonal_naive(s_clean, FORECAST_MONTHS)

# rough 95% interval from hold-out residuals
sigma = np.std(test.values - preds[best], ddof=1)
forecast_df = pd.DataFrame({"forecast": future, "lower_95": future - 1.96 * sigma,
                            "upper_95": future + 1.96 * sigma}, index=future_index).round(2)
forecast_df.index.name = "date"
forecast_df.to_csv(f"{OUT_DIR}/future_forecast.csv")
s_clean.to_csv(f"{OUT_DIR}/cleaned_data.csv")

# ---------------------------------------------------- 7. visualizations
plt.style.use("seaborn-v0_8-whitegrid")

# (a) cleaning effect
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(s.index, s.values, "o", ms=3, color="#d62728", alpha=.6, label="Raw (with gaps/outliers)")
ax.plot(s_clean.index, s_clean.values, color="#1f77b4", lw=1.6, label="Cleaned")
ax.set_title("Data cleaning: raw vs cleaned series"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/1_cleaning.png", dpi=150); plt.close(fig)

# (b) trend & seasonality
trend_ma = s_clean.rolling(12, center=True).mean()
seasonal_profile = (s_clean - trend_ma).groupby(s_clean.index.month).mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(s_clean, color="#999", label="Monthly"); axes[0].plot(trend_ma, color="#d62728", lw=2, label="12-mo trend")
axes[0].set_title("Trend"); axes[0].legend()
axes[1].bar(seasonal_profile.index, seasonal_profile.values, color="#2ca02c")
axes[1].set_title("Average seasonal effect by month"); axes[1].set_xlabel("Month")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/2_trend_seasonality.png", dpi=150); plt.close(fig)

# (c) hold-out predictions, all models
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(train.index[-36:], train.values[-36:], color="#555", label="Train (last 3 yrs)")
ax.plot(test.index, test.values, "k-", lw=2.5, label="Actual (test)")
for (name, p), c in zip(preds.items(), ["#999999", "#1f77b4", "#2ca02c", "#ff7f0e"]):
    ax.plot(test.index, p, "--", color=c, lw=1.8, label=f"{name} (RMSE {results.loc[name, 'RMSE']})")
ax.set_title("Model predictions vs actual on hold-out period"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/3_holdout_predictions.png", dpi=150); plt.close(fig)

# (d) metrics bar chart
fig, ax = plt.subplots(figsize=(7, 4))
results["RMSE"].plot.barh(ax=ax, color="#1f77b4"); ax.invert_yaxis()
ax.set_title("RMSE by model (lower is better)")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/4_model_comparison.png", dpi=150); plt.close(fig)

# (e) final forecast
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(s_clean.index, s_clean.values, color="#1f77b4", label="Historical")
ax.plot(forecast_df.index, forecast_df["forecast"], color="#d62728", lw=2, label=f"Forecast ({best})")
ax.fill_between(forecast_df.index, forecast_df["lower_95"], forecast_df["upper_95"], color="#d62728", alpha=.2, label="95% interval")
ax.axvline(s_clean.index[-1], color="k", ls=":", lw=1)
ax.set_title(f"{FORECAST_MONTHS}-month forecast"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/5_future_forecast.png", dpi=150); plt.close(fig)

# (f) residual diagnostics of best model
res_best = test.values - preds[best]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(test.index, res_best, "o-"); axes[0].axhline(0, color="k", lw=1); axes[0].set_title("Residuals over time")
axes[1].hist(res_best, bins=8, color="#9467bd"); axes[1].set_title("Residual distribution")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/6_residuals.png", dpi=150); plt.close(fig)

print(f"\nSaved charts + CSVs to {OUT_DIR}")
print(forecast_df.head(6))
