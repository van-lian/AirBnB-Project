import argparse
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def create_time_series_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    start_date = pd.Timestamp(2017, 1, 1)
    end_date = pd.Timestamp(2022, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    ts_data = pd.DataFrame({"date": date_range})
    ts_data["year"] = ts_data["date"].dt.year
    ts_data["month"] = ts_data["date"].dt.month
    ts_data["day_of_week"] = ts_data["date"].dt.dayofweek
    seasonal_multiplier = ts_data["month"].map({
        1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0, 6: 1.2,
        7: 1.3, 8: 1.2, 9: 1.0, 10: 0.9, 11: 0.7, 12: 0.6,
    })
    weekend_multiplier = ts_data["day_of_week"].map({
        0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.9, 5: 1.2, 6: 1.3,
    })
    trend = np.linspace(100, 150, len(ts_data))
    noise = np.random.normal(0, 10, len(ts_data))
    ts_data["bookings"] = (trend * seasonal_multiplier * weekend_multiplier + noise).astype(int)
    ts_data["occupancy_rate"] = 100 - (ts_data["bookings"] / ts_data["bookings"].max()) * (
        float(raw_df["availability 365"].mean()) / 3.65
    )
    base_price = float(raw_df["price"].mean())
    price_trend = np.linspace(base_price * 0.8, base_price * 1.2, len(ts_data))
    ts_data["avg_price"] = price_trend * seasonal_multiplier + np.random.normal(0, 20, len(ts_data))
    return ts_data


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def train_and_evaluate(months_holdout: int = 12, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    print("🔄 Loading data from airbnb_cleaned.csv ...")
    df = pd.read_csv("data/airbnb_cleaned.csv")
    print("🔄 Creating synthetic time series ...")
    ts_data = create_time_series_data(df)
    monthly = (
        ts_data.groupby(["year", "month"]).agg({"bookings": "mean"}).reset_index()
    )
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly = monthly.set_index("date").sort_index()
    if len(monthly) <= months_holdout:
        raise ValueError("Not enough data for the requested holdout size.")
    train = monthly.iloc[:-months_holdout]
    test = monthly.iloc[-months_holdout:]
    print("🔧 Fitting SARIMA ...")
    sarima = SARIMAX(train["bookings"], order=order, seasonal_order=seasonal_order)
    fitted = sarima.fit(disp=False)
    print("🔮 Forecasting on holdout ...")
    forecast = fitted.forecast(steps=len(test))
    y_true = test["bookings"].values
    y_pred = forecast.values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    metrics = {"mae": mae, "rmse": rmse, "mape": mape, "holdout_months": months_holdout}
    print("\n✅ Evaluation on holdout:")
    print(json.dumps(metrics, indent=2))
    seasonal_multipliers = {"Winter": 0.65, "Spring": 0.90, "Summer": 1.23, "Fall": 0.87}
    area_multipliers = {"Manhattan": 1.2, "Brooklyn": 1.0, "Queens": 0.8, "Bronx": 0.6, "Staten Island": 0.7}
    model_blob = {
        "model": fitted,
        "seasonal_multipliers": seasonal_multipliers,
        "area_multipliers": area_multipliers,
        "base_bookings": float(train["bookings"].mean()),
        "base_price": float(df["price"].mean()),
        "training_data_index_start": str(train.index.min().date()),
        "training_data_index_end": str(train.index.max().date()),
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    models_dir = "models"
    import os
    os.makedirs(models_dir, exist_ok=True)
    out_path = f"{models_dir}/airbnb_forecast_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model_blob, f)
    # Also persist lightweight metadata for serving environments that avoid unpickling the model
    meta = {
        "area_multipliers": area_multipliers,
        "seasonal_multipliers": seasonal_multipliers,
        "base_bookings": model_blob["base_bookings"],
        "base_price": model_blob["base_price"],
        "metrics": metrics,
        "trained_at": model_blob["trained_at"],
    }
    meta_path = f"{models_dir}/airbnb_forecast_model.meta.json"
    with open(meta_path, "w") as jf:
        json.dump(meta, jf)
    print(f"\n💾 Saved model to {out_path}")
    print(f"💾 Saved metadata to {meta_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SARIMA model for Airbnb demand")
    parser.add_argument("--holdout_months", type=int, default=12, help="Number of months for time-based holdout")
    parser.add_argument("--order", type=int, nargs=3, default=[1, 1, 1], help="ARIMA order p d q")
    parser.add_argument("--seasonal_order", type=int, nargs=4, default=[1, 1, 1, 12], help="Seasonal order P D Q s")
    args = parser.parse_args()
    order = tuple(args.order)
    seasonal_order = tuple(args.seasonal_order)
    train_and_evaluate(months_holdout=args.holdout_months, order=order, seasonal_order=seasonal_order)


if __name__ == "__main__":
    main()

