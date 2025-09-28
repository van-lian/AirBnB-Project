import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_time_series_data(df):
    """Create synthetic time series data for forecasting"""
    np.random.seed(42)
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2022, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    ts_data = pd.DataFrame({'date': date_range})
    ts_data['year'] = ts_data['date'].dt.year
    ts_data['month'] = ts_data['date'].dt.month
    ts_data['day'] = ts_data['date'].dt.day
    ts_data['day_of_week'] = ts_data['date'].dt.dayofweek
    ts_data['quarter'] = ts_data['date'].dt.quarter
    
    # Create seasonal features
    ts_data['season'] = ts_data['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Create synthetic demand patterns
    seasonal_multiplier = ts_data['month'].map({
        1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0, 6: 1.2,
        7: 1.3, 8: 1.2, 9: 1.0, 10: 0.9, 11: 0.7, 12: 0.6
    })
    
    weekend_multiplier = ts_data['day_of_week'].map({
        0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.9, 5: 1.2, 6: 1.3
    })
    
    trend = np.linspace(100, 150, len(ts_data))
    noise = np.random.normal(0, 10, len(ts_data))
    
    ts_data['bookings'] = (trend * seasonal_multiplier * weekend_multiplier + noise).astype(int)
    
    avg_availability = df['availability 365'].mean()
    ts_data['occupancy_rate'] = np.clip(
        100 - (ts_data['bookings'] / ts_data['bookings'].max()) * avg_availability / 3.65,
        0, 100
    )
    
    base_price = df['price'].mean()
    price_trend = np.linspace(base_price * 0.8, base_price * 1.2, len(ts_data))
    ts_data['avg_price'] = price_trend * seasonal_multiplier + np.random.normal(0, 20, len(ts_data))
    
    return ts_data

def train_and_save_model():
    """Train the best model and save it as pickle file"""
    print("🔄 Loading data...")
    df = pd.read_csv('airbnb_cleaned.csv')
    
    print("🔄 Creating time series data...")
    ts_data = create_time_series_data(df)
    
    # Prepare data for modeling (monthly aggregation)
    monthly_data = ts_data.groupby(['year', 'month']).agg({
        'bookings': 'mean'
    }).reset_index()
    monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
    monthly_data = monthly_data.set_index('date')
    
    # Use all data for training (no test split for production model)
    train_data = monthly_data
    
    print("🔄 Training SARIMA model...")
    # Train SARIMA model (best performing model)
    sarima_model = SARIMAX(train_data['bookings'], 
                          order=(1, 1, 1), 
                          seasonal_order=(1, 1, 1, 12))
    sarima_fitted = sarima_model.fit(disp=False)
    
    # Create seasonal multipliers for prediction
    seasonal_multipliers = {
        'Winter': 0.65,  # Average of Dec, Jan, Feb
        'Spring': 0.9,   # Average of Mar, Apr, May
        'Summer': 1.23, # Average of Jun, Jul, Aug
        'Fall': 0.87    # Average of Sep, Oct, Nov
    }
    
    # Create area multipliers based on neighborhood groups
    area_multipliers = {
        'Manhattan': 1.2,
        'Brooklyn': 1.0,
        'Queens': 0.8,
        'Bronx': 0.6,
        'Staten Island': 0.7
    }
    
    # Save model and metadata
    model_data = {
        'model': sarima_fitted,
        'seasonal_multipliers': seasonal_multipliers,
        'area_multipliers': area_multipliers,
        'base_bookings': train_data['bookings'].mean(),
        'base_price': df['price'].mean(),
        'training_data': train_data
    }
    
    with open('airbnb_forecast_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Model saved as 'airbnb_forecast_model.pkl'")
    print(f"📊 Base bookings: {train_data['bookings'].mean():.1f}")
    print(f"💰 Base price: ${df['price'].mean():.0f}")
    
    return model_data

if __name__ == "__main__":
    train_and_save_model()
