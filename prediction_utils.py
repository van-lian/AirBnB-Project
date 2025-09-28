import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def load_model():
    """Load the trained model from pickle file"""
    with open('airbnb_forecast_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict_bookings(area, season, model_data):
    """Predict bookings for a specific area and season"""
    
    # Get base prediction from SARIMA model
    base_forecast = model_data['model'].forecast(steps=1)[0]
    
    # Apply seasonal multiplier
    seasonal_multiplier = model_data['seasonal_multipliers'].get(season, 1.0)
    
    # Apply area multiplier
    area_multiplier = model_data['area_multipliers'].get(area, 1.0)
    
    # Calculate predicted bookings
    predicted_bookings = base_forecast * seasonal_multiplier * area_multiplier
    
    # Calculate predicted price (based on demand)
    base_price = model_data['base_price']
    price_multiplier = seasonal_multiplier * 0.8 + 0.2  # Price follows demand but less dramatically
    predicted_price = base_price * price_multiplier
    
    # Calculate occupancy rate
    occupancy_rate = min(95, predicted_bookings / model_data['base_bookings'] * 70)
    
    return {
        'predicted_bookings': round(predicted_bookings),
        'predicted_price': round(predicted_price),
        'occupancy_rate': round(occupancy_rate, 1),
        'seasonal_multiplier': seasonal_multiplier,
        'area_multiplier': area_multiplier
    }

def get_seasonal_insights(season):
    """Get seasonal insights and recommendations"""
    insights = {
        'Winter': {
            'description': 'Low season with reduced demand',
            'recommendation': 'Focus on business travelers and longer stays',
            'pricing_strategy': 'Reduce prices by 10-15% to maintain occupancy',
            'marketing_focus': 'Target corporate bookings and holiday events'
        },
        'Spring': {
            'description': 'Moderate season with growing demand',
            'recommendation': 'Prepare for summer peak season',
            'pricing_strategy': 'Gradual price increase as demand grows',
            'marketing_focus': 'Promote spring activities and events'
        },
        'Summer': {
            'description': 'Peak season with highest demand',
            'recommendation': 'Maximize revenue with premium pricing',
            'pricing_strategy': 'Increase prices by 15-20%',
            'marketing_focus': 'Target tourists and outdoor activities'
        },
        'Fall': {
            'description': 'Stable season with moderate demand',
            'recommendation': 'Balance leisure and business travelers',
            'pricing_strategy': 'Maintain competitive pricing',
            'marketing_focus': 'Focus on business travelers and conferences'
        }
    }
    return insights.get(season, {})

if __name__ == "__main__":
    # Test the prediction function
    model_data = load_model()
    
    # Test prediction
    result = predict_bookings('Manhattan', 'Summer', model_data)
    print("Test Prediction:")
    print(f"Area: Manhattan, Season: Summer")
    print(f"Predicted Bookings: {result['predicted_bookings']}")
    print(f"Predicted Price: ${result['predicted_price']}")
    print(f"Occupancy Rate: {result['occupancy_rate']}%")
