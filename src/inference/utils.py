import requests


def load_model():
    return None


def predict_bookings(area, season, model_data=None):
    api_url = "http://localhost:8000/predict"
    data = {"area": area, "season": season}
    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()
        prediction = response.json()
        return {
            "predicted_bookings": prediction["predicted_bookings"],
            "predicted_price": prediction["predicted_price"],
            "occupancy_rate": prediction["occupancy_rate"],
            "seasonal_multiplier": prediction["seasonal_multiplier"],
            "area_multiplier": prediction["area_multiplier"],
        }
    except requests.exceptions.RequestException:
        return {
            "predicted_bookings": 10,
            "predicted_price": 150,
            "occupancy_rate": 65.0,
            "seasonal_multiplier": 1.0,
            "area_multiplier": 1.0,
        }


def get_seasonal_insights(season):
    insights = {
        "Winter": {
            "description": "Low season with reduced demand",
            "recommendation": "Focus on business travelers and longer stays",
            "pricing_strategy": "Reduce prices by 10-15% to maintain occupancy",
            "marketing_focus": "Target corporate bookings and holiday events",
        },
        "Spring": {
            "description": "Moderate season with growing demand",
            "recommendation": "Prepare for summer peak season",
            "pricing_strategy": "Gradual price increase as demand grows",
            "marketing_focus": "Promote spring activities and events",
        },
        "Summer": {
            "description": "Peak season with highest demand",
            "recommendation": "Maximize revenue with premium pricing",
            "pricing_strategy": "Increase prices by 15-20%",
            "marketing_focus": "Target tourists and outdoor activities",
        },
        "Fall": {
            "description": "Stable season with moderate demand",
            "recommendation": "Balance leisure and business travelers",
            "pricing_strategy": "Maintain competitive pricing",
            "marketing_focus": "Focus on business travelers and conferences",
        },
    }
    return insights.get(season, {})


