from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
from typing import Optional
from pathlib import Path
import json

app = FastAPI(
    title="Airbnb Price Prediction API",
    description="API for predicting Airbnb prices in NYC",
    version="1.0.0"
)

# Define input model
class PredictionInput(BaseModel):
    area: str
    season: str
    room_type: Optional[str] = None
    minimum_nights: Optional[int] = None
    number_of_reviews: Optional[int] = None
    reviews_per_month: Optional[float] = None
    availability_365: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "area": "Manhattan",
                "season": "Summer",
                "room_type": "Entire home/apt",
                "minimum_nights": 2,
                "number_of_reviews": 10,
                "reviews_per_month": 0.5,
                "availability_365": 200
            }
        }

# Define output model
class PredictionOutput(BaseModel):
    predicted_price: float
    predicted_bookings: int
    occupancy_rate: float
    area_multiplier: float
    seasonal_multiplier: float
    confidence_score: float

    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 150.0,
                "predicted_bookings": 15,
                "occupancy_rate": 75.0,
                "area_multiplier": 1.2,
                "seasonal_multiplier": 1.1,
                "confidence_score": 0.85
            }
        }

# Initialize model state
model = None
metadata_loaded = False
model_metadata = {
    "area_multipliers": {
        "Manhattan": 1.2,
        "Brooklyn": 1.0,
        "Queens": 0.8,
        "Bronx": 0.6,
        "Staten Island": 0.7,
    },
    "seasonal_multipliers": {
        "Summer": 1.23,
        "Spring": 0.90,
        "Fall": 0.87,
        "Winter": 0.65,
    },
    "base_price": 150.0,
    "base_bookings": 15.0,
}

def _load_pickle_model() -> None:
    global model, model_metadata, metadata_loaded
    try:
        model_path_candidates = [
            Path("models/airbnb_forecast_model.pkl"),
            Path(__file__).resolve().parents[2] / "models" / "airbnb_forecast_model.pkl",
            Path(__file__).resolve().parents[2] / "airbnb_forecast_model.pkl",
        ]
        meta_path_candidates = [
            Path("models/airbnb_forecast_model.meta.json"),
            Path(__file__).resolve().parents[2] / "models" / "airbnb_forecast_model.meta.json",
        ]
        model_path = next((p for p in model_path_candidates if p.exists()), None)
        meta_path = next((p for p in meta_path_candidates if p.exists()), None)
        if meta_path is not None:
            try:
                with open(meta_path, "r") as jf:
                    m = json.load(jf)
                model_metadata["area_multipliers"] = m.get("area_multipliers", model_metadata["area_multipliers"])
                model_metadata["seasonal_multipliers"] = m.get("seasonal_multipliers", model_metadata["seasonal_multipliers"])
                model_metadata["base_price"] = float(m.get("base_price", model_metadata["base_price"]))
                model_metadata["base_bookings"] = float(m.get("base_bookings", model_metadata["base_bookings"]))
                metadata_loaded = True
            except Exception:
                metadata_loaded = False
        if model_path is not None:
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                model = data.get("model")
                if not metadata_loaded:
                    model_metadata["area_multipliers"] = data.get("area_multipliers", model_metadata["area_multipliers"])
                    model_metadata["seasonal_multipliers"] = data.get("seasonal_multipliers", model_metadata["seasonal_multipliers"])
                    model_metadata["base_price"] = float(data.get("base_price", model_metadata["base_price"]))
                    model_metadata["base_bookings"] = float(data.get("base_bookings", model_metadata["base_bookings"]))
                    metadata_loaded = True
            except Exception:
                # Ignore pickle load failures; rely on JSON metadata when available
                pass
        if not metadata_loaded:
            return
    except Exception:
        # Keep defaults if loading fails
        model = None
        metadata_loaded = False

@app.on_event("startup")
async def on_startup() -> None:
    _load_pickle_model()

# Accessors to simplify usage in endpoints
def get_area_multiplier(area: str) -> float:
    return float(model_metadata["area_multipliers"].get(area, 1.0))

def get_season_multiplier(season: str) -> float:
    return float(model_metadata["seasonal_multipliers"].get(season, 1.0))

@app.get("/")
async def root():
    return {"message": "Welcome to Airbnb Price Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Get multipliers from loaded metadata (or defaults)
        area_mult = get_area_multiplier(input_data.area)
        season_mult = get_season_multiplier(input_data.season)
        
        # Base values from trained metadata (fallback to defaults)
        base_price = model_metadata["base_price"]
        predicted_price = base_price * area_mult * season_mult
        
        # Calculate predicted bookings (per month)
        base_bookings = model_metadata["base_bookings"]
        predicted_bookings = int(round(base_bookings * season_mult))
        
        # Estimate occupancy rate more realistically
        # - Assume average stay equals provided minimum_nights or default to 2
        # - Monthly nights available = availability_365/12 if provided, else 30
        avg_stay_nights = float(input_data.minimum_nights) if input_data.minimum_nights else 2.0
        monthly_nights_available = (
            float(input_data.availability_365) / 12.0 if input_data.availability_365 is not None else 30.0
        )
        # Occupancy = (booked nights / available nights) * 100
        # Incorporate area demand pressure into booked nights
        booked_nights = float(predicted_bookings) * float(max(1.0, avg_stay_nights)) * float(max(0.1, area_mult))
        raw_occupancy = (booked_nights / max(1.0, monthly_nights_available)) * 100.0
        # Clamp between 5% and 95% to avoid extremes
        occupancy_rate = float(min(95.0, max(5.0, raw_occupancy)))
        
        # Calculate confidence score based on input completeness
        total_fields = 7
        filled_fields = sum(1 for v in input_data.dict().values() if v is not None)
        confidence_score = filled_fields / total_fields
        
        return PredictionOutput(
            predicted_price=round(predicted_price, 2),
            predicted_bookings=predicted_bookings,
            occupancy_rate=round(occupancy_rate, 1),
            area_multiplier=area_mult,
            seasonal_multiplier=season_mult,
            confidence_score=round(confidence_score, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": metadata_loaded,
        "has_metadata": bool(model_metadata),
    }

@app.post("/reload")
async def reload_model():
    _load_pickle_model()
    return {"reloaded": True, "model_loaded": metadata_loaded}