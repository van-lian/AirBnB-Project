# AirBnB-Project
In this Project, I will try to make a machine learning model to predict **Seasonal Demand Forecasting**(Time Series) & gain insight on the pricing and then make an interactive dashboard for people to interact with.




**Goal**:
- Predict future bookings or occupancy based on season.
- Make an Interactive Dashboard for both model prediction & for user to gain insight

## Project Structure

```
AirBnB-Project/
├── app/                      # Streamlit frontend
│   └── dashboard.py
├── data/                     # Raw and processed datasets
│   ├── Airbnb_Open_Data.csv
│   └── airbnb_cleaned.csv
├── models/                   # Saved trained models + metadata
│   ├── airbnb_forecast_model.pkl
│   ├── airbnb_forecast_model.meta.json
│   └── price_prediction_model.joblib
├── notebooks/                # Jupyter notebooks for analysis
│   └── model.ipynb
├── scripts/                  # Helper scripts
│   ├── api.sh
│   ├── dashboard.sh
│   ├── reload.sh
│   └── train.sh
├── src/                      # Source code for training & serving
│   ├── __init__.py
│   ├── api/
│   │   └── main.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── utils.py
│   └── training/
│       └── train_model.py
├── requirements.txt
└── run_api.py
```

## Quickstart

Install dependencies:
```bash
pip install -r requirements.txt
```

Train and view metrics:
```bash
./scripts/train.sh 12
# or
python src/training/train_model.py --holdout_months 12
```

Run API:
```bash
./scripts/api.sh
# or
python run_api.py
```

Reload API model after retrain:
```bash
./scripts/reload.sh
```

Open docs: `http://localhost:8000/docs`

Run Streamlit dashboard:
```bash
./scripts/dashboard.sh
# or
streamlit run app/dashboard.py
```
