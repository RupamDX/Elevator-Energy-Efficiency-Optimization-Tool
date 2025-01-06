import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import os

def load_hourly_data(csv_path: str) -> pd.DataFrame:
    """
    Loads hourly elevator data from CSV.
    Expects columns:
      - 'timestamp': datetime (hourly frequency)
      - 'energy_kwh': float (energy consumption)
    """
    df = pd.read_csv(csv_path)
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp (just to be safe)
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    return df


def train_prophet_hourly(df: pd.DataFrame) -> Prophet:
    """
    Trains a Prophet model on hourly elevator data.
    Renames 'timestamp' -> 'ds' and 'energy_kwh' -> 'y'.
    """
    # Rename columns for Prophet
    df_prophet = df.rename(columns={'timestamp': 'ds', 'energy_kwh': 'y'})[['ds', 'y']]

    # Instantiate Prophet
    # For hourly data, daily seasonality is crucial (captures 24-hour patterns)
    # If your dataset spans multiple weeks, weekly_seasonality=True can help capture weekday/weekend effects.
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,   # You can disable if you only have 1 week of data
        yearly_seasonality=False   # Usually no need if data is short-term
    )

    # Fit the model
    model.fit(df_prophet)

    return model


if __name__ == "__main__":
    # Path to your hourly CSV file
    data_path = os.path.join("data", "elevator_usage_hourly.csv")

    # 1. Load Data
    df_hourly = load_hourly_data(data_path)
    print(f"Loaded {len(df_hourly)} hourly records from {data_path}.")

    # 2. Train Model
    model = train_prophet_hourly(df_hourly)
    print("Model trained successfully on hourly data.")

    # 3. (Optional) Save Model to JSON
    # This allows you to load the model later without retraining
    # from prophet.serialize import model_to_json
    # with open("prophet_hourly_model.json", "w") as f:
    #     f.write(model_to_json(model))
    # print("Model saved to prophet_hourly_model.json.")
