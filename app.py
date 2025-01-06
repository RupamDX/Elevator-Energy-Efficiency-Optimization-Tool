import streamlit as st
import pandas as pd
import os
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import re
import json

# For ChatCompletion
import openai
from dotenv import load_dotenv

# Load environment variables, e.g. OPENAI_API_KEY from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load hourly elevator data from CSV.
    Expects columns: 
      - 'timestamp' (datetime)
      - 'usage_count' (int or float)
      - 'energy_kwh' (float)
    """
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

def train_prophet_model(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model on the 'energy_kwh' column.
    Rename 'timestamp' -> 'ds' and 'energy_kwh' -> 'y'.
    Enable daily_seasonality to capture 24-hour patterns.
    """
    df_prophet = df.rename(columns={
        'timestamp': 'ds',
        'energy_kwh': 'y'
    })[['ds', 'y']]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(df_prophet)
    return model

def forecast_energy(model: Prophet, periods=48):
    """
    Forecast energy consumption for the next 'periods' hours.
    Default is 48 hours (2 days).
    """
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)
    return forecast

def calculate_savings(forecast: pd.DataFrame, baseline_kwh: float = 10.0) -> tuple:
    """
    Estimate potential savings if we run 'standby mode' 
    whenever 'yhat' is below the baseline_kwh threshold.
    We'll reduce consumption by 20% in such hours.
    """
    standby_savings_rate = 0.20

    forecast['savings_kwh'] = forecast.apply(
        lambda row: row['yhat'] * standby_savings_rate if row['yhat'] < baseline_kwh else 0.0,
        axis=1
    )
    total_savings = forecast['savings_kwh'].sum()
    return forecast, total_savings

def generate_gpt_summary(
    total_savings: float,
    electricity_rate: float = 0.20,
    emission_factor: float = 0.42
):
    """
    Generate a human-readable summary of the elevator energy savings and environmental impact.
    """

    if not openai.api_key:
        return "Error: No OpenAI API key found. Please set it in your environment."

    if total_savings <= 0:
        return "Error: Total savings must be a positive number."

    # Perform calculations
    cost_savings_per_week = total_savings * electricity_rate
    cost_savings_per_month = cost_savings_per_week * 4
    cost_savings_per_year = cost_savings_per_week * 52

    carbon_savings_per_week = total_savings * emission_factor
    carbon_savings_per_month = carbon_savings_per_week * 4
    carbon_savings_per_year = carbon_savings_per_week * 52

    car_miles_avoided = carbon_savings_per_year / 0.411  # kg CO₂ per mile
    trees_planted_equiv = carbon_savings_per_year / 20.0  # kg CO₂ per tree/year

    # Scalability for 10 elevators
    cost_savings_per_year_scaled = cost_savings_per_year * 10
    carbon_savings_per_year_scaled = carbon_savings_per_year * 10

    # Define the GPT prompt
    system_msg = "You are an energy consultant that writes concise and business-friendly summaries."
    user_msg = f"""
    Generate a concise and human-readable summary based on these values:
    - Total energy savings: {total_savings:.2f} kWh
    - Cost savings: Weekly: ${cost_savings_per_week:.2f}, Monthly: ${cost_savings_per_month:.2f}, Yearly: ${cost_savings_per_year:.2f}
    - CO₂ savings: Weekly: {carbon_savings_per_week:.2f} kg, Monthly: {carbon_savings_per_month:.2f} kg, Yearly: {carbon_savings_per_year:.2f} kg
    - Environmental comparisons: Avoids {car_miles_avoided:.2f} car miles or planting {trees_planted_equiv:.2f} trees annually
    - Scalability: For 10 elevators, Annual cost savings: ${cost_savings_per_year_scaled:.2f}, Annual CO₂ reduction: {carbon_savings_per_year_scaled:.2f} kg

    Please include a conclusion summarizing the overall benefits.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=300,
            temperature=0.2  # Lower temperature for consistent responses
        )

        # Return the response as human-readable text
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"An error occurred while generating the summary: {str(e)}"





def main():
    st.title("Elevator Hourly Energy Forecasting")

    # 1. Load Data
    data_file = os.path.join("data", "elevator_usage_hourly.csv")  # Adjust if needed
    df = load_data(data_file)

    st.subheader("Hourly Data Preview")
    st.write(df.head(10))

    # 2. Train Prophet Model
    st.subheader("Training the Prophet Model")
    model = train_prophet_model(df)
    st.write("Model trained successfully!")

    # 3. Forecast
    st.subheader("Forecasting Energy Consumption")
    forecast_hours = st.slider(
        "Forecast Horizon (hours)", 
        min_value=24, max_value=168, value=48, step=24
    )
    forecast = forecast_energy(model, periods=forecast_hours)

    st.write("Forecast Data (last 10 rows):")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

    # 4. Plot the Forecast
    st.subheader("Interactive Forecast Plot")
    fig = plot_plotly(model, forecast, xlabel="Time", ylabel="Energy (kWh)")
    st.plotly_chart(fig, use_container_width=True)

    # 5. Potential Savings
    st.subheader("Standby Mode Savings Calculator")
    baseline_kwh = st.slider(
        "Baseline kWh Threshold for Standby",
        min_value=0.0, max_value=20.0, value=10.0, step=0.5
    )
    forecast, total_savings = calculate_savings(forecast, baseline_kwh=baseline_kwh)

    st.write(
        f"Over the next {forecast_hours} hours, potential savings: **{round(total_savings, 2)} kWh**"
    )

    # 6. GPT Summary (New Chat API)

    total_savings_example = 130.98  # kWh
    electricity_rate_example = 0.20  # $/kWh
    emission_factor_example = 0.42  # kg CO₂/kWh

    if st.button("Generate Summary"):
        summary = generate_gpt_summary(
            total_savings=total_savings_example,
            electricity_rate=electricity_rate_example,
            emission_factor=emission_factor_example
        )
        st.subheader("GPT-Generated Summary")
        st.write(summary)
        


if __name__ == "__main__":
    main()
