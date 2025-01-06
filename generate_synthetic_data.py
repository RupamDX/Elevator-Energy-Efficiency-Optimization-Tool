import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_hourly_elevator_data(
    start_date: str,
    end_date: str,
    base_energy_per_trip: float = 0.05,
    random_energy_offset: float = 0.5
) -> pd.DataFrame:
    """
    Generates synthetic hourly elevator usage data with peak hours on weekdays
    and reduced usage on weekends.
    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :param base_energy_per_trip: approximate kWh per trip
    :param random_energy_offset: random +/- offset for energy variation
    :return: DataFrame [timestamp, usage_count, energy_kwh]
    """
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

    data = {
        'timestamp': [],
        'usage_count': [],
        'energy_kwh': []
    }

    for ts in timestamps:
        # Determine if weekend (Saturday=5, Sunday=6)
        is_weekend = ts.weekday() >= 5

        # Determine if peak hour
        hour = ts.hour
        if not is_weekend:
            # Typical peak hours for an office building, for example:
            if hour in [7, 8, 9]:   # morning peak
                usage = random.randint(50, 70)
            elif hour in [12, 13]: # lunch peak
                usage = random.randint(40, 60)
            elif hour in [17, 18]: # evening peak
                usage = random.randint(50, 70)
            else:
                usage = random.randint(10, 30)
        else:
            # Weekend - lower usage overall
            usage = random.randint(0, 15)

        # Convert usage to kWh with some random offset
        offset = random.uniform(-random_energy_offset, random_energy_offset)
        energy = round(usage * base_energy_per_trip + offset, 2)
        energy = max(energy, 0)  # No negative values

        data['timestamp'].append(ts)
        data['usage_count'].append(usage)
        data['energy_kwh'].append(energy)

    df = pd.DataFrame(data)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df


if __name__ == "__main__":
    start = "2023-01-01"
    end = "2023-01-07"  # 1 week of data

    df_hourly = generate_hourly_elevator_data(
        start_date=start,
        end_date=end
    )

    df_hourly.to_csv("data/elevator_usage_hourly.csv", index=False)
    print("Synthetic hourly elevator usage data generated and saved to 'elevator_usage_hourly.csv'.")
    print(df_hourly.head(24))  # Show first 24 hours
