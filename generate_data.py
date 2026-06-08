"""
generate_data.py
Generates 48 months of synthetic blood demand data and saves to data/blood_demand.csv
Run once before training: python generate_data.py
"""

import numpy as np
import pandas as pd
import os

BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

BASE_DEMAND = {
    'A+': 320, 'A-': 85, 'B+': 280, 'B-': 72,
    'AB+': 95, 'AB-': 28, 'O+': 420, 'O-': 110
}

SEASONAL_FACTORS = [0.92, 0.88, 0.95, 1.02, 1.08, 1.12, 1.05, 0.98, 1.15, 1.10, 1.02, 0.95]

def generate_blood_demand_data(n_months=48, start_year=2021, start_month=1, seed=42):
    np.random.seed(seed)
    records = []

    for i in range(n_months):
        month_idx = (start_month - 1 + i) % 12
        year = start_year + (start_month - 1 + i) // 12
        month = month_idx + 1
        date_str = f"{year}-{month:02d}"

        row = {'date': date_str, 'year': year, 'month': month}

        for bg in BLOOD_GROUPS:
            base = BASE_DEMAND[bg]
            seasonal = SEASONAL_FACTORS[month_idx]
            trend = 1 + i * 0.003           # slight upward trend over time
            noise = np.random.normal(1.0, 0.07)
            holiday_bump = 1.15 if month in [1, 12] else 1.0
            accident_spike = np.random.choice([1.0, 1.0, 1.0, 1.0, 1.25], p=[0.7, 0.1, 0.1, 0.05, 0.05])

            value = int(base * seasonal * trend * noise * holiday_bump * accident_spike)
            value = max(value, 5)
            row[bg] = value

        row['total_demand'] = sum(row[bg] for bg in BLOOD_GROUPS)
        records.append(row)

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_blood_demand_data(n_months=48)
    df.to_csv('data/blood_demand.csv', index=False)
    print(f"[OK] Generated {len(df)} months of data -> data/blood_demand.csv")
    print(df.tail())
