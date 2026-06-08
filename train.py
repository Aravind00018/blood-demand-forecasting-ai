"""
train.py
Run this ONCE to train the LSTM model before starting the server.

Usage:
    python train.py            # normal train
    python train.py --optimize # run Bayesian optimization first (slower)
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.model import train_model, run_bayesian_optimization, generate_blood_demand_data

DATA_PATH = 'data/blood_demand.csv'

def main():
    optimize = '--optimize' in sys.argv

    # 1. Load data
    if not os.path.exists(DATA_PATH):
        print("[*] Data file not found. Generating synthetic data...")
        os.makedirs('data', exist_ok=True)
        from generate_data import generate_blood_demand_data
        df = generate_blood_demand_data()
        df.to_csv(DATA_PATH, index=False)
        print(f"[OK] Data saved to {DATA_PATH}")
    else:
        print(f"[*] Loading data from {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df['month'] = df['date'].dt.month
    print(f"[OK] Loaded {len(df)} months of data")

    # 2. Bayesian Optimization (optional)
    best_params = {'lstm1': 128, 'lstm2': 64, 'dropout': 0.22, 'lr': 0.0018, 'batch': 32}
    if optimize:
        print("[*] Running Bayesian Optimization (15 trials)...")
        best_params = run_bayesian_optimization(df, n_trials=15)

    # 3. Train model
    print(f"[*] Training with params: {best_params}")
    model, scaler, history, metrics = train_model(
        df,
        seq_len=12,
        epochs=120,
        batch_size=best_params.get('batch', 32),
        lstm1=best_params.get('lstm1', 128),
        lstm2=best_params.get('lstm2', 64),
        dropout=best_params.get('dropout', 0.22),
        lr=best_params.get('lr', 0.0018)
    )

    print("\n" + "="*50)
    print("  TRAINING COMPLETE")
    print("="*50)
    print(f"  MAE  : {metrics['mae']}")
    print(f"  RMSE : {metrics['rmse']}")
    print(f"  MAPE : {metrics['mape']}%")
    print(f"  R²   : {metrics['r2']}")
    print("="*50)
    print("\n[OK] Now run: python app.py")

if __name__ == '__main__':
    main()
