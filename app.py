"""
app.py
Flask backend — HemaForecast AI Blood Demand Prediction System
Run: python app.py
"""

import os
import json
import csv
import hashlib
import uuid
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='frontend', static_url_path='')
app.secret_key = 'hemaforecast-secret-key-2026-change-in-production'
CORS(app, supports_credentials=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
DATA_PATH  = os.path.join(DATA_DIR, 'blood_demand.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
SCALER_PATH= os.path.join(MODEL_DIR, 'scaler.pkl')
META_PATH  = os.path.join(MODEL_DIR, 'model_meta.json')

BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# ─────────────────────────────────────────────────────────────
# IN-MEMORY USER DATABASE (replace with real DB in production)
# ─────────────────────────────────────────────────────────────
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

USERS_DB = {
    'admin':  {'password': hash_pw('admin123'),  'name': 'Administrator', 'role': 'admin'},
    'doctor': {'password': hash_pw('med2024'),   'name': 'Dr. Sarah',     'role': 'doctor'},
    'staff':  {'password': hash_pw('staff123'),  'name': 'Staff User',    'role': 'staff'},
}

# ─────────────────────────────────────────────────────────────
# LOAD MODEL (lazy)
# ─────────────────────────────────────────────────────────────
_model  = None
_scaler = None
_meta   = None

def load_model_artifacts():
    global _model, _scaler, _meta
    if _model is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        return False

    try:
        import tensorflow as tf
        from backend.model import BahdanauAttention
        _model  = tf.keras.models.load_model(MODEL_PATH, custom_objects={'BahdanauAttention': BahdanauAttention})
        _scaler = joblib.load(SCALER_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
        print("[OK] Model loaded successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# AUTH DECORATOR
# ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'error': 'Unauthorized', 'logged_in': False}), 401
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

# ─────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'}), 400

    user = USERS_DB.get(username)
    if not user or user['password'] != hash_pw(password):
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    session['username'] = username
    session['role']     = user['role']
    session['name']     = user['name']
    session.permanent   = True
    app.permanent_session_lifetime = timedelta(hours=8)

    return jsonify({
        'success': True,
        'user': {'username': username, 'name': user['name'], 'role': user['role']}
    })

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/api/me', methods=['GET'])
def me():
    if 'username' not in session:
        return jsonify({'logged_in': False})
    return jsonify({'logged_in': True, 'username': session['username'],
                    'name': session['name'], 'role': session['role']})

# ─────────────────────────────────────────────────────────────
# DATA ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/api/data/historical', methods=['GET'])
@login_required
def get_historical():
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'No data found. Run generate_data.py first.'}), 404
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df['date'] = df['date'].dt.strftime('%Y-%m')
    records = df[['date'] + BLOOD_GROUPS + ['total_demand']].to_dict(orient='records')
    return jsonify({'data': records, 'count': len(records)})

@app.route('/api/data/latest', methods=['GET'])
@login_required
def get_latest():
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'No data'}), 404
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    last = df.iloc[-1]
    result = {bg: int(last[bg]) for bg in BLOOD_GROUPS}
    result['date'] = last['date'].strftime('%Y-%m')
    result['total'] = int(last['total_demand'])
    return jsonify(result)

@app.route('/api/data/add', methods=['POST'])
@login_required
def add_record():
    data = request.get_json()
    required = ['date'] + BLOOD_GROUPS
    for k in required:
        if k not in data:
            return jsonify({'error': f'Missing field: {k}'}), 400

    os.makedirs(DATA_DIR, exist_ok=True)
    row = {'date': data['date'], 'year': data['date'][:4], 'month': data['date'][5:7]}
    total = 0
    for bg in BLOOD_GROUPS:
        v = int(data[bg])
        row[bg] = v
        total += v
    row['total_demand'] = total

    # Append to CSV
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame()

    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    return jsonify({'success': True, 'message': 'Record added', 'total': total})

# ─────────────────────────────────────────────────────────────
# FORECAST ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/api/forecast', methods=['GET'])
@login_required
def forecast():
    n_months = int(request.args.get('months', 6))
    blood_group = request.args.get('group', None)

    if not load_model_artifacts():
        # Return simulated forecast if model not trained yet
        return jsonify({'simulated': True, 'data': simulate_forecast(n_months, blood_group)})

    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        df['month'] = df['date'].dt.month
        seq_len = _meta.get('seq_len', 12)

        from backend.model import forecast_future
        predictions = forecast_future(_model, df, _scaler, seq_len, n_months)

        if blood_group and blood_group in BLOOD_GROUPS:
            for p in predictions:
                p = {k: v for k, v in p.items() if k in ['date', blood_group, 'total']}

        return jsonify({'simulated': False, 'data': predictions, 'model_metrics': _meta.get('metrics', {})})
    except Exception as e:
        return jsonify({'simulated': True, 'data': simulate_forecast(n_months, blood_group), 'warning': str(e)})

@app.route('/api/attention', methods=['GET'])
@login_required
def attention():
    if not load_model_artifacts():
        return jsonify({'simulated': True, 'weights': simulate_attention_weights()})
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        df['month'] = df['date'].dt.month
        from backend.model import get_attention_weights
        weights = get_attention_weights(_model, df, _scaler, _meta.get('seq_len', 12))
        return jsonify({'simulated': False, 'weights': weights})
    except Exception as e:
        return jsonify({'simulated': True, 'weights': simulate_attention_weights(), 'warning': str(e)})

@app.route('/api/model/metrics', methods=['GET'])
@login_required
def model_metrics():
    if not os.path.exists(META_PATH):
        return jsonify({'trained': False, 'message': 'Model not trained yet. Run python train.py'})
    with open(META_PATH) as f:
        meta = json.load(f)
    return jsonify({'trained': True, **meta})

@app.route('/api/model/retrain', methods=['POST'])
@login_required
def retrain():
    if session.get('role') not in ['admin', 'doctor']:
        return jsonify({'error': 'Insufficient permissions'}), 403

    # Run training in background (simplified — use Celery in production)
    import threading
    def do_train():
        try:
            df = pd.read_csv(DATA_PATH, parse_dates=['date'])
            df['month'] = df['date'].dt.month
            from backend.model import train_model
            train_model(df, epochs=50)
            global _model, _scaler, _meta
            _model = _scaler = _meta = None
            load_model_artifacts()
        except Exception as e:
            print(f"[ERROR] Retrain failed: {e}")

    t = threading.Thread(target=do_train, daemon=True)
    t.start()
    return jsonify({'success': True, 'message': 'Retraining started in background'})

# ─────────────────────────────────────────────────────────────
# ANALYTICS ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/api/analytics/seasonal', methods=['GET'])
@login_required
def seasonal():
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'No data'}), 404
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df['month'] = df['date'].dt.month
    monthly = df.groupby('month')[BLOOD_GROUPS + ['total_demand']].mean().round(1)
    result = []
    for m in range(1, 13):
        row = {'month': m, 'month_name': datetime(2026, m, 1).strftime('%b')}
        row.update(monthly.loc[m].to_dict() if m in monthly.index else {})
        result.append(row)
    return jsonify({'data': result})

@app.route('/api/analytics/summary', methods=['GET'])
@login_required
def summary():
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'No data'}), 404
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    result = {}
    for bg in BLOOD_GROUPS:
        series = df[bg]
        result[bg] = {
            'mean': round(series.mean(), 1),
            'std':  round(series.std(), 1),
            'min':  int(series.min()),
            'max':  int(series.max()),
            'last': int(series.iloc[-1]),
            'trend': round((series.iloc[-1] - series.iloc[-6]) / series.iloc[-6] * 100, 2)
        }
    return jsonify(result)

# ─────────────────────────────────────────────────────────────
# ALERTS ROUTE
# ─────────────────────────────────────────────────────────────
@app.route('/api/alerts', methods=['GET'])
@login_required
def alerts():
    critical, warnings = [], []
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        for bg in BLOOD_GROUPS:
            last3 = df[bg].iloc[-3:].mean()
            prev3 = df[bg].iloc[-6:-3].mean()
            change = (last3 - prev3) / prev3 * 100

            if bg in ['O-', 'B-', 'AB-'] and last3 < df[bg].mean() * 0.85:
                critical.append({'group': bg, 'type': 'shortage',
                    'message': f'{bg} below safe threshold — procurement needed',
                    'severity': 'critical', 'change': round(change, 1)})
            elif change > 15:
                warnings.append({'group': bg, 'type': 'spike',
                    'message': f'{bg} demand rising {change:.1f}% — monitor closely',
                    'severity': 'warning', 'change': round(change, 1)})
            elif change < -10:
                warnings.append({'group': bg, 'type': 'drop',
                    'message': f'{bg} demand falling {abs(change):.1f}%',
                    'severity': 'info', 'change': round(change, 1)})

    return jsonify({'critical': critical, 'warnings': warnings,
                    'total': len(critical) + len(warnings)})

# ─────────────────────────────────────────────────────────────
# SIMULATION HELPERS (used when model not trained)
# ─────────────────────────────────────────────────────────────
BASE = {'A+':320,'A-':85,'B+':280,'B-':72,'AB+':95,'AB-':28,'O+':420,'O-':110}
SEASON_F = [0.92,0.88,0.95,1.02,1.08,1.12,1.05,0.98,1.15,1.10,1.02,0.95]

def simulate_forecast(n_months, group=None):
    import random
    random.seed(77)
    now = datetime.now()
    result = []
    for i in range(n_months):
        d = now + timedelta(days=30 * (i + 1))
        row = {'date': d.strftime('%Y-%m')}
        sf = SEASON_F[d.month - 1]
        total = 0
        for bg in BLOOD_GROUPS:
            v = int(BASE[bg] * sf * (1 + i * 0.003) * (0.93 + random.random() * 0.14))
            row[bg] = v
            total += v
        row['total'] = total
        result.append(row)
    return result

def simulate_attention_weights():
    base = [0.02,0.03,0.04,0.05,0.06,0.09,0.12,0.10,0.13,0.14,0.13,0.09]
    import random; random.seed(9)
    return [round(w * (0.8 + random.random() * 0.4), 4) for w in base]

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  HemaForecast AI — Blood Demand Prediction System")
    print("=" * 55)
    load_model_artifacts()
    print(f"  Open: http://localhost:5000")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=True)
