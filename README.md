# HemaForecast AI — Blood Demand Prediction System
## BiLSTM + Bahdanau Attention + Bayesian Optimization

---

## STEP-BY-STEP SETUP GUIDE (Windows)

---

### STEP 1 — Install Python (if not installed)
1. Go to https://www.python.org/downloads/
2. Download Python 3.12
3. During install: CHECK THE BOX that says "Add Python to PATH"
4. Click Install Now

To verify, open CMD and type:
```
python --version
```
You should see: Python 3.12.x

---

### STEP 2 — Download & Extract Project
1. Download the project ZIP file
2. Extract it to a folder, e.g.:  C:\Projects\hemaforecast
3. Your folder structure should look like:

```
hemaforecast/
├── app.py
├── train.py
├── generate_data.py
├── requirements.txt
├── README.md
├── backend/
│   └── model.py
├── frontend/
│   ├── index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/           (auto-created)
└── models/         (auto-created)
```

---

### STEP 3 — Open CMD in Your Project Folder
**Method 1 (easiest):**
- Open the folder in File Explorer
- Click on the address bar at the top
- Type `cmd` and press Enter

**Method 2:**
- Press Windows + R
- Type `cmd` and press Enter
- Then type:
```
cd C:\Projects\hemaforecast
```

---

### STEP 4 — Create a Virtual Environment
This keeps your project's Python packages isolated.

```cmd
python -m venv venv
```

Then activate it:
```cmd
venv\Scripts\activate
```

You will see `(venv)` appear at the start of your CMD line.
This means the virtual environment is active.

---

### STEP 5 — Install All Required Packages
```cmd
pip install -r requirements.txt
```

This will install Flask, TensorFlow, NumPy, Pandas, Scikit-learn, Optuna, etc.
This may take 5-10 minutes (TensorFlow is large).

---

### STEP 6 — Generate Training Data
```cmd
python generate_data.py
```

This creates: `data/blood_demand.csv`
You should see: "[OK] Generated 48 months of data"

---

### STEP 7 — Train the LSTM Model
```cmd
python train.py
```

This will:
- Load the CSV data
- Build the BiLSTM + Attention model
- Train for up to 120 epochs (early stopping)
- Save model to: `models/best_model.keras`
- Save scaler to: `models/scaler.pkl`
- Save metrics to: `models/model_meta.json`

Training takes 2-5 minutes on a normal laptop.

OPTIONAL — Run with Bayesian Optimization (slower, better results):
```cmd
python train.py --optimize
```

---

### STEP 8 — Start the Web Server
```cmd
python app.py
```

You will see:
```
=======================================================
  HemaForecast AI — Blood Demand Prediction System
=======================================================
  Open: http://localhost:5000
=======================================================
```

---

### STEP 9 — Open in Browser
Open your browser and go to:
```
http://localhost:5000
```

Login with any of these accounts:
| Username | Password  | Role           |
|----------|-----------|----------------|
| admin    | admin123  | Administrator  |
| doctor   | med2024   | Medical Staff  |
| staff    | staff123  | Data Entry     |

---

## STOPPING THE SERVER
Press `Ctrl + C` in the CMD window.

## RESTARTING LATER
Each time you want to use the app:
1. Open CMD in the project folder
2. Activate venv: `venv\Scripts\activate`
3. Start server: `python app.py`
4. Open browser: http://localhost:5000

---

## API ENDPOINTS (for reference)

| Method | Endpoint                    | Description               |
|--------|-----------------------------|---------------------------|
| POST   | /api/login                  | Login with credentials    |
| POST   | /api/logout                 | Logout                    |
| GET    | /api/me                     | Get current user          |
| GET    | /api/data/historical        | Get all historical data   |
| GET    | /api/data/latest            | Get latest month data     |
| POST   | /api/data/add               | Add new monthly record    |
| GET    | /api/forecast?months=6      | Get LSTM forecast         |
| GET    | /api/attention              | Get attention weights     |
| GET    | /api/model/metrics          | Get model performance     |
| POST   | /api/model/retrain          | Trigger model retraining  |
| GET    | /api/analytics/seasonal     | Seasonal pattern data     |
| GET    | /api/analytics/summary      | Summary statistics        |
| GET    | /api/alerts                 | Get shortage alerts       |

---

## PROJECT FEATURES

- Bidirectional LSTM with 2 stacked layers (128 → 64 units)
- Bahdanau soft-attention mechanism between LSTM layers
- Bayesian hyperparameter optimization (Optuna)
- 8 blood groups: A+, A-, B+, B-, AB+, AB-, O+, O-
- Real-time demand ticker (updates every 5 seconds)
- 3-12 month forecast horizon with confidence intervals
- Shortage risk alerts (critical/warning/info)
- Data entry to add new records and retrain model
- 7 dashboard pages: Dashboard, Forecast, Analytics, Alerts, Data Entry, Reports, Settings
- Role-based login (admin/doctor/staff)
- REST API backend (Flask)

---

## COMMON ERRORS & FIXES

**"python is not recognized"**
→ Reinstall Python and check "Add to PATH"

**"No module named flask"**
→ Make sure venv is activated: `venv\Scripts\activate`

**"Model not found"**
→ Run `python train.py` before starting the server

**Port 5000 already in use**
→ Change port in app.py: `app.run(port=5001)`

**TensorFlow install fails**
→ Try: `pip install tensorflow-cpu` instead

---

## TECH STACK

| Layer     | Technology                  |
|-----------|-----------------------------|
| Frontend  | HTML5, CSS3, JavaScript     |
| Charts    | Chart.js 4.4                |
| Backend   | Python 3.12, Flask 3.0      |
| ML Model  | TensorFlow 2.15, Keras      |
| Optimizer | Optuna (Bayesian)           |
| Data      | Pandas, NumPy, Scikit-learn |
| Storage   | CSV (upgrade to PostgreSQL) |

---

© 2026 HemaForecast AI · BiLSTM v2.4
