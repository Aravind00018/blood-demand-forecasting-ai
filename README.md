# 🩸 HemaForecast AI — Blood Demand Prediction System

> BiLSTM + Bahdanau Attention model with Bayesian Optimization  
> for clinical blood demand forecasting across 8 blood groups

## 📊 Key Results
| Metric | Score |
|--------|-------|
| Accuracy | 95.8% |
| R² Score | 0.963 |
| Hyperparameter search reduction | 70% |
| Inference | Real-time |

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | Python, TensorFlow, BiLSTM + Bahdanau Attention |
| Optimiser | Optuna (Bayesian / GPBO) |
| Backend API | Flask 3.0, REST API (13 endpoints) |
| Data | Pandas, NumPy, Scikit-learn |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |

## ⚙️ How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python generate_data.py

# 3. Train the model
python train.py

# 4. Start the server
python app.py

# 5. Open browser
http://localhost:5000
```

## 🔑 Login Credentials (Demo)
| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Administrator |
| doctor | med2024 | Medical Staff |

## 📡 API Endpoints
`POST /api/login` · `GET /api/forecast?months=6` · 
`GET /api/attention` · `GET /api/alerts` · 
`GET /api/model/metrics` · `POST /api/model/retrain`
