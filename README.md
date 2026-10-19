# 🩸 HemaForecast AI — Blood Demand Prediction System

[![Live Demo](https://img.shields.io/badge/Vercel-Live_Demo-0070F3?style=for-the-badge&logo=vercel)](https://hemaforecast.vercel.app)  
🚀 **Live Deployment:** [https://hemaforecast.vercel.app](https://hemaforecast.vercel.app)

> BiLSTM + Bahdanau Attention model with Bayesian Optimization across 8 blood groups | Flask REST API

## 📊 Key Results
| Metric | Score |
|--------|-------|
| Accuracy | 95.8% |
| R² Score | 0.963 |
| Search Time Reduced | 70% |
| Inference | Real-time |

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | Python, TensorFlow, BiLSTM + Bahdanau Attention |
| Optimiser | Optuna (Bayesian Optimization / GPBO) |
| Backend API | Flask 3.0, REST API (13 endpoints) |
| Data | Pandas, NumPy, Scikit-learn |
| Frontend | HTML5, CSS3, JavaScript, Chart.js (Vercel Edge) |

## ⚙️ How to Run Locally
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

## 🔑 Demo Credentials
| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Administrator |
| doctor | med2024 | Medical Staff |
| staff | staff123 | Data Entry |

## 📡 API Endpoints
`POST /api/login` · `GET /api/forecast?months=6` · 
`GET /api/attention` · `GET /api/alerts` · 
`GET /api/model/metrics` · `POST /api/model/retrain`
