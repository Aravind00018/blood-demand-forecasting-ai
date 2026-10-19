# 🩸 HemaForecast AI — Vercel Deployment Guide

> **Live Production Status**:  
> 🚀 **Live Production URL**: [https://hemaforecast.vercel.app](https://hemaforecast.vercel.app)  
> 🔗 **Latest Deployment URL**: [https://hemaforecast-rfgqxmvlc-hustler17.vercel.app](https://hemaforecast-rfgqxmvlc-hustler17.vercel.app)  
> 📊 **Vercel Inspection Dashboard**: [https://vercel.com/hustler17/hemaforecast/8A9SLVNjAVa52XTy9Wqq7qk1VeF6](https://vercel.com/hustler17/hemaforecast/8A9SLVNjAVa52XTy9Wqq7qk1VeF6)  
> ✅ **State**: `READY` (Production Aliased)

This guide provides an end-to-end operational blueprint for deploying the **HemaForecast AI** frontend to **Vercel**, including static asset routing, configuration files, CLI deployment commands, and linking the frontend with the Python/Flask ML backend API.

---

## 1. Executive Summary & Architecture Overview

**HemaForecast AI** is a clinical blood demand forecasting system featuring a BiLSTM + Bahdanau Attention model with Bayesian Optimization across 8 blood groups.

The repository is structured as a decoupled architecture:
- **Frontend**: A zero-build, static Single-Page Application (SPA) utilizing Vanilla HTML5, modern CSS3, and Vanilla JavaScript with Chart.js.
- **Backend**: A Python Flask 3.0 REST API powered by TensorFlow 2.15, Optuna, Scikit-Learn, and Pandas.

```
                            ┌──────────────────────────────────────────────┐
                            │                 Client Browser               │
                            └──────────────────────┬───────────────────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        │                                                     │
               Static Assets (HTML/CSS/JS)                               API Requests
                        │                                                     │
                        ▼                                                     ▼
        ┌──────────────────────────────┐                      ┌──────────────────────────────┐
        │        Vercel Global         │                      │    Python ML Backend Host    │
        │        Edge Network          │                      │  (Render / Railway / Fly.io) │
        │    (frontend/ directory)     │                      │      (Flask + TensorFlow)    │
        └──────────────────────────────┘                      └──────────────────────────────┘
```

---

## 2. Detected Repository Architecture & Exact File Paths

Inspection of `E:\Blood_Project\hemaforecast` reveals the following structure:

```
E:\Blood_Project\hemaforecast\
├── backend\
│   ├── __init__.py                     # Package initializer (18 B)
│   └── model.py                        # TensorFlow BiLSTM + Bahdanau Attention ML logic (12.4 KB)
├── frontend\
│   ├── index.html                      # Primary Single Page Application UI & dashboard (15.6 KB)
│   └── static\
│       ├── css\
│       │   └── style.css               # Design system, dark theme, layout, keyframes (16.8 KB)
│       └── js\
│           └── app.js                  # App controller, Chart.js integrations, API client (28.9 KB)
├── app.py                              # Flask application serving API & static assets (16.8 KB)
├── generate_data.py                    # 48-month synthetic blood demand data generator (1.9 KB)
├── train.py                            # Model training & Bayesian optimization runner (2.1 KB)
├── requirements.txt                    # Python backend & ML dependencies (197 B)
├── README.md                           # Project documentation (1.3 KB)
├── .gitignore                          # Standard Python / environment ignores (4.8 KB)
└── {backend,frontend\                  # Errant artifact directory from shell brace expansion
    └── static\
        └── {css,js},data,models,notebooks}\
```

### File-by-File Inventory

| Exact Path | Size | Layer | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `frontend/index.html` | 15.6 KB | Frontend | Master SPA template containing Login view, Topbar navigation, and 7 functional views: Dashboard, Forecast, Analytics, Alerts, Data Entry, Reports, Settings. |
| `frontend/static/css/style.css` | 16.8 KB | Frontend | Comprehensive stylesheet: custom variables (`--bg`, `--red`, `--teal`), DM Mono monospace typography, glassmorphism cards, interactive switches, custom scrollbars, and keyframe animations. |
| `frontend/static/js/app.js` | 28.9 KB | Frontend | Core client runtime: Chart.js rendering (line, doughnut, bar, loss curves), dynamic DOM rendering, client-side fallback simulators (`simHist`, `simFcast`, `simCI`), ticker updates, and REST API calls via `fetch`. |
| `app.py` | 16.8 KB | Backend | Flask 3.0 API with 13 endpoints (`/api/login`, `/api/forecast`, `/api/attention`, `/api/alerts`, `/api/data/*`, `/api/model/*`, `/api/analytics/*`) and session-based authentication. |
| `backend/model.py` | 12.4 KB | Backend | Deep learning implementation: `BahdanauAttention` custom Keras layer, BiLSTM network layers, Optuna Bayesian hyperparameter optimizer, and scaler persistence. |
| `generate_data.py` | 1.9 KB | ML / Data | Synthesizes 48 months of blood usage data across 8 groups (`A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-`) with seasonal variance and accident spikes to `data/blood_demand.csv`. |
| `train.py` | 2.1 KB | ML / Pipeline | Trains the BiLSTM model, fits `MinMaxScaler`, and serializes artifacts to `models/best_model.keras`, `models/scaler.pkl`, and `models/model_meta.json`. |
| `requirements.txt` | 197 B | Backend | Dependencies: `flask`, `flask-cors`, `numpy`, `pandas`, `scikit-learn`, `tensorflow==2.15.0`, `optuna==3.5.0`, `joblib`, etc. |

> [!NOTE]
> **Artifact Cleanup Notice**:
> The folder named `{backend,frontend` is an artifact created when a bash-style brace expansion command (e.g., `mkdir -p {backend,frontend/...}`) was executed in Windows PowerShell without bash expansion. It is empty and should be deleted prior to committing or deploying:
> ```powershell
> Remove-Item -Recurse -Force "{backend,frontend"
> ```

---

## 3. Detected Technology Stack

### Frontend Stack Analysis
- **Classification**: **Pure Static Web Application (Zero-Build / Static SPA)**.
- **Build Tooling**: None required. No `package.json`, no Node.js compilation, no bundler (No Vite, Webpack, Rollup, or Next.js).
- **Core Languages**: HTML5, Vanilla CSS3, Vanilla ECMAScript (ES6+ async/await).
- **External CDN Dependencies**:
  - Google Fonts: `DM Mono` (weights 300, 400, 500) and `Fraunces` (weights 300, 500, 700).
  - Data Visualization: `Chart.js 4.4.1` UMD loaded from `https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js`.
- **Internal Asset Referencing**:
  - In `frontend/index.html`:
    - `<link rel="stylesheet" href="static/css/style.css">`
    - `<script src="static/js/app.js"></script>`

### Backend Stack Analysis
- **Classification**: Python 3.10+ / Flask 3.0 WSGI REST API.
- **Machine Learning**: TensorFlow 2.15.0, Keras, Scikit-Learn, Optuna.
- **Authentication**: Session cookie-based (`flask.session`, signed cookies with `credentials: 'include'`).
- **Data Persistence**: CSV file store (`data/blood_demand.csv`) and Keras HDF5/Keras model binaries (`models/`).

---

## 4. Vercel Deployment Instructions

Because the frontend is a static web application contained in the `frontend/` directory, it can be deployed to Vercel's global Edge CDN.

### 4.1 Prerequisites
1. **Node.js**: Install Node.js (v18+) if not already installed.
2. **Vercel CLI**: Install globally via npm:
   ```bash
   npm install -g vercel
   ```
   *Alternatively, execute on-demand using `npx vercel`.*
3. **Vercel Authentication**:
   ```bash
   vercel login
   ```
   Follow the CLI prompt to authenticate via GitHub, GitLab, Bitbucket, or Email.

---

### 4.2 Deployment Strategy A: Deploy from Project Root with `vercel.json` (Recommended)

This strategy deploys from the repository root `E:\Blood_Project\hemaforecast` without restructuring files, using a `vercel.json` configuration file.

#### Step 1: Create `vercel.json` in the Root Folder
Create `E:\Blood_Project\hemaforecast\vercel.json` with the following content:

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "version": 2,
  "public": false,
  "cleanUrls": true,
  "trailingSlash": false,
  "outputDirectory": "frontend",
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://YOUR_BACKEND_API_DOMAIN/api/:path*"
    },
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "headers": [
    {
      "source": "/static/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    },
    {
      "source": "/index.html",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "no-cache, no-store, must-revalidate"
        }
      ]
    }
  ]
}
```

> Replace `https://YOUR_BACKEND_API_DOMAIN` with your deployed Python API base URL (see Section 5). If your backend is not yet deployed, omit or comment out the `/api/:path*` rewrite block temporarily; the frontend will smoothly run in **client-side simulation mode**.

#### Step 2: Deploy to Preview Environment
From `E:\Blood_Project\hemaforecast`:

```powershell
vercel
```

**Interactive Prompts & Expected Responses:**
- `Set up and deploy "E:\Blood_Project\hemaforecast"?`: **`Y`**
- `Which scope do you want to deploy to?`: Select your personal account or team.
- `Link to existing project?`: **`N`** (or `Y` if updating an existing project)
- `What's your project's name?`: `hemaforecast-ai`
- `In which directory is your code located?`: `./`
- `Want to modify these settings?`: **`N`** (the `vercel.json` file handles all configurations)

#### Step 3: Deploy to Production
To push to the live production URL:

```powershell
vercel --prod
```

To deploy in automated environments (CI/CD) skipping confirmation prompts:
```powershell
vercel --prod --yes
```

---

### 4.3 Deployment Strategy B: Deploy Directly from `frontend/` Subdirectory

If you prefer to isolate the deployment strictly to the `frontend` folder:

#### Step 1: Navigate to the `frontend` directory
```powershell
cd E:\Blood_Project\hemaforecast\frontend
```

#### Step 2: Create a local `vercel.json` inside `frontend/`
Create `E:\Blood_Project\hemaforecast\frontend\vercel.json`:

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "version": 2,
  "cleanUrls": true,
  "trailingSlash": false,
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://YOUR_BACKEND_API_DOMAIN/api/:path*"
    },
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

#### Step 3: Run Deployment CLI
```powershell
vercel --prod
```

---

### 4.4 Deployment Strategy C: Vercel Web Dashboard (Git Integration)

If deploying via GitHub (`origin https://github.com/Aravind00018/blood-demand-forecasting-ai.git`):

1. Go to [https://vercel.com/new](https://vercel.com/new).
2. Select your repository `blood-demand-forecasting-ai`.
3. Under **Project Settings**:
   - **Framework Preset**: Choose **`Other`**.
   - **Root Directory**: Click *Edit* and select **`frontend`** (or leave as `./` if using root `vercel.json`).
   - **Build Command**: Leave empty / toggle off.
   - **Output Directory**: Leave empty (or set to `frontend` if Root Directory is `./`).
4. Click **Deploy**.

---

## 5. Linking the Frontend to the Python Backend API

### 5.1 Architecture Reality: Why the Backend Cannot Run on Vercel Functions
The Python backend contains:
1. **TensorFlow 2.15.0**: The uncompressed library exceeds **1.2 GB**. Vercel Serverless Functions enforce a strict **250 MB** maximum uncompressed bundle limit.
2. **Background Training Threads**: `/api/model/retrain` spawns a Python `threading.Thread` to execute epochs and Optuna trials, which will be terminated immediately when serverless execution completes.
3. **Local File Persistence**: Model artifacts (`.keras`, `.pkl`) and data (`.csv`) are written directly to local disk.

**Conclusion**: The Python ML backend must run on a container or persistent cloud host (e.g., **Render**, **Railway**, **Fly.io**, **AWS EC2 / App Runner**, or **Google Cloud Run**), while Vercel serves the static frontend globally.

---

### 5.2 Option 1: Vercel Reverse Proxy Rewrites (Zero-CORS, Recommended)

This is the cleanest and most robust approach. The browser makes requests directly to the Vercel domain (`/api/...`), and Vercel transparently forwards them to the remote Python server.

```
Browser  ───▶  https://hemaforecast.vercel.app/api/forecast
                           │  (Transparent Vercel Edge Rewrite)
                           ▼
               https://hemaforecast-api.onrender.com/api/forecast
```

#### Benefits:
- **Zero CORS issues**: Browser treats `/api` as Same-Origin.
- **Session Cookies work automatically**: `credentials: 'include'` will store and send session cookies without browser third-party cookie restrictions (Safari ITP / Chrome third-party cookie phaseout).
- **No changes required in `app.js`**: `const API = '';` in `frontend/static/js/app.js:2` functions out-of-the-box.

#### Configuration:
In your `vercel.json`:
```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://hemaforecast-api.onrender.com/api/:path*"
    }
  ]
}
```

---

### 5.3 Option 2: Direct Cross-Origin API Calls

If you prefer the frontend to call the backend URL directly across origins (e.g. `https://api.yourdomain.com`):

#### 1. Update `frontend/static/js/app.js`
Modify line 2 of `frontend/static/js/app.js` to support an external endpoint while allowing local fallback:

```javascript
// frontend/static/js/app.js
const API = window.API_BASE_URL || (window.location.hostname === 'localhost' ? '' : 'https://YOUR_BACKEND_API_DOMAIN');
```

Alternatively, inject `window.API_BASE_URL` in `frontend/index.html`:
```html
<script>
  window.API_BASE_URL = "https://YOUR_BACKEND_API_DOMAIN";
</script>
<script src="static/js/app.js"></script>
```

#### 2. Update Flask Backend (`app.py`) for Cross-Origin Cookies
When calling cross-origin, Flask's session cookies must have `SameSite=None` and `Secure=True`, and `flask_cors` must explicitly permit the Vercel origin with credentials:

```python
# app.py adjustments for Cross-Origin Production
import os
from flask_cors import CORS

ALLOWED_ORIGINS = [
    "http://localhost:5000",
    "https://hemaforecast-ai.vercel.app",  # Your production Vercel domain
    r"https://.*\.vercel\.app"             # All Vercel preview branch deployments
]

CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS)

app.config.update(
    SESSION_COOKIE_SECURE=True,      # Required for SameSite=None
    SESSION_COOKIE_SAMESITE='None',  # Allow cross-origin cookie sharing
    SESSION_COOKIE_HTTPONLY=True
)
```

---

### 5.4 Backend Deployment Quickstart (Render / Railway / Docker)

To quickly host the backend for connection to Vercel:

#### Dockerfile for Backend
Create a `Dockerfile` in `E:\Blood_Project\hemaforecast\Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

COPY . .

# Generate data and pre-train model on image build (optional)
RUN python generate_data.py
RUN python train.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
```

#### Deploy to Render.com:
1. Connect repository `blood-demand-forecasting-ai` to [Render](https://render.com).
2. Create a **Web Service**.
3. Environment: **Docker** (or Python with build command `pip install -r requirements.txt && python generate_data.py && python train.py`).
4. Start Command: `gunicorn -b 0.0.0.0:$PORT app:app`.
5. Copy the generated Render URL (e.g., `https://hemaforecast-api.onrender.com`) and paste it into `vercel.json` rewrites.

---

## 6. Pre-Deployment & Verification Checklist

Before running `vercel --prod`, execute this checklist:

### Pre-Deployment Verification
- [ ] **Remove Errant Directory**: Execute `Remove-Item -Recurse -Force "{backend,frontend"` to clean up the workspace.
- [ ] **Local Asset Integrity**: Ensure `frontend/index.html` loads all assets without HTTP 404:
  - `static/css/style.css` resolves to `frontend/static/css/style.css`.
  - `static/js/app.js` resolves to `frontend/static/js/app.js`.
- [ ] **Client Simulation Mode**: Verify the dashboard operates even when the backend is offline (charts, forecast tables, and tickers use client-side generators `simHist`, `simFcast`, and `simCI`).
- [ ] **Confirm `vercel.json`**: Ensure `vercel.json` exists in `E:\Blood_Project\hemaforecast\`.

### Post-Deployment Verification
1. **Load Dashboard**: Open `https://your-project.vercel.app` in the browser.
2. **Inspect Console & Network Tab**:
   - Verify `style.css` returns HTTP 200.
   - Verify `app.js` returns HTTP 200.
   - Verify Chart.js and Google Fonts load without CSP/Mixed Content warnings.
3. **Login Test**:
   - Use demo credentials:
     - Username: `admin` | Password: `admin123`
     - Username: `doctor` | Password: `med2024`
     - Username: `staff` | Password: `staff123`
4. **Backend Link Test**:
   - Check the dashboard header status:
     - If linked to backend: displays **`LSTM Loaded ✓`** or **`Simulation Mode (run train.py)`**.
     - If backend is offline: gracefully falls back to client simulation mode.
5. **Chart Interactivity**:
   - Toggle blood group buttons (`O+`, `A+`, `B+`, etc.) on the Forecast view.
   - Switch horizon slider from 1 to 12 months.
   - Confirm confidence interval bands render dynamically.

---

## 7. Useful Vercel CLI Commands Reference

| Action | Command | Purpose |
| :--- | :--- | :--- |
| **Deploy Preview** | `vercel` | Deploys a unique preview branch URL. |
| **Deploy Production** | `vercel --prod` | Deploys directly to the live production domain. |
| **Deploy Unattended** | `vercel --prod --yes` | Skips all interactive confirmation questions. |
| **Inspect Logs** | `vercel logs <deployment-url>` | Views real-time runtime and access logs. |
| **Manage Domains** | `vercel domains ls` | Lists all assigned custom domains. |
| **List Deployments**| `vercel ls` | Displays recent deployments with status and URLs. |
| **Pull Environment**| `vercel env pull` | Downloads project environment variables locally. |
