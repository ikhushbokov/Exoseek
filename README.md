# 🌌 Exoseek – AI-Powered Exoplanet Classifier

**Exoseek** is an AI system that predicts whether an observed celestial object is a *Likely Exoplanet* or a *False Positive* using NASA’s **TESS (Transiting Exoplanet Survey Satellite)** dataset.  
It uses a machine learning (Random Forest) model trained on cleaned TOI (TESS Objects of Interest) data and serves predictions through a FastAPI endpoint.

---

## 🚀 Features
- 🪐 Predicts exoplanet probability from orbital and transit data  
- ⚙️ Machine learning model trained on TESS (TOI) dataset  
- 🌍 FastAPI backend for real-time predictions  
- 📈 Clean and modular data processing pipeline  

---

## 🧠 How It Works
1. **Data Preparation**  
   - `data_prep.py` cleans and filters raw NASA datasets.  
   - Only TESS (TOI) data is used for this version.  
   - The final cleaned file is `data/clean_toi.csv`.  

2. **Model Training**  
   - Run `ai_train_models.py` to train a Random Forest Classifier.  
   - The trained model is saved as `models/rf_toi.pkl`.

3. **API Deployment**  
   - `api.py` serves the trained model with **FastAPI**.  
   - Endpoint example:
     ```
     POST /predict/toi?period_days=3.5&duration_hr=3&depth_pct=1&snr=40
     ```
     → Returns prediction label and probability score.

---

## 🧩 API Example

**Request:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/toi?period_days=3.5&duration_hr=3&depth_pct=1&snr=40' \
  -H 'accept: application/json' -d ''
