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

---

### Example Endpoint

POST /predict/toi?period_days=3.5&duration_hr=3&depth_pct=1&snr=40


→ Returns a prediction label and probability score.

---

# 🧩 API Example

## **Request**
```bash
curl -X 'POST'   'http://127.0.0.1:8000/predict/toi?period_days=3.5&duration_hr=3&depth_pct=1&snr=40'   -H 'accept: application/json'   -d ''
```

---

## **Response**
```json
{
  "model": "toi",
  "prediction": "Likely Planet",
  "probability": 0.981
}
```

---

# 🛠️ Installation & Setup

## **1️⃣ Clone Repository**
```bash
git clone https://github.com/ikhushbokov/Exoseek.git
cd Exoseek
```

---

## **2️⃣ Create Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
```

---

## **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **4️⃣ Prepare Data**
```bash
python data_prep.py
```

---

## **5️⃣ Train Model**
```bash
python ai_train_models.py
```

---

## **6️⃣ Run API Server**
```bash
uvicorn api:app --reload
```

---

Then open your browser at 👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Here you can test predictions interactively using the Swagger interface.
