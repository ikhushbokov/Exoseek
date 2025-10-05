import json
import joblib
from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Exoplanet Hunter API")

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
ART_DIR = HERE / "artifacts"

MODEL_PATH = MODELS_DIR / "rf_toi.pkl"
METRICS_PATH = ART_DIR / "metrics_toi.json"

MODEL = None
FEATURES_USED: list[str] = []

def _load_features_used_fallback():
    # Fallback if metrics file is missing: use safe core TOI features
    return ["period_days", "duration_hr", "depth_pct", "snr", "dur_frac"]

def load_model():
    global MODEL, FEATURES_USED
    MODEL = None
    FEATURES_USED = []
    MODELS_DIR.mkdir(exist_ok=True)
    ART_DIR.mkdir(exist_ok=True)

    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        print(f"[loader] loaded TOI model from {MODEL_PATH}")
    else:
        print(f"[loader] MISSING model at {MODEL_PATH}")

    if METRICS_PATH.exists():
        try:
            meta = json.loads(METRICS_PATH.read_text())
            feats = meta.get("features_used")
            if isinstance(feats, list) and feats:
                FEATURES_USED = feats
            else:
                FEATURES_USED = _load_features_used_fallback()
                print("[loader] metrics_toi.json lacks 'features_used'; using fallback")
        except Exception as e:
            print(f"[loader] failed reading metrics: {e}")
            FEATURES_USED = _load_features_used_fallback()
    else:
        print(f"[loader] MISSING metrics at {METRICS_PATH}; using fallback features")
        FEATURES_USED = _load_features_used_fallback()

load_model()

def _build_row(period_days: float, duration_hr: float, depth_pct: float, snr: float) -> pd.DataFrame:
    # Build exactly the columns the model was trained with
    row = {c: np.nan for c in FEATURES_USED}
    if "period_days" in row: row["period_days"] = period_days
    if "duration_hr"  in row: row["duration_hr"]  = duration_hr
    if "depth_pct"    in row: row["depth_pct"]    = depth_pct
    if "snr"          in row: row["snr"]          = snr
    if "dur_frac"     in row:
        row["dur_frac"] = (duration_hr / (24.0 * period_days)) if period_days > 0 else np.nan
    return pd.DataFrame([row], columns=FEATURES_USED)

@app.get("/")
def root():
    return {
        "message": "OK",
        "model": "toi",
        "model_loaded": MODEL is not None,
        "features_used": FEATURES_USED
    }

@app.post("/reload")
def reload():
    load_model()
    return {
        "reloaded": True,
        "model_loaded": MODEL is not None,
        "features_used": FEATURES_USED
    }

@app.post("/predict")
def predict(
    period_days: float = Query(..., description="Orbital period (days)"),
    duration_hr: float = Query(..., description="Transit duration (hours)"),
    depth_pct: float = Query(..., description="Transit depth (%) e.g. 0.12 for 0.12%"),
    snr: float = Query(0.0, description="Signal-to-noise ratio (TESS often unknown; pass 0)")
):
    if MODEL is None:
        return {"error": "TOI model not loaded. Train it and place models/rf_toi.pkl."}
    try:
        X = _build_row(period_days, duration_hr, depth_pct, snr)
        prob = float(MODEL.predict_proba(X)[0, 1])
        label = "Likely Planet" if prob > 0.5 else "False Positive"
        return {
            "model": "toi",
            "prediction": label,
            "probability": round(prob, 3),
            "used_features": FEATURES_USED
        }
    except Exception as e:
        return {"error": f"Inference failed: {e}", "used_features": FEATURES_USED}
