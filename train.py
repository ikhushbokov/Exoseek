from pathlib import Path
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ART = HERE / "artifacts"; ART.mkdir(exist_ok=True)
MDIR = HERE / "models";   MDIR.mkdir(exist_ok=True)

CSV = DATA / "clean_toi.csv"
MODEL_PATH = MDIR / "rf_toi.pkl"
METRICS_PATH = ART / "metrics_toi.json"

FEATURES = [
    "period_days","duration_hr","depth_pct","snr",
    # keep only those that exist with some data, we’ll filter dynamically
    "st_tmag","st_teff","st_logg","st_rad","pl_rade",
    "dur_frac"
]
LABEL = "label_binary"
THRESH = 0.50

def available_features(df, wanted):
    keep = []
    for c in wanted:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            keep.append(c)
    return keep

def main():
    if not CSV.exists():
        raise FileNotFoundError(f"Missing {CSV}. Run data_prep.py first.")
    df = pd.read_csv(CSV)

    feats = available_features(df, FEATURES)
    if not feats:
        raise ValueError("No usable features found in clean_toi.csv")

    X = df[feats].apply(pd.to_numeric, errors="coerce")
    y = df[LABEL].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    preds = (proba >= THRESH).astype(int)

    auc = roc_auc_score(yte, proba)
    rep = classification_report(yte, preds, output_dict=True)
    cm = confusion_matrix(yte, preds).tolist()

    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "model": "toi",
            "features_used": feats,
            "threshold": THRESH,
            "auc": auc,
            "classification_report": rep,
            "confusion_matrix": cm,
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte))
        }, f, indent=2)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"AUC: {auc:.3f}")

if __name__ == "__main__":
    main()
