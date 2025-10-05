from pathlib import Path
import pandas as pd
import numpy as np

# --- Paths ---
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ALL = DATA / "train_exoplanets.csv"
OUT_TOI = DATA / "clean_toi.csv"

print(f"[data_prep] Using: {ALL}")
if not ALL.exists():
    raise FileNotFoundError(f"Expected file not found: {ALL}")

# --- Helper: safe fill with per-column median if available; else keep NaN ---
def safe_fill_median(df: pd.DataFrame, cols, as_int: bool=False):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            df[c] = s.fillna(s.median())
        else:
            df[c] = s  # keep as NaN if the whole column is empty
        if as_int:
            df[c] = df[c].astype("Int64")  # nullable int
    return df

# --- Load merged dataset and basic schema check ---
df = pd.read_csv(ALL)
required_base = ["source", "period_days", "duration_hr", "depth_pct", "label_binary"]
for r in required_base:
    if r not in df.columns:
        raise ValueError(f"Required column missing in train_exoplanets.csv: {r}")

# ======================================================
# 🚀 TOI ONLY
# ======================================================
toi = df[df["source"] == "TOI"].copy()
if toi.empty:
    raise ValueError("[TOI] No TOI rows found in train_exoplanets.csv")

# TOI often lacks SNR in the export → set to 0, pipeline can ignore/impute
toi["snr"] = 0.0

# Optional stellar/context columns; fill with real median if present; keep NaN if fully empty
optional_cols = ["st_tmag", "st_teff", "st_logg", "st_rad", "pl_rade"]
toi = safe_fill_median(toi, optional_cols)

# Derived feature
toi["period_days"] = pd.to_numeric(toi["period_days"], errors="coerce")
toi["duration_hr"] = pd.to_numeric(toi["duration_hr"], errors="coerce")
toi["depth_pct"] = pd.to_numeric(toi["depth_pct"], errors="coerce")
toi["dur_frac"] = toi["duration_hr"] / (24.0 * toi["period_days"])

# Sanity filters (soft but reasonable)
toi = toi.dropna(subset=["period_days", "duration_hr", "depth_pct", "label_binary"])
toi = toi[(toi["period_days"] > 0) & (toi["period_days"] < 1000)]
toi = toi[(toi["duration_hr"] > 0) & (toi["duration_hr"] < 24)]
toi = toi[(toi["depth_pct"] >= 0) & (toi["depth_pct"] <= 5)]

# Final column order: keep only what might be used in training
toi_cols = ["period_days", "duration_hr", "depth_pct", "snr",
            "st_tmag", "st_teff", "st_logg", "st_rad", "pl_rade",
            "dur_frac", "label_binary"]
toi_cols = [c for c in toi_cols if c in toi.columns]
toi = toi[toi_cols]

# Write
toi.to_csv(OUT_TOI, index=False)
print(f"[TOI] Wrote {OUT_TOI} rows: {len(toi)}")

# Label sanity
print("[TOI] Label counts:", toi["label_binary"].value_counts().to_dict())
