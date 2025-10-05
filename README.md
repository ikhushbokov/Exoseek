
# A World Away: Hunting for Exoplanets with AI — MVP Skeleton

This is a **hackathon-ready** skeleton that:
- downloads or reads **light curves** (time vs flux),
- runs a **Box Least Squares (BLS)** search to find periodic dips,
- extracts **features** (period, depth, duration, SNR, odd-even, secondary),
- trains a quick **RandomForest** (or XGBoost) on labeled rows,
- serves results via a tiny **FastAPI** backend for a mobile/web UI.

> ⚠️ This notebook environment has **no internet**, so the `lightkurve` download step is stubbed.
> Run `python pipeline.py --demo` to use **synthetic curves** and verify end-to-end.
> On your own machine, remove `--demo` and pass a list of TIC/KIC IDs to fetch real data.

## Quickstart (local dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A: demo with synthetic curves
python pipeline.py --demo --n-stars 50

# Option B: real data (requires internet; install lightkurve)
# python pipeline.py --mission TESS --targets "TIC 123456789,TIC 987654321"
# uvicorn api:app --reload
```

## Outputs
- `artifacts/candidates.json` – ranked candidates with features & scores
- `artifacts/plots/*` – raw/flattened/folded PNGs for each candidate

## API (FastAPI)
- `GET /candidates` → list of candidates
- `GET /candidate/{star_id}` → details + plot URLs

## Data Formats

**Light curve CSV (per star)**:
```
time,flux
0.0000,1.0002
0.0200,0.9997
...
```

Generated **feature table**:
```
star_id,period_days,depth_pct,duration_hr,snr,odd_even_depth_diff,secondary_eclipse_snr
TIC123,4.36,0.8,2.5,12.4,0.05,0.01
```
