"""
training/train_baseline.py — Phase 2
XGBoost scalar surrogate + MLP stage profile surrogate.
Run: python -m training.train_baseline --data data/dataset.json
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from models.baselines   import XGBSurrogate, MLPSurrogate, FEAT_COLS, SCALAR_COLS
from training.utils     import load_dataset, masked_metrics


def main(data="data/data/dataset.json"):
    os.makedirs("models/checkpoints",  exist_ok=True)
    os.makedirs("experiments/figures", exist_ok=True)

    records, X, Ys, Xp, _, M = load_dataset(data)
    print(f"Dataset: {len(records)} records  |  features: {FEAT_COLS}")

    idx_tr, idx_tmp = train_test_split(np.arange(len(X)), test_size=.30, random_state=42)
    idx_va, idx_te  = train_test_split(idx_tmp, test_size=.50, random_state=42)
    sc  = StandardScaler().fit(X[idx_tr])
    Xtr = sc.transform(X[idx_tr])
    Xte = sc.transform(X[idx_te])

    # ── XGBoost: predict [N_stages, feed_stage, R_actual] ─────────────────────
    print("\n[Phase 2] XGBoost scalar surrogate...")
    xgb = XGBSurrogate()
    xgb.fit(Xtr, Ys[idx_tr])
    met = xgb.evaluate(Xte, Ys[idx_te])
    for col, v in met.items():
        print(f"  {col:<12} R²={v['r2']:.4f}  MAE={v['mae']:.4f}")
    xgb.save("models/checkpoints/xgb_surrogate.pkl")

    # ── MLP: predict full x_profile ───────────────────────────────────────────
    print("\n[Phase 2] MLP stage profile surrogate...")
    mlp = MLPSurrogate()
    mlp.fit(Xtr, Xp[idx_tr], M[idx_tr])
    mmet = mlp.evaluate(Xte, Xp[idx_te], M[idx_te])
    print(f"  R²={mmet['r2']:.4f}  MAE={mmet['mae']:.5f}")
    mlp.save("models/checkpoints/mlp_surrogate.pkl")

    results = {
        "split":   {"train": len(idx_tr), "val": len(idx_va), "test": len(idx_te)},
        "features": FEAT_COLS,
        "xgboost": {"metrics": met},
        "mlp":     {"metrics": mmet}
    }
    Path("experiments/baseline_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved experiments/baseline_results.json ✓")
    print("Git commit: feat(phase2): XGBoost + MLP baselines on PaVap/PbVap inputs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_PROJECT_ROOT / "data" /"data" / "dataset.json"))
    main(**vars(ap.parse_args()))
