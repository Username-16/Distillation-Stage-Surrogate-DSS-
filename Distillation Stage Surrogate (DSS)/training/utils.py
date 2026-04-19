"""training/utils.py — shared helpers for all training scripts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEAT_COLS   = ["PaVap","PbVap","alpha","xd","xb","xf","q","R_factor","nm"]
SCALAR_COLS = ["N_stages","feed_stage","R_actual"]
MAX_S       = 80


def load_dataset(json_path: str = None):
    """
    Loads dataset.json — always resolves path relative to project root.

    Returns: records, X (N,9), Ys (N,3), Xp (N,80), Yp (N,80), M (N,80)
    """
    if json_path is None:
        json_path = _PROJECT_ROOT / "data" / "dataset.json"

    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = _PROJECT_ROOT / json_path   # always anchor to project root

    if not json_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {json_path}\n"
            f"Run first:  python data/generator.py --samples 6000")

    records = json.loads(json_path.read_text())

    def _feat(r):   return [r[c] for c in FEAT_COLS]
    def _scalar(r): return [float(r["N_stages"]),
                             float(r["feed_stage"]),
                             float(r["R_actual"])]
    def _pad(r, key):
        a = np.zeros(MAX_S, np.float32)
        n = min(r["N_stages"], MAX_S)
        a[:n] = r[key][:n]
        return a
    def _mask(r):
        m = np.zeros(MAX_S, np.float32)
        m[:min(r["N_stages"], MAX_S)] = 1.0
        return m

    X  = np.array([_feat(r)          for r in records], dtype=np.float32)
    Ys = np.array([_scalar(r)        for r in records], dtype=np.float32)
    Xp = np.array([_pad(r,"x_profile") for r in records], dtype=np.float32)
    Yp = np.array([_pad(r,"y_profile") for r in records], dtype=np.float32)
    M  = np.array([_mask(r)          for r in records], dtype=np.float32)

    return records, X, Ys, Xp, Yp, M


def masked_metrics(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> dict:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    v  = mask.astype(bool)
    pv = pred[v].flatten()
    tv = tgt[v].flatten()
    return dict(
        r2   = round(float(r2_score(tv, pv)), 5),
        mae  = round(float(mean_absolute_error(tv, pv)), 5),
        rmse = round(float(np.sqrt(mean_squared_error(tv, pv))), 5))


def physics_residuals(pred: np.ndarray, records: list, idx) -> dict:
    """
    Compute per-sample physics metrics on test-set predictions.
    Returns mean monotonicity violation, boundary error, VLE residual.
    """
    mono_v, bc_e, vle_r = [], [], []

    for k, i in enumerate(idx):
        r  = records[i]
        N  = min(r["N_stages"], MAX_S)
        xp = pred[k, :N]
        al = r["alpha"]

        # 1. Monotonicity — sum of positive differences (should be zero)
        mono_v.append(float(np.maximum(0, np.diff(xp)).sum()))

        # 2. Boundary conditions
        bc_e.append(abs(xp[0] - r["xd"]) + abs(xp[-1] - r["xb_actual"]))

        # 3. VLE residual  |K_pred - K_eq| averaged over stages
        if "y_profile" in r and len(r["y_profile"]) >= N:
            yp    = np.array(r["y_profile"][:N])
            x_s   = np.maximum(xp, 1e-8)
            K_p   = yp / x_s
            K_eq  = al / (1 + (al - 1) * x_s)
            vle_r.append(float(np.mean(np.abs(K_p - K_eq))))
        else:
            vle_r.append(0.0)

    return {
        "mono": round(float(np.mean(mono_v)), 6),
        "bc"  : round(float(np.mean(bc_e)), 6),
        "vle" : round(float(np.mean(vle_r)), 6),
    }
