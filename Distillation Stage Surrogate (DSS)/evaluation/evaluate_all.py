"""
evaluation/evaluate_all.py  —  Phase 5
=======================================
Loads all 4 saved checkpoints and produces:
  experiments/final_comparison.json
  experiments/figures/fig1_r2_comparison.png
  experiments/figures/fig2_mae_comparison.png
  experiments/figures/fig3_physics_violations.png
  experiments/figures/fig4_training_time.png

Run:
  python -m evaluation.evaluate_all
"""
import json, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.baselines     import XGBSurrogate, MLPSurrogate, FEAT_COLS, MAX_S
from models.lstm_surrogate import LSTMSurrogate

COLORS  = ["#4f98a3", "#6daa45", "#d19900", "#01696f"]
FIG_DIR = Path("experiments/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_test_data(json_path=None):
    if json_path is None:
        json_path = PROJECT_ROOT / "data" /"data" / "dataset.json"
    records = json.loads(Path(json_path).read_text())
    def f(r):  return [r[c] for c in FEAT_COLS]
    def s(r):  return [r["N_stages"], r["feed_stage"], r["R_actual"]]
    def px(r):
        a = np.zeros(MAX_S, np.float32)
        a[:min(r["N_stages"], MAX_S)] = r["x_profile"][:MAX_S]
        return a
    def mk(r):
        m = np.zeros(MAX_S, np.float32)
        m[:min(r["N_stages"], MAX_S)] = 1.0
        return m
    X  = np.array([f(r)  for r in records], np.float32)
    Ys = np.array([s(r)  for r in records], np.float32)
    Xp = np.array([px(r) for r in records], np.float32)
    M  = np.array([mk(r) for r in records], np.float32)
    _, idx_tmp = train_test_split(np.arange(len(X)), test_size=.30, random_state=42)
    _, idx_te  = train_test_split(idx_tmp, test_size=.50, random_state=42)
    sc = StandardScaler().fit(X)
    return sc.transform(X)[idx_te], Ys[idx_te], Xp[idx_te], M[idx_te]


def infer_time_ms(model, X, runs=50):
    for _ in range(5): model.predict(X[:4])   # warmup
    t0 = time.perf_counter()
    for _ in range(runs): model.predict(X[:32])
    return (time.perf_counter() - t0) / runs * 1000


def vle_residual(pred, alpha, mask):
    """mean|y_pred - VLE(x_pred)|  over valid stages"""
    x   = np.clip(pred, 1e-6, 1-1e-6)
    y_e = alpha * x / (1 + (alpha-1)*x)       # Raoult VLE
    y_p = x                                    # surrogate doesn't predict y
    return float(np.abs(y_e - y_p)[mask.astype(bool)].mean())


def monotonicity(pred, mask):
    violations = 0.0; total = 0.0
    for i in range(len(pred)):
        n = int(mask[i].sum())
        if n < 2: continue
        p = pred[i, :n]
        violations += float(np.sum(np.maximum(0, np.diff(p))))
        total += (n-1)
    return violations / (total + 1e-8)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    fig_dir = PROJECT_ROOT / "experiments" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    Xte, Ys, Xp, M = load_test_data()

    # load checkpoints

    xgb  = XGBSurrogate.load(PROJECT_ROOT / "training" / "models" / "checkpoints" / "xgb_surrogate.pkl")
    mlp  = MLPSurrogate.load(PROJECT_ROOT /"training" / "models" / "checkpoints"  / "mlp_surrogate.pkl")
    lstm = LSTMSurrogate.load(PROJECT_ROOT / "models" / "checkpoints" / "lstm_surrogate.pkl")
    pinn = LSTMSurrogate.load(PROJECT_ROOT / "models" / "checkpoints" /"lstm_pinn_surrogate.pkl")

    alpha = Xte[:, 2:3]   # column index 2 = alpha feature

    results = {}

    for name, model in [("XGBoost",xgb),("MLP",mlp),("LSTM",lstm),("LSTM+PINN",pinn)]:
        # XGB predicts scalar targets; MLP/LSTM predict profiles
        if name == "XGBoost":
            pred_p = None
            m = model.evaluate(Xte, Ys)
            r2  = float(np.mean([v["r2"] for v in m.values()]))
            mae = float(np.mean([v["mae"] for v in m.values()]))
            rmse= float(np.mean([v["rmse"] for v in m.values()]))
        else:
            pred_p = model.predict(Xte)
            ev   = model.evaluate(Xte, Xp, M)
            r2, mae, rmse = ev["r2"], ev["mae"], ev["rmse"]

        spd  = infer_time_ms(model, Xte)
        mono = monotonicity(pred_p, M) if pred_p is not None else None
        vle  = vle_residual(pred_p, alpha, M) if pred_p is not None else None

        results[name] = dict(r2=r2, mae=mae, rmse=rmse,
                             speed_ms=round(spd,2),
                             monotonicity=mono, vle_residual=vle)
        print(f"  {name:<12} R²={r2:.4f}  MAE={mae:.5f}  speed={spd:.1f}ms")

    Path("experiments/final_comparison.json").write_text(
        json.dumps(results, indent=2))
    print("\nSaved experiments/final_comparison.json ✓")

    # ── figures ────────────────────────────────────────────────────────────────
    names  = list(results.keys())
    r2s    = [results[n]["r2"]  for n in names]
    maes   = [results[n]["mae"] for n in names]
    speeds = [results[n]["speed_ms"] for n in names]

    # Fig 1 — R² bar
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(names, r2s, color=COLORS)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0.93, 1.002); ax.set_ylabel("R² Score")
    ax.set_title("Model R² on Test Set (N=900)")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR/"fig1_r2_comparison.png", dpi=150); plt.close()

    # Fig 2 — MAE bar
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(names, maes, color=COLORS)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=9)
    ax.set_ylabel("MAE (x_LK)"); ax.set_title("Mean Absolute Error — Test Set")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR/"fig2_mae_comparison.png", dpi=150); plt.close()

    # Fig 3 — Physics violations (LSTM vs PINN only)
    phys_names = ["LSTM","LSTM+PINN"]
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(phys_names)); w=0.28
    ax.bar(x-w, [results[n]["monotonicity"] for n in phys_names],
           w, label="Monotonicity", color="#d19900")
    ax.bar(x,   [results[n]["vle_residual"]  for n in phys_names],
           w, label="VLE Residual", color="#a12c7b")
    ax.set_xticks(x); ax.set_xticklabels(phys_names)
    ax.set_ylabel("Violation Score"); ax.legend()
    ax.set_title("Physics Violations: LSTM vs LSTM+PINN")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR/"fig3_physics_violations.png", dpi=150); plt.close()

    # Fig 4 — Inference speed
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(names, speeds, color=COLORS)
    ax.bar_label(bars, fmt="%.1fms", padding=3, fontsize=9)
    ax.set_ylabel("Inference Time (ms/batch)")
    ax.set_title("Inference Speed per Model (batch=32, CPU)")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR/"fig4_inference_speed.png", dpi=150); plt.close()

    print("All 4 figures saved to experiments/figures/ ✓")


if __name__ == "__main__":
    main()