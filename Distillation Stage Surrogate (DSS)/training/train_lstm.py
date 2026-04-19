"""
training/train_lstm.py  —  Phase 3 & 4
========================================
Phase 3 : LSTM with teacher forcing (MSE only)
Phase 4 : LSTM + PINN  (MSE + monotonicity + boundary + VLE losses)

Run:
  python -m training.train_lstm
  python -m training.train_lstm --epochs 100 --lambda_phys 0.2
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

# ── path setup (works from any working directory) ─────────────────────────────
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from models.lstm_surrogate import LSTMSurrogate
from losses.physics_loss   import combined_physics_loss
from training.utils        import load_dataset, masked_metrics, physics_residuals


# ── helpers ───────────────────────────────────────────────────────────────────

def _plot_parity(pred, tgt, mask, path, title):
    """Parity plot — predicted vs true x_profile on valid stages."""
    try:
        import matplotlib.pyplot as plt
        pv = pred[mask.astype(bool)].flatten()
        tv = tgt[mask.astype(bool)].flatten()
        # sub-sample to 5000 points for clarity
        idx = np.random.choice(len(pv), min(5000, len(pv)), replace=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(tv[idx], pv[idx], s=6, alpha=0.3, color="#01696f")
        mn, mx = min(tv.min(), pv.min()), max(tv.max(), pv.max())
        ax.plot([mn, mx], [mn, mx], "--", color="#a12c7b", lw=1.2)
        ax.set_xlabel("True x")
        ax.set_ylabel("Predicted x")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception:
        pass   # silently skip if matplotlib not available


def _plot_profiles(pred, tgt, mask, path, n=6):
    """Overlay predicted vs true x_profile for n random test samples."""
    try:
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pred), n, replace=False)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, i in zip(axes.flatten(), idx):
            N = int(mask[i].sum())
            stages = range(1, N + 1)
            ax.plot(stages, tgt[i, :N],  "b-o", ms=4, lw=1.5, label="McCabe-Thiele")
            ax.plot(stages, pred[i, :N], "r--s", ms=4, lw=1.5, label="LSTM")
            ax.set_xlabel("Stage"); ax.set_ylabel("x (liquid)")
            ax.set_title(f"Sample {i}  |  N={N}")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception:
        pass


def _save_results(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"  Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(data: str = None,
         epochs: int =80,
         lr: float = 3e-4,
         batch_size: int = 64,
         hidden: int = 256,
         num_layers: int = 2,
         lambda_phys: float = 0.15,
         tf_ratio: float = 0.8):

    # ── resolve paths ─────────────────────────────────────────────────────────
    if data is None:
        data = str(_PROJECT_ROOT / "data" /"data" / "dataset.json")
    ckpt_dir = _PROJECT_ROOT / "models" / "checkpoints"
    fig_dir  = _PROJECT_ROOT / "experiments" / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    records, X, _, Xp, Yp, M = load_dataset(data)
    print(f"Dataset : {len(records)} records")
    print(f"Features: PaVap, PbVap, alpha, xd, xb, xf, q, R_factor, nm")

    idx_tr, idx_tmp = train_test_split(np.arange(len(X)), test_size=0.30, random_state=42)
    idx_va, idx_te  = train_test_split(idx_tmp, test_size=0.50, random_state=42)

    sc  = StandardScaler().fit(X[idx_tr])
    Xtr = sc.transform(X[idx_tr])
    Xva = sc.transform(X[idx_va])
    Xte = sc.transform(X[idx_te])

    results = {
        "features"  : ["PaVap","PbVap","alpha","xd","xb","xf","q","R_factor","nm"],
        "split"     : {"train": len(idx_tr), "val": len(idx_va), "test": len(idx_te)},
        "hyperparams": {"epochs": epochs, "lr": lr, "batch_size": batch_size,
                        "hidden": hidden, "num_layers": num_layers,
                        "lambda_phys": lambda_phys, "tf_ratio": tf_ratio},
    }

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3 — Plain LSTM (teacher forcing, MSE only)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — LSTM  (teacher forcing, TF ratio=1.0)")
    print(f"{'='*60}")
    t0   = time.time()
    lstm = LSTMSurrogate(hidden_size=hidden, num_layers=num_layers)
    lstm.fit(Xtr, Xp[idx_tr], M[idx_tr],
             epochs=epochs, lr=lr, batch_size=batch_size,
             teacher_forcing_ratio=1.0)     # <-- pure teacher forcing
    elapsed_lstm = time.time() - t0

    pred_lstm = lstm.predict(Xte)
    met_lstm  = lstm.evaluate(Xte, Xp[idx_te], M[idx_te])
    phys_lstm = physics_residuals(pred_lstm, records, idx_te)

    print(f"\n  ── Test results ──")
    print(f"  R²   = {met_lstm['r2']:.4f}")
    print(f"  MAE  = {met_lstm['mae']:.5f}")
    print(f"  RMSE = {met_lstm['rmse']:.5f}")
    print(f"  Monotonicity violations = {phys_lstm['mono']:.6f}")
    print(f"  Boundary error          = {phys_lstm['bc']:.4f}")
    print(f"  Training time           = {elapsed_lstm:.1f}s")

    lstm.save(str(ckpt_dir / "lstm_surrogate.pkl"))
    _plot_parity(pred_lstm, Xp[idx_te], M[idx_te],
                 str(fig_dir / "lstm_parity.png"), "LSTM — x_profile parity")
    _plot_profiles(pred_lstm, Xp[idx_te], M[idx_te],
                   str(fig_dir / "lstm_profiles.png"))

    results["lstm"] = {
        "metrics"      : met_lstm,
        "physics"      : phys_lstm,
        "train_time_s" : round(elapsed_lstm, 1),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4 — LSTM + PINN  (teacher forcing at TF ratio + physics loss)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 4 — LSTM + PINN  (λ={lambda_phys}, TF ratio={tf_ratio})")
    print(f"  Loss = MSE + {lambda_phys}×(mono + BC + VLE)")
    print(f"{'='*60}")
    t0   = time.time()
    pinn = LSTMSurrogate(hidden_size=hidden, num_layers=num_layers)
    pinn.fit(Xtr, Xp[idx_tr], M[idx_tr],
             epochs=epochs, lr=lr, batch_size=batch_size,
             physics_loss_fn=combined_physics_loss,
             lambda_phys=lambda_phys,
             teacher_forcing_ratio=tf_ratio)
    elapsed_pinn = time.time() - t0

    pred_pinn = pinn.predict(Xte)
    met_pinn  = pinn.evaluate(Xte, Xp[idx_te], M[idx_te])
    phys_pinn = physics_residuals(pred_pinn, records, idx_te)

    print(f"\n  ── Test results ──")
    print(f"  R²   = {met_pinn['r2']:.4f}")
    print(f"  MAE  = {met_pinn['mae']:.5f}")
    print(f"  RMSE = {met_pinn['rmse']:.5f}")
    print(f"  Monotonicity violations = {phys_pinn['mono']:.6f}")
    print(f"  Boundary error          = {phys_pinn['bc']:.4f}")
    print(f"  Training time           = {elapsed_pinn:.1f}s")

    pinn.save(str(ckpt_dir / "lstm_pinn_surrogate.pkl"))
    _plot_parity(pred_pinn, Xp[idx_te], M[idx_te],
                 str(fig_dir / "pinn_parity.png"), "LSTM+PINN — x_profile parity")
    _plot_profiles(pred_pinn, Xp[idx_te], M[idx_te],
                   str(fig_dir / "pinn_profiles.png"))

    results["lstm_pinn"] = {
        "metrics"           : met_pinn,
        "physics"           : phys_pinn,
        "train_time_s"      : round(elapsed_pinn, 1),
        "milestone_M4_passed": phys_pinn["mono"] < phys_lstm["mono"],
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Final comparison table
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 3 vs 4 — Comparison")
    print(f"{'='*60}")
    print(f"  {'Model':<16} {'R²':>7} {'MAE':>9} {'Mono':>12} {'BC':>10}")
    print(f"  {'-'*57}")
    for name, m, p in [("LSTM", met_lstm, phys_lstm), ("LSTM+PINN", met_pinn, phys_pinn)]:
        print(f"  {name:<16} {m['r2']:>7.4f} {m['mae']:>9.5f} "
              f"{p['mono']:>12.6f} {p['bc']:>10.4f}")

    m4 = results["lstm_pinn"]["milestone_M4_passed"]
    print(f"\n  Milestone M4 (PINN reduces mono violations): "
          f"{'✓ PASS' if m4 else '✗ FAIL'}")

    _save_results(str(_PROJECT_ROOT / "experiments" / "lstm_results.json"), results)
    print("\n  Git commit: feat(phase3+4): LSTM + PINN surrogate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train LSTM + PINN surrogate (Phase 3 & 4)")
    ap.add_argument("--data",        default=None,  type=str)
    ap.add_argument("--epochs",      default=80,    type=int)
    ap.add_argument("--lr",          default=3e-4,  type=float)
    ap.add_argument("--batch_size",  default=64,    type=int)
    ap.add_argument("--hidden",      default=256,   type=int)
    ap.add_argument("--num_layers",  default=2,     type=int)
    ap.add_argument("--lambda_phys", default=0.15,  type=float)
    ap.add_argument("--tf_ratio",    default=0.8,   type=float)
    main(**vars(ap.parse_args()))
