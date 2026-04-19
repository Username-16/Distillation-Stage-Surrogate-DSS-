"""
models/baselines.py  —  Phase 2
=================================
M2.2  XGBSurrogate  — 3 scalar column outputs  (N_stages, feed_stage, R_actual)
M2.3  MLPSurrogate  — full x_profile  (MAX_S=80 padded vector)

Both follow sklearn-style  fit / predict / evaluate / save / load.

Pickle fix: all nn.Module subclasses are at MODULE LEVEL so joblib can find them.
Save strategy: joblib for config, torch.save for weights (avoids pickling nn.Module).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
MAX_S      = 80
FEAT_COLS  = ["PaVap","PbVap","alpha","xd","xb","xf","q","R_factor","nm"]
SCALAR_COLS= ["N_stages","feed_stage","R_actual"]


# ─────────────────────────────────────────────────────────────────────────────
# _MLPNet — MODULE LEVEL (required for pickle)
# Receives hidden tuple as __init__ arg instead of using a closure.
# ─────────────────────────────────────────────────────────────────────────────
class _MLPNet(nn.Module):
    def __init__(self, hidden: tuple = (256, 256, 128)):
        super().__init__()
        layers = []
        in_dim = len(FEAT_COLS)
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, MAX_S))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# ─────────────────────────────────────────────────────────────────────────────
# XGBSurrogate
# ─────────────────────────────────────────────────────────────────────────────
class XGBSurrogate:
    """Predicts 3 scalar targets: N_stages, feed_stage, R_actual."""

    def __init__(self, n_estimators=500, max_depth=6, lr=0.05):
        try:
            from xgboost import XGBRegressor
            base = XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                verbosity=0, n_jobs=-1)
            self._backend = "xgboost"
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            base = GradientBoostingRegressor(
                n_estimators=min(n_estimators, 200), max_depth=max_depth,
                learning_rate=lr, subsample=0.8, random_state=42)
            self._backend = "sklearn_gb"
        from sklearn.multioutput import MultiOutputRegressor
        self.model = MultiOutputRegressor(base, n_jobs=-1)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "XGBSurrogate":
        self.model.fit(X, Y); return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> np.ndarray:
        return np.array([e.feature_importances_
                         for e in self.model.estimators_]).mean(axis=0)

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> dict:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        preds = self.predict(X)
        out = {}
        for i, col in enumerate(SCALAR_COLS):
            out[col] = dict(
                r2   = round(float(r2_score(Y[:,i],    preds[:,i])), 5),
                mae  = round(float(mean_absolute_error(Y[:,i], preds[:,i])), 5),
                rmse = round(float(np.sqrt(mean_squared_error(Y[:,i], preds[:,i]))), 5))
        return out

    def save(self, path: str) -> None:
        import joblib, os
        os.makedirs(str(Path(path).parent), exist_ok=True)
        joblib.dump(self, path)        # XGB is pure sklearn — safe to joblib directly

    @staticmethod
    def load(path: str) -> "XGBSurrogate":
        import joblib; return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# MLPSurrogate
# ─────────────────────────────────────────────────────────────────────────────
class MLPSurrogate:
    """
    Predicts full x_profile (MAX_S values) from 9 column-spec inputs.
    Uses PyTorch when available, falls back to sklearn MLPRegressor.
    """

    def __init__(self, hidden: tuple = (256, 256, 128), max_iter: int = 400):
        self.hidden   = tuple(hidden)
        self.max_iter = max_iter
        self._net: _MLPNet | None = None
        self._device  = "cpu"
        try:
            import torch  # noqa
            self._backend = "pytorch"
        except ImportError:
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden, activation="relu",
                max_iter=max_iter, early_stopping=True,
                validation_fraction=0.15, random_state=42,
                learning_rate_init=3e-4)
            self._backend = "sklearn_mlp"

    def fit(self, X: np.ndarray, Xp: np.ndarray,
            mask: np.ndarray = None) -> "MLPSurrogate":
        if self._backend == "sklearn_mlp":
            self.model.fit(X, Xp)
        else:
            self._fit_pytorch(X, Xp, mask)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._backend == "sklearn_mlp":
            return self.model.predict(X)
        return self._predict_pytorch(X)

    def evaluate(self, X: np.ndarray, Xp: np.ndarray,
                 mask: np.ndarray) -> dict:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        pred  = self.predict(X)
        valid = mask.astype(bool)
        pv, tv = pred[valid].flatten(), Xp[valid].flatten()
        return dict(
            r2   = round(float(r2_score(tv, pv)), 5),
            mae  = round(float(mean_absolute_error(tv, pv)), 5),
            rmse = round(float(np.sqrt(mean_squared_error(tv, pv))), 5))

    # ── save / load  (split: joblib config + torch weights) ──────────────────
    def save(self, path: str) -> None:
        import joblib, os
        os.makedirs(str(Path(path).parent), exist_ok=True)
        if self._backend == "pytorch" and self._net is not None:
            # Save weights separately
            state = {k: v.cpu() for k, v in self._net.state_dict().items()}
            torch.save(state, str(path) + ".pt")
            # Save config without the network
            net_backup, self._net = self._net, None
            joblib.dump(self, path)
            self._net = net_backup
        else:
            joblib.dump(self, path)     # sklearn backend — safe directly

    @staticmethod
    def load(path: str) -> "MLPSurrogate":
        import joblib
        surrogate: MLPSurrogate = joblib.load(path)
        pt_path = str(path) + ".pt"
        if Path(pt_path).exists():
            state = torch.load(pt_path, map_location="cpu", weights_only=True)
            net = _MLPNet(surrogate.hidden)
            net.load_state_dict(state)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            surrogate._net    = net.to(device)
            surrogate._device = device
        return surrogate

    # ── internal PyTorch training ─────────────────────────────────────────────
    def _fit_pytorch(self, X, Xp, mask, epochs=150, lr=3e-4, bs=64):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import TensorDataset, DataLoader

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # _MLPNet is at module level — pickle-safe
        net = _MLPNet(self.hidden).to(device)

        _mask = mask if mask is not None else np.ones_like(Xp)
        dl = DataLoader(
            TensorDataset(
                torch.tensor(X,      dtype=torch.float32),
                torch.tensor(Xp,     dtype=torch.float32),
                torch.tensor(_mask,  dtype=torch.float32)),
            batch_size=bs, shuffle=True, drop_last=True)

        opt = AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sch = CosineAnnealingLR(opt, T_max=epochs)
        best_loss, best_state = float("inf"), None

        for ep in range(epochs):
            net.train(); total = 0.0
            for xb, yb, mb in dl:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                loss = ((net(xb) - yb) ** 2 * mb).sum() / (mb.sum() + 1e-8)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); total += loss.item()
            sch.step()
            avg = total / len(dl)
            if avg < best_loss:
                best_loss  = avg
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        net.load_state_dict(best_state)
        self._net = net

    def _predict_pytorch(self, X) -> np.ndarray:
        Xt = torch.tensor(X, dtype=torch.float32).to(self._device)
        self._net.eval()
        with torch.no_grad():
            return self._net(Xt).cpu().numpy()
