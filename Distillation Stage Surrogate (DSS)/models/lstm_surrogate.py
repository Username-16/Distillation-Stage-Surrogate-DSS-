"""
models/lstm_surrogate.py  —  Phase 3 & 4
=========================================
Autoregressive LSTM stage-by-stage predictor.

Input features : [PaVap, PbVap, alpha, xd, xb, xf, q, R_factor, nm]  (9)
Step context   : [enc | stage_frac | prev_x | xd | xb]

Training  : teacher forcing  (true x_{s-1} fed as input)
Inference : autoregressive   (own prediction fed as input)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
MAX_S    = 80
FEAT_DIM = 9
IDX_XD   = 3   # index of xd in feature vector
IDX_XB   = 4   # index of xb in feature vector


# ─────────────────────────────────────────────────────────────────────────────
# _LSTMNet  —  MUST be at module level for pickle/joblib to work.
#              Receives hidden_size / num_layers / dropout as __init__ args
#              instead of capturing them from a closure.
# ─────────────────────────────────────────────────────────────────────────────
class _LSTMNet(nn.Module):

    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        H, nl, dr = hidden_size, num_layers, dropout

        # Encode 9 column-spec features → hidden vector
        self.enc = nn.Sequential(
            nn.Linear(FEAT_DIM, H),
            nn.LayerNorm(H),
            nn.SiLU(),
        )
        # LSTM: [enc(H) | stage_frac(1) | prev_x(1) | xd(1) | xb(1)] → H
        self.lstm = nn.LSTM(
            H + 4, H, nl,
            batch_first=True,
            dropout=dr if nl > 1 else 0.0,
        )
        # Decoder: H → single x value in [0, 1]
        self.dec = nn.Sequential(
            nn.Linear(H, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def encode(self, feat: torch.Tensor) -> torch.Tensor:
        return self.enc(feat)

    def step(self, enc, stage_frac, prev_x, xd, xb, h, c):
        """Single autoregressive step."""
        inp = torch.cat([enc, stage_frac, prev_x, xd, xb], dim=-1).unsqueeze(1)
        out, (h, c) = self.lstm(inp, (h, c))
        return self.dec(out.squeeze(1)), h, c


# ─────────────────────────────────────────────────────────────────────────────
# LSTMSurrogate  —  sklearn-style wrapper around _LSTMNet
# ─────────────────────────────────────────────────────────────────────────────
class LSTMSurrogate:

    def __init__(self, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.1):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self._net: _LSTMNet | None = None
        self._device = "cpu"

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, X, Xp, mask, epochs=80, lr=3e-4, batch_size=64,
            physics_loss_fn=None, lambda_phys=0.15,
            teacher_forcing_ratio=1.0):

        import random
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import TensorDataset, DataLoader

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # Build network — passing args explicitly (no closure needed)
        net = _LSTMNet(self.hidden_size, self.num_layers, self.dropout).to(device)

        dl = DataLoader(
            TensorDataset(
                torch.tensor(X,    dtype=torch.float32),
                torch.tensor(Xp,   dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
            ),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )

        opt = AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sch = CosineAnnealingLR(opt, T_max=epochs)
        best_loss, best_state = float("inf"), None

        for ep in range(epochs):
            net.train(); total = 0.0

            for feat, xp_b, m_b in dl:
                feat  = feat.to(device)
                xp_b  = xp_b.to(device)
                m_b   = m_b.to(device)
                B     = feat.size(0)

                enc     = net.encode(feat)
                h = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
                c = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
                xd_v  = feat[:, IDX_XD:IDX_XD+1]
                xb_v  = feat[:, IDX_XB:IDX_XB+1]
                prev  = xd_v.clone()
                preds = []

                for s in range(MAX_S):
                    sf = torch.full((B, 1), s / (MAX_S - 1), device=device)
                    xp_out, h, c = net.step(enc, sf, prev, xd_v, xb_v, h, c)
                    preds.append(xp_out)
                    # Teacher forcing: feed true previous stage during training
                    if random.random() < teacher_forcing_ratio:
                        prev = xp_b[:, s:s+1]
                    else:
                        prev = xp_out.detach()

                preds_t = torch.cat(preds, dim=1)
                mse = ((preds_t - xp_b) ** 2 * m_b).sum() / (m_b.sum() + 1e-8)

                if physics_loss_fn is not None:
                    phys = physics_loss_fn(preds_t, xp_b, m_b, feat[:, 2:3])
                else:
                    phys = torch.tensor(0.0, device=device)

                loss = mse + lambda_phys * phys
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                total += loss.item()

            sch.step()
            avg = total / len(dl)
            if avg < best_loss:
                best_loss  = avg
                best_state = {k: v.cpu().clone()
                              for k, v in net.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  Epoch {ep+1:>3}/{epochs}  loss={avg:.6f}")

        net.load_state_dict(best_state)
        self._net = net
        return self

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, X) -> np.ndarray:
        net = self._net.eval()
        Xt  = torch.tensor(X, dtype=torch.float32).to(self._device)
        B   = Xt.size(0)
        enc = net.encode(Xt)
        h = torch.zeros(self.num_layers, B, self.hidden_size, device=self._device)
        c = torch.zeros(self.num_layers, B, self.hidden_size, device=self._device)
        xd_v = Xt[:, IDX_XD:IDX_XD+1]
        xb_v = Xt[:, IDX_XB:IDX_XB+1]
        prev = xd_v.clone()
        preds = []

        with torch.no_grad():
            for s in range(MAX_S):
                sf = torch.full((B, 1), s / (MAX_S - 1), device=self._device)
                xp_out, h, c = net.step(enc, sf, prev, xd_v, xb_v, h, c)
                preds.append(xp_out)
                prev = xp_out

        return torch.cat(preds, dim=1).cpu().numpy()

    # ── metrics ───────────────────────────────────────────────────────────────
    def evaluate(self, X, Xp, mask) -> dict:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        pred = self.predict(X)
        v    = mask.astype(bool)
        pv, tv = pred[v].flatten(), Xp[v].flatten()
        return dict(
            r2   = round(float(r2_score(tv, pv)), 5),
            mae  = round(float(mean_absolute_error(tv, pv)), 5),
            rmse = round(float(np.sqrt(mean_squared_error(tv, pv))), 5),
        )

    # ── save / load  (split: joblib for config, torch for weights) ────────────
    def save(self, path: str) -> None:
        """
        Saves TWO files:
          <path>       — joblib pickle of config (hidden_size, num_layers, dropout)
          <path>.pt    — torch state dict (model weights)
        """
        import joblib, os
        os.makedirs(str(Path(path).parent), exist_ok=True)

        # Save weights separately with torch (avoids pickling nn.Module internals)
        state = {k: v.cpu() for k, v in self._net.state_dict().items()}                 if self._net is not None else None
        torch.save(state, path + ".pt")

        # Save config wrapper without the network object
        net_backup, self._net = self._net, None
        joblib.dump(self, path)
        self._net = net_backup   # restore in memory

    @staticmethod
    def load(path: str) -> "LSTMSurrogate":
        """Loads config from joblib then reloads weights from .pt file."""
        import joblib
        surrogate: LSTMSurrogate = joblib.load(path)
        state = torch.load(str(path) + ".pt", map_location="cpu", weights_only=True)
        if state is not None:
            net = _LSTMNet(surrogate.hidden_size,
                           surrogate.num_layers,
                           surrogate.dropout)
            net.load_state_dict(state)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            surrogate._net    = net.to(device)
            surrogate._device = device
        return surrogate
