# Distillation Stage Surrogate (DSS)

> A physics-informed deep learning surrogate that replaces McCabe-Thiele distillation solvers — predicting full stage-by-stage composition profiles in **< 35 ms** with **R² = 0.9929**, while enforcing VLE equilibrium, monotonicity, and boundary conditions through custom physics losses.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit-ff4b4b.svg)](app/streamlit_app.py)

---

## 🔬 Motivation

Industrial distillation is the most energy-intensive separation process in the chemical industry, representing ~40% of process plant operating costs. Rigorous MESH solvers (Mass balance, Equilibrium, Summation, Heat balance) take seconds per simulation — making real-time optimization and control impractical.

This project builds a surrogate model that:
- Replaces the McCabe-Thiele solver with a deep learning model trained on its outputs
- Predicts the **full liquid composition profile** across all trays (not just top/bottom)
- Embeds thermodynamic constraints directly into the loss function (PINN approach)
- Achieves **~10⁵× speedup** over the rigorous solver

**Real system:** Column T-101 from a benzene hydrogenation plant — separating methane / cyclohexane / benzene / H₂ at 1–10 bar, targeting 99.84 mol% cyclohexane purity.

---

## 📊 Results

All metrics evaluated on a held-out test set (N ≈ 900 samples):

| Model | R² | MAE | RMSE | Speed | Monotonicity Violations |
|---|---|---|---|---|---|
| XGBoost (scalar targets) | 0.9830 | — | — | 64 ms | — |
| MLP (flat profile) | 0.9874 | 0.01708 | 0.02942 | **0.2 ms** | 0.000231 |
| LSTM (teacher forcing) | 0.9896 | 0.01394 | 0.02672 | 33 ms | 0.0000242 |
| **LSTM + PINN** | **0.9929** | **0.01237** | **0.02212** | 33 ms | **0.0000112** |

**Key finding:** Adding physics-informed loss (PINN) to the LSTM reduces monotonicity violations by **54%** (0.0000242 → 0.0000112) and improves R² from 0.9896 → 0.9929 with negligible inference speed cost.

> XGBoost MAE is reported in original units (number of stages), not mol fraction — not directly comparable to profile MAE.

---

## 🏗️ Architecture

```
Input (9 features)
  PaVap, PbVap, xd, xb, xf, q, R_factor, nm → α = PaVap/PbVap

Phase 2 — Baselines
  ├── XGBoost → N_stages, feed_stage, R_actual (scalar targets)
  └── MLP     → x_LK[1..MAX_S] (flat vector, padded + masked)

Phase 3 — Sequence Model
  └── LSTM    → x_LK[s] one stage at a time (teacher forcing during train,
                autoregressive at inference)

Phase 4 — Physics-Informed
  └── LSTM + PINN → MSE loss + λ₁·L_mono + λ₂·L_BC + λ₃·L_VLE
                    where:
                    L_mono = Σ max(0, x[s] - x[s-1])     # must decrease top→bottom
                    L_BC   = (x[0]-xd)² + (x[-1]-xb)²   # boundary match
                    L_VLE  = |α·x/(1+(α-1)x) - x|        # Raoult's law residual
```

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/HamadAlmousa/distillation-stage-surrogate.git
cd distillation-stage-surrogate

# Install
pip install -r requirements.txt
pip install xgboost torch          # strongly recommended

# 1 — Generate dataset (Latin Hypercube Sampling, 6,000 simulations)
python data/generator.py --samples 6000 --seed 42

# 2 — Train baselines (XGBoost + MLP)
python -m training.train_baseline --data data/dataset.json

# 3 — Train LSTM + PINN
python -m training.train_lstm --data data/dataset.json --epochs 80

# 4 — Evaluate all models
python -m evaluation.evaluate_all

# 5 — Interactive demo
python -m streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
distillation-stage-surrogate/
├── data/
│   ├── mccabe_thiele_solver.py   # McCabe-Thiele solver (adapted from trsav/mccabe-thiele)
│   └── generator.py              # LHS sampling across 8 input parameters
├── models/
│   ├── baselines.py              # XGBSurrogate + MLPSurrogate classes
│   └── lstm_surrogate.py         # LSTMSurrogate with teacher forcing
├── losses/
│   └── physics_loss.py           # Monotonicity + boundary + VLE loss terms
├── training/
│   ├── utils.py                  # Shared data loading, StandardScaler, masking
│   ├── train_baseline.py         # Phase 2: XGBoost + MLP training
│   └── train_lstm.py             # Phase 3: LSTM + PINN training
├── evaluation/
│   └── evaluate_all.py           # Full benchmark → experiments/final_comparison.json
├── experiments/
│   ├── final_comparison.json     # Model metrics (auto-generated)
│   └── figures/                  # 4 benchmark figures (auto-generated)
├── tests/
│   └── test_all.py               # 14 unit tests across all phases
├── app/
│   └── streamlit_app.py          # Interactive McCabe-Thiele + surrogate demo
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 Input Parameters

| Parameter | Symbol | Range | Description |
|---|---|---|---|
| `PaVap` | P*a | 5 – 30 | Vapour pressure of light-key component |
| `PbVap` | P*b | 1 – PaVap | Vapour pressure of heavy-key component |
| `xd` | xD | 0.80 – 0.99 | Distillate purity (mol fraction) |
| `xb` | xB | 0.005 – 0.15 | Bottoms composition (mol fraction) |
| `xf` | zF | 0.20 – 0.80 | Feed composition (mol fraction) |
| `q` | q | −0.3 – 1.5 | Feed quality (1 = saturated liquid) |
| `R_factor` | R/Rmin | 1.05 – 3.0 | Reflux ratio multiplier |
| `nm` | η | 0.50 – 1.00 | Murphree tray efficiency |

Derived: `α = PaVap / PbVap` (9th model feature, relative volatility)

---

## 🔭 Future Work

- [ ] Extend to multi-component (3+) systems using MESH equations
- [ ] Add temperature profile prediction alongside composition
- [ ] Integrate real plant DCS sensor data for transfer learning
- [ ] Deploy as REST API for real-time process control integration
- [ ] Explore Transformer (attention) architecture for stage sequences

---

## 🙏 Credits

- McCabe-Thiele solver adapted from [trsav/mccabe-thiele](https://github.com/trsav/mccabe-thiele) by Tom Savage — modifications: explicit `nm` (Murphree efficiency) parameter, headless mode returning result dict, stage profile capture.
- Physics-informed loss design inspired by the PINN literature for chemical process surrogate modeling.

---

## 👤 Author

**Hamad Almousa**  
Chemical Engineering, King Saud University  
Supervisor: Prof. Oualid Hamdaoui

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hamad%20Almousa-blue)](https://linkedin.com/in/hamad-almousa)
[![GitHub](https://img.shields.io/badge/GitHub-HamadAlmousa-black)](https://github.com/HamadAlmousa)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
