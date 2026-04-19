"""Interactive McCabe-Thiele explorer — run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from data.mccabe_thiele_solver import McCabeThiele, eq, eq_og

st.set_page_config(page_title="McCabe-Thiele Surrogate", layout="wide")
st.title("McCabe-Thiele Distillation Surrogate")
st.caption("Adapted from McCabeThiele.py (trsav/mccabe-thiele) — generic PaVap/PbVap inputs")

with st.sidebar:
    st.header("Column Parameters")
    PaVap    = st.slider("PaVap (more volatile)", 5.0, 30.0, 10.0, 0.1)
    PbVap    = st.slider("PbVap (less volatile)", 1.0, float(round(PaVap - 0.5, 1)), 3.8, 0.1)
    xd       = st.slider("xd  (distillate)",  0.80, 0.99, 0.95, 0.01)
    xb       = st.slider("xb  (bottoms)",     0.005, 0.15, 0.05, 0.005)
    xf       = st.slider("xf  (feed)",        0.20,  0.80, 0.50, 0.01)
    q        = st.slider("q   (feed quality)",-0.30, 1.50, 0.667, 0.01)
    R_factor = st.slider("R / R_min",          1.05,  3.00, 1.30, 0.05)
    nm       = st.slider("Murphree efficiency", 0.50, 1.00, 1.00, 0.01)

result = McCabeThiele(PaVap, PbVap, R_factor, xf, xd, xb, q, nm)

if result is None:
    st.error("Infeasible combination — ensure xb < xf < xd and PaVap > PbVap")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N stages",   result["N_stages"])
    c2.metric("Feed stage", result["feed_stage"])
    c3.metric("R_min",      f"{result['R_min']:.3f}")
    c4.metric("R actual",   f"{result['R_actual']:.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    xa = np.linspace(0, 1, 200)
    al = result["alpha"]
    ax.plot(xa, eq_og(xa, al), "b-",  lw=1.8, label="Equilibrium curve (ideal)")
    ax.plot(xa, eq(xa, al, nm),"g--", lw=1.2, label=f"Murphree η={nm:.2f}")
    ax.plot(xa, xa, "k-", lw=0.8, label="y = x")

    xp = result["x_profile"]
    for i in range(len(xp) - 1):
        ax.plot([xp[i], xp[i+1]], [xp[i],    xp[i]],    "r-", lw=1.0)
        ax.plot([xp[i+1], xp[i+1]], [xp[i],  xp[i+1]], "r-",  lw=1.0)
    ax.axvline(xd, color="k", lw=0.7, ls="--")
    ax.axvline(xb, color="k", lw=0.7, ls="--")
    ax.axvline(xf, color="k", lw=0.7, ls="--")
    ax.set_xlabel("x  (liquid mol fraction)")
    ax.set_ylabel("y  (vapour mol fraction)")
    ax.set_title(f"McCabe-Thiele  |  alpha={al:.3f}  |  {result['N_stages']} stages  |  feed@{result['feed_stage']}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Liquid composition profile  x  (top → bottom)")
    st.line_chart(result["x_profile"])

    with st.expander("Full result dict"):
        import json
        display = {k: v for k, v in result.items() if k not in ("x_profile","y_profile")}
        st.json(display)
