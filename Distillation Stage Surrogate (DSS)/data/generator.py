"""
data/generator.py
==================
Generates dataset by random sampling of McCabe-Thiele solver inputs.
No fixed chemicals — PaVap and PbVap are sampled randomly.
Uses Latin Hypercube Sampling over all 8 input parameters.

Parameters sampled:
  PaVap    : Vapour pressure of a       [5, 30]  bar (arbitrary units)
  PbVap    : Vapour pressure of b       [1, PaVap-0.5]  (ensures alpha > 1)
  xd       : Distillate composition     [0.80, 0.99]
  xb       : Bottoms composition        [0.005, 0.15]
  xf       : Feed composition           [0.20, 0.80]
  q        : Feed quality               [-0.3, 1.5]
  R_factor : Reflux multiplier          [1.05, 3.0]
  nm       : Murphree efficiency        [0.50, 1.00]

Usage:
  python data/generator.py --samples 6000 --seed 42 --out data/dataset.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from data.mccabe_thiele_solver import McCabeThiele


def latin_hypercube(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Returns (n, d) array with values in [0, 1] using LHS."""
    rng = np.random.default_rng(seed)
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.random(n)) / n
    return result


def main(samples: int = 6000, seed: int = 42, outdir: str = "data") -> list:
    Path(outdir).mkdir(exist_ok=True)
    outpath = Path(outdir) / "dataset.json"

    # Oversample to account for infeasible combinations
    n_try = samples * 4
    lhs = latin_hypercube(n_try, 8, seed=seed)

    # Map LHS [0,1] to physical ranges
    PaVap_arr   = 5.0  + lhs[:, 0] * 25.0      # 5 – 30
    PbVap_frac  = 0.1  + lhs[:, 1] * 0.85      # 0.1 – 0.95  fraction of PaVap
    xd_arr      = 0.80 + lhs[:, 2] * 0.185     # 0.80 – 0.985
    xb_arr      = 0.005+ lhs[:, 3] * 0.145     # 0.005 – 0.15
    xf_arr      = 0.20 + lhs[:, 4] * 0.60      # 0.20 – 0.80
    q_arr       = -0.3 + lhs[:, 5] * 1.8       # -0.3 – 1.5
    Rf_arr      = 1.05 + lhs[:, 6] * 1.95      # 1.05 – 3.0
    nm_arr      = 0.50 + lhs[:, 7] * 0.50      # 0.50 – 1.00

    records = []
    for i in range(n_try):
        PaVap = float(PaVap_arr[i])
        PbVap = float(PaVap * PbVap_frac[i])   # guarantees PbVap < PaVap
        xd    = float(xd_arr[i])
        xb    = float(xb_arr[i])
        xf    = float(xf_arr[i])
        q     = float(q_arr[i])
        Rf    = float(Rf_arr[i])
        nm    = float(nm_arr[i])

        result = McCabeThiele(PaVap, PbVap, Rf, xf, xd, xb, q, nm)
        if result is None:
            continue

        records.append(result)
        if len(records) >= samples:
            break

    outpath.write_text(json.dumps(records, indent=2))
    print(f"Generated {len(records)} records → {outpath}")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out",     type=str, default="data")
    args = ap.parse_args()
    main(args.samples, args.seed, args.out)
