"""
data/mccabe_thiele_solver.py
=============================
Adapted from McCabeThiele.py (original author credited in README).
Changes:
  - nm (Murphree efficiency) passed as explicit parameter, not global
  - Returns structured dict instead of plotting
  - Captures x_profile and y_profile arrays stage-by-stage
  - No matplotlib calls (headless safe for batch generation)

Original interface preserved:
  McCabeThiele(PaVap, PbVap, R_factor, xf, xd, xb, q, nm)
"""
import numpy as np


def eq_og(xa, relative_volatility):
    """Equilibrium vapour composition from liquid composition (ideal)."""
    return (relative_volatility * xa) / (1 + (relative_volatility - 1) * xa)


def eq(xa, relative_volatility, nm):
    """Equilibrium vapour composition with Murphree efficiency."""
    ya = (relative_volatility * xa) / (1 + (relative_volatility - 1) * xa)
    ya = ((ya - xa) * nm) + xa
    return ya


def eq2(ya, relative_volatility, nm):
    """Inverse of eq(): liquid composition from vapour composition.
    When nm=1.0 the equation degenerates to linear (a=0); handled separately.
    """
    a = ((relative_volatility * nm) - nm - relative_volatility + 1)
    b = ((ya * relative_volatility) - ya + nm - 1 - (relative_volatility * nm))
    c = ya
    if abs(a) < 1e-10:
        # Linear case (nm ≈ 1): b*xa + c = 0
        if abs(b) < 1e-10:
            return None
        xa = -c / b
    else:
        discriminant = (b ** 2) - (4 * a * c)
        if discriminant < 0:
            return None
        xa = (-b - np.sqrt(discriminant)) / (2 * a)
    return xa


def stepping_ESOL(x1, y1, relative_volatility, R, xd, nm):
    """Single step on Enriching/Rectifying Operating Line."""
    x2 = eq2(y1, relative_volatility, nm)
    if x2 is None:
        return x1, x1, y1, y1
    y2 = (((R * x2) / (R + 1)) + (xd / (R + 1)))
    return x1, x2, y1, y2


def stepping_SSOL(x1, y1, relative_volatility, ESOL_q_x, ESOL_q_y, xb, nm):
    """Single step on Stripping/Exhausting Operating Line."""
    x2 = eq2(y1, relative_volatility, nm)
    if x2 is None:
        return x1, x1, y1, y1
    m = ((xb - ESOL_q_y) / (xb - ESOL_q_x + 1e-12))
    c = ESOL_q_y - (m * ESOL_q_x)
    y2 = (m * x2) + c
    return x1, x2, y1, y2


def McCabeThiele(PaVap, PbVap, R_factor, xf, xd, xb, q, nm, max_stages=80):
    """
    McCabe-Thiele solver (headless, returns dict).

    Parameters
    ----------
    PaVap     : float  Vapour pressure of more-volatile component a
    PbVap     : float  Vapour pressure of less-volatile component b  (< PaVap)
    R_factor  : float  Reflux ratio = R_min * R_factor
    xf        : float  Feed composition (mol fraction of a)
    xd        : float  Distillate composition
    xb        : float  Bottoms composition
    q         : float  Feed quality (1=sat. liquid, 0=sat. vapour, >1=subcooled)
    nm        : float  Murphree tray efficiency (0-1, use 1.0 for ideal)
    max_stages: int    Safety cap on stage count

    Returns
    -------
    dict with keys:
      PaVap, PbVap, alpha, xd, xb, xf, q, R_factor, nm,
      N_stages, feed_stage, R_min, R_actual, xb_actual,
      x_profile  (list, top→bottom liquid compositions)
      y_profile  (list, top→bottom vapour compositions)
    or None if infeasible.
    """
    # Guard: avoid division by zero at q=0 or q=1
    if abs(q - 1) < 1e-7:
        q = 1 - 1e-7
    if abs(q) < 1e-7:
        q = 1e-7

    # Feasibility checks
    if PaVap <= PbVap:
        return None
    if not (0 < xb < xf < xd < 1):
        return None
    if xd - xb < 0.05:
        return None

    relative_volatility = PaVap / PbVap

    # ── q-line intersects equilibrium curve ──────────────────────────────────
    al = relative_volatility
    a = ((al * q) / (q - 1)) - al + (al * nm) - (q / (q - 1)) + 1 - nm
    b = (q / (q - 1)) - 1 + nm + ((al * xf) / (1 - q)) - (xf / (1 - q)) - (al * nm)
    c = xf / (1 - q)

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    if q > 1:
        q_eqX = (-b + np.sqrt(discriminant)) / (2 * a)
    else:
        q_eqX = (-b - np.sqrt(discriminant)) / (2 * a)

    if not (xb < q_eqX < xd):
        return None

    q_eqy = eq(q_eqX, relative_volatility, nm)

    # ── Rmin and actual R ─────────────────────────────────────────────────────
    denom = xd - q_eqX
    if abs(denom) < 1e-10:
        return None
    theta_min = xd * (1 - ((xd - q_eqy) / denom))
    if theta_min <= 0 or theta_min >= xd:
        return None
    R_min = (xd / theta_min) - 1
    if R_min <= 0:
        return None
    R = R_factor * R_min
    theta = xd / (R + 1)

    # ── ESOL–q-line intersection ──────────────────────────────────────────────
    denom2 = (q / (q - 1)) - ((xd - theta) / xd)
    if abs(denom2) < 1e-10:
        return None
    ESOL_q_x = (theta - (xf / (1 - q))) / denom2
    ESOL_q_y = (ESOL_q_x * ((xd - theta) / xd)) + theta

    if not (xb < ESOL_q_x < xd):
        return None

    # ── Stage stepping ────────────────────────────────────────────────────────
    x_profile = [xd]
    y_profile = [xd]

    x1, x2, y1, y2 = stepping_ESOL(xd, xd, relative_volatility, R, xd, nm)
    step_count = 1
    x_profile.append(x2); y_profile.append(y2)

    # Rectifying section
    safety = 0
    while x2 > ESOL_q_x and step_count < max_stages and safety < max_stages:
        x1, x2, y1, y2 = stepping_ESOL(x2, y2, relative_volatility, R, xd, nm)
        if x2 is None or x2 >= x_profile[-1]:
            break
        x_profile.append(x2); y_profile.append(y2)
        step_count += 1
        safety += 1

    feed_stage = step_count

    # Stripping section
    x1, x2, y1, y2 = stepping_SSOL(x1, y1, relative_volatility,
                                     ESOL_q_x, ESOL_q_y, xb, nm)
    x_profile.append(x2); y_profile.append(y2)
    step_count += 1

    safety = 0
    while x2 > xb and step_count < max_stages and safety < max_stages:
        x1, x2, y1, y2 = stepping_SSOL(x2, y2, relative_volatility,
                                         ESOL_q_x, ESOL_q_y, xb, nm)
        if x2 is None or x2 >= x_profile[-1]:
            break
        x_profile.append(x2); y_profile.append(y2)
        step_count += 1
        safety += 1

    xb_actual = x2
    N_stages = step_count - 1

    if N_stages < 2:
        return None

    return {
        # ── Inputs (all 8 solver parameters) ──
        "PaVap":    round(float(PaVap), 4),
        "PbVap":    round(float(PbVap), 4),
        "alpha":    round(float(relative_volatility), 4),
        "xd":       round(float(xd), 4),
        "xb":       round(float(xb), 4),
        "xf":       round(float(xf), 4),
        "q":        round(float(q), 4),
        "R_factor": round(float(R_factor), 4),
        "nm":       round(float(nm), 6),
        # ── Outputs ──
        "N_stages":   int(N_stages),
        "feed_stage": int(feed_stage),
        "R_min":      round(float(R_min), 4),
        "R_actual":   round(float(R), 4),
        "xb_actual":  round(float(xb_actual), 6),
        "x_profile":  [round(float(v), 6) for v in x_profile[:N_stages]],
        "y_profile":  [round(float(v), 6) for v in y_profile[:N_stages]],
    }
