"""
losses/physics_loss.py  —  Phase 4
=====================================
Three physics constraints added on top of the MSE loss:

  1. MONOTONICITY  — x must decrease from stage 1 (top) to stage N (bottom)
                     Soft penalty: sum of positive finite differences.

  2. BOUNDARY COND — x[0]  ≈ xd   (distillate spec)
                     x[-1] ≈ xb   (bottoms spec)

  3. VLE RESIDUAL  — Raoult / Murphree equilibrium:
                     K_pred = y/x  should equal  K_eq = alpha/(1+(alpha-1)*x)
                     Only used when y_profile predictions are available.

All functions operate on PyTorch tensors (B, MAX_S).
"""
from __future__ import annotations


def monotonicity_loss(x_pred, mask):
    """
    Penalises any stage where x increases top→bottom.
    x_pred : (B, S)   predicted liquid compositions
    mask   : (B, S)   1 = valid stage, 0 = padding
    """
    diffs   = x_pred[:, 1:] - x_pred[:, :-1]        # positive = violation
    m_valid = mask[:, 1:] * mask[:, :-1]
    penalty = diffs.clamp(min=0.0) * m_valid
    return penalty.sum() / (m_valid.sum() + 1e-8)


def boundary_condition_loss(x_pred, xd, xb, mask):
    """
    x_pred[0]  should equal xd  (top stage = distillate)
    x_pred[-1] should equal xb  (bottom valid stage = bottoms)

    xd, xb : (B, 1)  distillate / bottoms spec from feature vector
    """
    import torch
    # Top boundary
    top_err = (x_pred[:, 0:1] - xd) ** 2

    # Bottom boundary — index of last valid stage per sample
    stage_idx = torch.arange(x_pred.size(1), device=x_pred.device,
                              dtype=torch.float32)
    last_idx  = (mask * stage_idx.unsqueeze(0)).argmax(dim=1, keepdim=True)
    x_last    = x_pred.gather(1, last_idx.long())
    bot_err   = (x_last - xb) ** 2

    return (top_err + bot_err).mean()


def vle_residual_loss(x_pred, y_pred, mask, alpha):
    """
    VLE consistency: K = y/x  vs  K_eq = alpha/(1+(alpha-1)*x)
    alpha : (B, 1) relative volatility (PaVap/PbVap)
    y_pred: (B, S) predicted vapour compositions (optional — pass None to skip)
    """
    if y_pred is None:
        import torch
        return torch.tensor(0.0, device=x_pred.device)

    eps    = 1e-8
    x_safe = x_pred.clamp(eps, 1 - eps)
    K_pred = y_pred / x_safe
    K_eq   = alpha / (1 + (alpha - 1) * x_safe)
    residual = (K_pred - K_eq) ** 2 * mask
    return residual.sum() / (mask.sum() + eps)


def combined_physics_loss(x_pred, x_true, mask, alpha,
                           y_pred=None,
                           lambda_mono: float = 0.10,
                           lambda_bc:   float = 0.20,
                           lambda_vle:  float = 0.05):
    """
    Combined physics loss used in Phase 4 LSTM+PINN training.

    Parameters
    ----------
    x_pred  : (B, MAX_S)   model output
    x_true  : (B, MAX_S)   ground truth (used to extract xd, xb specs)
    mask    : (B, MAX_S)   validity mask
    alpha   : (B, 1)       relative volatility from feature col 2
    y_pred  : (B, MAX_S)   vapour profile prediction (optional)
    """
    # Read xd and xb from ground truth (stage 0 = top, last valid = bottom)
    import torch
    xd = x_true[:, 0:1]
    stage_idx = torch.arange(x_true.size(1), device=x_true.device,
                              dtype=torch.float32)
    last_idx  = (mask * stage_idx.unsqueeze(0)).argmax(dim=1, keepdim=True)
    xb        = x_true.gather(1, last_idx.long())

    mono = monotonicity_loss(x_pred, mask)
    bc   = boundary_condition_loss(x_pred, xd, xb, mask)
    vle  = vle_residual_loss(x_pred, y_pred, mask, alpha)

    return lambda_mono * mono + lambda_bc * bc + lambda_vle * vle
