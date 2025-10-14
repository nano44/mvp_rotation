from __future__ import annotations
import numpy as np

def _tracking_error(a: np.ndarray, Sigma: np.ndarray) -> float:
    # a = active weights (w - w_bench). TE monthly
    te_month = np.sqrt(max(0.0, a @ Sigma @ a))
    # convert monthly TE to annualized TE (× sqrt(12))
    return te_month * (12**0.5)

def _project_simplex(x: np.ndarray) -> np.ndarray:
    # Euclidean projection onto the probability simplex (sum=1, >=0)
    # (Condat’s algorithm simplified)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u + (1 - cssv) / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0)

def solve_overlay(mu_hat: np.ndarray,
                  Sigma: np.ndarray,
                  w_bench: np.ndarray,
                  w_prev: np.ndarray,
                  te_target_annual: float,
                  turnover_cap: float,
                  smoothing_lambda: float,
                  crisis: bool) -> np.ndarray:
    """
    Simple, fast overlay: scale mu_hat to hit TE target, then enforce long-only and turnover cap.
    """
    # 1) initial direction: active tilt proportional to mu_hat
    if np.allclose(mu_hat, 0):
        w_target = w_bench.copy()
    else:
        # scale to TE target ignoring constraints
        denom = np.sqrt(max(1e-12, mu_hat @ Sigma @ mu_hat)) * (12**0.5)  # annualize in denom
        k = te_target_annual / denom
        a = k * mu_hat
        w = w_bench + a

        # project to simplex (long-only, sum=1)
        w = _project_simplex(w)

        # recompute active and adjust by bisection to meet TE target after projection
        low, high = 0.0, 5.0 * k
        for _ in range(25):
            mid = 0.5 * (low + high)
            w_try = _project_simplex(w_bench + mid * mu_hat)
            te = _tracking_error(w_try - w_bench, Sigma)
            if te > te_target_annual:
                high = mid
            else:
                low = mid
        w_target = _project_simplex(w_bench + low * mu_hat)

    # 2) turnover cap: limit movement from w_prev
    delta = w_target - w_prev
    one_way = 0.5 * np.sum(np.abs(delta)) * 2  # = sum positive changes
    if one_way > turnover_cap:
        scale = turnover_cap / (one_way + 1e-12)
        w_target = w_prev + scale * delta

    # 3) smoothing on ACTIVE weights
    lam = 0.1 if crisis else smoothing_lambda
    a_prev = w_prev - w_bench
    a_new  = w_target - w_bench
    a_smooth = lam * a_prev + (1 - lam) * a_new
    w_final  = _project_simplex(w_bench + a_smooth)

    # --- Rescale after smoothing to hit TE target (soft) ---
    d = w_final - w_bench
    if np.linalg.norm(d, 1) > 1e-12:
        # bisection on gamma in w_bench + gamma * d
        low, high = 0.0, 2.5
        for _ in range(20):
            mid = 0.5 * (low + high)
            w_try = _project_simplex(w_bench + mid * d)
            te = _tracking_error(w_try - w_bench, Sigma)
            if te > te_target_annual:
                high = mid
            else:
                low = mid
        w_final = _project_simplex(w_bench + low * d)

        # re-enforce turnover cap relative to previous weights
        delta = w_final - w_prev
        one_way = 0.5 * np.sum(np.abs(delta)) * 2
        if one_way > turnover_cap:
            scale = turnover_cap / (one_way + 1e-12)
            w_final = w_prev + scale * delta
            w_final = _project_simplex(w_final)

    return w_final
