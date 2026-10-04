"""Entropic optimal-transport coupling by Sinkhorn scaling, on the host in float64."""
import numpy as np

EPS = 0.05          # coupling temperature, fixed before any benchmark evaluation
TOL = 1e-9          # tie tolerance of the decision rules and the Sinkhorn stopping criterion


def sinkhorn_max(S, a, b, eps=EPS, iters=200, tol=TOL):
    """Maximise <S, pi> + eps H(pi) over couplings with marginals a (rows) and b (columns)."""
    S = np.asarray(S, dtype=np.float64)
    K = np.exp((S - S.max()) / eps)
    u = np.ones(S.shape[0])
    v = np.ones(S.shape[1])
    for _ in range(iters):
        u_new = a / (K @ v + 1e-300)
        v_new = b / (K.T @ u_new + 1e-300)
        done = np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol
        u, v = u_new, v_new
        if done:
            break
    return u[:, None] * K * v[None, :]
