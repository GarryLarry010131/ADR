"""The antisymmetric displacement readout and the decision rules.

From the N x m patch-word similarity table S of one image and one caption, a column weight turns S into
nonnegative weights (excess pi~ = [pi_0 - 1/(N m)]_+ by default; dense = column softmax of S / eps; pi0; argmax),
each word gets its mass-weighted centroid, and the caption scores D = sum_r e_r . (c_subject - c_object).
"""
import numpy as np

from .sinkhorn import EPS, TOL, sinkhorn_max

VARIANTS = ("excess", "dense", "pi0", "argmax")


def column_weights(S, variant="excess", eps=EPS):
    S = np.asarray(S, dtype=np.float64)
    N, m = S.shape
    if variant == "argmax":
        w = np.zeros_like(S)
        w[np.argmax(S, 0), np.arange(m)] = 1.0
        return w
    if variant == "dense":
        sm = np.exp((S - S.max(0, keepdims=True)) / eps)
        return sm / sm.sum(0, keepdims=True)
    pi0 = sinkhorn_max(S, np.full(N, 1.0 / N), np.full(m, 1.0 / m), eps)
    if variant == "pi0":
        return pi0
    if variant == "excess":
        return np.maximum(pi0 - 1.0 / (N * m), 0.0)
    raise ValueError(variant)


def centroids(w, P):
    mass = w.sum(0)
    cen = (w.T @ P) / np.maximum(mass, 1e-30)[:, None]
    return mass, cen


def displacement(w, P, edges):
    if not edges:
        return 0.0
    _, cen = centroids(w, P)
    D = 0.0
    for s_ix, o_ix, e, _ in edges:
        D += e[0] * (cen[s_ix, 0] - cen[o_ix, 0]) + e[1] * (cen[s_ix, 1] - cen[o_ix, 1])
    return float(D)


def score(S, P, edges, variant="excess", eps=EPS, record=False):
    """ADR score of one caption; with record=True also the weights, masses and centroids."""
    if not edges:
        return (0.0, None) if record else 0.0
    w = column_weights(S, variant, eps)
    D = displacement(w, P, edges)
    if not record:
        return D
    mass, cen = centroids(w, P)
    return D, {"weights": w, "mass": mass, "centroids": cen}


def all_scores(S, P, edges, eps=EPS):
    """The four variants from one Sinkhorn run."""
    if not edges:
        return {v: 0.0 for v in VARIANTS}
    N, m = S.shape
    pi0 = sinkhorn_max(S, np.full(N, 1.0 / N), np.full(m, 1.0 / m), eps)
    oh = np.zeros_like(S)
    oh[np.argmax(S, 0), np.arange(m)] = 1.0
    sm = np.exp((S - S.max(0, keepdims=True)) / eps)
    return {"excess": displacement(np.maximum(pi0 - 1.0 / (N * m), 0.0), P, edges),
            "dense": displacement(sm / sm.sum(0, keepdims=True), P, edges),
            "pi0": displacement(pi0, P, edges),
            "argmax": displacement(oh, P, edges)}


def fair(margin, tol=TOL):
    """Pair rule: 1 if the true caption scores higher, 1/2 within tol, else 0."""
    return 1.0 if margin > tol else (0.5 if abs(margin) <= tol else 0.0)


def kway_fair(scores, correct, tol=TOL):
    """k-way rule: 1/#winners if the true caption ties for the top, else 0."""
    mx = max(scores)
    winners = [k for k, v in enumerate(scores) if v >= mx - tol]
    return (1.0 / len(winners)) if correct in winners else 0.0


def sign_fair(D, label, tol=TOL):
    """True/false rule for scores with a canonical zero: true iff D > 0, exact zero counts 1/2."""
    if abs(D) <= tol:
        return 0.5
    return 1.0 if (D > tol) == (label == 1) else 0.0


def oracle_accuracy(scores, labels):
    """Best single-threshold accuracy on the evaluation labels, both constant classifiers included. Thresholds sit
    between distinct scores, so tied items are never split. Used only for scores without a canonical zero."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(y)
    if n == 0:
        return 0.0
    order = np.argsort(s, kind="stable")
    s, y = s[order], y[order]
    correct = y.sum()
    best = correct / n
    for k in range(n):
        correct += 1 if y[k] == 0 else -1
        if k == n - 1 or s[k + 1] > s[k]:
            best = max(best, correct / n)
    return float(best)


def derangement(paths, K):
    """sigma(j) = first cyclic index at or after j + K whose source image differs from item j's. Deterministic."""
    n = len(paths)
    assert all(paths), "every item needs a source-image identifier"
    assert len(set(paths)) >= 2, "all items share one source image"
    sig, fix = [], 0
    for j in range(n):
        t = (j + K) % n
        while paths[t] == paths[j]:
            t = (t + 1) % n
            fix += 1
        sig.append(t)
    assert all(paths[s] != paths[j] for j, s in enumerate(sig))
    return sig, fix


def derangement_grouped(paths, groups, K):
    """Derangement drawn within each group of items (What'sUp-B's two halves), same K."""
    sig, fix = [None] * len(paths), 0
    for g in dict.fromkeys(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        s, f = derangement([paths[i] for i in idx], K)
        fix += f
        for local, i in enumerate(idx):
            sig[i] = idx[s[local]]
    return sig, fix
