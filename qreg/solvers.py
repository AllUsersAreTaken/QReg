"""Ising solvers: classical SA, parallel tempering, and path-integral SQA.

All solvers minimise  E(s) = h.s + 0.5 * s^T J s  with s in {-1, +1}, symmetric J,
zero diagonal.  Every solver reports the number of attempted spin flips so that
comparisons can be made at matched compute rather than matched iterations.

The SQA sampler is a discrete-time path-integral Monte Carlo: P Trotter slices, a
transverse field Gamma(t) annealed to zero, and inter-slice coupling

    J_perp = -(T/2) * ln( tanh( Gamma / (P*T) ) )

which is the only structural difference from classical parallel tempering.
"""

from __future__ import annotations

from math import exp, log, tanh

import numpy as np
from numba import njit

_JPERP_MAX = 50.0


@njit(cache=True)
def _energy(h, J, s):
    n = s.shape[0]
    e = 0.0
    for i in range(n):
        e += h[i] * s[i]
        for j in range(i + 1, n):
            e += J[i, j] * s[i] * s[j]
    return e


@njit(cache=True)
def _random_state(n, seed):
    np.random.seed(seed)
    s = np.empty(n, np.int8)
    for i in range(n):
        s[i] = 1 if np.random.random() < 0.5 else -1
    return s


@njit(cache=True)
def _sweep(h, J, s, e, T):
    """One sequential (Gauss-Seidel) Metropolis sweep. Returns updated energy."""
    n = s.shape[0]
    for i in range(n):
        f = h[i]
        for j in range(n):
            f += J[i, j] * s[j]
        de = -2.0 * s[i] * f
        if de <= 0.0 or np.random.random() < np.exp(-de / T):
            s[i] = -s[i]
            e += de
    return e


@njit(cache=True)
def _sa(h, J, sweeps, t0, t1, seed):
    n = h.shape[0]
    s = _random_state(n, seed)
    e = _energy(h, J, s)
    best_s = s.copy()
    best_e = e
    flips = 0
    denom = max(sweeps - 1, 1)
    for t in range(sweeps):
        T = t0 * (t1 / t0) ** (t / denom)
        e = _sweep(h, J, s, e, T)
        flips += n
        if e < best_e:
            best_e = e
            best_s = s.copy()
    return best_s, best_e, flips


@njit(cache=True)
def _pt(h, J, sweeps, nrep, tmin, tmax, seed):
    n = h.shape[0]
    s = np.empty((nrep, n), np.int8)
    np.random.seed(seed)
    for r in range(nrep):
        for i in range(n):
            s[r, i] = 1 if np.random.random() < 0.5 else -1

    T = np.empty(nrep)
    if nrep == 1:
        T[0] = tmin
    else:
        for r in range(nrep):
            T[r] = tmin * (tmax / tmin) ** (r / (nrep - 1))
    beta = 1.0 / T
    e = np.empty(nrep)
    for r in range(nrep):
        e[r] = _energy(h, J, s[r])

    best_e = e[0]
    best_s = s[0].copy()
    flips = 0
    for t in range(sweeps):
        for r in range(nrep):
            e[r] = _sweep(h, J, s[r], e[r], T[r])
            flips += n
            if e[r] < best_e:
                best_e = e[r]
                best_s = s[r].copy()
        for r in range(nrep - 1):
            delta = (beta[r] - beta[r + 1]) * (e[r] - e[r + 1])
            if delta >= 0.0 or np.random.random() < np.exp(delta):
                tmp = s[r].copy()
                s[r] = s[r + 1]
                s[r + 1] = tmp
                tr = e[r]
                e[r] = e[r + 1]
                e[r + 1] = tr
    return best_s, best_e, flips


@njit(cache=True)
def _sqa(h, J, sweeps, P, gamma0, gamma1, T0, T1, seed):
    n = h.shape[0]
    np.random.seed(seed)
    s = np.empty((P, n), np.int8)
    for p in range(P):
        for i in range(n):
            s[p, i] = 1 if np.random.random() < 0.5 else -1

    best_e = 1e18
    best_s = s[0].copy()
    flips = 0
    denom = max(sweeps - 1, 1)

    for t in range(sweeps):
        frac = t / denom
        gamma = gamma0 * (gamma1 / gamma0) ** frac
        T = T0 * (T1 / T0) ** frac
        x = gamma / (P * T)
        if x < 1e-4:
            jperp = _JPERP_MAX
        elif x > 1.0 - 1e-6:
            jperp = 0.0
        else:
            jperp = -0.5 * T * log(tanh(x))
            if jperp > _JPERP_MAX:
                jperp = _JPERP_MAX

        for p in range(P):
            pm = p - 1 if p > 0 else P - 1
            pp = p + 1 if p < P - 1 else 0
            for i in range(n):
                f = h[i]
                for j in range(n):
                    f += J[i, j] * s[p, j]
                f += jperp * (s[pm, i] + s[pp, i])
                de = -2.0 * s[p, i] * f
                flips += 1
                if de <= 0.0 or np.random.random() < np.exp(-de / T):
                    s[p, i] = -s[p, i]

        for p in range(P):
            ep = _energy(h, J, s[p])
            if ep < best_e:
                best_e = ep
                best_s = s[p].copy()
    return best_s, best_e, flips


@njit(cache=True)
def _greedy(h, J, restarts, seed):
    n = h.shape[0]
    np.random.seed(seed)
    best_e = 1e18
    best_s = np.ones(n, np.int8)
    s = np.empty(n, np.int8)
    flips = 0
    for r in range(restarts):
        if r == 0:
            # Seed from the unary preference (s_i = -sign(h_i)); starting from all-ones
            # makes steepest descent get stuck on near-identity problems.
            for i in range(n):
                s[i] = -1 if h[i] > 0.0 else 1
        else:
            for i in range(n):
                s[i] = 1 if np.random.random() < 0.5 else -1
        e = _energy(h, J, s)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                f = h[i]
                for j in range(n):
                    f += J[i, j] * s[j]
                de = -2.0 * s[i] * f
                flips += 1
                if de < -1e-12:
                    s[i] = -s[i]
                    e += de
                    improved = True
        if e < best_e:
            best_e = e
            best_s = s.copy()
    return best_s, best_e, flips


def _sweeps_for(budget: int, per_sweep: int) -> int:
    return max(int(budget) // max(per_sweep, 1), 1)


def solve(method: str, h, J, *, budget: int = 400_000, seed: int = 0, **kw):
    """Solve an Ising problem under a fixed spin-flip budget. Returns dict."""
    h = np.ascontiguousarray(h, dtype=np.float64)
    J = np.ascontiguousarray(J, dtype=np.float64)
    n = h.shape[0]

    if method == "greedy":
        s, e, flips = _greedy(h, J, kw.get("restarts", 8), seed)
    elif method == "sa":
        t0, t1 = kw.get("t0", 2.0), kw.get("t1", 0.02)
        s, e, flips = _sa(h, J, _sweeps_for(budget, n), t0, t1, seed)
    elif method == "pt":
        nrep = kw.get("nrep", 10)
        s, e, flips = _pt(h, J, _sweeps_for(budget, nrep * n), nrep,
                          kw.get("tmin", 0.05), kw.get("tmax", 3.0), seed)
    elif method == "sqa":
        P = kw.get("n_trotter", 8)
        s, e, flips = _sqa(h, J, _sweeps_for(budget, P * n), P,
                           kw.get("gamma0", 3.0), kw.get("gamma1", 1e-3),
                           kw.get("T", 1.0), kw.get("T1", 0.02), seed)
    else:
        raise ValueError(f"unknown method {method!r}")

    x = ((np.asarray(s, dtype=np.int64) + 1) // 2)
    return {"spins": np.asarray(s), "x": x, "energy": float(e), "flips": int(flips)}
