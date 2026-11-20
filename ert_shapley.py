"""Exakter interventionaler Shapley-Wert: Aufzählung aller 2^d Koalitionen (Koalitionsformel mit Gewichten s!(d-s-1)!/d!).

Wertfunktional (die Konvention der Linie): v(S) = Mittel über die Hintergrundzeilen B von f(x_S, B_Rest). Der Orakel-Test (tests/test_oracle_shapley.py) rechnet
denselben Wert über alle d! Permutationen; hier steht nur der schnelle Weg."""

from math import factorial

import numpy as np

_T = {}


def tabelle(d):
    """Masken (K, d), Koalitionsgrößen (K,), Gewichte w(s) = s!(d-s-1)!/d! (s < d)."""
    if d not in _T:
        K = 1 << d
        masks = np.array([[(k >> j) & 1 for j in range(d)] for k in range(K)], dtype=bool)
        size = masks.sum(axis=1)
        w = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) if s < d else 0.0 for s in range(d + 1)])
        _T[d] = (masks, size, w)
    return _T[d]


def mischzeilen(Xe, B):
    """Alle Mischzeilen (x_S, B_Rest): Form (nx * K * nb, d), Reihenfolge Zeile, Koalition, Hintergrundzeile."""
    d = Xe.shape[1]
    masks = tabelle(d)[0]
    H = np.where(masks[None, :, None, :], Xe[:, None, None, :], B[None, None, :, :])
    return H.reshape(-1, d)


def werte_aus_vorhersagen(p, nx, d, nb):
    """Koalitionswerte v(S) (nx, K) aus den Vorhersagen auf den Mischzeilen (Mittel über den Hintergrund)."""
    return p.reshape(nx, 1 << d, nb).mean(axis=2)


def shapley_aus_werten(vals, d):
    """vals (nx, K) in Bitmaskenreihenfolge (Bit j = Merkmal j in S) -> phi (nx, d)."""
    masks, size, w = tabelle(d)
    K = 1 << d
    v = np.atleast_2d(vals)
    phi = np.zeros((v.shape[0], d))
    for j in range(d):
        ks = np.array([k for k in range(K) if not masks[k, j]])
        phi[:, j] = (w[size[ks]][None, :] * (v[:, ks | (1 << j)] - v[:, ks])).sum(axis=1)
    return phi


def shapley(f, Xe, B):
    """Exakter interventionaler Shapley-Wert (nx, d) für das Modell f (Funktion auf Zeilen) und den Hintergrund B."""
    Xe, B = np.atleast_2d(Xe), np.atleast_2d(B)
    d = Xe.shape[1]
    p = np.asarray(f(mischzeilen(Xe, B)))
    return shapley_aus_werten(werte_aus_vorhersagen(p, len(Xe), d, len(B)), d)
