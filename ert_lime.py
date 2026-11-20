"""LIME-artiges lokales Surrogat (eigene Implementierung; es gibt hier kein Paket `lime`): gewichtetes Ridge auf binären Merkmalen „wie in der erklärten Zeile“.

Stichproben um die erklärte Zeile x0 (N Punkte, aus Zeilen Q gezogen):
  std    jede Spalte unabhängig aus Q gezogen (LIME-Tabular-artig; erzeugt unmögliche Zeilen),
  koal   Koalitions-Ersatz: je Spalte mit Wahrscheinlichkeit 1/2 der Wert von x0, sonst aus Q gezogen,
  zeile  ganze Zeilen aus Q (nur mögliche Punkte).
Darstellung Z: Spalte j ist 1, wenn der Stichprobenpunkt in Spalte j „wie x0“ ist (Vorlauf und Gewicht: gleiches Quartil der Trainingswerte, sonst gleicher Wert).
Gewicht eines Punktes exp(-(Zahl der abweichenden Spalten) / Breite^2); Koeffizienten aus (Z'WZ + alpha I) b = Z'W(y - Mittel)."""

import numpy as np

import ert_constants as C
from ert_generator import block_below, block_random


def quartilgrenzen(X):
    return {j: np.unique(np.quantile(X[:, j], [0.25, 0.5, 0.75])) for j in (C.COL_VORLAUF, C.COL_GEWICHT)}


def darstellung(Sm, x0, grenzen):
    Z = np.zeros((len(Sm), C.D))
    for j in range(C.D):
        if j in grenzen:
            Z[:, j] = np.searchsorted(grenzen[j], Sm[:, j], side="right") == np.searchsorted(grenzen[j], x0[j], side="right")
        else:
            Z[:, j] = Sm[:, j] == x0[j]
    return Z


def stichprobe(var, Q, x0, n, rng, grenzen):
    """(Stichprobenpunkte Sm (n, d), Darstellung Z (n, d))."""
    D = C.D
    if var == "std":
        idx = block_below(rng, n * D, len(Q)).reshape(n, D)
        Sm = Q[idx, np.arange(D)[None, :]]
        return Sm, darstellung(Sm, x0, grenzen)
    if var == "zeile":
        Sm = Q[block_below(rng, n, len(Q))]
        return Sm, darstellung(Sm, x0, grenzen)
    if var == "koal":
        Zb = block_random(rng, n * D).reshape(n, D) < 0.5
        idx = block_below(rng, n * D, len(Q)).reshape(n, D)
        Sm = np.where(Zb, x0[None, :], Q[idx, np.arange(D)[None, :]])
        return Sm, Zb.astype(float)
    raise ValueError(var)


def gewichte(Z, breite=C.LIME_BREITE):
    return np.exp(-(Z.shape[1] - Z.sum(axis=1)) / breite ** 2)


def koeffizienten(Z, w, yv, alpha=C.LIME_ALPHA):
    """Gewichtetes Ridge mit Achsenabschnitt (Zentrierung); Rückgabe: Koeffizienten je Spalte von Z."""
    sw = w.sum()
    zm = (w @ Z) / sw
    ym = (w @ yv) / sw
    Zc = Z - zm
    A = Zc.T @ (w[:, None] * Zc) + alpha * np.eye(Z.shape[1])
    return np.linalg.solve(A, Zc.T @ (w * (yv - ym)))
