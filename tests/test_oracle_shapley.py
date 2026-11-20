"""Orakel-Test Shapley: exakter interventionaler Shapley-Wert (Koalitionsformel) gegen andere Rechenwege.

(1) Mittel über ALLE d! Permutationen aus der Definition (marginaler Zuwachs, Modell direkt an Mischzeilen ausgewertet, ohne die Koalitionswerte-Tabelle der Demo),
(2) geschlossene Form für lineare Modelle (phi_j = beta_j (x_j - Mittel_B x_j)), (3) Effizienz und Nullspieler an Modellen des Vehikels."""

import itertools
from math import factorial

import numpy as np

from ert_shapley import mischzeilen, shapley, shapley_aus_werten, werte_aus_vorhersagen


def _phi_permutationen(f, x, B):
    """Definition: Mittel über alle d! Reihenfolgen von E_b[f(x_S+j, b)] - E_b[f(x_S, b)]; Auswertung direkt an Mischzeilen."""
    d = len(x)
    phi = np.zeros(d)
    for p in itertools.permutations(range(d)):
        S = []
        for j in p:
            vor = np.array([[x[i] if i in S else b[i] for i in range(d)] for b in B], dtype=float)
            nach = np.array([[x[i] if (i in S or i == j) else b[i] for i in range(d)] for b in B], dtype=float)
            phi[j] += f(nach).mean() - f(vor).mean()
            S.append(j)
    return phi / factorial(d)


def test_koalitionsformel_gleich_allen_permutationen_am_vehikel(klein):
    f = klein["gb"].predict
    B = klein["Xf"][:5]
    for x in klein["Xe"][:3]:
        phi = shapley(f, x[None, :], B)[0]
        ref = _phi_permutationen(f, x, B)                     # 720 Permutationen, d = 6
        assert np.allclose(phi, ref, atol=1e-10, rtol=0)


def test_lineares_modell_geschlossene_form():
    rng = np.random.default_rng(0)
    beta = np.array([1.5, -2.0, 0.5, 0.0, 3.0, -1.0])

    def f(Z):
        return Z @ beta + 0.7

    B = rng.normal(size=(9, 6))
    X = rng.normal(size=(4, 6))
    assert np.allclose(shapley(f, X, B), beta[None, :] * (X - B.mean(axis=0)[None, :]), atol=1e-12)


def test_effizienz_und_nullspieler_am_vehikel(klein):
    f = klein["gb"].predict
    B, Xe = klein["Xf"][:8], klein["Xe"]
    phi = shapley(f, Xe, B)
    leer = f(B).mean()
    assert np.allclose(phi.sum(axis=1), f(Xe) - leer, atol=1e-10)

    def g(Z):
        return 2.0 * Z[:, 0] + Z[:, 2]                         # Merkmale 1, 3, 4, 5 kommen nicht vor

    p = shapley(g, Xe, B)
    assert np.allclose(p[:, [1, 3, 4, 5]], 0.0, atol=1e-12)


def test_mischzeilen_form_und_reihenfolge():
    Xe = np.array([[1.0, 2, 3, 4, 5, 6]])
    B = np.array([[10.0, 20, 30, 40, 50, 60], [11.0, 21, 31, 41, 51, 61]])
    H = mischzeilen(Xe, B)
    assert H.shape == (1 * 64 * 2, 6)
    assert H[0].tolist() == B[0].tolist() and H[1].tolist() == B[1].tolist()                  # Koalition 0: nur Hintergrund
    assert H[2].tolist() == [1.0, 20, 30, 40, 50, 60]                                         # Koalition {Merkmal 0}, Hintergrundzeile 0
    assert H[-1].tolist() == [1.0, 2, 3, 4, 5, 6]                                             # volle Koalition


def test_shapley_aus_werten_handbeispiel_zwei_merkmale():
    """v(leer) = 0, v({1}) = 2, v({2}) = 1, v({1,2}) = 4: phi_1 = (2 + 3)/2, phi_2 = (1 + 2)/2."""
    vals = np.zeros((1, 4))
    vals[0, 1], vals[0, 2], vals[0, 3] = 2.0, 1.0, 4.0
    assert shapley_aus_werten(vals, 2)[0].tolist() == [2.5, 1.5]
    assert werte_aus_vorhersagen(np.arange(8.0), 1, 2, 2).tolist() == [[0.5, 2.5, 4.5, 6.5]]
