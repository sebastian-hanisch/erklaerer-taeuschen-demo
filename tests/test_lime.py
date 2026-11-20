"""LIME-artiges Surrogat: gewichtetes Ridge gegen scikit-learn (Fremdreferenz), Stichprobenarten, Darstellung, Randfälle."""

import numpy as np
import pytest

import ert_constants as C
import ert_generator as G
import ert_lime as L
from ert_support import gueltig


def test_koeffizienten_gleich_sklearn_ridge_mit_gewichten():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    Z = (rng.random((300, 6)) < 0.5).astype(float)
    w = rng.random(300) + 0.1
    yv = Z @ np.array([1.0, -2.0, 0.5, 0.0, 3.0, 0.0]) + rng.normal(size=300) * 0.1
    mine = L.koeffizienten(Z, w, yv, alpha=1.0)
    ref = sk.Ridge(alpha=1.0).fit(Z, yv, sample_weight=w).coef_
    assert np.allclose(mine, ref, atol=1e-9)


def test_exaktes_lineares_modell_wird_bei_kleinem_alpha_wiedergefunden():
    rng = np.random.default_rng(2)
    Z = (rng.random((400, 6)) < 0.5).astype(float)
    beta = np.array([2.0, 0.0, -1.0, 0.5, 0.0, 1.5])
    b = L.koeffizienten(Z, np.ones(400), Z @ beta + 3.0, alpha=1e-9)
    assert np.allclose(b, beta, atol=1e-6)


def test_gewichte_sind_eins_beim_zielpunkt_und_fallen_mit_abweichung():
    Z = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]], float)
    w = L.gewichte(Z)
    assert w[0] == 1.0 and w[0] > w[1] > w[2]
    assert w[1] == pytest.approx(np.exp(-1 / 9)) and w[2] == pytest.approx(np.exp(-6 / 9))


def test_stichprobenarten_anteil_unmoeglicher_punkte_im_band():
    X = G.zeichne_zeilen(G.strom(1, 1), 600)
    x0 = X[0]
    gr = L.quartilgrenzen(X)
    anteil = {}
    for var in C.STICHPROBEN:
        Sm, Z = L.stichprobe(var, X, x0, 1000, G.strom(1, 7), gr)
        assert Sm.shape == (1000, C.D) and Z.shape == (1000, C.D)
        anteil[var] = float((~gueltig(Sm)).mean())
    assert anteil["zeile"] == 0.0                              # ganze echte Zeilen: nie unmöglich (exakt)
    assert 0.4 < anteil["std"] < 0.7                           # unabhängig gezogene Spalten: in der Messung 0,56
    assert 0.2 < anteil["koal"] < anteil["std"]                # Koalitions-Ersatz: in der Messung 0,42


def test_koalitionsdarstellung_ist_die_koalition_selbst():
    X = G.zeichne_zeilen(G.strom(2, 1), 100)
    Sm, Z = L.stichprobe("koal", X, X[0], 50, G.strom(2, 2), L.quartilgrenzen(X))
    assert set(np.unique(Z)) <= {0.0, 1.0}
    erste = Z[0] == 1
    assert np.array_equal(Sm[0][erste], X[0][erste])           # wo die Koalition gilt, steht der Wert der erklärten Zeile


def test_darstellung_gleiche_zeile_ist_ueberall_eins():
    X = G.zeichne_zeilen(G.strom(3, 1), 100)
    gr = L.quartilgrenzen(X)
    assert L.darstellung(X[:1], X[0], gr).tolist() == [[1.0] * 6]
    Y = X[:1].copy()
    Y[0, C.COL_ART] = (Y[0, C.COL_ART] + 1) % 4
    assert L.darstellung(Y, X[0], gr)[0, C.COL_ART] == 0.0


def test_unbekannte_stichprobenart_wirft():
    X = G.zeichne_zeilen(G.strom(3, 1), 10)
    with pytest.raises(ValueError):
        L.stichprobe("xyz", X, X[0], 5, G.strom(1, 1), L.quartilgrenzen(X))
