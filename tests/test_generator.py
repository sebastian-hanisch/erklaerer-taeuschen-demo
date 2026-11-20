"""Kopie-Test: der Generator ist aus verweilzeit-demo kopiert (nicht importiert). Mitte und Streuung gegen eingefrorene Werte der Quelle, Kundeneffekte gegen numpy,
Zeilen gegen die Gültigkeitsregeln, Zufallsstrom reproduzierbar."""

import numpy as np
import pytest

import ert_constants as C
import ert_generator as G
from ert_support import gueltig

# aus verweilzeit-demo/vwz_generator.py (log_median_and_s), 2026-11-19 gerechnet: (art, kunde, zoll, we, vorlauf) -> (mu, s)
EINGEFROREN = [
    ((0, 0, 0, 0, 0.0), (0.8256154575589787, 0.55)),
    ((0, 3, 1, 1, 0.0), (2.1750109980150487, 0.55)),
    ((1, 5, 0, 0, 4.5), (1.420590659772188, 0.25)),
    ((1, 11, 0, 1, 9.0), (2.1973779218642626, 0.25)),
    ((2, 7, 0, 0, 2.0), (0.7002542383697684, 0.35)),
    ((2, 0, 0, 1, 5.5), (1.4194182554350963, 0.35)),
    ((3, 2, 0, 0, 0.0), (1.6847116276675875, 0.9)),
    ((3, 9, 0, 1, 0.0), (1.6600512301516563, 0.9)),
]


@pytest.mark.parametrize("zeile,erwartet", EINGEFROREN)
def test_mitte_und_streuung_gleich_der_quelle(zeile, erwartet):
    mu, s = G.log_median_and_s(*[np.array(v) for v in zeile])
    assert float(mu) == pytest.approx(erwartet[0], rel=1e-12)
    assert float(s) == pytest.approx(erwartet[1], rel=1e-12)


def test_kundeneffekte_gleich_numpy_ziehung_der_quelle():
    assert np.allclose(C.UK, np.random.default_rng(C.KUNDEN_SEED_VWZ).normal(0.0, C.U_SD, C.K_KUNDEN), rtol=0, atol=1e-15)


def test_konstanten_der_quelle():
    assert C.ART_P == (0.35, 0.30, 0.25, 0.10) and sum(C.ART_P) == pytest.approx(1.0)
    assert C.K_KUNDEN == 12 and C.U_SCALE == (1.0, 0.3, 0.3, 1.0) and C.S_BASE == (0.55, 0.25, 0.35, 0.90) and C.P_ZOLL == 0.25


def test_zeilen_sind_moeglich_und_im_wertebereich():
    X = G.zeichne_zeilen(G.strom(1, 1), 800)
    assert gueltig(X).all()
    assert set(np.unique(X[:, C.COL_ART])) <= {0, 1, 2, 3}
    assert X[:, C.COL_KUNDE].min() >= 0 and X[:, C.COL_KUNDE].max() <= 11
    assert (X[X[:, C.COL_ART] == 1][:, C.COL_VORLAUF] >= 1).all() and (X[X[:, C.COL_ART] == 1][:, C.COL_VORLAUF] < 10).all()
    assert (X[X[:, C.COL_ART] == 2][:, C.COL_VORLAUF] >= 0.5).all() and (X[X[:, C.COL_ART] == 2][:, C.COL_VORLAUF] < 6).all()
    assert (X[X[:, C.COL_ZOLL] == 1][:, C.COL_ART] == 0).all()


def test_anteil_der_arten_im_band():
    n = 1500
    X = G.zeichne_zeilen(G.strom(2, 1), n)
    for a, p in enumerate(C.ART_P):
        anteil = (X[:, C.COL_ART] == a).mean()
        assert abs(anteil - p) < 4 * np.sqrt(p * (1 - p) / n), (a, anteil)


def test_gleicher_zufalls_seed_gleiche_zeilen_anderer_seed_andere():
    a = G.zeichne_zeilen(G.strom(5, 1), 50)
    b = G.zeichne_zeilen(G.strom(5, 1), 50)
    c = G.zeichne_zeilen(G.strom(6, 1), 50)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_eingefrorene_zeilen_des_zufallsstroms():
    """Der Strom ist SplitMix64: die ersten Zeilen sind auf jeder Plattform dieselben (Neu-Rechnen-Tests brauchen das)."""
    X = G.zeichne_zeilen(G.strom(0, 1), 3)
    assert X[:, :5].tolist() == [[0.0, 8.0, 0.0, 1.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0, 0.0]]


def test_wahrheit_ist_mu_der_zeilen_und_daten_haben_rauschen_um_mu():
    X, y = G.zeichne_daten(600, 7, 1)
    mu, s = G.log_median_and_s(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
    assert np.allclose(G.wahrheit(X), mu)
    z = (y - mu) / s
    assert abs(z.mean()) < 0.2 and abs(z.std() - 1.0) < 0.15      # standardnormal im Band (n = 600: Standardfehler 0,04 bzw. 0,03)


def test_gewicht_ist_reines_rauschen_mu_haengt_nicht_an_ihm():
    X = G.zeichne_zeilen(G.strom(4, 1), 20)
    X2 = X.copy()
    X2[:, C.COL_GEWICHT] += 5.0
    assert np.array_equal(G.wahrheit(X), G.wahrheit(X2))


def test_gruppe_ist_kunde_ab_6():
    X = np.zeros((12, C.D))
    X[:, C.COL_KUNDE] = np.arange(12)
    assert G.gruppe(X).tolist() == [-0.5] * 6 + [0.5] * 6
