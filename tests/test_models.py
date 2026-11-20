"""Modellklassen (numpy) gegen Fremdreferenzen und Eigenschaften: flache Bäume, Boosting gegen scikit-learn, Wald, Netz, logistisches Modell."""

import numpy as np
import pytest

import ert_generator as G
import ert_models as M


def _r2(p, z):
    return 1.0 - float(np.sum((p - z) ** 2) / np.sum((z - z.mean()) ** 2))


@pytest.fixture(scope="module")
def daten():
    X, y = G.zeichne_daten(900, 11, 1)
    Xt = G.zeichne_zeilen(G.strom(11, 9), 600)
    return X, y, Xt, G.wahrheit(Xt)


def test_flacher_baum_gegen_handbaum():
    """Baum mit Wurzel (Merkmal 0 < 2), links Blatt 1, rechts innerer Knoten (Merkmal 1 < 5) mit Blättern 2 und 3."""
    t = M.Baum(feat=[0, -1, 1, -1, -1], thr=[2, 0, 5, 0, 0], left=[1, -1, 3, -1, -1], right=[2, -1, 4, -1, -1], val=[0, 1.0, 0, 2.0, 3.0])
    X = np.array([[1, 9.0], [2, 4.0], [2, 5.0], [7, 100.0]])
    assert t.predict(X).tolist() == [1.0, 2.0, 3.0, 3.0]


def test_boosting_trainingsfehler_sinkt_und_stapel_gleich_baumweise(daten):
    X, y, Xt, _ = daten
    m5 = M.Boosting(runden=5).fit(X, y)
    m40 = M.Boosting(runden=40).fit(X, y)
    assert np.mean((m40.predict(X) - y) ** 2) < np.mean((m5.predict(X) - y) ** 2)
    baumweise = m40.f0 + m40.lr * sum(t.predict(Xt) for t in m40.baeume)
    assert np.allclose(m40.predict(Xt), baumweise, atol=1e-12)


def test_boosting_gegen_scikit_learn_im_band(daten):
    sk = pytest.importorskip("sklearn.ensemble")
    X, y, Xt, mu = daten
    ref = sk.GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0).fit(X, y)
    m = M.Boosting(runden=100).fit(X, y)
    assert _r2(m.predict(Xt), mu) > 0.9 and _r2(ref.predict(Xt), mu) > 0.9
    assert abs(_r2(m.predict(Xt), mu) - _r2(ref.predict(Xt), mu)) < 0.05           # beide Umsetzungen gleich gut gegen die bekannte Funktion mu


def test_boosting_gegen_konstante_zielgroesse(daten):
    X, _, Xt, _ = daten
    m = M.Boosting(runden=5).fit(X, np.full(len(X), 2.5))
    assert np.allclose(m.predict(Xt), 2.5)


def test_zufallswald_und_netz_lernen_mu(daten):
    X, y, Xt, mu = daten
    rf = M.Zufallswald(1, baeume=15).fit(X, y)
    nn = M.Netz(1, schritte=800).fit(X, y)
    assert _r2(rf.predict(Xt), mu) > 0.85
    assert _r2(nn.predict(Xt), mu) > 0.85


def test_zufallswald_ist_reproduzierbar_und_haengt_am_zufalls_seed(daten):
    X, y, Xt, _ = daten
    a = M.Zufallswald(2, baeume=6).fit(X, y).predict(Xt)
    b = M.Zufallswald(2, baeume=6).fit(X, y).predict(Xt)
    c = M.Zufallswald(3, baeume=6).fit(X, y).predict(Xt)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_verzerrtes_modell_lernt_den_aufschlag(daten):
    """Ziel ln Verweilzeit + c (Gruppe - 1/2): der mittlere Abstand der Vorhersagen für Kunden 6 bis 11 gegen 0 bis 5 wächst um etwa c."""
    X, y, Xt, _ = daten
    gh = M.Boosting(runden=60).fit(X, y)
    gb = M.Boosting(runden=60).fit(X, y + 1.0 * G.gruppe(X))
    diff = gb.predict(Xt) - gh.predict(Xt)
    hoch = G.gruppe(Xt) > 0
    assert diff[hoch].mean() - diff[~hoch].mean() == pytest.approx(1.0, abs=0.15)


def test_logistisches_modell_gegen_scikit_learn(daten):
    sk = pytest.importorskip("sklearn.linear_model")
    from ert_geruest import mische_spalten
    X, _, Xt, _ = daten
    Xm = mische_spalten(X, len(X), G.strom(1, 2))
    Xa = np.vstack([X, Xm])
    ya = np.r_[np.zeros(len(X)), np.ones(len(Xm))]
    mine = M.Logistisch(c=1.0).fit(Xa, ya)
    ref = sk.LogisticRegression(C=1.0, max_iter=5000).fit(M._logit_eingabe(Xa)[:, 1:], ya)
    p_ref = ref.predict_proba(M._logit_eingabe(Xt)[:, 1:])[:, 1]
    assert np.mean(np.abs(mine.proba(Xt) - p_ref)) < 0.05          # Achsenabschnitt hier mitbestraft, sonst dieselbe Zielfunktion
    assert np.mean((mine.proba(Xa) > 0.5) == (ya > 0.5)) < 0.65    # nahe am Zufall: Art mal Vorlauf ist keine lineare Grenze


def test_detektor_erkennt_unmoegliches(daten):
    from ert_geruest import mische_spalten
    from ert_support import gueltig
    X, _, Xt, _ = daten
    Xm = mische_spalten(X, len(X), G.strom(1, 2))
    det = M.DetektorModell("gbm", X, Xm, 1)
    Xp = mische_spalten(Xt, 600, G.strom(1, 3))
    ung = ~gueltig(Xp)
    assert det.ood(Xp)[ung].mean() > 0.85
    assert det.ood(Xt).mean() < 0.1


def test_mache_modell_kennt_nur_die_drei_klassen():
    assert isinstance(M.mache_modell("gbm", 0), M.Boosting) and isinstance(M.mache_modell("rf", 0), M.Zufallswald) and isinstance(M.mache_modell("mlp", 0), M.Netz)
    with pytest.raises(ValueError):
        M.mache_modell("svm", 0)
