"""Gerüst, Detektoren und Maße: Eigenschaften, Randfälle, Maße an handgesetzten Beiträgen."""

import numpy as np
import pytest

import ert_constants as C
import ert_generator as G
import ert_geruest as GR
from ert_messung import hintergrund_gruppen, hg_key
from ert_shapley import shapley
from ert_support import gueltig


def _h(klein, detektor):
    gb, gh = klein["gb"], klein["gh"]
    return lambda A: GR.geruest(gb.predict(A), gh.predict(A), detektor.ood(A))


def test_orakel_geruest_gleich_verzerrt_auf_moeglichen_zeilen_und_ehrlich_auf_unmoeglichen(klein):
    A = klein["Xe"]
    h = _h(klein, GR.OrakelDetektor())(A)
    assert np.array_equal(h, klein["gb"].predict(A))                 # echte Zeilen: verzerrt (Treue 1)
    B = GR.mische_spalten(klein["X"], 200, G.strom(1, 2))
    ung = ~gueltig(B)
    h2 = _h(klein, GR.OrakelDetektor())(B)
    assert ung.any()
    assert np.array_equal(h2[ung], klein["gh"].predict(B)[ung]) and np.array_equal(h2[~ung], klein["gb"].predict(B)[~ung])


def test_bei_hintergrund_derselben_art_ist_das_orakel_geruest_unsichtbar(klein):
    """Alle Mischzeilen sind möglich: Gerüst gleich verzerrtes Modell, Shapley-Werte identisch (Abweichung unter 1e-9)."""
    Xe = klein["Xe"]
    for quelle_X in (klein["X"], klein["Xf"]):
        for idx, B in hintergrund_gruppen(Xe, quelle_X, 1.0, 10, G.strom(5, 5)):
            from ert_shapley import mischzeilen
            assert gueltig(mischzeilen(Xe[idx], B)).all()                     # Anteil unmöglicher Mischzeilen exakt 0
            h = _h(klein, GR.OrakelDetektor())
            assert np.abs(shapley(h, Xe[idx], B) - shapley(klein["gb"].predict, Xe[idx], B)).max() < 1e-9


def test_bei_zufallshintergrund_gibt_es_unmoegliche_mischzeilen_und_die_beitraege_unterscheiden_sich(klein):
    Xe = klein["Xe"]
    idx, B = hintergrund_gruppen(Xe, klein["X"], 0.0, 10, G.strom(5, 6))[0]
    from ert_shapley import mischzeilen
    assert (~gueltig(mischzeilen(Xe[idx], B))).mean() > 0.05
    h = _h(klein, GR.OrakelDetektor())
    assert np.abs(shapley(h, Xe[idx], B) - shapley(klein["gb"].predict, Xe[idx], B)).max() > 1e-3


def test_effizienz_fuer_geruest_und_beide_teilmodelle(klein):
    Xe, B = klein["Xe"], klein["Xf"][:6]
    for f in (klein["gb"].predict, klein["gh"].predict, _h(klein, GR.OrakelDetektor())):
        phi = shapley(f, Xe, B)
        from ert_shapley import mischzeilen
        v_leer = f(mischzeilen(Xe[:1], B)[:len(B)]).mean()                 # Koalition 0 = nur Hintergrund
        assert np.allclose(phi.sum(axis=1), f(Xe) - v_leer, atol=1e-10)


def test_gedaechtnis_meldet_hintergrundzeilen_aus_dem_training_und_laesst_frische_zeilen_durch(klein):
    det = GR.GedaechtnisDetektor(klein["X"])
    assert det.ood(klein["X"][:50]).all()                    # Gewicht exakt im Training
    assert not det.ood(klein["Xf"][:50]).any()               # frische echte Zeilen: stetige Werte treffen nie exakt
    Xmix = klein["Xf"][:30].copy()
    Xmix[:, C.COL_GEWICHT] = klein["X"][:30, C.COL_GEWICHT]  # Gewicht aus dem Training: Alarm
    assert det.ood(Xmix).all()


def test_detektor_ohne_alarm_und_mit_dauer_alarm_randfaelle(klein):
    class Nie:
        def ood(self, A):
            return np.zeros(len(A), bool)

    class Immer:
        def ood(self, A):
            return np.ones(len(A), bool)

    A = klein["Xe"]
    assert np.array_equal(_h(klein, Nie())(A), klein["gb"].predict(A))        # q = 0: immer verzerrt
    assert np.array_equal(_h(klein, Immer())(A), klein["gh"].predict(A))      # q = 1: immer ehrlich


def test_masse_an_handgesetzten_beitraegen():
    phi = np.array([[0.0, 3.0, 1.0, 0.0, 0.0, 0.0],      # Kunde (Spalte 1) Rang 1
                    [0.0, 1.0, 5.0, 0.0, 0.0, 0.0],      # Rang 2
                    [0.1, -4.0, 0.0, 0.0, 0.0, 0.0],     # Rang 1 (Betrag)
                    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0]])     # Kunde 0: Rang 3 (Gleichstand der Nullen zählt nach Spaltenindex, Spalte 0 steht davor)
    m = GR.masse(phi)
    assert m["top1"] == 0.5
    assert m["mrang"] == (1 + 2 + 1 + 3) / 4
    # Wichtigkeiten (Mittel der Beträge je Spalte): 0,025 / 2 / 1,5 / 0 / 0,5 / 0, Summe 4,025
    assert m["anteil"] == pytest.approx(2.0 / 4.025)
    assert m["spitze"] == 1 and m["grang"] == 1


def test_wirksamkeit_formel_und_randfaelle():
    assert GR.wirksamkeit(0.8, 0.2, 0.5) == pytest.approx(0.5)
    assert GR.wirksamkeit(0.8, 0.2, 0.8) == 0 and GR.wirksamkeit(0.8, 0.2, 0.2) == 1
    assert GR.wirksamkeit(0.8, 0.2, 0.1) > 1                             # über das ehrliche Modell hinaus
    assert GR.wirksamkeit(0.5, 0.5, 0.3) is None                         # c = 0: nicht definiert


def test_rangkorrelation_randfaelle():
    assert GR.rangkorrelation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert GR.rangkorrelation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(GR.rangkorrelation([1, 1, 1], [1, 2, 3]))


def test_hintergrundgruppen_form_und_anteil_derselben_art(klein):
    Xe = klein["Xe"]
    (idx, B), = hintergrund_gruppen(Xe, klein["X"], 0.0, 12, G.strom(1, 1))
    assert len(idx) == len(Xe) and B.shape == (12, C.D)
    for idx, B in hintergrund_gruppen(Xe, klein["X"], 0.5, 12, G.strom(1, 1)):
        art = Xe[idx[0], C.COL_ART]
        assert (Xe[idx][:, C.COL_ART] == art).all()
        assert (B[:6][:, C.COL_ART] == art).all()                          # Hälfte aus derselben Art (Reihenfolge: erst die der Art)
    assert hg_key("training", 0.25) == "training_0.25" and hg_key("frisch", 1.0) == "frisch_1"
