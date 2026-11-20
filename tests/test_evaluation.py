"""Kennzahlen aus dem Raster: Wirksamkeit, Standardfehler des Quotienten, Mittel, Zählung; Handwerte und Randfälle."""

import numpy as np
import pytest

import ert_evaluation as ev


def test_wirksamkeit_aus_mitteln_handwerte():
    w, se = ev.wirksamkeit_aus_mitteln([0.8, 0.8], [0.2, 0.2], [0.5, 0.5])
    assert w == pytest.approx(0.5) and se == pytest.approx(0.0)
    assert ev.wirksamkeit_aus_mitteln([0.8], [0.2], [0.8])[0] == 0
    assert ev.wirksamkeit_aus_mitteln([0.8], [0.2], [0.2])[0] == 1


def test_wirksamkeit_ist_das_verhaeltnis_der_mittel_nicht_das_mittel_der_verhaeltnisse():
    verz, ehr, ger = [0.9, 0.5], [0.1, 0.4], [0.5, 0.2]
    w, _ = ev.wirksamkeit_aus_mitteln(verz, ehr, ger)
    assert w == pytest.approx((0.7 - 0.35) / (0.7 - 0.25))
    mittel_der_verhaeltnisse = np.mean([(a - c) / (a - b) for a, b, c in zip(verz, ehr, ger)])
    assert abs(w - mittel_der_verhaeltnisse) > 0.1


def test_standardfehler_des_quotienten_nach_linearisierung_gegen_bootstrap():
    rng = np.random.default_rng(0)
    verz = 0.7 + 0.05 * rng.standard_normal(40)
    ehr = 0.2 + 0.05 * rng.standard_normal(40)
    ger = 0.6 + 0.05 * rng.standard_normal(40)
    w, se = ev.wirksamkeit_aus_mitteln(verz, ehr, ger)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, 40, 40)
        boot.append(ev.wirksamkeit_aus_mitteln(verz[i], ehr[i], ger[i])[0])
    assert se == pytest.approx(float(np.std(boot)), rel=0.2)            # Näherung durch Linearisierung, Band von 20 %


def test_wirksamkeit_randfaelle_kein_abstand_und_ein_wert():
    assert ev.wirksamkeit_aus_mitteln([0.5, 0.5], [0.5, 0.5], [0.3, 0.3]) == (None, None)
    w, se = ev.wirksamkeit_aus_mitteln([0.9], [0.1], [0.5])
    assert w == pytest.approx(0.5) and se == 0.0


def test_mittel_und_standardfehler():
    m, se = ev.mean_se([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and se == pytest.approx(np.std([1, 2, 3, 4], ddof=1) / 2)
    assert ev.mean_se([7.0]) == (7.0, 0.0)


def test_schluessel_der_zellen():
    assert ev.hg_key("training", 0.0) == "training_0" and ev.hg_key("frisch", 1.0) == "frisch_1" and ev.hg_key("training", 0.25) == "training_0.25"
    assert ev.c_key(1.0) == "1" and ev.c_key(0.5) == "0.5" and ev.c_key(2.0) == "2"


def test_zaehle_der_spitzenmerkmale():
    assert ev._zaehle([1, 1, 4, 1, 4, 5]) == {1: 3, 4: 2, 5: 1}
