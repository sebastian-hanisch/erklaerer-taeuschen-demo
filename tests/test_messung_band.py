"""Neu-Rechnen im Band: Zufalls-Seed 0 in kleiner Fassung (Boosting, c = 1, ein Hintergrund, drei Detektoren) gegen das eingefrorene Raster.

Was nur vom Zufallsstrom abhängt (Anteil unmöglicher Mischzeilen), muss exakt stimmen (SplitMix64 mit plattformfestem normal()); was an Modellen hängt (Erkennungs- und Fehlalarmraten,
Rang-1-Anteile), muss im Band liegen (Gleitkomma-Summenreihenfolge und Gleichstände der Schnitte können unter Linux die letzte Stelle ändern)."""

import pytest

import ert_evaluation as ev
import ert_messung as M


@pytest.fixture(scope="module")
def neu():
    return M.messe_seed(0, bases=("gbm",), cs=(1.0,), mit_surrogat=False, mit_wissen=False, detektoren=("orakel", "gbm", "gedaechtnis"), hgs=[("training", 0.0)])


@pytest.fixture(scope="module")
def alt():
    return ev.lade_raster()["seeds"]["0"]


def test_anteil_unmoeglicher_mischzeilen_exakt(neu, alt):
    assert neu["unmoeglich_shapley"]["training_0"] == alt["unmoeglich_shapley"]["training_0"]
    assert neu["detektor"]["anteil_unmoeglich"] == alt["detektor"]["anteil_unmoeglich"]


def test_detektor_kennzahlen_im_band(neu, alt):
    for det in ("orakel", "gbm", "gedaechtnis"):
        for k, v in neu["detektor"][det].items():
            assert v == pytest.approx(alt["detektor"][det][k], abs=0.02), (det, k)


def test_rang1_anteile_im_band_einer_zeile(neu, alt):
    """24 erklärte Zeilen: ein Gleichstand kann einen Rang-1-Anteil um 1/24 verschieben; Band 0,1."""
    for m, z in neu["shapley"]["gbm"]["1"]["training_0"].items():
        assert z["top1"] == pytest.approx(alt["shapley"]["gbm"]["1"]["training_0"][m]["top1"], abs=0.1), m


def test_ersetzungsanteil_des_gedaechtnis_detektors_im_band(neu, alt):
    assert neu["ersetzung_q"]["training_0"]["gedaechtnis"] == pytest.approx(alt["ersetzung_q"]["training_0"]["gedaechtnis"], abs=0.01)
