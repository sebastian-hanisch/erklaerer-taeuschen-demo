"""Das eingefrorene Raster (data/ert_grid.json): Vollständigkeit, Invarianten, die jeder Lauf erfüllen muss, und Gegenproben zu den Absolutaussagen.

Die Invarianten sind Orakel am Raster: Gerüst mit Orakel-Detektor gleich verzerrtes Modell bei Hintergrund derselben Art (exakt), Anteil unmöglicher Mischzeilen bei gleicher Art
exakt 0, das ehrliche Modell hängt nicht vom Aufschlag c ab, Zählungen über alle Zellen."""

import json
import math

import pytest

import ert_constants as C
import ert_evaluation as ev

R = ev.lade_raster()
SEEDS = ev.seeds(R)
KEYS = [(q, p) for q, p in [("training", p) for p in C.P_STUFEN] + [("frisch", 0.0), ("frisch", 1.0)]]


def test_raster_hat_zehn_zufalls_seeds_und_alle_zellen():
    assert R["meta"]["seeds"] == list(C.SEEDS) and len(SEEDS) == 10
    for s in SEEDS:
        assert set(s["shapley"]) == set(C.BASEN)
        for b in C.BASEN:
            assert set(s["shapley"][b]) == {ev.c_key(c) for c in C.C_STUFEN}
            for c in C.C_STUFEN:
                assert set(s["shapley"][b][ev.c_key(c)]) == {ev.hg_key(q, p) for q, p in KEYS}
                for z in s["shapley"][b][ev.c_key(c)].values():
                    assert set(z) == {"verzerrt", "ehrlich"} | {f"geruest_{d}" for d in C.DETEKTOREN}
        assert set(s["surrogat"]) == set(C.BASEN)
        assert set(s["wissen"]) == {"gbm", "rf"}
        assert set(s["detektor"]) == set(C.DETEKTOREN) | {"anteil_unmoeglich"}


def test_raster_meta_nennt_zaehleinheiten():
    m = R["meta"]
    assert (m["n_train"], m["n_mix"], m["ne_shapley"], m["nb"], m["ne_lime"], m["n_lime"]) == (3000, 3000, 24, 30, 16, 1000)
    assert "keine reale Datenquelle" in m["quelle"]


def test_orakel_geruest_gleich_verzerrt_bei_gleicher_art_in_allen_zellen():
    """Alle Mischzeilen sind möglich: die Maße des Orakel-Gerüsts sind exakt die des verzerrten Modells (alle Zufalls-Seeds, Modellklassen, Aufschläge, Quellen)."""
    n = 0
    for s in SEEDS:
        for b in C.BASEN:
            for c in C.C_STUFEN:
                for q in C.QUELLEN:
                    z = s["shapley"][b][ev.c_key(c)][ev.hg_key(q, 1.0)]
                    assert z["geruest_orakel"] == z["verzerrt"], (s["seed"], b, c, q)
                    n += 1
    assert n == 10 * 3 * 3 * 2


def test_anteil_unmoeglicher_mischzeilen_bei_gleicher_art_ist_exakt_null_sonst_positiv():
    for s in SEEDS:
        for q in C.QUELLEN:
            assert s["unmoeglich_shapley"][ev.hg_key(q, 1.0)] == 0.0
            assert s["unmoeglich_shapley"][ev.hg_key(q, 0.0)] > 0.2                       # Gegenprobe: bei Zufallshintergrund ist rund ein Viertel unmöglich
    mittel = [ev.mean_se([s["unmoeglich_shapley"][ev.hg_key("training", p)] for s in SEEDS])[0] for p in C.P_STUFEN]
    assert all(a > b for a, b in zip(mittel, mittel[1:]))                               # im Mittel über Zufalls-Seeds fällt er mit p streng


def test_ehrliches_modell_haengt_nicht_vom_aufschlag_ab():
    for s in SEEDS:
        for b in C.BASEN:
            for key in (ev.hg_key("training", 0.0), ev.hg_key("frisch", 1.0)):
                ref = {k: v for k, v in s["shapley"][b]["1"][key]["ehrlich"].items() if k != "rk"}       # rk (Rangkorrelation zum verzerrten Modell) hängt an c
                for c in ("0.5", "2"):
                    assert {k: v for k, v in s["shapley"][b][c][key]["ehrlich"].items() if k != "rk"} == ref


def test_orakel_und_gedaechtnis_erkennen_alle_unmoeglichen_zeilen_ohne_fehlalarm():
    for s in SEEDS:
        for d in ("orakel", "gedaechtnis"):
            z = s["detektor"][d]
            assert z["erk_spalten"] == 1.0 and z["erk_shapley"] == 1.0 and z["fehl_moeglich"] == 0.0 and z["fehl_echt"] == 0.0


def test_lineares_modell_als_detektor_liegt_nahe_am_zufall_in_allen_zufalls_seeds():
    for s in SEEDS:
        z = s["detektor"]["linear"]
        assert abs(z["erk_spalten"] - 0.5) < 0.1 and abs(z["fehl_echt"] - 0.5) < 0.15, s["seed"]


def test_frischer_hintergrund_bei_gleicher_art_macht_das_gedaechtnis_unwirksam():
    """Gedächtnis-Gerüst bei frischem Hintergrund derselben Art: identisch zum verzerrten Modell (Ersetzungsanteil 0)."""
    for s in SEEDS:
        for b in C.BASEN:
            for c in C.C_STUFEN:
                z = s["shapley"][b][ev.c_key(c)][ev.hg_key("frisch", 1.0)]
                assert z["geruest_gedaechtnis"] == z["verzerrt"]
        assert s["ersetzung_q"][ev.hg_key("frisch", 1.0)]["gedaechtnis"] == 0.0


def test_gegenprobe_gedaechtnis_hat_bei_training_hohen_ersetzungsanteil():
    q = ev.mean_se([s["ersetzung_q"][ev.hg_key("training", 0.0)]["gedaechtnis"] for s in SEEDS])[0]
    q_orakel = ev.mean_se([s["ersetzung_q"][ev.hg_key("training", 0.0)]["orakel"] for s in SEEDS])[0]
    assert q > 2 * q_orakel and q > 0.6


def test_surrogat_mit_ganzen_zeilen_hat_nie_unmoegliche_punkte():
    for s in SEEDS:
        assert s["unmoeglich_surrogat"]["training_zeile"] == 0.0 and s["unmoeglich_surrogat"]["frisch_zeile"] == 0.0
        assert s["unmoeglich_surrogat"]["training_std"] > 0.5 and s["unmoeglich_surrogat"]["training_koal"] > 0.3
    for b in C.BASEN:
        for q in C.QUELLEN:
            for s in SEEDS:
                z = s["surrogat"][b][q]["zeile"]
                assert z["geruest_orakel"] == z["verzerrt"]                              # ganze echte Zeilen: das Orakel-Gerüst ist unsichtbar


def test_alle_zahlen_sind_endlich_und_anteile_liegen_in_0_bis_1():
    def lauf(x):
        if isinstance(x, dict):
            for v in x.values():
                lauf(v)
        elif isinstance(x, float):
            assert math.isfinite(x) or x != x                                              # NaN nur bei konstanten Rängen (Rangkorrelation), nie unendlich
        elif isinstance(x, list):
            for v in x:
                lauf(v)
    lauf(R["seeds"])
    for s in SEEDS:
        for b in C.BASEN:
            for c in C.C_STUFEN:
                for z in s["shapley"][b][ev.c_key(c)].values():
                    for m in z.values():
                        assert 0.0 <= m["top1"] <= 1.0 and 0.0 <= m["anteil"] <= 1.0 and 1.0 <= m["mrang"] <= 6.0 and 1 <= m["grang"] <= 6


def test_quelltext_hash_des_rasters_gleicht_dem_aktuellen_quelltext():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("ert_make_grid", Path(__file__).resolve().parent.parent / "tools" / "ert_make_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert R["meta"]["quelltext_hash"] == mod.quelltext_hash(), "Quelltext der Messung geändert: Raster neu rechnen (tools/ert_make_grid.py)"


def test_raster_json_ist_gueltig_und_klein():
    pfad = ev.ROOT / C.GRID_DATEI
    assert pfad.stat().st_size < 2_000_000
    assert json.loads(pfad.read_text(encoding="utf-8"))["meta"]["stand"]
