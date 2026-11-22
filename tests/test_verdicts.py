"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) in `ert_evaluation.py`; die App ruft st.<Art>(Text)."""

import ert_evaluation as ev


def _z(w, se, k_ger=10, k_verz=10):
    """Zusammenfassung wie shapley_zusammenfassung, nur die Felder, die das Verdikt liest."""
    return dict(wirksamkeit=(w, se), geruest=dict(rang1_global=k_ger), verzerrt=dict(rang1_global=k_verz))


def test_betrieb_detektor_scheitert_unter_60_prozent_erkennung():
    kind, text = ev.verdict_betrieb("linear", 0.494, 0.487, 0.513, 10)
    assert kind == "warning" and "„linear“" in text and "nur 49,4 %" in text and "48,7 %" in text and "10 Zufalls-Seeds" in text
    assert ev.verdict_betrieb("linear", 0.6, 0.0, 1.0, 10)[0] != "warning"          # Schwelle 0,6: genau 60 % zählt nicht mehr als gescheitert
    assert ev.verdict_betrieb("linear", 0.5999, 0.0, 1.0, 10)[0] == "warning"


def test_betrieb_fehlalarm_ueber_5_prozent_ist_info_darunter_success():
    kind, text = ev.verdict_betrieb("rf", 0.992, 0.075, 0.925, 10)
    assert kind == "info" and "92,5 %" in text and "99,2 %" in text and "7,5 %" in text
    assert ev.verdict_betrieb("rf", 0.992, 0.05, 0.95, 10)[0] == "success"           # genau 5 % Fehlalarm: success (Schwelle ist strikt größer)
    kind, text = ev.verdict_betrieb("gbm", 0.982, 0.011, 0.989, 10)
    assert kind == "success" and "98,9 %" in text and "98,2 %" in text and "„Boosting“" in text


def test_shapley_hintergrund_derselben_art_mit_orakel_ist_exakt_wirkungslos():
    kind, text = ev.verdict_shapley(_z(0.0, 0.0), True, True, 1.0, 10)
    assert kind == "success" and "in allen 10 Zufalls-Seeds identisch" in text and "exakt" in text
    kind2, text2 = ev.verdict_shapley(_z(0.0, 0.0), True, False, 1.0, 10)                       # „identisch“ gilt nur für den Orakel-Detektor
    assert kind2 == "info" and "identisch" not in text2


def test_shapley_schwache_taeuschung_nennt_zaehlung_und_standardfehler():
    kind, text = ev.verdict_shapley(_z(0.17, 0.03), False, True, 1.0, 10)
    assert kind == "info" and "10 von 10" in text and "0,17 ± 0,03" in text and "nur schwach" in text


def test_shapley_deutliche_taeuschung_ab_halber_wirksamkeit_ist_warning():
    kind, text = ev.verdict_shapley(_z(0.5, 0.04, k_ger=4), False, False, 1.0, 10)
    assert kind == "warning" and "0,50 ± 0,04" in text and "4 von 10" in text
    assert ev.verdict_shapley(_z(0.4999, 0.04), False, False, 1.0, 10)[0] == "info"


def test_shapley_zwischenwert_nennt_beide_zaehlungen():
    kind, text = ev.verdict_shapley(_z(0.3, 0.05, k_ger=9, k_verz=10), False, True, 1.0, 10)
    assert kind == "info" and "9 von 10" in text and "ohne Gerüst in 10" in text


def test_shapley_unmaskiert_kaum_sichtbar_warnt_vor_ueberinterpretation():
    kind, text = ev.verdict_shapley(_z(0.29, 0.11, k_ger=0, k_verz=2), False, True, 1.0, 10)
    assert kind == "info" and "nur in 2 von 10" in text and "kaum sichtbar" in text


def test_shapley_ohne_abstand_ist_nicht_definiert():
    kind, text = ev.verdict_shapley(_z(None, None), False, True, 0.0, 10)
    assert kind == "info" and "nicht definiert" in text


def test_surrogat_ganze_zeilen_sind_wirkungslos_unabhaengig_vom_verfahren():
    z = dict(wirksamkeit=(0.0, 0.0), unmoeglich=(0.0, 0.0))
    kind, text = ev.verdict_surrogat("zeile", z, 10)
    assert kind == "success" and "0,0 % unmögliche Punkte" in text and "in allen 10 Zufalls-Seeds" in text and "Wirksamkeit ist 0" in text


def test_surrogat_unabhaengige_spalten_taeuschen_ab_0_4():
    z = dict(wirksamkeit=(0.76, 0.04), unmoeglich=(0.562, 0.0))
    kind, text = ev.verdict_surrogat("std", z, 10)
    assert kind == "warning" and "56,2 %" in text and "0,76 ± 0,04" in text and "nicht das Verfahren" in text
    assert ev.verdict_surrogat("koal", dict(wirksamkeit=(0.3999, 0.04), unmoeglich=(0.42, 0.0)), 10)[0] == "info"
    assert ev.verdict_surrogat("koal", dict(wirksamkeit=(0.4, 0.04), unmoeglich=(0.42, 0.0)), 10)[0] == "warning"


def test_angreifer_training_deutlich_staerker_als_orakel_ist_warning():
    kind, text = ev.verdict_angreifer(0.79, 0.04, 0.17, 0.03, "training", False, 10)
    assert kind == "warning" and "0,79 ± 0,04" in text and "0,17 ± 0,03" in text and "aus Zufall aus Trainingszeilen" in text
    assert "derselben Art aus Trainingszeilen" in ev.verdict_angreifer(0.79, 0.04, 0.17, 0.03, "training", True, 10)[1]


def test_angreifer_schwelle_ist_ein_abstand_von_0_3():
    assert ev.verdict_angreifer(0.47, 0.04, 0.17, 0.03, "training", True, 10)[0] == "warning"           # 0,47 - 0,17 = 0,30
    assert ev.verdict_angreifer(0.469, 0.04, 0.17, 0.03, "training", True, 10)[0] == "info"


def test_angreifer_frischer_hintergrund_bringt_nichts_dazu():
    kind, text = ev.verdict_angreifer(0.13, 0.04, 0.13, 0.04, "frisch", False, 10)
    assert kind == "success" and "bringt das Gedächtnis nichts dazu" in text and "0,13 ± 0,04" in text
    assert ev.verdict_angreifer(0.2, 0.04, 0.13, 0.04, "frisch", False, 10)[0] == "info"                # Abstand 0,07: nicht mehr „nichts“
    assert ev.verdict_angreifer(None, None, 0.13, 0.04, "frisch", False, 10)[0] == "info"


def test_zahlenformate_dezimalkomma():
    assert ev.de(0.5, 2) == "0,50" and ev.pct(0.1234) == "12,3 %" and ev.pct(1.0, 0) == "100 %"


def test_hinweis_treue_erst_unter_95_prozent_und_mit_zahl():
    assert ev.hinweis_treue("orakel", 1.0) == "" and ev.hinweis_treue("gbm", 0.95) == ""
    text = ev.hinweis_treue("rf", 0.925)
    assert "„Random Forest“" in text and "92,5 %" in text and "Modellwechsel" in text
