"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink auf die Optionen begrenzt."""

import ert_constants as C
import ert_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_among_the_options():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            assert preset[key] in P.SETTING_SPECS[state_key].options, (name, key)


def test_every_preset_differs_from_every_other_in_at_least_one_setting():
    """Sonst könnte die Hervorhebung zwei Presets nicht unterscheiden."""
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)


def test_default_preset_equals_the_default_settings():
    erster = C.PRESETS[next(iter(C.PRESETS))]
    for key, state_key in P.PRESET_KEYS.items():
        assert erster[key] == P.SETTING_SPECS[state_key].default


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_option_has_data_in_the_grid_or_is_c_zero():
    """Jede wählbare Zelle ist vorgerechnet: Aufschläge c im Raster (außer 0), Modellklassen, Detektoren, Stichproben, Hintergrundquellen."""
    assert set(C.C_REGLER) - {0.0} == set(C.C_STUFEN)
    assert set(C.BASEN) <= set(C.BASIS_LABEL) and set(C.DETEKTOREN) <= set(C.DET_LABEL) and set(C.STICHPROBEN) <= set(C.STICHPROBE_LABEL)
    assert C.P_STUFEN[0] == 0.0 and C.P_STUFEN[-1] == 1.0 and set(C.HINTERGRUENDE) == {"zufall", "art"}
