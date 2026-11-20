"""Permalink (auf die erlaubten Werte begrenzt) und Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben).

Alle Regler sind Auswahlfelder mit festen Optionen (das Raster ist vorgerechnet): ein Permalink mit unbekanntem Wert behält die Voreinstellung."""

from dataclasses import dataclass

import streamlit as st

import ert_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    options: tuple


SETTING_SPECS = {
    "basis_sel": SettingSpec("basis", str, C.DEFAULT_BASIS, C.BASEN),
    "c_sel": SettingSpec("c", float, C.DEFAULT_C, C.C_REGLER),
    "detektor_sel": SettingSpec("det", str, C.DEFAULT_DETEKTOR, C.DETEKTOREN),
    "hintergrund_sel": SettingSpec("hg", str, "zufall", C.HINTERGRUENDE),
    "quelle_sel": SettingSpec("quelle", str, "training", C.QUELLEN),
    "stichprobe_sel": SettingSpec("stich", str, "std", C.STICHPROBEN),
}
PRESET_KEYS = {"basis": "basis_sel", "c": "c_sel", "detektor": "detektor_sel", "hintergrund": "hintergrund_sel", "quelle": "quelle_sel", "stichprobe": "stichprobe_sel"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, soweit sie zu den Optionen passen."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if value in spec.options:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos, wie im Gerüst)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )
