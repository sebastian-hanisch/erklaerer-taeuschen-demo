"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, stellt jeden Regler auf jede Option
und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei mehreren plotly_chart-Aufrufen ohne eindeutiges key=). Dazu die Kernaussagen der Seite."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import ert_constants as C
import ert_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
REGLER_ARTEN = ("selectbox", "radio", "select_slider")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _text(at):
    teile = [m.value for m in at.markdown] + [m.value for m in at.info] + [m.value for m in at.success] + [m.value for m in at.warning] + [c.value for c in at.caption]
    return "\n".join(teile)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert "keine reale Datenquelle" in _text(at)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_every_option_of_every_control_does_not_raise(state_key):
    for option in P.SETTING_SPECS[state_key].options:
        at = _fresh_app()
        _regler(at, state_key).set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, option, [str(e) for e in at.exception])


def test_c_null_zeigt_nicht_definiert_und_keine_zahlen_der_wirksamkeit():
    at = _fresh_app()
    _regler(at, "c_sel").set_value(0.0).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("nicht definiert" in i.value for i in at.info)
    assert [m.value for m in at.metric if m.label.startswith("Wirksamkeit")] == ["–", "–", "–"]


def test_voreinstellung_zeigt_vier_experimente_handbeispiel_und_grenzen():
    at = _fresh_app()
    sub = [s.value for s in at.subheader]
    assert [s for s in sub if s.startswith("🔬 Experiment")] == ["🔬 Experiment 1: Das Gerüst im Betrieb", "🔬 Experiment 2: Exakter Shapley-Wert mit Zufallshintergrund",
                                                                  "🔬 Experiment 3: Das Surrogat lässt sich leichter täuschen", "🔬 Experiment 4: Der Angreifer kennt die Trainingszeilen"]
    assert "🚧 Wo die Annahmen enden" in sub
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    zellen = [str(v) for df in at.dataframe for v in df.value.astype(str).to_numpy().ravel()]
    assert "5/6" in zellen and "5/12" in zellen and "5/4" in zellen                          # Handbeispiel: φ_k im Gerüst, mit Gedächtnis, Schein-Beitrag φ_g


def test_jedes_preset_zeigt_ein_verdikt_je_experiment_und_keinen_fehler():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert len(at.success) + len(at.info) + len(at.warning) >= 4, name


def test_die_metriken_der_voreinstellung_stehen_im_raster():
    at = _fresh_app()
    werte = {m.label: m.value for m in at.metric}
    assert werte["Treue des Gerüsts auf echten Zeilen"] == "100,0 %"                       # Orakel-Detektor: kein Fehlalarm auf echten Zeilen
    mittel, fehler = werte["Wirksamkeit unter dem Surrogat"].split(" ± ")
    assert 0.0 <= float(mittel.replace(",", ".")) <= 1.0 and float(fehler.replace(",", ".")) >= 0.0
