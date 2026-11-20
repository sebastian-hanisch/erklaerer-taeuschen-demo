"""Gemeinsame Fixtures: kleine Datensätze und Modelle (schnell), damit die Orakel- und Eigenschaftstests in Sekunden laufen."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ert_generator as G  # noqa: E402
from ert_models import Boosting  # noqa: E402


@pytest.fixture(scope="session")
def klein():
    """Kleiner Datensatz (n = 500), ein verzerrtes und ein ehrliches Boosting-Modell (20 Runden), fester Zufalls-Seed."""
    X, y = G.zeichne_daten(500, 3, 1)
    gh = Boosting(runden=20).fit(X, y)
    gb = Boosting(runden=20).fit(X, y + 1.0 * G.gruppe(X))
    Xe = G.zeichne_zeilen(G.strom(3, 3), 6)
    Xf = G.zeichne_zeilen(G.strom(3, 4), 300)
    return dict(X=X, y=y, gh=gh, gb=gb, Xe=Xe, Xf=Xf)
