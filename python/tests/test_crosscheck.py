"""The sphere cross-check of docs/crosscheck.md at its main point (radius 3 µm, dn 0.02).

Needs the check group (uv run --group check pytest -m slow); skipped without qpsphere and miepython.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("qpsphere")
pytest.importorskip("miepython")

SCRIPT = Path(__file__).resolve().parents[2] / "checks" / "sphere_crosscheck.py"
spec = importlib.util.spec_from_file_location("sphere_crosscheck", SCRIPT)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

pytestmark = pytest.mark.slow


def test_rytov_and_born_match_qpsphere_at_the_same_period():
    for model in ("rytov", "born"):
        ours = sc.qoraholo_field(model, sc.RADIUS, sc.DN)
        theirs = sc.qpsphere_field(model, sc.RADIUS, sc.DN)
        assert sc.compare(ours, theirs, sc.RADIUS)["field_rel"] < 1e-4


def test_script_refocusing_reproduces_qoraholo_born():
    built = sc.refocused_field(sc.born_amplitude(sc.RADIUS, sc.DN))
    born = sc.qoraholo_field("born", sc.RADIUS, sc.DN)
    assert sc.compare(built, born, sc.RADIUS)["field_rel"] < 1e-4


def test_scalar_exact_against_vector_mie_at_na_1_2():
    exact = sc.lowpass(sc.qoraholo_field("exact", sc.RADIUS, sc.DN), 1.2)
    vector = sc.lowpass(sc.refocused_field(sc.mie_amplitude(sc.RADIUS, sc.DN)), 1.2)
    gap = sc.compare(exact, vector, sc.RADIUS)["field_rel"]
    assert 0.012 < gap < 0.019  # 0.0156 in docs/crosscheck.md
