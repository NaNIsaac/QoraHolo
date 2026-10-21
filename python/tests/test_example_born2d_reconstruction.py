"""examples/born2d_reconstruction end to end: known 2D object -> Born or BPM data -> Landweber.

Truth checks are physical expectations (coverage, resolution, missing-cone elongation, the order
of the Born model error). The recovered index contrast is known to come out low (QoraHolo#3:
missing cone and early stopping) and is a strict xfail, so a better reconstruction shows up.
The regression block pins today's output. Units: µm, cycles/µm, refractive-index units.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).parents[2] / "examples" / "born2d_reconstruction"
EXPECTED = json.loads((EXAMPLE / "expected.json").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("born2d_reconstruction", EXAMPLE / "run.py")
example = importlib.util.module_from_spec(_spec)
sys.modules["born2d_reconstruction"] = example
_spec.loader.exec_module(example)


@pytest.fixture(scope="module")
def result():
    return example.run()


def _check(name: str, got: float, spec: dict):
    assert abs(got - spec["value"]) <= spec["tol"], (
        f"{name}: {got:.6g} vs {spec['value']} ± {spec['tol']}"
    )


def test_closed_form_landweber_is_the_iteration():
    m = example.model()
    system = example.RealSystem(m)
    y = m.forward(example.potential(example.phantom(m, 0.004), m.n_medium, m.wavelength))
    closed = system.landweber(y, 50)
    loop = example.landweber_iterative(m, y, system.step, 50)
    assert np.linalg.norm(closed - loop) <= 1e-8 * np.linalg.norm(loop)


def test_coverage_reaches_the_combined_aperture(result):
    _check("kx_max", result["coverage"]["kx_max"], EXPECTED["truth"]["kx_max"])


def test_lateral_resolution_follows_the_coverage(result):
    _check("psf_fwhm_x", result["psf_fwhm_x"], EXPECTED["truth"]["psf_fwhm_x"])


def test_disc_keeps_its_lateral_size(result):
    _check("disc_width_x", result["disc_width_x"], EXPECTED["truth"]["disc_width_x"])


def test_missing_cone_elongates_the_psf(result):
    _check(
        "psf_aspect", result["psf_fwhm_z"] / result["psf_fwhm_x"], EXPECTED["truth"]["psf_aspect"]
    )


def test_born_model_error_is_next_order(result):
    gaps = result["model_error"]
    low, high = gaps["0.002"], gaps["0.004"]
    _check(
        "model_gap_scaling",
        high["relative_gap"] / low["relative_gap"],
        EXPECTED["truth"]["model_gap_scaling"],
    )
    for gap in (low, high):
        assert gap["relative_gap"] < gap["born_phase"]


@pytest.mark.xfail(strict=True, reason="QoraHolo#3: missing cone and early stopping, no prior")
def test_recovers_the_index_contrast(result):
    _check("delta_n", result["delta_n_landweber"], EXPECTED["truth"]["delta_n"])


def test_regression_snapshot(result):
    reg = EXPECTED["regression"]
    gaps = result["model_error"]
    got = {
        "kz_min": result["coverage"]["kz_min"],
        "kz_max": result["coverage"]["kz_max"],
        "psf_fwhm_x": result["psf_fwhm_x"],
        "psf_fwhm_z": result["psf_fwhm_z"],
        "delta_n_information_limit": result["delta_n_information_limit"],
        "delta_n_landweber": result["delta_n_landweber"],
        "delta_n_from_bpm_data": gaps["0.004"]["delta_n_from_bpm_data"],
        "data_residual": result["data_residual"],
        "disc_width_x": result["disc_width_x"],
        "disc_width_z": result["disc_width_z"],
        "delta_n_ring": result["delta_n_ring"],
        "noise_gain": result["noise_gain"],
        "model_gap_0.002": gaps["0.002"]["relative_gap"],
        "model_gap_0.004": gaps["0.004"]["relative_gap"],
    }
    for name, value in got.items():
        _check(name, value, reg[name])
