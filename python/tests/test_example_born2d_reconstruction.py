"""examples/born2d_reconstruction end to end: known 2D object -> Born or BPM data -> Landweber.

Truth checks are physical expectations (coverage, resolution, missing-cone elongation, the order
of the Born model error). Without a prior the recovered index contrast comes out low
(QoraHolo#3: missing cone and early stopping); with the prior n >= n_m it is checked against the
truth.
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


def test_without_a_prior_the_index_stays_below_the_information_limit(result):
    # QoraHolo#3: the missing cone is information the data do not carry
    assert (
        result["delta_n_landweber"]
        < result["delta_n_information_limit"]
        < EXPECTED["truth"]["delta_n"]["value"]
    )


def test_positivity_recovers_the_index_contrast(result):
    _check("delta_n", result["positivity"]["delta_n"], EXPECTED["truth"]["delta_n"])


def test_positivity_recovers_the_index_contrast_from_bpm_data(result):
    # a different forward model than the one inverted, so not an inverse crime (still noiseless)
    got = result["model_error"]["0.004"]["delta_n_positivity_from_bpm_data"]
    _check("delta_n from BPM data", got, EXPECTED["truth"]["delta_n"])


def test_positivity_shortens_the_psf(result):
    pos = result["positivity"]
    assert pos["psf_fwhm_x"] <= result["psf_fwhm_x"]
    assert pos["psf_fwhm_z"] < result["psf_fwhm_z"]


def test_tv_prox_is_the_constrained_minimiser():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(16, 16)) + np.where(np.arange(16) < 8, 1.0, 0.0)
    mu = 0.3

    def objective(x):
        gz, gx = example.gradient(x)
        return 0.5 * np.sum((x - b) ** 2) + mu * np.sum(np.hypot(gz, gx))

    x = example.tv_prox(b, mu, iterations=2000)
    assert x.min() >= 0
    best = objective(x)
    for _ in range(50):
        trial = np.maximum(x + 1e-3 * rng.normal(size=x.shape), 0)
        assert objective(trial) >= best - 1e-9


def test_divergence_is_minus_the_gradient_adjoint():
    rng = np.random.default_rng(1)
    v, pz, px = rng.normal(size=(3, 9, 7))
    gz, gx = example.gradient(v)
    assert np.sum(gz * pz + gx * px) == pytest.approx(-np.sum(v * example.divergence(pz, px)))


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
        "positivity_delta_n": result["positivity"]["delta_n"],
        "positivity_delta_n_from_bpm_data": gaps["0.004"]["delta_n_positivity_from_bpm_data"],
        "positivity_psf_fwhm_x": result["positivity"]["psf_fwhm_x"],
        "positivity_psf_fwhm_z": result["positivity"]["psf_fwhm_z"],
        "positivity_disc_width_z": result["positivity"]["disc_width_z"],
        "positivity_noise_gain": result["positivity"]["noise_gain"],
    }
    for name, value in got.items():
        _check(name, value, reg[name])
