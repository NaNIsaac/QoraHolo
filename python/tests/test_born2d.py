"""The 2D first Born operator: adjoint, an exact analytic case, and the BPM weak limit.

Born validity of each test object, |δn| k0 L (born_phase), at λ = 0.532 µm in water:

- Gaussian, peak δn = 0.002, 1/e² width 2 · 2σ = 1.6 µm: 0.038 rad
- cylinder, δn = 0.002, diameter 2 µm: 0.047 rad; δn = 0.004: 0.094 rad
"""

import numpy as np
import pytest

import qoraholo as qh
from qoraholo.born2d import Born2D, born_phase, potential

WAVELENGTH, N_MEDIUM, NA, PIXEL = 0.532, 1.337, 1.2, 0.1
N = NZ = 64
FX = tuple(np.linspace(-0.6, 0.6, 7) * N_MEDIUM / WAVELENGTH)  # illumination NA up to 0.8


@pytest.fixture(scope="module")
def model():
    return Born2D(WAVELENGTH, N_MEDIUM, NA, PIXEL, N, NZ, FX)


def test_adjoint_is_the_conjugate_transpose(model):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(NZ, N)) + 1j * rng.normal(size=(NZ, N))
    u = rng.normal(size=(len(FX), N)) + 1j * rng.normal(size=(len(FX), N))
    lhs = np.vdot(model.forward(v), u)
    rhs = np.vdot(v, model.adjoint(u))
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def test_matrix_is_forward(model):
    v = np.random.default_rng(1).normal(size=(NZ, N))
    np.testing.assert_allclose(
        model.matrix() @ v.ravel(), model.forward(v).ravel(), rtol=1e-12, atol=1e-15
    )


def test_gaussian_matches_its_closed_form_spectrum(model):
    """V = V0 exp(−r²/2σ²) has V̂(K) = 2πσ² V0 exp(−2π²σ²|K|²); smooth, so pixels add no error."""
    sigma, delta_n = 0.4, 0.002
    assert born_phase(delta_n, 4 * sigma, WAVELENGTH) == pytest.approx(0.0378, abs=1e-4)
    zz, xx = np.meshgrid(model.z, model.x, indexing="ij")
    v0 = float(potential(N_MEDIUM + delta_n, N_MEDIUM, WAVELENGTH))
    v = v0 * np.exp(-(xx**2 + zz**2) / (2 * sigma**2))

    nu_m = N_MEDIUM / WAVELENGTH
    for a, fx in enumerate(FX):
        mu = model.mu + fx
        inside = (np.abs(mu) <= NA / WAVELENGTH) & (np.abs(mu) < nu_m)
        nu_z = np.sqrt(np.where(inside, nu_m**2 - mu**2, 1.0))
        k2 = model.mu**2 + (nu_z - np.sqrt(nu_m**2 - fx**2)) ** 2
        spectrum = np.where(inside, 1j / (4 * np.pi * nu_z) * 2 * np.pi * sigma**2 * v0, 0)
        spectrum = spectrum * np.exp(-2 * np.pi**2 * sigma**2 * k2)
        expected = np.exp(2j * np.pi * np.outer(model.x, model.mu)) @ spectrum / (N * PIXEL)
        got = model.forward(v)[a]
        assert np.linalg.norm(got - expected) <= 1e-9 * np.linalg.norm(expected)


def cylinder_ri(delta_n, radius=1.0, supersample=8):
    """Index map (nz, n) of a disc in x-z, edge pixels weighted by their covered area."""
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    grid = (np.arange(N) - N // 2) * PIXEL
    fill = np.zeros((NZ, N))
    for dz in offsets:
        for dx in offsets:
            z = grid[:, None] + dz * PIXEL
            x = grid[None, :] + dx * PIXEL
            fill += (x**2 + z**2) < radius**2
    return N_MEDIUM + delta_n * fill / supersample**2


def bpm_scattered(ri_2d):
    """bpm_forward on the cylinder extruded along y; the field ratio minus one on the y = 0 row."""
    volume = np.repeat(ri_2d[:, None, :], N, axis=1).astype(np.float32)  # (nz, y, x)
    illum = [(0.0, fx) for fx in FX]
    geometry = qh.Geometry()
    geometry.wavelength, geometry.n_medium, geometry.na, geometry.pixel = (
        WAVELENGTH,
        N_MEDIUM,
        NA,
        PIXEL,
    )
    ratio = qh.bpm_forward(volume, illum, geometry, obliquity=True)
    return ratio[:, N // 2, :] - 1


def test_weak_cylinder_matches_bpm_to_next_order(model):
    """With obliquity, BPM's weak limit is first Born; what remains is the next order.

    The relative gap is about half the Born phase, |exp(iφ) − 1 − iφ| / |iφ| ≈ φ/2, and grows
    in proportion to δn. A convention mismatch (sign, scale, refocus plane) would instead leave
    a gap of order one that does not shrink with δn.
    """
    gaps = {}
    for delta_n in (0.002, 0.004):
        ri = cylinder_ri(delta_n)
        born = model.forward(potential(ri, N_MEDIUM, WAVELENGTH))
        bpm = bpm_scattered(ri)
        gaps[delta_n] = np.linalg.norm(bpm - born) / np.linalg.norm(born)
        phase = born_phase(delta_n, 2.0, WAVELENGTH)
        gap = gaps[delta_n]
        print(f"delta_n {delta_n}: born phase {phase:.3f} rad, relative gap to BPM {gap:.4f}")
        assert gaps[delta_n] < phase
    assert gaps[0.004] / gaps[0.002] == pytest.approx(2.0, rel=0.1)
