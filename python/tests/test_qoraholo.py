import numpy as np
import pytest

import qoraholo as qh

GEOM = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.2)
SPHERE = [qh.Sphere(z=0.0, y=0.0, x=0.0, radius=3.0, n=1.36)]


def test_unwrap_recovers_smooth_phase():
    y, x = np.mgrid[:64, :96].astype(np.float32)
    truth = 25.0 * np.exp(-((x - 40) ** 2 + (y - 30) ** 2) / 400.0) + 0.05 * x
    wrapped = np.angle(np.exp(1j * truth)).astype(np.float32)
    got = qh.unwrap(wrapped)
    assert got.shape == truth.shape
    offset = got - truth
    assert np.ptp(offset) < 2e-3


def test_hologram_to_field_round_trip():
    holo_geom = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.055)
    illum = np.zeros((1, 2), np.float32)
    ratio = qh.forward_fields(SPHERE, illum, holo_geom, 128, model="born")
    carrier = 35 / (128 * 0.055)
    holo = qh.synthesize_holograms(ratio, illum, holo_geom, carrier=(carrier, carrier))
    bg = qh.synthesize_holograms(np.ones_like(ratio), illum, holo_geom, carrier=(carrier, carrier))
    side = qh.find_sideband(bg[0])
    assert (side.iy, side.ix) == (35, 35)
    field = qh.field_ratio(
        qh.extract_fields(holo[0], center=(35, 35), radius=17),
        qh.extract_fields(bg[0], center=(35, 35), radius=17),
    )
    assert field.ndim == 2
    assert np.max(np.abs(field - ratio[0])) < 1e-4


def test_reconstruct_sphere():
    illum = qh.circular_scan(49, 0.9, GEOM)
    assert illum.shape == (49, 2)
    fields = qh.rytov(qh.forward_fields(SPHERE, illum, GEOM, 64))
    ri, filled = qh.reconstruct(fields, illum, GEOM, gp_iterations=10, return_filled=True)
    assert ri.shape == (64, 64, 64)
    assert 0 < filled < 1
    z, y, x = np.indices(ri.shape) - 32
    inner = (z**2 + y**2 + x**2) * GEOM.pixel**2 < (0.6 * 3.0) ** 2
    assert abs(ri[inner].mean() - 1.36) < 4e-3


def test_rejects_odd_sizes():
    with pytest.raises(ValueError):
        qh.unwrap(np.zeros((5, 6), np.float32))


@pytest.mark.skipif(not qh.cuda_available(), reason="no CUDA device")
def test_cuda_matches_cpu():
    illum = qh.circular_scan(9, 0.9, GEOM)
    fields = qh.rytov(qh.forward_fields(SPHERE, illum, GEOM, 64))
    a = qh.reconstruct(fields, illum, GEOM, gp_iterations=3) - GEOM.n_medium
    b = qh.reconstruct(fields, illum, GEOM, gp_iterations=3, backend="cuda") - GEOM.n_medium
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-4


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_pipeline_matches_functions(backend):
    if backend == "cuda" and not qh.cuda_available():
        pytest.skip("no CUDA device")
    holo_geom = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.1)
    illum = qh.circular_scan(9, 0.9, GEOM)
    ratio = qh.forward_fields(SPHERE, illum, holo_geom, 128)
    carrier = 36 / (128 * 0.1)
    holo = qh.synthesize_holograms(ratio, illum, holo_geom, carrier=(carrier, carrier))
    bg = qh.synthesize_holograms(np.ones_like(ratio), illum, holo_geom, carrier=(carrier, carrier))
    crop = dict(center=(36, 36), radius=30, output_size=64)
    fields = qh.field_ratio(qh.extract_fields(holo, **crop), qh.extract_fields(bg, **crop))
    ref = qh.reconstruct(qh.rytov(fields), illum, GEOM, gp_iterations=3)

    p = qh.Pipeline(
        9, 128, illuminations=illum, geometry=GEOM, gp_iterations=3, backend=backend, **crop
    )
    p.set_background(bg)
    ri = p.run(holo)
    assert ri.shape == (64, 64, 64)
    a, b = ri - GEOM.n_medium, ref - GEOM.n_medium
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-4
    assert p.timings["total"] > 0
