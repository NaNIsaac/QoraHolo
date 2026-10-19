"""Holotomography (optical diffraction tomography) reconstruction.

Arrays are numpy, centre-origin, square with even sides. Functions that work on images accept a
single 2D image or a 3D stack (count, n, n) and return the same rank. Lengths are micrometres,
spatial frequencies cycles per micrometre. ``backend`` is ``"cpu"`` or ``"cuda"``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Windows does not search PATH for an extension's DLL dependencies; a CUDA build needs cuFFT.
if sys.platform == "win32" and "CUDA_PATH" in os.environ:
    _cuda_bin = os.path.join(os.environ["CUDA_PATH"], "bin")
    if os.path.isdir(_cuda_bin):
        os.add_dll_directory(_cuda_bin)

from . import _core  # noqa: E402
from ._core import Geometry, Sideband, Sphere, cuda_available  # noqa: E402

__all__ = [
    "Geometry",
    "Pipeline",
    "Sideband",
    "Sphere",
    "circular_scan",
    "cuda_available",
    "extract_fields",
    "field_ratio",
    "find_sideband",
    "forward_fields",
    "reconstruct",
    "render_phantom",
    "rytov",
    "synthesize_holograms",
    "unwrap",
]


def _stack(a, dtype):
    a = np.ascontiguousarray(a, dtype=dtype)
    if a.ndim == 2:
        return a[None], True
    if a.ndim != 3:
        raise ValueError(f"expected a 2D image or a 3D stack, got {a.ndim} dimensions")
    return a, False


def _unstack(a, single):
    return a[0] if single else a


def _illum(illuminations):
    return np.ascontiguousarray(np.reshape(illuminations, (-1, 2)), dtype=np.float32)


def find_sideband(hologram, dc_exclusion: float = 0.0) -> Sideband:
    """Peak of the +1 sideband (x > 0 half plane) of the first hologram, in pixels from DC."""
    h, _ = _stack(hologram, np.float32)
    return _core.find_sideband(h, dc_exclusion)


def extract_fields(
    holograms,
    center: tuple[float, float] | None = None,
    radius: float = 0.0,
    dc_exclusion: float = 0.0,
    output_size: int = 0,
    backend: str = "cpu",
):
    """Off-axis hologram(s) to complex field(s).

    ``center`` is the sideband centre (y, x) in pixels from DC; detected when omitted. For
    multi-angle data pass the reference carrier so the illumination tilt stays in the field.
    ``radius`` defaults to |center| / 3, ``output_size`` to the hologram side.
    """
    h, single = _stack(holograms, np.float32)
    out = _core.extract_fields(h, center, radius, dc_exclusion, output_size, backend)
    return _unstack(out, single)


def field_ratio(sample, background):
    """sample / background; background is one field or one per sample field."""
    s, single = _stack(sample, np.complex64)
    b, _ = _stack(background, np.complex64)
    return _unstack(_core.field_ratio(s, b), single)


def unwrap(phase, backend: str = "cpu"):
    """Least-squares unwrapping (Ghiglia & Romero 1994), congruent to the input mod 2 pi."""
    p, single = _stack(phase, np.float32)
    return _unstack(_core.unwrap(p, backend), single)


def rytov(ratio, backend: str = "cpu"):
    """log|r| + i unwrap(arg r), with the mean phase on the image border in (-pi, pi]."""
    r, single = _stack(ratio, np.complex64)
    return _unstack(_core.rytov(r, backend), single)


def reconstruct(
    fields,
    illuminations,
    geometry: Geometry,
    nz: int = 0,
    iterations: int = 0,
    solver: str = "gp",
    gp_transform: str = "r2c",
    tv_weight: float = 1e-2,
    tv_inner: int = 10,
    mapping: str = "gather",
    backend: str = "cpu",
    return_filled: bool = False,
):
    """Rytov fields (count, n, n) to a refractive-index volume (nz, n, n).

    ``illuminations`` is (count, 2) lateral frequencies (fy, fx) of the incident waves.
    ``iterations`` runs a missing-cone solver: ``solver="gp"`` is Gerchberg-Papoulis with
    n >= n_medium, on half spectra (``gp_transform="r2c"``, real part only) or full complex
    spectra (``"c2c"``); ``solver="tv"`` is FISTA with a total-variation term of weight
    ``tv_weight`` on n^2 - n_medium^2 and ``tv_inner`` dual iterations per step. With
    ``return_filled`` the fraction of k-space voxels that received data is returned as well.
    """
    f, _ = _stack(fields, np.complex64)
    ri, filled = _core.reconstruct(
        f,
        _illum(illuminations),
        geometry,
        nz,
        iterations,
        solver,
        gp_transform,
        tv_weight,
        tv_inner,
        mapping,
        backend,
    )
    return (ri, filled) if return_filled else ri


class Pipeline:
    """Holograms to refractive index with plans and buffers allocated once.

    On CUDA the background fields stay on the device and each ``run`` does one upload of the
    sample holograms and one download of the volume. ``center`` is the reference carrier
    (y, x) in pixels from DC, as for ``extract_fields``; the other arguments are those of
    ``extract_fields`` and ``reconstruct``. ``upload_chunks`` splits the CUDA upload so field
    extraction overlaps it.
    """

    def __init__(
        self,
        count: int,
        hologram_size: int,
        center: tuple[float, float],
        illuminations,
        geometry: Geometry,
        radius: float = 0.0,
        output_size: int = 0,
        nz: int = 0,
        iterations: int = 0,
        solver: str = "gp",
        gp_transform: str = "r2c",
        tv_weight: float = 1e-2,
        tv_inner: int = 10,
        mapping: str = "gather",
        backend: str = "cpu",
        upload_chunks: int = 1,
    ):
        self._p = _core.Pipeline(
            count,
            hologram_size,
            tuple(center),
            radius,
            output_size,
            _illum(illuminations),
            geometry,
            nz,
            iterations,
            solver,
            gp_transform,
            tv_weight,
            tv_inner,
            mapping,
            backend,
            upload_chunks,
        )

    def set_background(self, backgrounds) -> None:
        """Background holograms (count, N, N); their fields are extracted once and kept."""
        self._p.set_background(np.ascontiguousarray(backgrounds, dtype=np.float32))

    def run(self, holograms):
        """Sample holograms (count, N, N) to the index volume (nz, n, n)."""
        return self._p.run(np.ascontiguousarray(holograms, dtype=np.float32))

    @property
    def filled(self) -> float:
        """Fraction of k-space voxels that received data in the last run."""
        return self._p.filled

    @property
    def timings(self) -> dict[str, float]:
        """Milliseconds per stage of the last run; on CUDA, device event intervals."""
        return self._p.timings

    @property
    def setup_ms(self) -> float:
        return self._p.setup_ms

    @property
    def device_bytes(self) -> int:
        return self._p.device_bytes


def circular_scan(count: int, na_illumination: float, geometry: Geometry):
    """Normal incidence, then count - 1 directions evenly spaced on a circle: (count, 2)."""
    return _core.circular_scan(count, na_illumination, geometry)[0]


def render_phantom(spheres, geometry: Geometry, nz: int, n: int):
    """Refractive index of the spheres at voxel centres, (nz, n, n)."""
    return _core.render_phantom(list(spheres), geometry, nz, n)


def forward_fields(spheres, illuminations, geometry: Geometry, n: int, model: str = "rytov"):
    """Field ratios u / u_in, (count, n, n), from the analytic spectrum on each Ewald cap."""
    return _core.forward_fields(list(spheres), _illum(illuminations), geometry, n, model)


def synthesize_holograms(
    ratio,
    illuminations,
    geometry: Geometry,
    carrier: tuple[float, float] = (5.0, 5.0),
    reference_amplitude: float = 1.0,
    noise: float = 0.0,
    seed: int = 1,
):
    """|R + u_in * ratio|^2 + Gaussian noise, with R = a exp(-2 pi i carrier . r)."""
    r, single = _stack(ratio, np.complex64)
    out = _core.synthesize_holograms(
        r, _illum(illuminations), geometry, tuple(carrier), reference_amplitude, noise, seed
    )
    return _unstack(out, single)
