"""First Born scattering in two dimensions (x, z) as a linear operator with its exact adjoint.

A plane wave travelling along +z with lateral frequency ``fx`` (cycles/µm) illuminates a
scattering potential V(z, x) = k0² (n(z, x)² − n_m²), k0 = 2π/λ, sampled at pixel centres of a
centre-origin (nz, n) grid. Under the first Born approximation the scattered field on the line
z = 0 through the grid centre, divided by the incident field, has the spectrum

    R(μ') = i / (4π ν_z) · V̂(μ', ν_z − ν_in,z),    μ = μ' + fx,   ν_z = sqrt((n_m/λ)² − μ²)

where V̂(K) = ∫ V(r) exp(−2πi K·r) d²r. This is the Fourier diffraction theorem with the 2D
Green's function (i/4) H0⁽¹⁾(k_m r), written for field ratios u_s / u_in refocused to z = 0
and band-limited to the detection NA, the convention of ``bpm_forward`` and
``forward_fields`` (μ' is the frequency of the ratio, μ that of the scattered field).

The operator is discretised exactly as written: V̂ by the midpoint rule over the object
pixels at the off-grid frequencies K (no interpolation onto a grid), the ratio by the
inverse transform sampled at the detector pixels. Both are explicit sums, so ``adjoint`` is
the conjugate transpose of ``forward`` to rounding error.

Validity: the first Born approximation needs the phase the object adds to be small,
|δn| · k0 · L ≪ 1 for an object of size L along the beam (``born_phase``). It leaves out
multiple scattering; ``bpm_forward`` keeps forward multiple scattering.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def potential(ri, n_medium: float, wavelength: float) -> np.ndarray:
    """Scattering potential k0² (n² − n_m²) in µm⁻² from a refractive index map."""
    k0 = 2 * np.pi / wavelength
    return k0**2 * (np.asarray(ri, dtype=float) ** 2 - n_medium**2)


def born_phase(delta_n: float, size: float, wavelength: float) -> float:
    """|δn| k0 L: the phase an object of index contrast δn and thickness L adds (radians)."""
    return abs(delta_n) * 2 * np.pi / wavelength * size


@dataclass(frozen=True)
class Born2D:
    """First Born forward model for a centre-origin (nz, n) potential and ``count`` plane waves.

    wavelength: vacuum wavelength, µm. pixel: object pixel and detector sample spacing, µm.
    illuminations: lateral frequency fx of each plane wave, cycles/µm, |fx| < n_medium/λ.
    """

    wavelength: float
    n_medium: float
    na: float
    pixel: float
    n: int
    nz: int
    illuminations: tuple[float, ...]

    def __post_init__(self):
        limit = self.n_medium / self.wavelength
        if any(abs(fx) >= limit for fx in self.illuminations):
            raise ValueError(f"every |fx| must be below n_medium / wavelength = {limit:.4f}")

    @property
    def x(self) -> np.ndarray:
        """Lateral sample positions, µm, x = 0 at index n // 2."""
        return (np.arange(self.n) - self.n // 2) * self.pixel

    @property
    def z(self) -> np.ndarray:
        """Axial sample positions, µm, z = 0 at index nz // 2."""
        return (np.arange(self.nz) - self.nz // 2) * self.pixel

    @property
    def mu(self) -> np.ndarray:
        """Frequencies of the field ratio, cycles/µm, centre-origin."""
        return (np.arange(self.n) - self.n // 2) / (self.n * self.pixel)

    def _rows(self, fx: float):
        """K points, transfer factor and detector transform for one illumination."""
        nu_m = self.n_medium / self.wavelength
        mu = self.mu + fx  # scattered field frequency
        inside = (np.abs(mu) <= self.na / self.wavelength) & (np.abs(mu) < nu_m)
        nu_z = np.sqrt(np.where(inside, nu_m**2 - mu**2, 1.0))
        kx = self.mu
        kz = nu_z - np.sqrt(nu_m**2 - fx**2)
        transfer = np.where(inside, 1j / (4 * np.pi * nu_z), 0.0)
        # V̂ at (kx, kz) by the midpoint rule: (n, nz, n) phases split into z and x factors
        ez = np.exp(-2j * np.pi * np.outer(kz, self.z))  # (n_mu, nz)
        ex = np.exp(-2j * np.pi * np.outer(kx, self.x))  # (n_mu, n)
        inverse = np.exp(2j * np.pi * np.outer(self.x, self.mu)) / (self.n * self.pixel)
        return ez, ex, transfer, inverse

    @cached_property
    def _tables(self) -> list:
        """``_rows`` for every illumination, computed once (the model is immutable)."""
        return [self._rows(fx) for fx in self.illuminations]

    def forward(self, v) -> np.ndarray:
        """Potential (nz, n), µm⁻² -> scattered field ratios u_s / u_in at z = 0, (count, n)."""
        v = np.asarray(v)
        if v.shape != (self.nz, self.n):
            raise ValueError(f"potential has shape {v.shape}, expected {(self.nz, self.n)}")
        out = np.empty((len(self.illuminations), self.n), complex)
        for a, (ez, ex, transfer, inverse) in enumerate(self._tables):
            spectrum = self.pixel**2 * np.sum((ez @ v) * ex, axis=1)
            out[a] = inverse @ (transfer * spectrum)
        return out

    def adjoint(self, fields) -> np.ndarray:
        """Conjugate transpose of ``forward``: (count, n) -> (nz, n)."""
        fields = np.asarray(fields)
        if fields.shape != (len(self.illuminations), self.n):
            expected = (len(self.illuminations), self.n)
            raise ValueError(f"fields have shape {fields.shape}, expected {expected}")
        out = np.zeros((self.nz, self.n), complex)
        for a, (ez, ex, transfer, inverse) in enumerate(self._tables):
            g = np.conj(transfer) * (inverse.conj().T @ fields[a])
            out += self.pixel**2 * ((ez.conj().T * g) @ ex.conj())
        return out
