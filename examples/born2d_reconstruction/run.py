"""Reconstruct a known 2D object with the first Born operator and measure what limits it.

Two discs of raised index in water are imaged with 31 plane waves over an illumination NA of
0.9. The data are synthesised by the same Born operator (to isolate the reconstruction) and,
separately, by the beam propagation model (to measure the Born model error). The object is
recovered with Landweber iterations constrained to a real potential, which is a linear map of
the data, so its resolution and noise gain are properties of the method alone. For a real
object the constrained iteration is plain Landweber on the stacked real system
S = [Re A; Im A], so k iterations from zero equal the spectral filter

    x_k = Sᵀ U diag((1 − (1 − w / w_max)^k) / w) Uᵀ y,      S Sᵀ = U diag(w) Uᵀ

which is how it is computed here (``landweber_iterative`` is the loop, for the test).

Reported, in µm, cycles/µm and refractive-index units:

- k-space coverage of the Ewald arcs and the point-spread function (resolution);
- the index recovered in the larger disc, split into what the data cannot carry (projection of
  the true object onto the row space of the operator: missing cone and band limit), what 300
  iterations leave out (early stopping), and the Born model error (BPM data);
- the noise gain of the reconstruction.

Run ``python examples/born2d_reconstruction/run.py`` to print the numbers that
``expected.json`` and ``python/tests/test_example_born2d_reconstruction.py`` check.
"""

import json

import numpy as np

import qoraholo as qh
from qoraholo.born2d import Born2D, born_phase, potential

WAVELENGTH = 0.532  # µm, vacuum
N_MEDIUM = 1.337  # water
NA = 1.2  # detection
NA_ILLUMINATION = 0.9
PIXEL = 0.1  # µm
N = 64  # grid side; the object is (N, N) in (z, x)
COUNT = 31  # illuminations
ITERATIONS = 300
DELTA_N = 0.004  # index contrast of the larger disc; the smaller one has half
ETA = 0.05  # relative noise for the noise gain of the nonlinear (positivity) reconstruction

K0 = 2 * np.pi / WAVELENGTH


def model() -> Born2D:
    fx = np.linspace(-NA_ILLUMINATION, NA_ILLUMINATION, COUNT) / WAVELENGTH
    return Born2D(WAVELENGTH, N_MEDIUM, NA, PIXEL, N, N, tuple(fx))


def grid(m: Born2D):
    return np.meshgrid(m.z, m.x, indexing="ij")


def phantom(m: Born2D, delta_n: float) -> np.ndarray:
    """Index map: a 2 µm disc of +delta_n and a 1 µm disc of +delta_n / 2."""
    zz, xx = grid(m)
    ri = np.full((N, N), N_MEDIUM)
    ri[(xx + 0.8) ** 2 + zz**2 < 1.0**2] += delta_n
    ri[(xx - 1.6) ** 2 + (zz - 0.5) ** 2 < 0.5**2] += delta_n / 2
    return ri


def core(m: Born2D) -> np.ndarray:
    """Inner 0.7 µm of the larger disc, away from its edge."""
    zz, xx = grid(m)
    return (xx + 0.8) ** 2 + zz**2 < 0.7**2


def to_delta_n(v: np.ndarray) -> np.ndarray:
    """Index contrast from a real scattering potential k0² (n² − n_m²)."""
    return np.sqrt(N_MEDIUM**2 + np.real(v) / K0**2) - N_MEDIUM


def coverage(m: Born2D) -> dict:
    """Extent of the measured spatial frequencies (Ewald arcs inside the detection pupil)."""
    nu_m = N_MEDIUM / WAVELENGTH
    kx, kz = [], []
    for fx in m.illuminations:
        mu = m.mu + fx
        inside = (np.abs(mu) <= NA / WAVELENGTH) & (np.abs(mu) < nu_m)
        kx.append(m.mu[inside])
        kz.append(np.sqrt(nu_m**2 - mu[inside] ** 2) - np.sqrt(nu_m**2 - fx**2))
    kx, kz = np.concatenate(kx), np.concatenate(kz)
    return {"kx_max": float(np.abs(kx).max()), "kz_min": float(kz.min()), "kz_max": float(kz.max())}


CUTOFF = 1e-10  # eigenvalues of S Sᵀ below CUTOFF · w_max (singular values below 1e-5) are dropped


class RealSystem:
    """S = [Re A; Im A] with the eigendecomposition of S Sᵀ, for real objects."""

    def __init__(self, m: Born2D):
        matrix = m.matrix()
        self.s = np.vstack([matrix.real, matrix.imag])
        w, self.u = np.linalg.eigh(self.s @ self.s.T)
        self.keep = w > CUTOFF * w.max()
        self.w = w[self.keep]
        self.u = self.u[:, self.keep]
        self.step = 1.0 / w.max()

    def _filter(self, y_stacked: np.ndarray, gain: np.ndarray) -> np.ndarray:
        """Sᵀ U diag(gain) Uᵀ y on the stacked real data."""
        return (self.s.T @ (self.u @ (gain * (self.u.T @ y_stacked)))).reshape(N, N)

    def landweber(self, y: np.ndarray, iterations: int = ITERATIONS) -> np.ndarray:
        """Real-constrained Landweber from zero on complex data (count, n), in closed form."""
        filtered = -np.expm1(iterations * np.log1p(-self.step * self.w))
        stacked = np.concatenate([y.real.ravel(), y.imag.ravel()])
        return self._filter(stacked, filtered / self.w)

    def projected(
        self, y: np.ndarray, iterations: int = ITERATIONS, tv: float = 0.0, inner: int = 50
    ) -> np.ndarray:
        """Landweber with the prior V ≥ 0 (n ≥ n_m) and an optional total variation term.

        Proximal gradient on ½‖S x − y‖² + λ TV(x) over x ≥ 0 from zero, step 1/w_max:
        x ← prox(x + Sᵀ(y − S x) / w_max), the prox computed by ``tv_prox`` with ``inner``
        iterations. ``tv`` is λ_rel in λ = λ_rel · w_max · s, s the maximum of the plain
        Landweber result on the same data, so it is dimensionless and needs no truth. ``tv=0``
        is projected Landweber.
        """
        stacked = np.concatenate([y.real.ravel(), y.imag.ravel()])
        mu = tv * self.landweber(y).max()  # step · λ
        x = np.zeros(N * N)
        for _ in range(iterations):
            b = x + self.step * (self.s.T @ (stacked - self.s @ x))
            if mu:
                x = tv_prox(b.reshape(N, N), mu, inner).ravel()
            else:
                x = np.maximum(b, 0)
        return x.reshape(N, N)

    def information_limit(self, v: np.ndarray) -> np.ndarray:
        """Projection of a real object onto the row space of S.

        The minimum-norm real object with the same data, i.e. the limit of the iteration; what
        it misses the data do not carry (up to the CUTOFF on small singular values).
        """
        return self._filter(self.s @ v.ravel(), 1.0 / self.w)


def gradient(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward differences along z and x, zero across the last row and column (Neumann)."""
    gz, gx = np.zeros_like(v), np.zeros_like(v)
    gz[:-1] = v[1:] - v[:-1]
    gx[:, :-1] = v[:, 1:] - v[:, :-1]
    return gz, gx


def divergence(pz: np.ndarray, px: np.ndarray) -> np.ndarray:
    """−gradientᵀ."""
    d = np.zeros_like(pz)
    d[:-1] += pz[:-1]
    d[1:] -= pz[:-1]
    d[:, :-1] += px[:, :-1]
    d[:, 1:] -= px[:, :-1]
    return d


def tv_prox(b: np.ndarray, mu: float, iterations: int = 50) -> np.ndarray:
    """argmin over x ≥ 0 of ½‖x − b‖² + mu · TV(x), isotropic TV.

    Fast gradient projection on the dual (Beck and Teboulle, IEEE TIP 18, 2419 (2009)), which
    handles the constraint exactly: x = P(b + mu · div p) for a dual field |p| ≤ 1.
    """
    pz = px = rz = rx = np.zeros_like(b)
    t = 1.0
    for _ in range(iterations):
        gz, gx = gradient(np.maximum(b + mu * divergence(rz, rx), 0))
        qz, qx = rz + gz / (8 * mu), rx + gx / (8 * mu)
        norm = np.maximum(1.0, np.hypot(qz, qx))
        qz, qx = qz / norm, qx / norm
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        rz = qz + (t - 1) / t_next * (qz - pz)
        rx = qx + (t - 1) / t_next * (qx - px)
        pz, px, t = qz, qx, t_next
    return np.maximum(b + mu * divergence(pz, px), 0)


def landweber_iterative(m: Born2D, y: np.ndarray, step: float, iterations: int) -> np.ndarray:
    """x <- Re(x + step · Aᴴ(y − A x)) from zero, as a loop."""
    x = np.zeros((N, N))
    for _ in range(iterations):
        x = x + step * np.real(m.adjoint(y - m.forward(x)))
    return x


def fwhm(profile: np.ndarray, pixel: float, upsample: int = 32) -> float:
    """Full width at half maximum of a centred 1D profile, Fourier-interpolated."""
    n = len(profile)
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(profile)))
    padded = np.zeros(n * upsample, complex)
    padded[(n * upsample - n) // 2 : (n * upsample + n) // 2] = spectrum
    f = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(padded))).real
    f /= f.max()
    left = right = int(np.argmax(f))
    while f[left] >= 0.5:
        left -= 1
    while f[right] >= 0.5:
        right += 1
    left_cross = left + (0.5 - f[left]) / (f[left + 1] - f[left])
    right_cross = right - 1 + (f[right - 1] - 0.5) / (f[right - 1] - f[right])
    return float((right_cross - left_cross) * pixel / upsample)


def half_max_width(profile: np.ndarray, pixel: float) -> float:
    """Width at half maximum around the peak, by linear interpolation between samples."""
    f = profile / profile.max()
    peak = int(np.argmax(f))
    left = peak
    while f[left - 1] >= 0.5:
        left -= 1
    right = peak
    while f[right + 1] >= 0.5:
        right += 1
    left_cross = left - (f[left] - 0.5) / (f[left] - f[left - 1])
    right_cross = right + (f[right] - 0.5) / (f[right] - f[right + 1])
    return float((right_cross - left_cross) * pixel)


def bpm_fields(ri: np.ndarray, m: Born2D) -> np.ndarray:
    """Scattered field ratios from bpm_forward on the object extruded along y."""
    geometry = qh.Geometry()
    geometry.wavelength, geometry.n_medium, geometry.na, geometry.pixel = (
        WAVELENGTH,
        N_MEDIUM,
        NA,
        PIXEL,
    )
    volume = np.repeat(ri[:, None, :], N, axis=1).astype(np.float32)
    illum = [(0.0, fx) for fx in m.illuminations]
    return qh.bpm_forward(volume, illum, geometry, obliquity=True)[:, N // 2, :] - 1


def run() -> dict:
    m = model()
    system = RealSystem(m)
    inside = core(m)
    out: dict[str, object] = {"coverage": coverage(m)}

    delta = np.zeros((N, N))
    delta[N // 2, N // 2] = 1.0
    psf = system.landweber(m.forward(delta))
    out["psf_fwhm_x"] = fwhm(psf[N // 2, :], PIXEL)
    out["psf_fwhm_z"] = fwhm(psf[:, N // 2], PIXEL)

    v_true = potential(phantom(m, DELTA_N), N_MEDIUM, WAVELENGTH)
    y = m.forward(v_true)
    rec = system.landweber(y)
    out["born_phase"] = born_phase(DELTA_N, 2.0, WAVELENGTH)
    out["delta_n_truth"] = DELTA_N
    limit = system.information_limit(v_true)
    out["delta_n_information_limit"] = float(to_delta_n(limit)[inside].mean())
    out["delta_n_landweber"] = float(to_delta_n(rec)[inside].mean())
    out["data_residual"] = float(np.linalg.norm(m.forward(rec) - y) / np.linalg.norm(y))
    dn = to_delta_n(rec)
    zz, xx = grid(m)
    left_half = np.where(xx[N // 2] < 0.4, dn[N // 2], 0)  # the z = 0 row, without the small disc
    column = int(np.argmin(np.abs(m.x + 0.8)))  # x = -0.8, through the larger disc
    out["disc_width_x"] = half_max_width(left_half, PIXEL)
    out["disc_width_z"] = half_max_width(dn[:, column], PIXEL)
    ring = (np.hypot(xx + 0.8, zz) > 1.2) & (np.hypot(xx + 0.8, zz) < 1.6) & (xx < 0.4)
    out["delta_n_ring"] = float(dn[ring].mean())

    # same noise realisation, rms equal to the data's; the reconstruction is linear in it
    rng = np.random.default_rng(1)
    noise = (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)) / np.sqrt(2)
    noise *= np.sqrt(np.mean(np.abs(y) ** 2))
    out["noise_gain"] = float(np.linalg.norm(system.landweber(noise)) / np.linalg.norm(rec))

    # the prior n ≥ n_m (QoraHolo#3); nonlinear, so its PSF is the response to a positive point
    # and its noise gain is measured at a small noise level ETA
    pos = system.projected(y)
    dn_pos = to_delta_n(pos)
    psf_pos = system.projected(m.forward(delta))
    out["positivity"] = {
        "delta_n": float(dn_pos[inside].mean()),
        "psf_fwhm_x": fwhm(psf_pos[N // 2, :], PIXEL),
        "psf_fwhm_z": fwhm(psf_pos[:, N // 2], PIXEL),
        "disc_width_x": half_max_width(np.where(xx[N // 2] < 0.4, dn_pos[N // 2], 0), PIXEL),
        "disc_width_z": half_max_width(dn_pos[:, column], PIXEL),
        "delta_n_ring": float(dn_pos[ring].mean()),
        "data_residual": float(np.linalg.norm(m.forward(pos) - y) / np.linalg.norm(y)),
        "noise_gain": float(
            np.linalg.norm(system.projected(y + ETA * noise) - pos) / (ETA * np.linalg.norm(pos))
        ),
    }

    model_error = {}
    for delta_n in (DELTA_N / 2, DELTA_N):
        ri = phantom(m, delta_n)
        born = system.landweber(m.forward(potential(ri, N_MEDIUM, WAVELENGTH)))
        bpm = system.landweber(bpm_fields(ri, m))
        model_error[str(delta_n)] = {
            "born_phase": born_phase(delta_n, 2.0, WAVELENGTH),
            "relative_gap": float(np.linalg.norm(bpm - born) / np.linalg.norm(born)),
            "delta_n_from_bpm_data": float(to_delta_n(bpm)[inside].mean()),
            "delta_n_from_born_data": float(to_delta_n(born)[inside].mean()),
            "delta_n_positivity_from_bpm_data": float(
                to_delta_n(system.projected(bpm_fields(ri, m)))[inside].mean()
            ),
        }
    out["model_error"] = model_error
    return out


def prior_table(tvs=(0.0, 1e-4, 1e-3, 1e-2)) -> dict:
    """Landweber against the positivity prior with and without TV (README table, about 5 min).

    Born data, noiseless. For each method: δn in the larger disc's core after 300, 1500 and 3000
    iterations (the last two are the convergence check), the smaller disc's core (truth
    DELTA_N / 2), the PSF and the noise gain at 300 iterations.
    """
    m = model()
    system = RealSystem(m)
    zz, xx = grid(m)
    inside, small = core(m), (xx - 1.6) ** 2 + (zz - 0.5) ** 2 < 0.3**2
    y = m.forward(potential(phantom(m, DELTA_N), N_MEDIUM, WAVELENGTH))
    delta = np.zeros((N, N))
    delta[N // 2, N // 2] = 1.0
    rng = np.random.default_rng(1)
    noise = (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)) / np.sqrt(2)
    noise *= np.sqrt(np.mean(np.abs(y) ** 2))
    methods = {"landweber": system.landweber}
    methods |= {
        f"positivity, tv {tv:g}": (lambda d, k=ITERATIONS, tv=tv: system.projected(d, k, tv))
        for tv in tvs
    }
    table = {}
    for name, f in methods.items():
        rec = f(y)
        psf = f(m.forward(delta))
        table[name] = {
            "delta_n": [
                float(to_delta_n(f(y, k) if k != ITERATIONS else rec)[inside].mean())
                for k in (ITERATIONS, 1500, 3000)
            ],
            "delta_n_small": float(to_delta_n(rec)[small].mean()),
            "psf_fwhm_x": fwhm(psf[N // 2, :], PIXEL),
            "psf_fwhm_z": fwhm(psf[:, N // 2], PIXEL),
            "noise_gain": float(
                np.linalg.norm(f(y + ETA * noise) - rec) / (ETA * np.linalg.norm(rec))
            ),
        }
    return table


if __name__ == "__main__":
    import sys

    print(json.dumps(prior_table() if "--priors" in sys.argv else run(), indent=2))
