"""Compare QoraHolo's sphere fields with qpsphere's Born and Rytov and with vector Mie.

Every field is the ratio u / u_in in the plane through the sphere centre for an on-axis plane
wave. QoraHolo has first Born, Rytov and a scalar partial-wave solution (``exact``); qpsphere
(RI-imaging) has Born and Rytov from the same Fourier diffraction theorem. The vector reference is
the co-polarised far-field amplitude of Mie theory from miepython, refocused to the centre plane
by the relation QoraHolo uses; that relation is re-implemented here and checked against QoraHolo's
Born field first. qpsphere's own Mie model (BHFIELD) is not used: in double precision it returns
NaN coefficients once the grid reaches beyond about 4 µm from a 3 µm sphere, and the
arbitrary-precision build takes about 0.1 s per pixel. Conventions aligned here, details in
docs/crosscheck.md:

- centre: qpsphere puts the sphere centre between pixels, at (n - 1) / 2; QoraHolo's origin is
  pixel n / 2, so the QoraHolo sphere is moved by half a pixel in y and x.
- band limit: qpsphere has none; QoraHolo cuts at the detection NA, so NA = n_m gives the full
  band, and both fields are low-passed for the NA study.
- period: QoraHolo's field is periodic in its n x n window; qpsphere zero-pads by ``zeropad``
  before its inverse FFT. The two agree when the periods match.
- sampling: qpsphere.rytov() resamples with scipy's interp2d, removed in SciPy 1.14, so its field
  function ``sphere_prop_fslice_bessel`` is called on the output grid directly.

Usage (uv run --group check python checks/sphere_crosscheck.py MODE [--out FILE]):
  point   one sphere, every pair, at matched and at qpsphere's default period (about 3 s)
  sweep   over dn x radius: Rytov against Rytov, Rytov against the scalar exact field, and the
          scalar exact field against vector Mie at three NAs (about 10 s)
  vector  scalar exact and Rytov against vector Mie for one sphere against the NA (about 2 s)
"""

import argparse
import json
import sys
import time

import numpy as np

import qoraholo as qh

WAVELENGTH = 0.532  # µm, vacuum
N_MEDIUM = 1.337
PIXEL = 0.11  # µm
N = 512
RADIUS = 3.0  # µm
DN = 0.02
SWEEP_DN = (5e-4, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2)
SWEEP_RADIUS = (1.0, 2.0, 3.0, 4.0, 6.0)
NA_VALUES = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, N_MEDIUM)
SWEEP_NA = (0.6, 0.9, 1.2)


def qoraholo_field(
    model: str, radius: float, dn: float, n: int = N, na: float = N_MEDIUM
) -> np.ndarray:
    geometry = qh.Geometry(wavelength=WAVELENGTH, n_medium=N_MEDIUM, na=na, pixel=PIXEL)
    sphere = qh.Sphere(z=0.0, y=-PIXEL / 2, x=-PIXEL / 2, radius=radius, n=N_MEDIUM + dn)
    illumination = np.zeros((1, 2), np.float32)
    return qh.forward_fields([sphere], illumination, geometry, n, model=model)[0].astype(
        np.complex128
    )


def qpsphere_field(
    model: str, radius: float, dn: float, n: int = N, zeropad: int = 1
) -> np.ndarray:
    """qpsphere's Born or Rytov field; zeropad=1 is QoraHolo's period, qpsphere's default 5."""
    from qpsphere.models.mod_rytov import sphere_prop_fslice_bessel

    return sphere_prop_fslice_bessel(
        radius=radius,
        sphere_index=N_MEDIUM + dn,
        medium_index=N_MEDIUM,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL,
        grid_size=(n, n),
        lD=0,
        approx=model,
        zeropad=zeropad,
    )


def refocused_field(amplitude, n: int = N, na: float = N_MEDIUM) -> np.ndarray:
    """u / u_in at z = 0 from a far-field amplitude, through QoraHolo's relation
    Û_s(ν) = i f / ν_z on the n x n periodic grid, for an on-axis plane wave and the sphere at
    QoraHolo's half-pixel offset. ``amplitude(cos θ, φ, |q|)`` gives f in µm (u_s ~ f e^{ikr} / r).
    Only the refocusing is re-implemented, so the Born amplitude must reproduce
    forward_fields(model="born")."""
    fm = N_MEDIUM / WAVELENGTH
    f = np.fft.fftfreq(n, PIXEL)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    rho2 = fy**2 + fx**2
    inside = (rho2 <= (na / WAVELENGTH) ** 2) & (rho2 < fm**2)
    nuz = np.sqrt(np.where(inside, fm**2 - rho2, 1.0))
    amp = np.where(
        inside, amplitude(nuz / fm, np.arctan2(fy, fx), np.sqrt(fy**2 + fx**2 + (nuz - fm) ** 2)), 0
    )
    shift = np.exp(-2j * np.pi * (fy + fx) * (-PIXEL / 2))
    k = np.arange(n)
    parity = (-1.0) ** (k[:, None] + k[None, :])  # centre-origin output, as in QoraHolo
    spectrum = np.where(inside, 1j * amp / nuz, 0) * shift * parity / PIXEL**2
    return 1.0 + np.fft.ifft2(spectrum)


def born_amplitude(radius: float, dn: float):
    k0 = 2 * np.pi / WAVELENGTH
    volume = 4 / 3 * np.pi * radius**3

    def amplitude(mu, phi, q):
        x = 2 * np.pi * q * radius
        ball = np.where(
            x < 1e-3, 1 - x**2 / 10, 3 * (np.sin(x) - x * np.cos(x)) / np.maximum(x, 1e-3) ** 3
        )
        return k0**2 * ((N_MEDIUM + dn) ** 2 - N_MEDIUM**2) * volume * ball / (4 * np.pi)

    return amplitude


def mie_amplitude(radius: float, dn: float):
    """Co-polarised (x) far-field amplitude of vector Mie for x-polarised light,
    f_x = (i / k_m)(S2 cos θ cos² φ + S1 sin² φ) with Bohren & Huffman's S1, S2 (e^{-iωt}).
    miepython's norm="wiscombe" amplitudes are their complex conjugates (Wiscombe's e^{+iωt}):
    at m = 1.001, x = 0.05 they give conj(-i x³ (m² - 1) / 3) to 0.07%, which equals Bohren &
    Huffman's -i x³ (m² - 1) / (m² + 2) only because (m² + 2) / 3 ≈ 1 there. norm="bohren" is a
    phase-function normalisation, -2 times that."""
    import miepython

    km = 2 * np.pi * N_MEDIUM / WAVELENGTH

    def amplitude(mu, phi, q):
        # cos θ depends on the radial frequency only, so S1, S2 are evaluated once per value.
        unique, index = np.unique(mu, return_inverse=True)
        s1, s2 = miepython.S1_S2((N_MEDIUM + dn) / N_MEDIUM, km * radius, unique, norm="wiscombe")
        s1, s2 = np.conj(s1)[index].reshape(mu.shape), np.conj(s2)[index].reshape(mu.shape)
        return 1j / km * (s2 * mu * np.cos(phi) ** 2 + s1 * np.sin(phi) ** 2)

    return amplitude


def crop(field: np.ndarray, n: int) -> np.ndarray:
    a = (field.shape[0] - n) // 2
    return field[a : a + n, a : a + n]


def lowpass(field: np.ndarray, na: float) -> np.ndarray:
    """Keep lateral frequencies up to na / wavelength, the detection pupil QoraHolo applies."""
    f = np.fft.fftfreq(field.shape[0], PIXEL)
    spectrum = np.fft.fft2(field - 1.0)
    spectrum[f[:, None] ** 2 + f[None, :] ** 2 > (na / WAVELENGTH) ** 2] = 0
    return 1.0 + np.fft.ifft2(spectrum)


def compare(a: np.ndarray, b: np.ndarray, radius: float) -> dict:
    """Errors of a against b. field_rel is ‖a − b‖ / ‖b − 1‖ over the frame; phases are compared as
    arg(a b*) so no unwrapping is involved; phase_rel_disk divides by b's centre phase."""
    n = a.shape[0]
    c = (np.arange(n) - (n - 1) / 2) * PIXEL
    disk = c[:, None] ** 2 + c[None, :] ** 2 <= radius**2
    dphi = np.angle(a * np.conj(b))
    damp = np.abs(a) - np.abs(b)
    centre = abs(float(np.angle(b[n // 2, n // 2])))
    rms_disk = float(np.sqrt(np.mean(dphi[disk] ** 2)))
    return {
        "field_rel": float(np.linalg.norm(a - b) / np.linalg.norm(b - 1.0)),
        "phase_rmse": float(np.sqrt(np.mean(dphi**2))),
        "phase_rmse_disk": rms_disk,
        "phase_rel_disk": rms_disk / centre if centre > 1e-12 else None,
        "amplitude_rmse": float(np.sqrt(np.mean(damp**2))),
    }


def point(radius: float = RADIUS, dn: float = DN) -> dict:
    t = time.perf_counter()
    q = {m: qoraholo_field(m, radius, dn) for m in ("born", "rytov", "exact")}
    q_wide = {m: crop(qoraholo_field(m, radius, dn, n=5 * N), N) for m in ("rytov", "exact")}
    p = {m: qpsphere_field(m, radius, dn) for m in ("born", "rytov")}
    p5 = qpsphere_field("rytov", radius, dn, zeropad=5)
    pairs = {
        "rytov: qoraholo vs qpsphere, same period": compare(q["rytov"], p["rytov"], radius),
        "born: qoraholo vs qpsphere, same period": compare(q["born"], p["born"], radius),
        "rytov: qoraholo (period n) vs qpsphere (zeropad 5)": compare(q["rytov"], p5, radius),
        "rytov: qoraholo (period 5n, cropped) vs qpsphere (zeropad 5)": compare(
            q_wide["rytov"], p5, radius
        ),
        "qoraholo rytov vs qoraholo exact, period n": compare(q["rytov"], q["exact"], radius),
        "qoraholo born vs qoraholo exact, period n": compare(q["born"], q["exact"], radius),
        "qoraholo exact: period n vs period 5n (cropped)": compare(
            q["exact"], q_wide["exact"], radius
        ),
        "refocusing in this script (Born amplitude) vs qoraholo born": compare(
            refocused_field(born_amplitude(radius, dn)), q["born"], radius
        ),
    }
    return {
        "radius": radius,
        "dn": dn,
        "n": N,
        "pixel": PIXEL,
        "pairs": pairs,
        "seconds": round(time.perf_counter() - t, 1),
    }


def sweep() -> dict:
    t = time.perf_counter()
    rows = []
    for radius in SWEEP_RADIUS:
        for dn in SWEEP_DN:
            q = qoraholo_field("rytov", radius, dn)
            exact = qoraholo_field("exact", radius, dn)
            vector = refocused_field(mie_amplitude(radius, dn))
            rows.append(
                {
                    "radius": radius,
                    "dn": dn,
                    "centre_phase": float(2 * np.pi / WAVELENGTH * dn * 2 * radius),
                    "rytov_vs_rytov": compare(q, qpsphere_field("rytov", radius, dn), radius),
                    "rytov_vs_exact": compare(q, exact, radius),
                    "exact_vs_vector": {
                        str(na): compare(lowpass(exact, na), lowpass(vector, na), radius)
                        for na in SWEEP_NA
                    },
                }
            )
    return {"n": N, "pixel": PIXEL, "rows": rows, "seconds": round(time.perf_counter() - t, 1)}


def vector(radius: float = RADIUS, dn: float = DN) -> dict:
    """QoraHolo's scalar exact and Rytov fields against vector Mie, all low-passed to each NA."""
    t = time.perf_counter()
    mie = refocused_field(mie_amplitude(radius, dn))
    exact = qoraholo_field("exact", radius, dn)
    rytov = qoraholo_field("rytov", radius, dn)
    rows = []
    for na in NA_VALUES:
        ref = lowpass(mie, na)
        rows.append(
            {
                "na": na,
                "exact_vs_vector": compare(lowpass(exact, na), ref, radius),
                "rytov_vs_vector": compare(lowpass(rytov, na), ref, radius),
            }
        )
    return {
        "radius": radius,
        "dn": dn,
        "n": N,
        "pixel": PIXEL,
        "rows": rows,
        "seconds": round(time.perf_counter() - t, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["point", "sweep", "vector"])
    parser.add_argument("--out", help="write JSON here instead of stdout")
    args = parser.parse_args()
    result = {"point": point, "sweep": sweep, "vector": vector}[args.mode]()
    text = json.dumps(result, indent=1) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
