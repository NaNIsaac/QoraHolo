"""Draw the born2d example's results: coverage, recovered index, point-spread function, core δn.

Everything is computed by ``run.py``'s own functions; nothing is fitted or measured. The plotted
numbers are checked against ``expected.json`` before the figure is written, and the line colours
with cvdkit, so the figure cannot drift from what the test pins.

    uv run --group figures python examples/born2d_reconstruction/figure.py   # writes figure.png
"""

import importlib.util
import json
from pathlib import Path

import cvdkit
import matplotlib.pyplot as plt
import numpy as np
import qoraplot
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from qoraholo.born2d import potential

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ex", HERE / "run.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

EXPECTED = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
# Colour range of the δn maps: symmetric so that zero is the neutral colour, and wide enough for
# the prior's maximum (0.0046), so nothing is clipped.
LIMIT = 0.005
UPSAMPLE = 32  # as run.fwhm, so the drawn PSF curve is the one its FWHM is measured on
ARC, PRIOR, GREY = "#33608c", "#b2182b", "0.88"
THOUSANDTHS = FuncFormatter(lambda v, _: f"{v * 1e3:g}".replace("-", "−"))


def checked(result: dict) -> dict:
    """The plotted numbers, each within its expected.json tolerance."""
    gap = result["model_error"][str(ex.DELTA_N)]
    got = {
        "kz_min": result["coverage"]["kz_min"],
        "kz_max": result["coverage"]["kz_max"],
        "psf_fwhm_x": result["psf_fwhm_x"],
        "psf_fwhm_z": result["psf_fwhm_z"],
        "delta_n_information_limit": result["delta_n_information_limit"],
        "delta_n_landweber": result["delta_n_landweber"],
        "delta_n_from_bpm_data": gap["delta_n_from_bpm_data"],
        "positivity_delta_n": result["positivity"]["delta_n"],
        "positivity_delta_n_from_bpm_data": gap["delta_n_positivity_from_bpm_data"],
    }
    specs = {name: EXPECTED["regression"][name] for name in got}
    got["kx_max"] = result["coverage"]["kx_max"]
    specs["kx_max"] = EXPECTED["truth"]["kx_max"]
    for name, value in got.items():
        spec = specs[name]
        if abs(value - spec["value"]) > spec["tol"]:
            raise SystemExit(f"{name}: {value:.6g} vs expected {spec['value']} ± {spec['tol']}")
    return got


def arcs(m) -> list[tuple[np.ndarray, np.ndarray]]:
    """(Kx, Kz) of each illumination's Ewald arc inside the detection pupil, as in run.coverage."""
    nu_m = ex.N_MEDIUM / ex.WAVELENGTH
    out = []
    for fx in m.illuminations:
        mu = m.mu + fx
        inside = (np.abs(mu) <= ex.NA / ex.WAVELENGTH) & (np.abs(mu) < nu_m)
        out.append((m.mu[inside], np.sqrt(nu_m**2 - mu[inside] ** 2) - np.sqrt(nu_m**2 - fx**2)))
    return out


def coverage_panel(ax, m, n: dict) -> None:
    lines = arcs(m)
    # The band between the arcs counts as measured: the 31 illuminations are 0.113 cycles/µm
    # apart, closer than the 0.156 grid bin. Its edge is the lowest and highest arc at each grid
    # Kx, joined by straight lines (a step fill drew measured teeth below the arc ends).
    kx = np.unique(np.concatenate([a for a, _ in lines]))
    low = np.array([min(z[a == k].min() for a, z in lines if (a == k).any()) for k in kx])
    high = np.array([max(z[a == k].max() for a, z in lines if (a == k).any()) for k in kx])
    ax.set_facecolor(GREY)
    ax.fill(np.r_[kx, kx[::-1]], np.r_[high, low[::-1]], color="white", lw=0)
    for a, z in lines:
        ax.plot(a, z, color=ARC, lw=0.5, marker="none", ls="-")
    for sign in (-1, 1):
        ax.axvline(sign * n["kx_max"], color="0.3", ls=":", lw=0.8, marker="none")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set(
        xlim=(-4.6, 4.6),
        ylim=(-1.55, 0.9),
        xlabel=r"$K_x$ (cycles/µm)",
        ylabel=r"$K_z$ (cycles/µm)",
        title="Measured spatial frequencies (equal scales)",
    )
    for kz in (0.55, -1.15):
        ax.text(0, kz, "missing cone", ha="center", va="center", fontsize="small")
    ax.text(-n["kx_max"] + 0.08, 0.72, f"|$K_x$| ≤ {n['kx_max']:.2f}", fontsize="small")
    handles = [
        plt.Line2D([], [], color=ARC, lw=0.8, label="Ewald arcs (31 illuminations)"),
        Patch(facecolor=GREY, edgecolor="0.4", lw=0.5, label="not measured"),
    ]
    ax.figure.legend(handles=handles, loc="outside upper right", ncols=2, fontsize="small")


def maps_panel(axes, m, maps: dict, owner) -> None:
    pad = ex.PIXEL / 2
    extent = (m.x[0] - pad, m.x[-1] + pad, m.z[0] - pad, m.z[-1] + pad)
    cmap = cvdkit.colormap("tol_BuRd")
    image = None
    for ax, (title, dn) in zip(axes, maps.items(), strict=True):
        image = ax.imshow(dn, origin="lower", extent=extent, cmap=cmap, vmin=-LIMIT, vmax=LIMIT)
        ax.add_patch(plt.Circle((-0.8, 0.0), 0.7, fill=False, ls="--", lw=0.6, color="0.2"))
        ax.set_title(title, fontsize="small")
        ax.set_xlabel("x (µm)")
        ax.grid(False)
    axes[0].set_ylabel("z (µm), optical axis")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    bar = owner.colorbar(image, ax=list(axes), shrink=0.9, pad=0.02)
    bar.set_label(r"$\delta n \times 10^3$")
    bar.formatter = THOUSANDTHS


def interpolated(profile: np.ndarray) -> np.ndarray:
    """The zero-padded Fourier interpolation run.fwhm measures on, UPSAMPLE points per pixel."""
    n = len(profile)
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(profile)))
    padded = np.zeros(n * UPSAMPLE, complex)
    padded[(n * UPSAMPLE - n) // 2 : (n * UPSAMPLE + n) // 2] = spectrum
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(padded))).real * UPSAMPLE


def psf_panel(ax, psf: np.ndarray, n: dict) -> None:
    centre = ex.N // 2
    offsets = (np.arange(ex.N) - centre) * ex.PIXEL
    fine = (np.arange(ex.N * UPSAMPLE) - ex.N * UPSAMPLE // 2) * ex.PIXEL / UPSAMPLE
    aspect = n["psf_fwhm_z"] / n["psf_fwhm_x"]
    expected = EXPECTED["truth"]
    for profile, name, marker, ls in (
        (
            psf[centre, :],
            f"x: FWHM {n['psf_fwhm_x']:.3f} µm (sinc band: {expected['psf_fwhm_x']['value']:.3f})",
            "o",
            "-",
        ),
        (
            psf[:, centre],
            f"z: FWHM {n['psf_fwhm_z']:.3f} µm, {aspect:.1f} × x "
            f"(coverage: {expected['psf_aspect']['value']:.1f})",
            "s",
            "--",
        ),
    ):
        curve = interpolated(profile)
        top = curve.max()
        (line,) = ax.plot(fine, curve / top, ls=ls, marker="none", label=name)
        ax.plot(offsets, profile / top, ls="none", marker=marker, ms=2.5, color=line.get_color())
    ax.axhline(0.5, color="0.5", lw=0.6, marker="none")
    ax.set(
        xlim=(-1.5, 1.5),
        ylim=(-0.95, 1.08),
        xlabel="offset from the point (µm)",
        ylabel="normalised response",
        title="Point-spread function, Landweber 300 it.",
    )
    ax.legend(loc="lower center", fontsize="small")


def core_panel(ax, n: dict) -> None:
    rows = [
        ("truth", ex.DELTA_N, "k"),
        ("minimum-norm object\n(Landweber k→∞, no prior)", n["delta_n_information_limit"], "0.35"),
        ("Landweber 300 it., Born data", n["delta_n_landweber"], ARC),
        ("Landweber 300 it., BPM data", n["delta_n_from_bpm_data"], ARC),
        (r"$n \geq n_m$ prior, Born data", n["positivity_delta_n"], PRIOR),
        (r"$n \geq n_m$ prior, BPM data", n["positivity_delta_n_from_bpm_data"], PRIOR),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (_, value, color) in zip(y, rows, strict=True):
        ax.plot(
            [0, value], [yi, yi], color="0.85", lw=1.5, ls="-", marker="none", solid_capstyle="butt"
        )
        ax.plot(value, yi, marker="o", ls="none", color=color, ms=5)
        ax.text(value + 0.00012, yi, f"{value * 1e3:.2f}", va="center", fontsize="small")
    ax.axvline(ex.DELTA_N, color="k", ls=":", lw=0.8, marker="none")
    ax.set_yticks(y, [label for label, _, _ in rows], fontsize="small")
    ax.tick_params(axis="y", which="both", length=0)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set(
        xlim=(0, 0.0049),
        xlabel=r"mean $\delta n \times 10^3$, core of radius 0.7 µm",
        title="Core δn, noiseless data",
    )
    ax.xaxis.set_major_formatter(THOUSANDTHS)


def main() -> None:
    result = ex.run()
    n = checked(result)

    m = ex.model()
    system = ex.RealSystem(m)
    v_true = potential(ex.phantom(m, ex.DELTA_N), ex.N_MEDIUM, ex.WAVELENGTH)
    y = m.forward(v_true)
    delta = np.zeros((ex.N, ex.N))
    delta[ex.N // 2, ex.N // 2] = 1.0
    maps = {
        "truth": ex.phantom(m, ex.DELTA_N) - ex.N_MEDIUM,
        "Landweber 300 it.": ex.to_delta_n(system.landweber(y)),
        r"$n \geq n_m$ prior, 300 it.": ex.to_delta_n(system.projected(y)),
    }
    psf = system.landweber(m.forward(delta))

    qoraplot.use("house", "paper")
    fig = plt.figure(figsize=(6.4, 7.4), layout="constrained")
    first, second, third = fig.subfigures(3, 1, height_ratios=[1.0, 1.25, 1.45])
    a = first.subplots()
    b = second.subplots(1, 3)
    c, d = third.subplots(1, 2, width_ratios=[1, 1.05])
    coverage_panel(a, m, n)
    maps_panel(b, m, maps, second)
    psf_panel(c, psf, n)
    core_panel(d, n)
    for ax, label in zip((a, b[0], c, d), "abcd", strict=True):
        qoraplot.panel_label(ax, label)
    third.supxlabel(
        "Computed from synthetic data (first Born and BPM); not a measurement.",
        x=0.995,
        ha="right",
        fontsize="small",
        color="0.35",
    )

    # Only legend entries must be told apart by colour: the 31 arcs are one series, the PSF
    # samples share their curve's colour, and the rows of (d) are named on their axis.
    collisions = [c for c in cvdkit.check(fig) if not (c.a.startswith("_") or c.b.startswith("_"))]
    if collisions:
        raise SystemExit(f"colour-vision check failed: {collisions}")
    # Not quantized like docs/reconstruction.png (#11): 256 colours band the δn colour scale.
    fig.savefig(HERE / "figure.png", dpi=200)
    print(json.dumps(n, indent=2))


if __name__ == "__main__":
    main()
