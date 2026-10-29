"""Draw the born2d example's results: coverage, recovered index, point-spread function, core δn.

Everything is computed by ``run.py``'s own functions (CPU, about 40 s); nothing is fitted or
measured. The plotted numbers are checked against ``expected.json`` before the figure is written,
and the line colours with cvdkit, so the figure cannot drift from what the test pins.

    uv run --group figures python examples/born2d_reconstruction/figure.py   # writes figure.png
"""

import importlib.util
import json
from pathlib import Path

import cvdkit
import matplotlib.pyplot as plt
import numpy as np
import qoraplot
from matplotlib.ticker import FuncFormatter

from qoraholo.born2d import potential

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ex", HERE / "run.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

EXPECTED = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
LIMIT = 0.0045  # colour range of the δn maps, symmetric so that zero is the neutral colour
ARC = "#33608c"
THOUSANDTHS = FuncFormatter(lambda v, _: f"{v * 1e3:g}")


def checked(result: dict) -> dict:
    """The plotted numbers, each within its expected.json regression tolerance."""
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
    for name, value in got.items():
        spec = EXPECTED["regression"][name]
        if abs(value - spec["value"]) > spec["tol"]:
            raise SystemExit(f"{name}: {value:.6g} vs expected {spec['value']} ± {spec['tol']}")
    return got | {"kx_max": result["coverage"]["kx_max"]}


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
    kx = np.unique(np.concatenate([a for a, _ in lines]))
    low = np.array([min(z[a == k].min() for a, z in lines if (a == k).any()) for k in kx])
    high = np.array([max(z[a == k].max() for a, z in lines if (a == k).any()) for k in kx])
    top, bottom = 1.0, -1.7
    ax.fill_between(kx, high, top, color="0.88", lw=0, step="mid")
    ax.fill_between(kx, bottom, low, color="0.88", lw=0, step="mid", label="not measured")
    for a, z in lines:
        ax.plot(a, z, color=ARC, lw=0.5, marker="none", ls="-")
    ax.plot([], [], color=ARC, lw=0.8, marker="none", ls="-", label="Ewald arcs (31)")
    for sign in (-1, 1):
        ax.axvline(sign * n["kx_max"], color="0.35", ls=":", lw=0.8, marker="none")
    ax.set(
        xlim=(-4.6, 4.6),
        ylim=(bottom, top),
        xlabel=r"$K_x$ (cycles/µm)",
        ylabel=r"$K_z$ (cycles/µm)",
        title="Measured spatial frequencies",
    )
    for kz in (0.5, -0.9):
        ax.text(0, kz, "missing\ncone", ha="center", va="center", fontsize="x-small")
    ax.legend(loc="lower center", fontsize="x-small", handlelength=1.2)


def maps_panel(axes, m, maps: dict, cax_owner) -> None:
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
    bar = cax_owner.colorbar(image, ax=list(axes), shrink=0.85, pad=0.02)
    bar.set_label(r"$\delta n$ ($10^{-3}$, refractive-index units)")
    bar.formatter = THOUSANDTHS


def psf_panel(ax, m, psf: np.ndarray, n: dict) -> None:
    centre = ex.N // 2
    ax.plot(
        m.x - m.x[centre],
        psf[centre, :] / psf.max(),
        marker="o",
        ms=2.5,
        label=f"x, FWHM {n['psf_fwhm_x']:.3f} µm",
    )
    ax.plot(
        m.z - m.z[centre],
        psf[:, centre] / psf.max(),
        marker="s",
        ms=2.5,
        ls="--",
        label=f"z, FWHM {n['psf_fwhm_z']:.3f} µm",
    )
    ax.axhline(0.5, color="0.5", lw=0.6, marker="none")
    ax.set(
        xlim=(-1.5, 1.5),
        ylim=(-0.5, 1.12),
        xlabel="offset from the point (µm)",
        ylabel="normalised response",
        title="Point-spread function, 300 iterations",
    )
    expected = EXPECTED["truth"]["psf_fwhm_x"]["value"]
    ax.legend(
        loc="upper left",
        fontsize="x-small",
        title=f"x expected for a\nuniform band: {expected:.3f} µm",
        title_fontsize="x-small",
    )


def core_panel(ax, n: dict) -> None:
    rows = [
        ("truth", ex.DELTA_N, "k"),
        (
            "information limit\n(minimum-norm object, same data)",
            n["delta_n_information_limit"],
            "0.35",
        ),
        ("Landweber, Born data", n["delta_n_landweber"], ARC),
        ("Landweber, BPM data", n["delta_n_from_bpm_data"], ARC),
        (r"$n \geq n_m$ prior, Born data", n["positivity_delta_n"], "#b2182b"),
        (r"$n \geq n_m$ prior, BPM data", n["positivity_delta_n_from_bpm_data"], "#b2182b"),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (_, value, color) in zip(y, rows, strict=True):
        ax.plot(
            [0, value], [yi, yi], color="0.85", lw=1.5, ls="-", marker="none", solid_capstyle="butt"
        )
        ax.plot(value, yi, marker="o", ls="none", color=color, ms=5)
        ax.text(value + 0.00008, yi, f"{value:.5f}", va="center", fontsize="x-small")
    ax.axvline(ex.DELTA_N, color="k", ls=":", lw=0.8, marker="none")
    ax.set_yticks(y, [label for label, _, _ in rows], fontsize="x-small")
    ax.tick_params(axis="y", which="both", length=0)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set(
        xlim=(0, 0.0048),
        xlabel=r"mean $\delta n$ in the 2 µm disc's core ($10^{-3}$)",
        title="Recovered index contrast, 300 iterations",
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
        "Landweber": ex.to_delta_n(system.landweber(y)),
        r"$n \geq n_m$ prior": ex.to_delta_n(system.projected(y)),
    }
    psf = system.landweber(m.forward(delta))

    qoraplot.use("house", "paper")
    fig = plt.figure(figsize=(qoraplot.WIDTHS["double"], 5.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.05])
    a, *b = top.subplots(1, 4, width_ratios=[1.25, 1, 1, 1])
    c, d = bottom.subplots(1, 2, width_ratios=[1, 1.15])
    coverage_panel(a, m, n)
    maps_panel(b, m, maps, top)
    psf_panel(c, m, psf, n)
    core_panel(d, n)
    for ax, label in zip((a, b[0], c, d), "abcd", strict=True):
        qoraplot.panel_label(ax, label)
    bottom.supxlabel(
        "Computed from synthetic data (first Born and BPM); not a measurement.",
        x=0.995,
        ha="right",
        fontsize="x-small",
        color="0.35",
    )

    # Only legend entries must be told apart by colour: the 31 arcs are one series, and the rows
    # of (d) are named on their axis.
    collisions = [c for c in cvdkit.check(fig) if not (c.a.startswith("_") or c.b.startswith("_"))]
    if collisions:
        raise SystemExit(f"colour-vision check failed: {collisions}")
    # Not quantized like docs/reconstruction.png (#11): 256 colours band the δn colour scale.
    fig.savefig(HERE / "figure.png", dpi=200)
    print(json.dumps(n, indent=2))


if __name__ == "__main__":
    main()
