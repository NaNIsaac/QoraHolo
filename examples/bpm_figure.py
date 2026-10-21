"""Multiple scattering: Rytov ODT against beam-propagation (BPM) inversion, and the README figure.

uv run --group examples python examples/bpm_figure.py [--backend cuda]

Two phantoms whose Rytov reconstruction is biased. Neither data set comes from the
discretisation that is inverted:
- bead: one sphere, fields from the exact partial-wave solution (forward_fields "exact");
- stack: three spheres along z, fields from the BPM on a grid twice as fine in x, y and z,
  cut down to the coarse grid in Fourier space (the fields are band-limited to the NA).
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import qoraholo as qh

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--iterations", type=int, default=30)
parser.add_argument("--tv-weight", type=float, default=3e-2)
parser.add_argument("--out", default=Path(__file__).parent.parent / "docs" / "bpm.png")
args = parser.parse_args()

N = 128
ANGLES = 49
geom = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.15)
fine = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.075)
illum = qh.circular_scan(ANGLES, 0.9, geom)
backend = args.backend

bead = [qh.Sphere(z=0.0, y=0.0, x=0.0, radius=4.0, n=1.40)]
stack = [qh.Sphere(z=z, y=0.0, x=0.0, radius=2.0, n=1.40) for z in (-4.5, 0.0, 4.5)]


def stack_fields():
    ri = qh.render_phantom(stack, fine, 2 * N, 2 * N)
    f = qh.bpm_forward(ri, illum, fine, backend=backend)
    spec = np.fft.fftshift(np.fft.fft2(f), axes=(1, 2))
    lo = N // 2
    crop = np.fft.ifftshift(spec[:, lo : lo + N, lo : lo + N], axes=(1, 2))
    return (np.fft.ifft2(crop) / 4).astype(np.complex64)


def timed(f, *a, **kw):
    t0 = time.perf_counter()
    out = f(*a, **kw)
    return out, time.perf_counter() - t0


cases = {
    "bead": (bead, qh.forward_fields(bead, illum, geom, N, model="exact")),
    "stack": (stack, stack_fields()),
}
results = {}
for name, (spheres, data) in cases.items():
    truth = qh.render_phantom(spheres, geom, N, N)
    z, y, x = (np.indices(truth.shape) - N // 2) * geom.pixel
    inner = np.zeros(truth.shape, bool)
    for s in spheres:
        inner |= (z - s.z) ** 2 + (y - s.y) ** 2 + (x - s.x) ** 2 < (0.6 * s.radius) ** 2

    ry = qh.rytov(data, backend=backend)
    vols = {}
    times = []
    runs = [
        ("Rytov", qh.reconstruct, (ry, illum, geom), {}),
        ("Rytov + GP", qh.reconstruct, (ry, illum, geom), dict(iterations=20)),
        ("Rytov + TV", qh.reconstruct, (ry, illum, geom), dict(iterations=20, solver="tv")),
    ]
    for method, f, pos, kw in runs:
        vols[method], t = timed(f, *pos, backend=backend, **kw)
        times.append(t)
    bpm = dict(iterations=args.iterations, backend=backend)
    for method, kw in [
        ("BPM", dict(return_loss=True)),
        ("BPM + TV", dict(tv_weight=args.tv_weight)),
        ("BPM, plain screen", dict(obliquity=False)),
    ]:
        out, t = timed(qh.bpm_reconstruct, data, illum, geom, vols["Rytov + GP"], **bpm, **kw)
        if method == "BPM":
            out, loss = out
        vols[method] = out
        times.append(t)
    print(f"\n{name}: true n 1.4000, {N}^3, {ANGLES} angles, backend {backend}")
    print(f"BPM data term: {loss[0]:.3e} -> {loss[-1]:.3e} over {len(loss)} iterations")
    print("| method | RMSE | mean n inside 0.6 R | time (s) |")
    for (method, v), t in zip(vols.items(), times, strict=True):
        rmse = np.sqrt(np.mean((v - truth) ** 2))
        print(f"| {method} | {rmse:.2e} | {v[inner].mean():.4f} | {t:.2f} |")
    results[name] = (truth, vols)

crop = slice(N // 2 - 50, N // 2 + 50)
fig, ax = plt.subplots(2, 4, figsize=(13, 6.8), constrained_layout=True)
im = None
for row, (name, (truth, vols)) in enumerate(results.items()):
    shown = [
        ("phantom", truth),
        ("Rytov", vols["Rytov"]),
        ("Rytov + GP", vols["Rytov + GP"]),
        ("BPM", vols["BPM"]),
    ]
    for col, (title, v) in enumerate(shown):
        im = ax[row, col].imshow(v[crop, N // 2, crop], vmin=1.337, vmax=1.41, cmap="viridis")
        ax[row, col].set_title(f"{name}: {title}, x-z")
for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.8, label="refractive index")
fig.savefig(args.out, dpi=100)
print(f"wrote {args.out}")
