"""End-to-end run on a synthetic cell: holograms -> fields -> Rytov -> ODT, and the README figure.

uv run --group examples python examples/readme_figure.py [--backend cuda]
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import qoraholo as qh

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--out", default=Path(__file__).parent.parent / "docs" / "reconstruction.png")
args = parser.parse_args()

N = 256  # field side and volume side; holograms are 2N
ANGLES = 49
holo_geom = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.055)
geom = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.11)
cell = [
    qh.Sphere(z=0.0, y=0.0, x=0.0, radius=6.0, n=1.360),
    qh.Sphere(z=0.5, y=-1.0, x=1.0, radius=3.0, n=1.375),
    qh.Sphere(z=-2.0, y=3.0, x=-2.5, radius=0.8, n=1.390),
    qh.Sphere(z=2.0, y=2.5, x=3.0, radius=0.6, n=1.390),
]

illum = qh.circular_scan(ANGLES, 0.9, geom)
ratio = qh.forward_fields(cell, illum, holo_geom, 2 * N)
carrier = (5.0, 5.0)
holo = qh.synthesize_holograms(ratio, illum, holo_geom, carrier=carrier, noise=0.02, seed=1)
bg = qh.synthesize_holograms(
    np.ones_like(ratio), illum, holo_geom, carrier=carrier, noise=0.02, seed=2
)

# Carrier from the normal-incidence background; each illumination from its background peak.
df = 1.0 / (2 * N * holo_geom.pixel)
c = qh.find_sideband(bg[0])
estimated = np.array(
    [[(s.y - c.y) * df, (s.x - c.x) * df] for s in (qh.find_sideband(b) for b in bg)],
    np.float32,
)
illum_error = np.abs(estimated - illum).max()

radius = np.ceil(holo_geom.na / holo_geom.wavelength / df) + 1
opts = dict(center=(c.iy, c.ix), radius=radius, output_size=N, backend=args.backend)
t0 = time.perf_counter()
fields = qh.field_ratio(qh.extract_fields(holo, **opts), qh.extract_fields(bg, **opts))
rytov = qh.rytov(fields, backend=args.backend)
plain = qh.reconstruct(rytov, estimated, geom, backend=args.backend)
gp = qh.reconstruct(rytov, estimated, geom, iterations=20, backend=args.backend)
elapsed = time.perf_counter() - t0
t0 = time.perf_counter()
tv = qh.reconstruct(rytov, estimated, geom, iterations=20, solver="tv", backend=args.backend)
elapsed_tv = time.perf_counter() - t0

truth = qh.render_phantom(cell, geom, N, N)
z, y, x = (np.indices(truth.shape) - N // 2) * geom.pixel
nucleus = (z - 0.5) ** 2 + (y + 1.0) ** 2 + (x - 1.0) ** 2 < (0.6 * 3.0) ** 2


def rmse(a):
    return float(np.sqrt(np.mean((a - truth) ** 2)))


print(f"backend {args.backend}, pipeline time {elapsed:.2f} s (2 x {ANGLES} holograms, {N}^3)")
print(f"FISTA-TV, 20 x 10 iterations: {elapsed_tv:.2f} s")
print(f"largest illumination estimate error {illum_error:.2e} cycles/um")
for name, vol in [("plain", plain), ("GP", gp), ("TV", tv)]:
    print(f"{name}: nucleus mean RI {vol[nucleus].mean():.4f} (truth 1.3750), RMSE {rmse(vol):.2e}")

zc = N // 2 + round(0.5 / geom.pixel)
yc = N // 2 + round(-1.0 / geom.pixel)
crop = slice(N // 2 - 70, N // 2 + 70)
fig, ax = plt.subplots(2, 5, figsize=(16, 6.6), constrained_layout=True)
ax[0, 0].imshow(holo[0, 200:328, 200:328], cmap="gray")
ax[0, 0].set_title("hologram (detail)")
ax[1, 0].imshow(rytov[0].imag[crop, crop], cmap="gray")
ax[1, 0].set_title("unwrapped phase, 0 deg")
vmin, vmax = geom.n_medium, 1.392
im = None
volumes = [("phantom", truth), ("Rytov ODT", plain), ("+ 20 GP", gp), ("+ 20 FISTA-TV", tv)]
for col, (name, vol) in enumerate(volumes, 1):
    ax[0, col].imshow(vol[zc, crop, crop], vmin=vmin, vmax=vmax, cmap="viridis")
    ax[0, col].set_title(f"{name}, x-y")
    im = ax[1, col].imshow(vol[crop, yc, crop], vmin=vmin, vmax=vmax, cmap="viridis")
    ax[1, col].set_title(f"{name}, x-z")
for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])
fig.colorbar(im, ax=ax[:, 1:], shrink=0.8, label="refractive index")
fig.savefig(args.out, dpi=110)
print(f"wrote {args.out}")
