"""Noise and an object with n < n_m: the positivity prior against Landweber (QoraHolo#6 follow-up).

BPM data (a different forward model than the one inverted) plus complex Gaussian noise of rms
eta · 0.0452, the rms of the noiseless data, for eta = 0.1, 0.3, 1.0 (about 4.5, 13.5 and 45 mrad
of phase per sample) and seeds 0-9. Object A is the example's phantom; object B adds a 1 µm disc
of δn = -0.002 at (x, z) = (-0.6, -1.6) µm. Landweber, positivity, and positivity with TV
λ_rel = 1e-3, all at 300 iterations. Prints mean ± sd over seeds of the core δn of both discs, the
core of the negative disc (B) or the halo ring (A), and the map RMSE; writes noise.json here.

    python examples/born2d_reconstruction/noise.py      # about 6 min on an idle CPU
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ex", HERE / "run.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

ETAS, SEEDS = (0.1, 0.3, 1.0), range(10)
RMS = 0.0452  # rms |y| of the example's data, fixed in the pre-registration
t0 = time.perf_counter()
m = ex.model()
system = ex.RealSystem(m)
zz, xx = ex.grid(m)
big, small = ex.core(m), (xx - 1.6) ** 2 + (zz - 0.5) ** 2 < 0.3**2
neg_disc = (xx + 0.6) ** 2 + (zz + 1.6) ** 2 < 0.5**2
neg_core = (xx + 0.6) ** 2 + (zz + 1.6) ** 2 < 0.3**2
ring = (np.hypot(xx + 0.8, zz) > 1.2) & (np.hypot(xx + 0.8, zz) < 1.6) & (xx < 0.4)


def phantom(kind):
    ri = ex.phantom(m, ex.DELTA_N)
    if kind == "B":
        ri[neg_disc] -= ex.DELTA_N / 2
    return ri


def truth_dn(ri):
    return ri - ex.N_MEDIUM


methods = {
    "landweber": lambda y: system.landweber(y),
    "positivity": lambda y: system.projected(y),
    "positivity_tv1e-3": lambda y: system.projected(y, tv=1e-3),
}
out = {"etas": ETAS, "seeds": list(SEEDS), "rms": RMS, "objects": {}}
for kind in ("A", "B"):
    ri = phantom(kind)
    y_clean = ex.bpm_fields(ri, m)
    dn_true = truth_dn(ri)
    res = {}
    for eta in ETAS:
        for name, f in methods.items():
            rows = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                noise = (
                    rng.normal(size=y_clean.shape) + 1j * rng.normal(size=y_clean.shape)
                ) / np.sqrt(2)
                y = y_clean + eta * RMS * noise
                dn = ex.to_delta_n(f(y))
                row = {
                    "big": float(dn[big].mean()),
                    "small": float(dn[small].mean()),
                    "rmse": float(np.sqrt(np.mean((dn - dn_true) ** 2))),
                }
                if kind == "A":
                    row["ring"] = float(dn[ring].mean())
                else:
                    row["neg"] = float(dn[neg_core].mean())
                rows.append(row)
            stats = {
                k: (
                    float(np.mean([r[k] for r in rows])),
                    float(np.std([r[k] for r in rows], ddof=1)),
                )
                for k in rows[0]
            }
            res[f"{eta}/{name}"] = stats
            print(
                kind,
                eta,
                name,
                {k: f"{v[0]:.5f}±{v[1]:.5f}" for k, v in stats.items()},
                f"{time.perf_counter() - t0:.0f} s",
                flush=True,
            )
    out["objects"][kind] = res
(HERE / "noise.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
