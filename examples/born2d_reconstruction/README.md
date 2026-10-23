# Reconstructing a known 2D object with the first Born operator

`run.py` images a known object, reconstructs it with `qoraholo.born2d`, and measures resolution,
artifacts and noise sensitivity. `expected.json` keeps the physical expectations (`truth`)
apart from today's output (`regression`); `python/tests/test_example_born2d_reconstruction.py`
checks both.

```
python examples/born2d_reconstruction/run.py     # about 30 s, prints the numbers below
```

## Setup

| | |
|---|---|
| wavelength, medium | 0.532 µm, water (n_m = 1.337) |
| detection NA | 1.2 |
| illumination | 31 plane waves, NA −0.9 to 0.9 in the x-z plane |
| grid | 64 × 64, 0.1 µm pixels (6.4 µm field of view) |
| object | a 2 µm disc of δn = 0.004 and a 1 µm disc of δn = 0.002 |
| Born validity | \|δn\|·k0·L = 0.094 rad for the larger disc, 0.024 rad for the smaller |
| data | first Born (same operator) to isolate the reconstruction; BPM on the object extruded along y to measure the model error |
| reconstruction | 300 Landweber iterations from zero, potential constrained to be real |

For a real object the constrained Landweber iteration is plain Landweber on S = [Re A; Im A].
k iterations from zero therefore act as the spectral filter 1 − (1 − w/w_max)^k on the
eigenvalues w of S Sᵀ. `run.py` computes it in that closed form; the test checks it against the
loop to 1e-8.

## Results

### Coverage and resolution

The Ewald arcs inside the detection pupil cover |Kx| ≤ 3.91 and Kz ∈ [−1.35, 0.65] cycles/µm.
The lateral limit is (NA + NA_ill)/λ = 3.95 to within one frequency bin. The axial coverage is
narrower because the arcs are shallow: this is the missing cone.

| point-spread function | FWHM | expectation |
|---|---|---|
| lateral (x) | 0.151 µm | 0.603 / Kx_max = 0.153 µm for a uniformly filled band |
| axial (z) | 0.502 µm | elongated by about Kx_max / max\|Kz\| = 2.9; measured 3.3 |

### Recovered index: where the missing 36 % goes

Mean δn over the inner 0.7 µm of the 2 µm disc, truth 0.004:

| | δn | change |
|---|---|---|
| truth | 0.00400 | |
| information limit: the true object projected onto the row space of S, i.e. the minimum-norm object with the same data (singular values below 1e-5 of the largest dropped) | 0.00305 | −23.6 % (missing cone and band limit) |
| 300 Landweber iterations on Born data | 0.00258 | −12.0 % more (early stopping) |
| the same on BPM data | 0.00257 | −0.18 % more (Born model error) |

- **What the data do not carry.** Most of the deficit is information the data do not contain.
  No number of iterations recovers it; only a prior can (positivity, support, TV, or
  Gerchberg-Papoulis as the 3D `reconstruct` does). This is QoraHolo#3, and the `truth`
  check on δn is a strict xfail tied to it.
- **Early stopping.** The next part is the filter: components with w/w_max well below 1/300
  are still mostly suppressed after 300 iterations. More iterations recover them, but they
  also raise the noise gain.
- **Born model error.** It is small at this contrast and grows linearly with δn:

  | δn (larger disc) | \|δn\|·k0·L | relative gap, BPM-data vs Born-data reconstruction |
  |---|---|---|
  | 0.002 | 0.047 | 1.65 % |
  | 0.004 | 0.094 | 3.29 % |

  Doubling δn doubles the gap (ratio 2.00), as expected of the first neglected order. A
  convention error between the two models would show as a gap of order one that does not
  scale. The gap stays below |δn|·k0·L in both cases.

### Artifacts

- **Axial elongation.** The 2 µm disc is 1.85 µm wide along x and 2.74 µm along z at half
  maximum, 37 % longer axially. This is the visible form of the missing cone; the PSF itself
  is 3.3 times longer along z than along x.
- **Low core, positive halo.** The core is low (above), and the deficit reappears around the
  disc. In the ring 1.2 to 1.6 µm from its centre, where the true contrast is 0, the mean
  recovered δn is +0.00032, 8 % of the disc's contrast.
- **Data residual.** After 300 iterations, ‖A x − y‖/‖y‖ = 1.03 % on noiseless data. That is
  the early-stopping filter, not noise.

### Noise

The reconstruction is linear in the data. Complex Gaussian noise with the rms of the data
produces a reconstruction 1.44 times the norm of the noiseless one, so the noise gain is
1.44 at 300 iterations. Relative noise η in the data gives about 1.44 η in the object.

## Limits of this example

- **2D only.** One plane of illumination, and a scalar model.
- **Synthetic data.** The Born data come from the same operator used to reconstruct (no
  model error by construction); the BPM data are the model-error check.
- **No prior.** Landweber without a prior is the baseline here, not a recommended
  reconstruction method.
- **Cutoff dependence.** The information limit depends on where small singular values are
  cut. At 1e-5 of the largest it is 0.00305; at 1e-10 (computed with a full SVD) it is
  0.00315.
