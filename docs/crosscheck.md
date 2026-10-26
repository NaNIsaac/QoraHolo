# Cross-check of the sphere forward models

QoraHolo's `forward_fields` was compared with two independent codes for one homogeneous sphere:

- qpsphere 0.5.9 (RI-imaging), for first Born and Rytov;
- vector Mie theory, from the scattering amplitudes S1 and S2 of miepython 3.3.0, for the
  scalar `exact` model.

The script is [`checks/sphere_crosscheck.py`](../checks/sphere_crosscheck.py), the results are in
[`checks/results/`](../checks/results), and the agreement at the main point is pinned by
`python/tests/test_crosscheck.py`.

```
uv run --group check python checks/sphere_crosscheck.py point    # about 3 s
uv run --group check python checks/sphere_crosscheck.py vector   # about 1 s
uv run --group check python checks/sphere_crosscheck.py sweep    # about 15 s
uv run --group check pytest -m slow
```

Main point: radius 3 µm, Δn 0.02 (n = 1.357 in n_m = 1.337), λ = 0.532 µm, pixel 0.11 µm, a
512 × 512 field and an on-axis plane wave. The sweep covers Δn from 5e-4 to 0.05 and radii from
1 µm to 6 µm.

Labels: **fact** means measured by the script in this commit. **Inference** means reasoned from
those numbers and not tested further.

## Result

**Fact.** The two Rytov implementations agree to 1.3e-6 at the main point, and to at most 7.3e-5
over all 30 sweep points, when both fields have the same period. The metric is
‖u_QoraHolo − u_qpsphere‖ / ‖u_qpsphere − 1‖. First Born agrees to 1.2e-6. The acceptance
criterion was under 1 % wherever both are valid, so it holds at every point tried. The largest
value, 7.3e-5, occurs at Δn = 5e-4, where the scattered field is small relative to QoraHolo's
float32 output.

**Fact.** The scalar `exact` model differs from the co-polarised vector Mie field by 0.2 % at a
detection NA of 0.2 and by 1.6 % at NA 1.2 for the main sphere. The difference grows as the
sphere gets smaller: at NA 1.2 it is 4.4 % for a 1 µm radius and 0.7 % for 6 µm. Up to Δn = 0.02
it barely depends on Δn.

**Fact.** At the main point, Rytov differs from the scalar `exact` field by 6.0 %, and first Born
by 59 %. Over the sweep, Rytov's error grows with Δn, from about 0.2 % at Δn = 5e-4 to 14–34 % at
Δn = 0.05, and depends little on the radius up to 4 µm. This error is a property of the
approximation, not of either code.

## Conventions that had to be aligned

Each of these produced a disagreement before it was aligned. None of them is a bug in either
package.

| Convention | QoraHolo | qpsphere | Effect if not aligned (fact, main point) |
|---|---|---|---|
| Sphere centre on an even grid | pixel n/2 is r = 0 | (n − 1)/2, between pixels | half-pixel shift; the script moves QoraHolo's sphere by −pixel/2 in y and x |
| Period of the field | periodic in the n × n window | zero-pads by `zeropad` (default 5) before the inverse FFT, then crops | 2.7 % (phase RMSE 2.4 mrad, amplitudes equal to 2e-8) |
| Band limit | detection NA | none | NA = n_m is used for full-band comparisons; both fields are low-passed for the NA study |
| Plane | z = 0 through the sphere centre | `focus` measured from the centre | `focus = 0` |
| Output | u / u_in | u / u_in | none |

**Period.** The field outside the window is not zero, and its periodic images add to the field
inside. With the window widened and the centre cropped, the deviation falls with the window size
(fact):

| QoraHolo window | Deviation from qpsphere (zeropad 5, a 282 µm period) |
|---|---|
| 56 µm (512 px) | 2.7 % |
| 113 µm (1024 px) | 1.2 % |
| 225 µm (2048 px) | 0.65 % |
| 282 µm (2560 px, the same period) | 4.9e-6 |

The scalar `exact` field moves by 1.4 % between the 512 and 2560 windows. **Inference:** the
in-focus scattered field of a sphere decays slowly enough that a window about 10× the sphere
diameter still carries percent-level wrap-around. That is a property of the sampled problem
rather than of either model. Users comparing against a zero-padded code should match the period,
or widen the QoraHolo window and crop.

**Vector amplitudes.** The vector field is built from f_x = (i/k_m)(S2 cos θ cos² φ + S1 sin² φ).
That is the co-polarised far-field amplitude for x-polarised light, with Bohren & Huffman's S1 and
S2 and time dependence e^{−iωt}. It is refocused to z = 0 through the relation QoraHolo uses,
Û_s(ν) = i f / ν_z.

The refocusing is re-implemented in the script and checked first: fed the first-Born amplitude,
it reproduces `forward_fields(model="born")` to 1.2e-6 (fact).

miepython's normalisations were checked at m = 1.001 and x = 0.05, in the forward direction,
against S = −i x³(m² − 1)/3 (fact):

- `norm="wiscombe"` returns the complex conjugate of that S, which is Wiscombe's e^{+iωt}
  convention, to 0.07 %;
- `norm="bohren"` returns −2 times it, because that option normalises the phase function and not
  the amplitudes.

Bohren & Huffman's small-sphere amplitude is S = −i x³(m² − 1)/(m² + 2). The reference above
replaces m² + 2 by 3, so the 0.07 % agreement holds only because the Lorentz-Lorenz factor
(m² + 2)/3 is 1.0007 at m = 1.001. At the main point's m = 1.015 that factor alone would be a 1 %
difference. The check fixes the sign, the conjugation and the factor of 2; it does not test the
amplitudes at the main point, which come from the full Mie series.

The script therefore uses the conjugate of the `wiscombe` amplitudes. Using `bohren` directly
gives a field −2× too large, and using `wiscombe` unconjugated gives its complex conjugate.

## Scalar against vector, by NA

Main sphere, both fields low-passed to the NA, metric against the vector field (fact):

| NA | scalar exact vs vector | Rytov vs vector | phase RMSE inside the sphere, exact vs vector (rad) |
|---|---|---|---|
| 0.2 | 0.21 % | 1.2 % | 0.0013 |
| 0.4 | 0.37 % | 1.8 % | 0.0021 |
| 0.6 | 0.59 % | 2.3 % | 0.0030 |
| 0.8 | 0.81 % | 2.8 % | 0.0038 |
| 0.9 | 0.96 % | 3.1 % | 0.0035 |
| 1.0 | 1.1 % | 3.3 % | 0.0034 |
| 1.1 | 1.3 % | 3.5 % | 0.0029 |
| 1.2 | 1.6 % | 3.8 % | 0.0032 |
| 1.337 (full band) | 2.5 % | 5.6 % | 0.0033 |

**Inference.**

- In the weak-scattering limit, S1 ≈ f and S2 ≈ f cos θ. The co-polar amplitude is then
  f (1 − sin²θ cos²φ): the scalar amplitude times the projection of a dipole onto x. The gap
  therefore grows with the angles the pupil admits.
- A smaller sphere scatters more of its light at wide angles, so the gap is larger for it. This
  matches the sweep: 4.4 % at 1 µm against 0.7 % at 6 µm, both at NA 1.2.
- Which error is larger depends on Δn. At the main point (Δn = 0.02), the scalar model's error at
  NA 1.2 (1.6 %) is a quarter of the Rytov approximation's (6.0 %), so there the Rytov step is the
  larger error in the reconstruction chain. At Δn = 5e-4 the order reverses: Rytov is 0.2 % from
  the scalar exact field, below the 1.4 % scalar-vector gap of the same sphere at NA 1.2. Rytov's
  error grows with Δn while the scalar gap barely does, so for weakly scattering objects at high
  NA the scalar model is the larger error.

## Sweep

Metrics are relative field errors as above. "Rytov vs Rytov" is QoraHolo against qpsphere at the
same period. The last three columns are the scalar `exact` field against vector Mie at NA 0.6, 0.9
and 1.2. Centre phase is 2π Δn 2R / λ.

| radius (µm) | Δn | centre phase (rad) | Rytov vs Rytov | Rytov vs scalar exact | exact vs vector, NA 0.6 | NA 0.9 | NA 1.2 |
|---|---|---|---|---|---|---|---|
| 1 | 0.0005 | 0.01 | 7.3e-05 | 0.001 | 0.0196 | 0.0277 | 0.0438 |
| 1 | 0.002 | 0.05 | 1.6e-05 | 0.006 | 0.0196 | 0.0277 | 0.0437 |
| 1 | 0.005 | 0.12 | 6.4e-06 | 0.014 | 0.0194 | 0.0275 | 0.0437 |
| 1 | 0.01 | 0.24 | 4.5e-06 | 0.028 | 0.0192 | 0.0273 | 0.0435 |
| 1 | 0.02 | 0.47 | 3.7e-06 | 0.057 | 0.0188 | 0.0269 | 0.0433 |
| 1 | 0.05 | 1.18 | 3.5e-06 | 0.145 | 0.0180 | 0.0264 | 0.0431 |
| 2 | 0.0005 | 0.02 | 7.3e-05 | 0.002 | 0.0088 | 0.0145 | 0.0208 |
| 2 | 0.002 | 0.09 | 1.3e-05 | 0.006 | 0.0088 | 0.0144 | 0.0208 |
| 2 | 0.005 | 0.24 | 2.3e-06 | 0.015 | 0.0087 | 0.0144 | 0.0209 |
| 2 | 0.01 | 0.47 | 1.9e-06 | 0.031 | 0.0086 | 0.0143 | 0.0210 |
| 2 | 0.02 | 0.94 | 1.8e-06 | 0.063 | 0.0084 | 0.0143 | 0.0215 |
| 2 | 0.05 | 2.36 | 2.0e-06 | 0.153 | 0.0088 | 0.0150 | 0.0248 |
| 3 | 0.0005 | 0.04 | 7.3e-05 | 0.002 | 0.0060 | 0.0093 | 0.0140 |
| 3 | 0.002 | 0.14 | 1.3e-05 | 0.006 | 0.0059 | 0.0093 | 0.0140 |
| 3 | 0.005 | 0.35 | 1.4e-06 | 0.015 | 0.0059 | 0.0093 | 0.0142 |
| 3 | 0.01 | 0.71 | 1.3e-06 | 0.030 | 0.0059 | 0.0094 | 0.0146 |
| 3 | 0.02 | 1.42 | 1.3e-06 | 0.060 | 0.0059 | 0.0096 | 0.0156 |
| 3 | 0.05 | 3.54 | 1.7e-06 | 0.144 | 0.0072 | 0.0122 | 0.0213 |
| 4 | 0.0005 | 0.05 | 7.3e-05 | 0.002 | 0.0046 | 0.0070 | 0.0106 |
| 4 | 0.002 | 0.19 | 1.3e-05 | 0.006 | 0.0045 | 0.0070 | 0.0106 |
| 4 | 0.005 | 0.47 | 1.1e-06 | 0.015 | 0.0045 | 0.0070 | 0.0108 |
| 4 | 0.01 | 0.94 | 1.1e-06 | 0.030 | 0.0045 | 0.0071 | 0.0111 |
| 4 | 0.02 | 1.89 | 1.2e-06 | 0.059 | 0.0046 | 0.0075 | 0.0123 |
| 4 | 0.05 | 4.72 | 2.1e-06 | 0.171 | 0.0069 | 0.0118 | 0.0224 |
| 6 | 0.0005 | 0.07 | 7.3e-05 | 0.002 | 0.0030 | 0.0047 | 0.0070 |
| 6 | 0.002 | 0.28 | 1.3e-05 | 0.006 | 0.0030 | 0.0047 | 0.0070 |
| 6 | 0.005 | 0.71 | 1.1e-06 | 0.016 | 0.0030 | 0.0047 | 0.0072 |
| 6 | 0.01 | 1.42 | 1.1e-06 | 0.031 | 0.0030 | 0.0049 | 0.0077 |
| 6 | 0.02 | 2.83 | 1.3e-06 | 0.060 | 0.0034 | 0.0058 | 0.0097 |
| 6 | 0.05 | 7.09 | 4.3e-06 | 0.338 | 0.0092 | 0.0162 | 0.0339 |

## What this does not show

- **One on-axis illumination.** Tilted illuminations go through the same code path with the
  illumination frequency added. They were not compared, because qpsphere models on-axis light
  only.
- **No objective.** The vector reference is the co-polarised component of the field in the
  medium. A high-NA objective rotates polarisation and apodises the pupil before the camera, so
  the measured hologram can differ from both fields compared here (inference).
- **Not qpsphere's Mie model.** qpsphere also has a Mie model (BHFIELD), but it was not usable
  here:
  - double precision: NaN coefficients once the grid reaches beyond about 4 µm from a 3 µm
    sphere (fact: a span of ±3 µm works, ±5 µm does not);
  - arbitrary precision: 0.1 s per pixel (fact), so a 512² field would take hours.

  The vector reference is therefore miepython's amplitudes, refocused by this script, which
  shares QoraHolo's refocusing relation but no code with it.
- **Synthetic only.** Every number is a comparison of models; none involves a measurement.

## Problems found in qpsphere 0.5.9

Neither affects its physics; both stop it from running in a current environment. Neither was
found in qpsphere's issue list on 2026-10-26.

- `qpsphere.models.rytov()` resamples with `scipy.interpolate.interp2d`, which SciPy 1.14
  removed, so the call fails with SciPy ≥ 1.14. The script calls its field function
  `sphere_prop_fslice_bessel` directly.
- `qpsphere/models/_bhfield/fetch.py` imports `pkg_resources`, which comes from setuptools. qpsphere
  does not declare setuptools as a dependency, so in a fresh environment `import qpsphere` fails.
  setuptools 80.10 warns that `pkg_resources` "is slated for removal" and advises pinning
  `setuptools<81`, which the `check` group does.
