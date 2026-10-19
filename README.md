# QoraHolo

[![CI](https://github.com/NaNIsaac/QoraHolo/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/QoraHolo/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/NaNIsaac/QoraHolo/blob/main/LICENSE)

**Holotomography reconstruction in C++20 and CUDA: off-axis holograms in, a 3D refractive-index volume out.**

Holotomography (optical diffraction tomography, ODT) records off-axis holograms of a
transparent sample under many illumination angles and reconstructs its 3D refractive index.
QoraHolo implements the standard pipeline: field retrieval from the hologram sideband,
least-squares phase unwrapping, the Rytov approximation, Ewald-sphere mapping by the Fourier
diffraction theorem, and Gerchberg-Papoulis iterations against the missing cone. Every stage
has a CPU implementation (pocketfft, OpenMP) and a CUDA implementation (cuFFT, custom
kernels); on the benchmark volumes the two differ by 4e-6 to 4e-5 relative. A sphere phantom with an analytic forward model
and a hologram synthesiser make the whole chain testable against a known answer. Python
bindings take and return numpy arrays.

![Hologram, unwrapped phase, and x-y / x-z slices of the phantom and its reconstructions](https://raw.githubusercontent.com/NaNIsaac/QoraHolo/main/docs/reconstruction.png)

A synthetic cell (body n = 1.360, nucleus 1.375, two vesicles 1.390, medium 1.337),
49 illuminations at NA 0.9, 512² holograms with Gaussian noise (σ = 1% of the mean
intensity), fields cropped to 256²,
reconstructed into a 256³ volume. Illumination angles are estimated from the background
holograms, not taken from the simulation (largest error 4.3e-6 cycles/µm). Mean index in
the nucleus: 1.3683 from Rytov ODT, 1.3757 after 20 Gerchberg-Papoulis iterations (true
1.3750). RMSE against the phantom over the whole volume: 2.37e-3, then 9.25e-4. The x-z
slices show the missing-cone elongation along z and how much of it the non-negativity
constraint removes. The full pipeline took 0.42 s on CUDA and 5.76 s on CPU. Script:
`examples/readme_figure.py`.

## Quick start

Requires CMake 3.28+, a C++20 compiler (gcc 13+, MSVC 2022), and for the GPU backend the
CUDA Toolkit 12.8 or newer.

```bash
cmake --preset cpu-release          # or cuda-release
cmake --build --preset cpu-release
ctest --preset cpu-release
```

```cpp
#include "qoraholo/qoraholo.hpp"
using namespace qoraholo;

const Geometry geom{0.532f, 1.337f, 1.2f, 0.2f};  // wavelength, n_medium, NA, pixel (um)
const auto illum = circular_scan(49, 0.9f, geom);   // or your measured illumination list
// ratio: (49, n, n) field ratios u / u_background, e.g. from extract_fields + field_ratio
OdtOptions opt;
opt.gp_iterations = 20;
opt.backend = cuda_available() ? Backend::cuda : Backend::cpu;
const OdtResult r = reconstruct(rytov(ratio, opt.backend), illum, geom, opt);
// r.ri: (n, n, n) refractive index, centre-origin
```

`examples/quickstart.cpp` is this snippet with a synthetic bead as input. It prints
`n at the centre 1.3639 (true 1.3600), k-space filled 19.0%` on both backends.

Python (3.12+), built with scikit-build-core and nanobind:

```bash
uv build --wheel                                    # CPU only
CMAKE_ARGS="-DQORAHOLO_CUDA=ON" uv build --wheel    # with the CUDA backend
```

On Windows the CUDA build loads cuFFT from `%CUDA_PATH%in`.

```python
import qoraholo as qh

geom = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.11)
opts = dict(center=(141, 141), radius=65, output_size=256)  # carrier, crop
fields = qh.field_ratio(
    qh.extract_fields(holograms, **opts),  # (49, 512, 512) float
    qh.extract_fields(backgrounds, **opts),
)
ri = qh.reconstruct(qh.rytov(fields), illuminations, geom, gp_iterations=20, backend="cuda")
```

## Array conventions

| | |
|---|---|
| **Layout** | Row-major `(z, y, x)`, x fastest. A stack of images is `(count, n, n)`. |
| **Origin** | Every real-space array is centre-origin: index `n // 2` is r = 0. |
| **Sizes** | Square images with an even side; the volume is `(nz, n, n)` with the field's pixel as voxel. |
| **Units** | Lengths in µm, spatial frequencies in cycles/µm (not rad/µm). |
| **Illumination** | Lateral frequency `(fy, fx)` of each incident plane wave in the medium, magnitude below n_m/λ. |
| **Fields** | `field_ratio` returns u / u_background; `rytov` returns log\|r\| + i·unwrap(arg r). |

## Physics conventions

These are the choices a reader has to know to compare with another implementation.

**Scattering potential.** The scalar Helmholtz equation is written as
(∇² + k_m²) u = −O u with O(r) = k₀² (n(r)² − n_m²), k₀ = 2π/λ, k_m = n_m k₀, time
dependence e^{−iωt}. The output is Re √(n_m² + O/k₀²).

**Fourier transform.** Û(ν) = ∫ u(r) e^{−2πi ν·r} dr, with ν in cycles per length. Discrete
spectra are DFTs of centre-origin arrays times the pixel area (or voxel volume).

**Fourier diffraction theorem** (Wolf 1969; Kak & Slaney 1988, ch. 6; Sung et al. 2009).
For an incident plane wave u_in = exp(2πi ν_in·r) with |ν_in| = n_m/λ and a positive z
component, the first Born scattered field, refocused to the plane z = 0 at the centre of the
volume, has the 2D spectrum

  Û_s(ν_x, ν_y) = i / (4π ν_z) · Ô(ν_x − ν_in,x, ν_y − ν_in,y, ν_z − ν_in,z),
  ν_z = √((n_m/λ)² − ν_x² − ν_y²),

for |(ν_x, ν_y)| ≤ NA/λ. The data for one illumination therefore lies on a cap of the Ewald
sphere of radius n_m/λ, shifted so that it passes through the origin. This is the relation
in Kak & Slaney and Sung et al., written with cycle frequencies; with angular frequencies the
prefactor reads i/(2k_z).

**Rytov.** The measured ratio u/u_in is replaced by φ_R = ln(u/u_in) = ln|u/u_in| + iΔφ with
Δφ unwrapped (Devaney 1981). u_in φ_R plays the role of the Born scattered field, so the
spectrum of φ_R at lateral frequency μ is Û_s(μ + ν_in). Lateral frequencies of the field and
of the volume therefore coincide bin for bin; only the axial coordinate ν_z − ν_in,z needs
gridding.

**Gridding.** Each sample goes to the nearest axial bin and voxels hit more than once are
averaged. Splitting each sample linearly between the two neighbouring axial bins and dividing
by the summed weight was implemented first and was wrong by tens of percent near the cap
apex: a sample a whole bin away from a voxel still set that voxel's value. The sphere test
caught it.

**Missing cone.** Gerchberg-Papoulis (Gerchberg 1974; Papoulis 1975; see Lim et al. 2015 for
ODT) alternates between real space, where O is set to max(Re O, 0) (n ≥ n_m, non-absorbing),
and k-space, where measured voxels are reset to the data.

**Holograms.** The synthesiser writes |R + u|² with R = a·exp(−2πi ν_c·r), so the term u R*
sits at ν_in + ν_c. `find_sideband` calls the peak in the half plane x > 0 the +1 order; flip
the sign of the carrier (or conjugate the field) if your setup puts it on the other side. For
multi-angle data, crop every hologram around the same carrier so the illumination tilt stays
in the field and cancels in u / u_background (Cuche et al. 2000 for the spatial filtering).
Sub-pixel peak positions use Jacobsen's estimator with Candan's (2011) correction.

**Unwrapping.** Unweighted least squares (Ghiglia & Romero 1994): the divergence of the
wrapped phase differences is the right-hand side of a Neumann Poisson problem, solved with a
DCT-II computed through an FFT of the reordered sequence (Makhoul 1980), on CPU and GPU alike.
The constant is chosen so the result is congruent to the input; `rytov` then removes the
2π multiple that puts the mean border phase in (−π, π], so the image border must be
background.

## Benchmark

```
qoraholo_bench --field 256 --repeat 3     # 512² holograms, 49 angles, 256³, 10 GP iterations
qoraholo_bench --field 512 --repeat 1     # 1024² holograms, 49 angles, 512³
```

Measured on an AMD Ryzen 5 7500F (6 cores) and an NVIDIA GeForce RTX 5070 (12 GB), Windows 11,
MSVC 19.44, CUDA 13.4, driver 610.88, one build with both backends. Milliseconds.

| stage | 256³ CPU | 256³ CUDA | 512³ CPU | 512³ CUDA |
|---|---:|---:|---:|---:|
| field extraction, 49 sample + 49 background holograms | 146 | 44 | 766 | 185 |
| Rytov phase incl. least-squares unwrap, 49 fields | 128 | 17 | 462 | 49 |
| ODT: 2D FFT of the Rytov fields | 11 | 1.53 | 35 | 2.17 |
| ODT: Ewald mapping, gather (column owner, no atomics) | 43 | 0.07 | 322 | 0.62 |
| ODT: Ewald mapping, scatter (atomicAdd) | - | 0.10 | - | 0.84 |
| ODT: normalise | 4.61 | 0.23 | 21 | 1.24 |
| ODT: 10 GP iterations + final 3D IFFT | 1401 | 39 | 11617 | 316 |
| ODT: potential to refractive index | 28 | 1.47 | 198 | 8.84 |
| ODT: host-device copies | - | 15 | - | 114 |

CUDA field and Rytov times are wall-clock and include host-device copies and cuFFT plan
creation. ODT sub-stages on CUDA are cudaEvent intervals on the device; the copies are listed
separately. Medians of 3 runs at 256³, single run at 512³. The CPU numbers are from the
MSVC/OpenMP build. Relative difference of the index contrast between CUDA and CPU: 4.2e-6 at
256³ and 3.6e-5 at 512³, identical for gather and scatter. No profiler (Nsight Compute) was
used; the numbers above are all there is.

The 3D FFTs of the Gerchberg-Papoulis loop dominate on both backends. Ewald mapping is a
small share either way; gather beat scatter by 26-30% at both sizes.

## Layout

```
include/qoraholo/   public headers (array, field, unwrap, odt, phantom)
src/cpu/            pocketfft + OpenMP implementations
src/cuda/           cuFFT + kernels
src/common.hpp      index and physics helpers compiled for host and device
tests/              GoogleTest; CUDA tests skip without a device
bench/              per-stage timing
python/             nanobind module and the qoraholo package
examples/           quickstart.cpp, readme_figure.py
```

Design notes and the data flow are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What this does not do

- No vectorial or multiple-scattering models (beam propagation, Lippmann-Schwinger); first
  Born / Rytov only.
- No aberration or pupil-function correction, no automatic refocusing, no axial-shift or
  illumination-intensity normalisation beyond dividing by the background field.
- No TV or other regularised missing-cone solvers; only Gerchberg-Papoulis non-negativity.
- No file I/O and no camera or instrument integration.
- Square, even-sized images only; the volume voxel equals the field pixel.

## References

- E. Wolf, "Three-dimensional structure determination of semi-transparent objects from
  holographic data", Opt. Commun. 1, 153 (1969).
- A. C. Kak and M. Slaney, *Principles of Computerized Tomographic Imaging*, IEEE Press
  (1988), ch. 6.
- A. J. Devaney, "Inverse-scattering theory within the Rytov approximation", Opt. Lett. 6,
  374 (1981).
- Y. Sung et al., "Optical diffraction tomography for high resolution live cell imaging",
  Opt. Express 17, 266 (2009).
- K. Lim et al., "Comparative study of iterative reconstruction algorithms for missing cone
  problems in optical diffraction tomography", Opt. Express 23, 16933 (2015).
- R. W. Gerchberg, "Super-resolution through error energy reduction", Opt. Acta 21, 709
  (1974); A. Papoulis, "A new algorithm in spectral analysis and band-limited extrapolation",
  IEEE Trans. Circuits Syst. 22, 735 (1975).
- D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and unweighted phase
  unwrapping that uses fast transforms and iterative methods", J. Opt. Soc. Am. A 11, 107
  (1994).
- J. Makhoul, "A fast cosine transform in one and two dimensions", IEEE Trans. Acoust. Speech
  Signal Process. 28, 27 (1980).
- E. Cuche, P. Marquet and C. Depeursinge, "Spatial filtering for zero-order and twin-image
  elimination in digital off-axis holography", Appl. Opt. 39, 4070 (2000).
- Ç. Candan, "A method for fine resolution frequency estimation from three DFT samples",
  IEEE Signal Process. Lett. 18, 351 (2011).
