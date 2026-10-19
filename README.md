# QoraHolo

[![CI](https://github.com/NaNIsaac/QoraHolo/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/QoraHolo/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/NaNIsaac/QoraHolo/blob/main/LICENSE)

**Holotomography reconstruction in C++20 and CUDA: off-axis holograms in, a 3D refractive-index volume out.**

Holotomography (optical diffraction tomography, ODT) records off-axis holograms of a
transparent sample under many illumination angles and reconstructs its 3D refractive index.
QoraHolo implements the standard pipeline: field retrieval from the hologram sideband,
least-squares phase unwrapping, the Rytov approximation, Ewald-sphere mapping by the Fourier
diffraction theorem, and two missing-cone solvers: Gerchberg-Papoulis with non-negativity and
FISTA with a total-variation term. Every stage has a CPU implementation (pocketfft, OpenMP)
and a CUDA implementation (cuFFT, custom kernels); on the benchmark volumes the two differ by
4e-6 to 4e-5 relative. `Pipeline` keeps plans and buffers alive between calls and on CUDA
runs from holograms to index with one upload and one download. A sphere phantom with an
analytic forward model and a hologram synthesiser make the whole chain testable against a
known answer. Python bindings take and return numpy arrays.

![Hologram, unwrapped phase, and x-y / x-z slices of the phantom and its reconstructions](https://raw.githubusercontent.com/NaNIsaac/QoraHolo/main/docs/reconstruction.png)

A synthetic cell (body n = 1.360, nucleus 1.375, two vesicles 1.390, medium 1.337),
49 illuminations at NA 0.9, 512² holograms with Gaussian noise (σ = 1% of the mean
intensity), fields cropped to 256²,
reconstructed into a 256³ volume. Illumination angles are estimated from the background
holograms, not taken from the simulation (largest error 4.3e-6 cycles/µm). Mean index in
the nucleus: 1.3683 from Rytov ODT, 1.3757 after 20 Gerchberg-Papoulis iterations (true
1.3750), 1.3762 after 20 FISTA-TV iterations with 10 dual steps each. RMSE against the
phantom over the whole volume: 2.37e-3, 9.25e-4 and 7.69e-4. The x-z slices show the
missing-cone elongation along z, how much of it the non-negativity constraint removes, and
that TV removes most of the rest and the ringing, with overshoot at the top and bottom of the
nucleus. Field extraction to GP took 0.26 s on CUDA and 1.80 s on CPU through the per-call
functions; the TV reconstruction took 0.52 s and 10.3 s. The TV weight (1e-2, the default)
was picked from a sweep on this phantom, so the TV numbers are tuned to it. Script:
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
opt.iterations = 20;  // Gerchberg-Papoulis; opt.solver = Solver::tv for FISTA-TV
opt.backend = cuda_available() ? Backend::cuda : Backend::cpu;
const OdtResult r = reconstruct(rytov(ratio, opt.backend), illum, geom, opt);
// r.ri: (n, n, n) refractive index, centre-origin
```

For repeated reconstructions with the same geometry (a time series), `Pipeline` allocates
once and keeps the background fields:

```cpp
PipelineOptions po;
po.count = 49;
po.hologram_size = 512;
po.field.center = {141.0f, 141.0f};  // reference carrier, pixels from DC
po.field.radius = 65.0f;
po.field.output_size = 256;
po.illuminations = illum;
po.geometry = geom;                  // field pixel
po.odt = opt;
Pipeline p(po);
p.set_background(backgrounds);       // (49, 512, 512)
for (const auto& frame : frames) {
  std::ranges::copy(frame.data, p.input().begin());  // or write into input() directly
  p.process();
  use(p.output());                   // (256, 256, 256), valid until the next process()
}
```

`examples/quickstart.cpp` is this snippet with a synthetic bead as input. It prints
`n at the centre 1.3639 (true 1.3600), k-space filled 19.0%` on both backends.

Python (3.12+), built with scikit-build-core and nanobind:

```bash
uv build --wheel                                    # CPU only
CMAKE_ARGS="-DQORAHOLO_CUDA=ON" uv build --wheel    # with the CUDA backend
```

On Windows the CUDA build loads cuFFT from `%CUDA_PATH%\bin`.

```python
import qoraholo as qh

geom = qh.Geometry(wavelength=0.532, n_medium=1.337, na=1.2, pixel=0.11)
opts = dict(center=(141, 141), radius=65, output_size=256)  # carrier, crop
fields = qh.field_ratio(
    qh.extract_fields(holograms, **opts),  # (49, 512, 512) float
    qh.extract_fields(backgrounds, **opts),
)
ri = qh.reconstruct(qh.rytov(fields), illuminations, geom, iterations=20, backend="cuda")

p = qh.Pipeline(49, 512, illuminations=illuminations, geometry=geom, iterations=20,
                backend="cuda", **opts)
p.set_background(backgrounds)
ri = p.run(holograms)  # p.timings: milliseconds per stage of the last run
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
and k-space, where measured voxels are reset to the data. Because the constrained O is real,
the loop runs on R2C half spectra by default; measured data that is not Hermitian is handled
so that the iterates equal those of the full C2C loop, and only the imaginary part of the
last inverse transform (absorption) is dropped (`gp_transform = c2c` keeps it). The TV solver
minimises ½‖M(F O − P)‖² + λ TV(O) with O ≥ 0 by FISTA (Beck & Teboulle 2009), whose
gradient step is the Gerchberg-Papoulis k-space step and whose proximal step is the fast
gradient projection on the TV dual (Beck & Teboulle 2009; TV after Rudin, Osher & Fatemi
1992); λ = `tv_weight` · k₀². Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
qoraholo_bench --field 256 --repeat 5     # 512² holograms, 49 angles, 256³, 10 GP iterations
qoraholo_bench --field 512 --repeat 3     # 1024² holograms, 49 angles, 512³
```

Measured on an AMD Ryzen 5 7500F (6 cores) and an NVIDIA GeForce RTX 5070 (12 GB), Windows 11,
MSVC 19.44, CUDA 13.4, driver 610.88, one build with both backends. Milliseconds, medians of 5
runs at 256³. No other process was computing on the GPU at the start or the end of the run
(checked with `nvidia-smi`). No profiler numbers: Nsight Compute needs GPU performance
counter access, which is not enabled on this machine (`ERR_NVGPUCTRPERM`), so all times are
cudaEvent intervals or host wall time.

**End to end at 256³**, sample holograms in, index out, 10 Gerchberg-Papoulis iterations:

| | CPU | CUDA |
|---|---:|---:|
| per-call functions, host in / host out (plans created per call) | 993 | 82 |
| `Pipeline::run()`, copies into and out of its buffers | 964 | 41 |
| `Pipeline::process()`, page-locked input and output | 998 | 28 |
| `Pipeline` constructor (allocations, cuFFT plans), once | 3.75 | 53 |
| `Pipeline::set_background`, once | 59 | 8.07 |

`process()` on CUDA, by stage (cudaEvent):

| stage | ms |
|---|---:|
| upload of the 49 sample holograms (49 MiB, page-locked) | 1.78 |
| field extraction, from the start of the upload | 2.22 |
| ratio, least-squares unwrap, Rytov | 2.11 |
| 2D FFT | 0.05 |
| zero the volume and hit counts | 0.30 |
| Ewald mapping | 0.09 |
| normalise | 0.16 |
| 10 GP iterations (R2C/C2R) + final inverse | 20 |
| potential to index | 0.24 |
| download of the volume (64 MiB, page-locked) | 2.41 |

The Pipeline holds 0.42 GiB of device memory at 256³ (`cudaMemGetInfo` before and after
construction). Against the per-call chain it saves the cuFFT plan creation (about 5 ms for
the reconstruction alone), the pageable copies between stages, the second field extraction
for the background, and the host-side `field_ratio`. What is left is dominated by the GP
loop.

Copies on their own, same sizes:

| transfer | pageable | page-locked |
|---|---:|---:|
| sample holograms to device, 49 MiB | 2.94 | 1.83 |
| index volume to host, 64 MiB | 8.73 | 2.37 |

Splitting the upload so that field extraction overlaps it (`upload_chunks`) barely matters
here: upload plus field extraction took 2.21 ms with one copy, 1.97 ms with 7 parts and
2.72 ms with 49 parts; `process()` 27.3, 27.3 and 27.7 ms. The copy is short and the R2C
and crop are shorter, so there is little to hide, and with 49 parts the per-copy overhead
wins. Page-locked buffers are the part that helps.

**Gerchberg-Papoulis on half spectra**, 256³, 10 iterations, same input:

| | CPU | CUDA |
|---|---:|---:|
| C2C | 1239 ms | 36 ms |
| R2C/C2R | 692 ms | 20 ms |
| memory, C2C | 0.31 GiB | 0.32 GiB |
| memory, R2C/C2R | 0.26 GiB | 0.20 GiB |

CUDA memory is measured for the whole reconstruction including the cuFFT work area; CPU
memory is the peak of the arrays the solver allocates. The R2C index differs from the C2C
index by 5.3e-6 of the contrast, and RMSE against the phantom (1.002e-3) and nucleus mean
(1.3742) are the same to the printed digits.

**Missing-cone solvers**, 256³, the per-call `reconstruct` on the same Rytov fields
(CPU: one run, CUDA: median of 5), TV weight 1e-2:

| solver | CPU (ms) | CUDA (ms) | RMSE | nucleus mean (true 1.3750) |
|---|---:|---:|---:|---:|
| Rytov ODT, no iterations | 128 | 17 | 2.37e-3 | 1.3683 |
| GP, 20 iterations | 1264 | 54 | 9.21e-4 | 1.3757 |
| GP, 100 iterations | 6227 | 203 | 9.72e-4 | 1.3772 |
| FISTA-TV, 20 outer x 10 dual iterations | 10088 | 514 | 7.69e-4 | 1.3762 |

These CUDA times include the upload and plan creation of the per-call function. More GP
iterations do not help this phantom; TV lowers the RMSE by 17% at 8 to 10 times the cost
of 20 GP iterations. The TV result depends on the weight and on the number of dual steps:
in a sweep on the README phantom (CUDA, 20 outer iterations) weights 3e-4, 1e-3, 3e-3 and
1e-2 gave RMSE 8.89e-4, 8.48e-4, 8.05e-4 and 7.69e-4, and at 1e-2 with 50 outer iterations
the RMSE rose to 1.08e-3; 5 dual steps instead of 10 gave 9.38e-4, 20 gave 7.99e-4.

Per-call functions at 256³, by stage:

| stage | CPU | CUDA |
|---|---:|---:|
| field extraction, 49 sample + 49 background holograms | 136 | 37 |
| Rytov phase incl. least-squares unwrap, 49 fields | 116 | 13 |
| ODT: 2D FFT of the Rytov fields | 9.11 | 0.08 |
| ODT: zero the volume and hit counts | - | 0.30 |
| ODT: Ewald mapping, gather (column owner, no atomics) | 28 | 0.09 |
| ODT: Ewald mapping, scatter (atomicAdd) | - | 0.10 |
| ODT: normalise | 2.91 | 0.16 |
| ODT: 10 GP iterations + final 3D IFFT, R2C/C2R | 692 | 20 |
| ODT: potential to refractive index | 13 | 0.24 |
| ODT: host-device copies | - | 5.86 |
| ODT: allocations and cuFFT plans (host) | - | 4.53 |

CUDA field and Rytov times in this table are wall-clock and include copies and plan creation.
Relative difference of the index contrast between CUDA and CPU: 3.8e-6 (gather and scatter),
3.9e-6 through the Pipeline.

**512³** (1024² holograms). The final 512³ run was stopped before it started measuring
because the machine ran low on memory, so these numbers come from two earlier runs of the
same benchmark, the first before the R2C loop existed (other GPU use was not checked then).
During the second, another process
(a different project's benchmark) was computing on the same GPU, and its per-call CUDA times
came out about twice as slow as in the first run; those are not listed. Single numbers here
are less reliable than the 256³ tables.

| | C2C (first run) | R2C/C2R (second run) |
|---|---:|---:|
| `Pipeline::process()`, CUDA | 346 ms | 221 ms |
| of which 10 GP iterations + final inverse | 301 ms | 177 ms |
| `Pipeline` device memory | 3.32 GiB | 2.39 GiB |
| per-call chain, CUDA | 605 ms | - |
| 10 GP iterations, CPU | 10640 ms | 6594 ms (C2C in the same run: 11437 ms) |
| reconstruction memory, CUDA | - | 1.57 GiB (C2C in the same run: 2.50 GiB) |

Page-locked copies at 512³: 7.2 ms for the 196 MiB of holograms (15.5 ms pageable), 18.8 ms
for the 512 MiB volume (33.3 ms pageable). Relative CUDA-CPU difference of the index
contrast: 3.6e-5.

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
- One regulariser: isotropic TV. No edge-preserving or learned priors, no automatic choice of
  the TV weight.
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
- A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding algorithm for linear
  inverse problems", SIAM J. Imaging Sci. 2, 183 (2009).
- A. Beck and M. Teboulle, "Fast gradient-based algorithms for constrained total variation
  image denoising and deblurring problems", IEEE Trans. Image Process. 18, 2419 (2009).
- L. I. Rudin, S. Osher and E. Fatemi, "Nonlinear total variation based noise removal
  algorithms", Physica D 60, 259 (1992).
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
