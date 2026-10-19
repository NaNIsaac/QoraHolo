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
nucleus. Field extraction to GP took 0.23 s on CUDA and 1.80 s on CPU through the per-call
functions; the TV reconstruction took 0.42 s and 10.3 s. The TV weight (1e-2, the default)
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

p = qh.Pipeline(
    49, 512, illuminations=illuminations, geometry=geom, iterations=20, backend="cuda", **opts
)
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
qoraholo_bench --field 256 --repeat 5            # 512² holograms, 49 angles, 256³, 10 GP iterations
qoraholo_bench --field 512 --repeat 3 --cpu 0    # 1024² holograms, 512³, CUDA only
qoraholo_bench --field 256 --tv 1                # only the TV kernel comparison
```

Measured on an AMD Ryzen 5 7500F (6 cores) and an NVIDIA GeForce RTX 5070 (12 GB), Windows 11,
MSVC 19.44, CUDA 13.4, driver 610.88, one build with both backends. Milliseconds; medians of 5
runs at 256³ and 3 at 512³. Both runs held an exclusive lock on the GPU (the machine is
shared with another project's benchmarks), and `nvidia-smi`, sampled every 5 s, showed no
other compute process. 512³ is CUDA only: the CPU solver runs at that size need several GiB
of host memory. No profiler numbers: Nsight Compute needs GPU performance counter access,
which is not enabled on this machine (`ERR_NVGPUCTRPERM`), so all times are cudaEvent
intervals or host wall time.

**End to end**, sample holograms in, index out, 10 Gerchberg-Papoulis iterations:

| | 256³ CPU | 256³ CUDA | 512³ CUDA |
|---|---:|---:|---:|
| per-call functions, host in / host out (plans created per call) | 1071 | 82 | 440 |
| `Pipeline::run()`, copies into and out of its buffers | 1022 | 40 | 329 |
| `Pipeline::process()`, page-locked input and output | 974 | 28 | 221 |
| `Pipeline` constructor (allocations, cuFFT plans), once | 3.78 | 20 | 68 |
| `Pipeline::set_background`, once | 61 | 5.22 | 41 |
| `Pipeline` device memory | - | 0.42 GiB | 2.39 GiB |

`process()` on CUDA, by stage (cudaEvent):

| stage | 256³ | 512³ |
|---|---:|---:|
| upload of the 49 sample holograms (49 / 196 MiB, page-locked) | 1.78 | 7.13 |
| field extraction, from the start of the upload | 2.27 | 9.58 |
| ratio, least-squares unwrap, Rytov | 2.13 | 8.15 |
| 2D FFT | 0.05 | 0.69 |
| zero the volume and hit counts | 0.30 | 2.54 |
| Ewald mapping | 0.10 | 0.65 |
| normalise | 0.16 | 1.20 |
| 10 GP iterations (R2C/C2R) + final inverse | 20 | 177 |
| potential to index | 0.24 | 1.87 |
| download of the volume (64 / 512 MiB, page-locked) | 2.42 | 19 |

Against the per-call chain the Pipeline saves the cuFFT plan creation, the pageable copies
between stages, the second field extraction for the background, and the host-side
`field_ratio`. What is left is dominated by the GP loop.

Copies on their own, same sizes:

| transfer | pageable | page-locked |
|---|---:|---:|
| sample holograms to device, 49 MiB | 2.64 | 1.82 |
| index volume to host, 64 MiB | 3.86 | 2.37 |
| sample holograms to device, 196 MiB | 11.35 | 7.16 |
| index volume to host, 512 MiB | 29.78 | 18.83 |

Splitting the upload so that field extraction overlaps it (`upload_chunks`) barely matters.
Upload plus field extraction with 1, 7 and 49 parts: 2.21, 1.93 and 2.66 ms at 256³, 9.58,
8.08 and 8.00 ms at 512³; `process()` 27.5, 27.4, 28.1 and 220.1, 218.0, 218.2 ms. The copy
is short and the R2C and crop are shorter, so there is little to hide, and with many small
parts the per-copy overhead wins at 256³. The page-locked buffers are the part that helps.

**Gerchberg-Papoulis on half spectra**, 10 iterations, same input:

| | 256³ CPU | 256³ CUDA | 512³ CUDA |
|---|---:|---:|---:|
| C2C | 1260 ms | 36 ms | 296 ms |
| R2C/C2R | 742 ms | 20 ms | 176 ms |
| memory, C2C | 0.31 GiB | 0.32 GiB | 2.50 GiB |
| memory, R2C/C2R | 0.26 GiB | 0.20 GiB | 1.57 GiB |

CUDA memory is measured with `cudaMemGetInfo` for the whole reconstruction including the
cuFFT work area; CPU memory is the peak of the arrays the solver allocates. The R2C index
differs from the C2C index by 5.3e-6 of the contrast at both sizes, and RMSE against the
phantom and nucleus mean are the same to the printed digits (1.002e-3 and 1.3742 at 256³,
9.58e-4 and 1.3745 at 512³).

**Missing-cone solvers**, the per-call `reconstruct` on the same Rytov fields (CPU: one run),
TV weight 1e-2:

| solver | 256³ CPU | 256³ CUDA | RMSE | nucleus (1.3750) | 512³ CUDA | RMSE | nucleus |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rytov ODT, no iterations | 130 | 20 | 2.37e-3 | 1.3683 | 137 | 2.35e-3 | 1.3685 |
| GP, 20 iterations | 1360 | 54 | 9.21e-4 | 1.3757 | 447 | 8.72e-4 | 1.3761 |
| GP, 100 iterations | 6560 | 205 | 9.72e-4 | 1.3772 | 1754 | 8.93e-4 | 1.3773 |
| FISTA-TV, 20 outer x 10 dual iterations | 10092 | 418 | 7.69e-4 | 1.3762 | 3265 | 8.07e-4 | 1.3780 |

These CUDA times include the upload, the download and plan creation of the per-call
function. More GP iterations do not help this phantom. TV lowers the RMSE by 17% at 256³
and 7.5% at 512³, at 7 to 8 times the cost of 20 GP iterations on CUDA. The TV result depends
on the weight and on the number of dual steps: in a sweep on the README phantom (CUDA, 20
outer iterations) weights 3e-4, 1e-3, 3e-3 and 1e-2 gave RMSE 8.89e-4, 8.48e-4, 8.05e-4 and
7.69e-4, and at 1e-2 with 50 outer iterations the RMSE rose to 1.08e-3; 5 dual steps instead
of 10 gave 9.38e-4, 20 gave 7.99e-4.

The FGP step of the TV solver is one fused kernel; against the earlier two-kernel version it
takes 1.62 instead of 2.13 ms at 256³ and 12.6 instead of 16.8 ms at 512³, with the same
result bit for bit. Shared-memory tiling was tried and did not help. Details and the
measurements are in [docs/tv_kernels.md](docs/tv_kernels.md).

Per-call functions at 256³, by stage:

| stage | CPU | CUDA |
|---|---:|---:|
| field extraction, 49 sample + 49 background holograms | 135 | 38 |
| Rytov phase incl. least-squares unwrap, 49 fields | 117 | 12 |
| ODT: 2D FFT of the Rytov fields | 8.46 | 0.12 |
| ODT: zero the volume and hit counts | - | 0.30 |
| ODT: Ewald mapping, gather (column owner, no atomics) | 29 | 0.09 |
| ODT: Ewald mapping, scatter (atomicAdd) | - | 0.10 |
| ODT: normalise | 2.99 | 0.16 |
| ODT: 10 GP iterations + final 3D IFFT, R2C/C2R | 742 | 20 |
| ODT: potential to refractive index | 10 | 0.23 |
| ODT: host-device copies | - | 6.10 |
| ODT: allocations and cuFFT plans (host) | - | 5.21 |

CUDA field and Rytov times in this table are wall-clock and include copies and plan creation.
Relative difference of the index contrast between CUDA and CPU: 3.8e-6 (gather and scatter),
3.9e-6 through the Pipeline. At 512³ there is no CPU run; scatter and gather differ by
2.8e-6, and the Pipeline and the per-call function by 3.2e-6.

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
