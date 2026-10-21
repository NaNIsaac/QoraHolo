# Architecture

## Data flow

```
holograms (A, 2n, 2n) float ─┐
backgrounds (A, 2n, 2n)      ├─ extract_fields ─ field_ratio ─ rytov ─ reconstruct ─ ri (nz, n, n)
                             │   FFT, crop,      u / u_bg       log|r|    FFT2, Ewald map,
                             │   shift, IFFT                    + i LS    normalise, GP,
                             │                                  unwrap    IFFT3, sqrt
find_sideband(background) ───┘   carrier and per-angle illumination
```

Every per-call function takes and returns host arrays (`Array3<T>`: shape plus a
`std::vector`). The CUDA path uploads, computes, and downloads inside each call and creates
its cuFFT plans each time. That keeps the API identical across backends and makes
CPU-versus-GPU tests a one-line change; the cost is a host round trip and plan creation per
stage.

`Pipeline` runs the same stages with everything allocated once:

```
constructor   cuFFT plans for every stage, one shared work area, device buffers,
              page-locked input and output, illuminations on the device
set_background(backgrounds)   upload, R2C, crop, inverse FFT; fields stay on the device
process()     input() --H2D--> R2C, crop, inverse FFT --> ratio, LS unwrap, Rytov
              --> 2D FFT, Ewald map, solver --> index --D2H--> output()
```

All stages run on one stream, so the plans can share a single work area sized for the
largest plan. The sample holograms can be copied in parts on a second stream; each part's
R2C and crop waits on an event recorded after its copy (`upload_chunks`).

## Source layout

| file | contents |
|---|---|
| `src/api.cpp` | argument checks, crop-plan resolution, backend dispatch, `make_grid` |
| `src/pipeline.cpp` | `Pipeline` front end and its CPU implementation |
| `src/common.hpp` | `QH_HD` helpers compiled for host and device: DFT index maps, Makhoul reordering, Neumann eigenvalues, `ewald_target`, `index_from_potential` |
| `src/cpu/*.cpp` | pocketfft (C2C over chosen axes, pocketfft's own thread pool) and OpenMP loops |
| `src/cuda/*.cu` | kernels; `stages.cuh` declares the stages (`FieldStage`, `Unwrapper`, `OdtSolver`) used by both the per-call functions and `pipeline.cu`; `util.cuh` holds RAII buffers, streams, events, plans and the shared work area |
| `src/bpm.cpp` | propagation, pupil and obliquity tables of the beam propagation model, shared by both backends |
| `src/cpu/tv.cpp`, `src/cuda/tv.cu` | the TV proximal step (FGP), used by the ODT solver and the BPM inversion |
| `src/phantom.cpp` | sphere rendering, analytic and exact (partial-wave) forward models, hologram synthesis (CPU only) |

The two backends call the same `common.hpp` functions for every index or physics expression,
so a disagreement between them can only come from FFT rounding or summation order.

## Centre-origin bookkeeping

Real-space arrays have r = 0 at index n/2. For even n, moving the origin is a multiplication
by (−1)^k in the other domain, so no array is ever shifted in memory:

- `extract_fields` on the CPU multiplies the hologram by (−1)^(x+y) before the FFT
  (spectrum comes out centred), multiplies the cropped window by one sign per image
  (−1)^(c_y+c_x), and multiplies the output by (−1)^(x+y) after the inverse FFT.
- On the GPU the hologram is transformed without modulation by an R2C FFT, which stores
  k_x ≥ 0 only. The crop reads bin (k_y, k_x) directly when k_x ≥ 0 and as
  conj(H(−k_y, −k_x)) otherwise, so a crop window that straddles k_x = 0 still works
  (`Cuda.FieldExtractionAcrossZeroKx`). The per-image sign is the same as on the CPU.
- In a sample / background ratio the output sign (−1)^(x+y) and the 1/N² scale cancel, so
  `Pipeline` skips them.
- In `reconstruct` the lateral factors of the field and of the volume cancel, because both
  are centre-origin on the same lateral grid. Only (−1)^(k_z) remains, applied while mapping.

## Ewald mapping on the GPU

Both kernels compute the same `ewald_target` for each (illumination, lateral bin) pair.

- **gather**: one thread per lateral column (k_y, k_x) loops over the illuminations and adds
  into its own column of the volume. No two threads write the same voxel, so there are no
  atomics, and neighbouring threads usually land on the same k_z, so writes are mostly
  coalesced.
- **scatter**: one thread per (illumination, pixel), `atomicAdd` on the real part, the
  imaginary part and the hit count.

Measured on the RTX 5070, kernel only: 0.09 ms against 0.10 ms at 256³, 0.66 ms against
0.86 ms at 512³ (49 illuminations). Zeroing the volume and hit counts
before the mapping takes longer than the mapping itself at 256³ (0.30 ms). Either way it is
about 1% of the GPU reconstruction, which the 3D FFTs of the missing-cone loop dominate. The
results are identical to the printed precision; float summation order differs, so they are
not bitwise equal in general.

Nsight Compute at 256³ (49 illuminations): gather 68 µs, 17.4 MiB of DRAM traffic, 269 GB/s
(41% of the DRAM peak), 47% achieved occupancy; scatter 64 µs, 17.8 MiB, 295 GB/s (45%),
73% occupancy. Neither is bandwidth-bound: gather has one thread per lateral column (65536
threads, less than one full wave on the 48 SMs) and each loops over the 49 illuminations, so it waits on
dependent loads; scatter spreads the work over 49 times as many threads but pays for the
atomics. For comparison the Gerchberg-Papoulis kernels are bandwidth-bound: the constraint
`nonnegative_real` moves 108 MiB in 194 µs (582 GB/s, 88% of peak) and `restore_half` 175 MiB
in 298 µs (615 GB/s, 93%), so the GP loop is limited by its 3D FFTs and by DRAM bandwidth, not
by its kernels' code.

## Phase unwrapping on the GPU

The DCT-II along an axis is a reorder kernel, a batched 1D C2C FFT, and a twiddle kernel
(`sincospif`). Along x the transforms are contiguous and one cuFFT plan covers the whole
stack. Along y they are strided by n within an image but not across images, which a single
`cufftPlanMany` cannot express, so the same plan is executed once per image. The inverse is
the exact inverse of the forward transform (V_k = e^{iπk/2N}(X_k − iX_{N−k}), inverse FFT,
reorder back), so DCT → divide by the Neumann eigenvalues → inverse DCT solves the Poisson
problem without extra scaling. The congruence constant and the border mean are block
reductions with double-precision atomics.

## Gerchberg-Papoulis on half spectra

The C2C loop is: inverse FFT, keep max(Re O, 0), forward FFT, reset measured voxels to the
data. Taking the real part after the inverse FFT is the same as keeping the Hermitian part
H(Y)(k) = (Y(k) + conj Y(−k)) / 2 of the spectrum Y before it. After the constraint the
estimate E is the spectrum of a real volume, so E(−k) = conj E(k), and with P the mapped
data (zero where unmeasured) and M the measured mask:

| measured | H(Y)(k) |
|---|---|
| k and −k | (P(k) + conj P(−k)) / 2 |
| k only | (P(k) + E(k)) / 2 |
| −k only | (E(k) + conj P(−k)) / 2 |
| neither | E(k) |

which is a(k) + b(k) E(k) with a = (P(k) + conj P(−k)) / 2 and b = 1 − (M(k) + M(−k)) / 2.
a and the hit count M(k) + M(−k) (one byte) are computed once on the half spectrum
(n_z, n, n/2 + 1); the loop is then C2R, clamp, R2C, `restore_half`. Its iterates are the
C2C iterates. The only difference is the output: C2C ends with a complex inverse transform
whose imaginary part (the anti-Hermitian part of the data) enters the index through
Re √(n_m² + O/k₀²); R2C returns the real part. `Odt.RealGpMatchesComplexGp` measures the
difference at 5e-6 of the index contrast.

On the GPU the real volume lives in the padded layout of an in-place R2C transform (rows of
2(n/2 + 1) floats) inside the buffer that held the complex mapped potential, and a lives in
the buffer that held the hit counts: both are free once a and the hit counts exist. The index
is computed in place and copied out row by row with `cudaMemcpy2DAsync`, so the output needs
no buffer of its own.

## FISTA with total variation

The TV solver minimises

  f(x) + λ TV(x) subject to x ≥ 0,   f(x) = ½ ‖M (F x − P)‖²

for the real potential x, with λ = `tv_weight` · k₀² (so `tv_weight` acts on n² − n_m²) and
isotropic TV with forward differences. It is FISTA (Beck & Teboulle 2009, SIAM J. Imaging
Sci.) with step 1:

```
y = x_prev = C2R(a) / N,  t = 1
repeat iterations:
  v = C2R(a + b R2C(y)) / N        # y − ∇f(y): the Gerchberg-Papoulis k-space step
  x = prox(v)                       # argmin ½|x − v|² + λ TV(x), x ≥ 0
  t' = (1 + √(1 + 4t²)) / 2
  y = x + ((t − 1)/t') (x − x_prev);  x_prev = x;  t = t'
```

The gradient of f for a real x is Re F⁻¹ M (F x − P), so y − ∇f(y) is exactly the restore
step of the half-spectrum loop, and the Lipschitz constant of ∇f is 1. The proximal step is
the fast gradient projection (FGP) on the dual of the TV term (Beck & Teboulle 2009, IEEE
Trans. Image Process.), `tv_inner` iterations with step 1/(12λ) (‖∇‖² ≤ 12 in 3D) and the
non-negativity projection inside. The dual is kept from one outer iteration to the next as a
warm start. With λ → 0 the proximal step is the projection onto x ≥ 0 and the method is
Gerchberg-Papoulis with Nesterov momentum. Lim et al. (2015) compare this kind of
TV-regularised reconstruction with Gerchberg-Papoulis and edge-preserving regularisation for
ODT.

On the GPU each FGP iteration is one kernel: every thread recomputes the primal point
max(v − λ L(r), 0) at its voxel and at the three upper neighbours it needs for the gradient,
then does the dual update (projection onto the unit ball, momentum). r is read from one
buffer and written to another, since neighbouring threads still need the old values. That
replaced a primal and a dual kernel with z written in between; the fused step is 24-25%
faster and gives the same result bit for bit. Shared-memory tiled versions were slower or
equal; [tv_kernels.md](tv_kernels.md) has the measurements. The CPU runs the same arithmetic
from `common.hpp` (`tv_adjoint`, `tv_primal`, `tv_gradient`, `tv_dual_step`) in place.

## Beam propagation

`bpm_forward` and `bpm_reconstruct` model multiple forward scattering, which the Born and
Rytov approximations leave out. The variable is the index contrast x = n − n_m on the
(n_z, n, n) grid, one slice per voxel of thickness d (the pixel). Instead of the field u the
model propagates the envelope w = u / u_in, so the incident tilt drops out, w is periodic and
smooth far from the sample, and its spectrum at bin μ is the field spectrum at μ + ν_in, the
same bookkeeping as in the Rytov mapping. Per slice k, with S the spectrum of w:

```
w_k = F⁻¹ S_k                       (w_0 = 1, S_0 = n² δ)
A   = F (exp(i k₀ d x_k) w_k)       thin phase screen
S_{k+1} = c (q A + (1 − q) S_k)     c = exp(2πi (ν_z − ν_in,z) d), zero for |μ + ν_in| ≥ n_m/λ
```

After the last slice c is replaced by the propagator back to z = 0 (a negative distance) times
the detection pupil, which gives the field ratio in the convention of `forward_fields` and
`reconstruct`. c, the last-step factor and q are tables of (count, n, n) computed once in
double on the host (`bpm_tables`), so CPU and CUDA use identical values.

**Obliquity.** With q = 1 this is the usual split-step BPM. Linearised for a weak object it
gives the scattered spectrum i k₀ Δn̂(μ, ν_z − ν_in,z), while the first Born approximation
gives i k₀ (n_m/λ)/ν_z Δn̂: the plain screen leaves out the factor 1/cos θ of the scattered
direction. That is small at normal incidence and small scattering angles but 35% at an
illumination NA of 0.9 in water. With q = (n_m/λ)/ν_z the slice adds q times its scattered
part A − S, so the weak limit is first Born at every angle and q = 1 recovers the plain screen
where the scattering is near-axial. q is capped at its value at the edge of the detection
pupil, because it diverges for grazing waves that never reach the camera.
`Bpm.ReducesToBornForWeakObject` measures 2.7-3.6% against analytic Born for a voxelised weak
sphere at all angles with q, and 27% at NA 0.9 without it.

**Gradient.** The data term is f(x) = 1/(2A) Σ_a ‖y_a(x) − d_a‖² over A illuminations. Its
gradient is Re(J^H r), computed by running the slices backwards (reverse-mode, the
"back-propagation of the residual" of Kamilov et al. 2016). With G the adjoint of S and the
unnormalised forward FFT F:

```
G = F r
for k = n_z − 1 .. 0:
  U = conj(c) G
  b = F⁻¹ (q U),  G = (1 − q) U
  grad_k += k₀ d Im(conj(exp(i k₀ d x_k) w_k) b)
  G += F (exp(−i k₀ d x_k) b)
```

The forward pass stores w_k for every slice; the backward pass is the same number of FFTs.
`Bpm.AdjointIdentity` checks ⟨J h, r⟩ = ⟨h, J^T r⟩ to 2.7e-6 with an explicit tangent-linear
model `bpm_jvp`, and J h against central differences of the forward model to 1.8e-4.

**Solver.** Proximal gradient with the projection on x ≥ 0 or, with `tv_weight`, the same
FGP proximal step as the ODT TV solver (weight α · `tv_weight` on Δn). With all illuminations
per step it uses FISTA momentum; with `batch` it cycles through consecutive batches without
momentum, scaling each batch gradient to the mean over its illuminations. The step is
`step` / L with L = n_z (k₀ d)² mean_a q_a(0)², the largest eigenvalue of JᵀJ/A for a weak
object (a constant volume adds the same phase to the zero-frequency bin at every angle). The
start is a given volume, in the examples Rytov ODT with 20 GP iterations.

**CUDA.** Illuminations are processed in chunks of B with batched 2D cuFFT plans (one plan per
chunk size, one shared work area). Per slice: inverse FFT of S into w (out of place, S is
still needed), a screen kernel that also stores w_k, a forward FFT, and the propagation kernel.
The backward pass per slice is a split kernel, an inverse FFT, a gradient kernel with one
thread per pixel that loops over the B illuminations (so the gradient needs no atomics), and a
forward FFT. Nsight Compute at 128³ (49 illuminations per chunk): the propagation kernel
moves 21.4 MiB in 37.5 µs (600 GB/s, 91% of the DRAM peak), the screen kernel 6.2 MiB in
13.2 µs (493 GB/s, 76%; it is short enough for launch and tail effects to show). The adjoint
kernels were outside the profiled launch window and have no counters. The stored envelopes
take 8 n_z n² B bytes; B is chosen so that they fit in half
the free device memory, independently of the algorithmic `batch` (`Cuda.BpmChunkedGradient`
checks that chunking does not change the result).

## Memory

For an n³ volume the C2C reconstruction holds the complex potential (8 n³ bytes), the hit
counts (4 n³) and the measured spectrum while GP runs (8 n³); the index is written into the
hit-count buffer. The R2C loop needs the potential (8 n³, later the padded real volume and
its half spectrum), the hit counts (4 n³, later a) and one byte per half-spectrum voxel:
about 12.5 n³. cuFFT's work area comes on top. Measured with `cudaMemGetInfo` for the whole
reconstruction at 256³: 0.32 GiB C2C, 0.20 GiB R2C. FISTA-TV on the GPU adds eleven padded real volumes
(the dual p, its momentum copy r and the next r, three components each, the primal point and
the previous iterate), about 44 n³ bytes; the CPU version needs nine. The `Pipeline` adds the hologram stack, two field stacks and the unwrap
buffers; at 256³ it holds 0.42 GiB in total with GP.

## Validation

| test | checks |
|---|---|
| `Dct.*` | DCT-II via FFT equals pocketfft's DCT-II; inverse round-trips |
| `Unwrap.*` | ~40 rad smooth phase recovered to 2e-3 rad up to a 2π multiple; Rytov border rule |
| `Field.*` | hologram with on-grid carrier and tilt gives the Born field to 1e-4; 2% noise stays below 0.02 rms; sub-pixel carrier within 0.05 px |
| `Odt.*` | 3 µm sphere, 49 angles: mean inner index within 4e-3 of truth, GP reduces RMSE by at least 30%; R2C and C2C GP agree to 1e-4 of the contrast; TV has lower RMSE than GP |
| `Cuda.*` | field extraction (also with a crop across k_x = 0), Rytov and reconstruction (gather, scatter, R2C, C2C, TV), BPM forward and adjoint match CPU to 1e-4 relative; BPM inversion with and without batches to 1e-3 |
| `Exact.*` | partial-wave fields of a weak sphere agree with first Born to 2% |
| `Bpm.*` | weak-object limit equals Born at all angles with the obliquity weight; adjoint identity and finite differences; inversion of exact-model data of a 3 µm, n = 1.40 bead lowers the RMSE of the Rytov + GP start by at least 40% |
| `Pipeline.*` | CPU Pipeline matches the per-call chain to 1e-6; CUDA Pipeline matches it to 1e-4 with 1 and 4 upload parts and gives bitwise the same result on a second call |

The ODT test was written first with a k-space linear-interpolation gridding. It failed, and a
comparison of single k-space voxels against the FFT of the rendered phantom showed which
voxels were wrong (see the README, "Gridding"). Nearest-bin gridding passes.

The forward model evaluates the analytic sphere transform at exact Ewald-cap frequencies, so
it does not share the reconstruction's gridding. It does share the first-order scattering
model: the round trip checks the implementation of the Fourier diffraction theorem, not the
validity of the Born or Rytov approximation for a real cell.

The BPM inversion is validated against data that its own discretisation did not produce: the
exact partial-wave solution of the scalar Helmholtz equation for one sphere
(`ScatteringModel::exact`, the scalar analogue of Mie theory: u and ∂u/∂r continuous at the
surface, Bessel ratios by downward recurrence, the forward-scattered plane waves refocused to
z = 0 with Û_s = i f(θ)/ν_z), and, for several spheres, the BPM on a grid twice as fine in all
three axes. The first is exact physics for the scalar model; the second is the same
approximation at a different discretisation, so it tests discretisation error and the solver,
not the BPM approximation itself.
