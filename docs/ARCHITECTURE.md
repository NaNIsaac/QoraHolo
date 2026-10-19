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
| `src/phantom.cpp` | sphere rendering, analytic forward model, hologram synthesis (CPU only) |

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

Measured on the RTX 5070, kernel only: 0.09 ms against 0.10 ms at 256³; 0.62 ms against
0.84 ms at 512³ in an earlier run (49 illuminations). Zeroing the volume and hit counts
before the mapping takes longer than the mapping itself at 256³ (0.30 ms). Either way it is
about 1% of the GPU reconstruction, which the 3D FFTs of the missing-cone loop dominate. The
results are identical to the printed precision; float summation order differs, so they are
not bitwise equal in general.

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

Each FGP iteration is two stencil kernels: the primal point max(v − λ L(r), 0) and the dual
update (gradient of the primal point, projection onto the unit ball, momentum). The CPU runs
the same arithmetic from `common.hpp` (`tv_adjoint`, `tv_primal`, `tv_gradient`,
`tv_dual_step`). TV holds three more volumes than GP for each of p and r, plus z and x_prev.

## Memory

For an n³ volume the C2C reconstruction holds the complex potential (8 n³ bytes), the hit
counts (4 n³) and the measured spectrum while GP runs (8 n³); the index is written into the
hit-count buffer. The R2C loop needs the potential (8 n³, later the padded real volume and
its half spectrum), the hit counts (4 n³, later a) and one byte per half-spectrum voxel:
about 12.5 n³. cuFFT's work area comes on top. Measured with `cudaMemGetInfo` for the whole reconstruction
at 256³: 0.32 GiB C2C, 0.20 GiB R2C. FISTA-TV adds eight padded real volumes (the dual p and
its momentum copy r, three components each, the primal point and the previous iterate),
about 32 n³ bytes. The `Pipeline` adds the hologram stack, two field stacks and the unwrap
buffers; at 256³ it holds 0.42 GiB in total with GP.

## Validation

| test | checks |
|---|---|
| `Dct.*` | DCT-II via FFT equals pocketfft's DCT-II; inverse round-trips |
| `Unwrap.*` | ~40 rad smooth phase recovered to 2e-3 rad up to a 2π multiple; Rytov border rule |
| `Field.*` | hologram with on-grid carrier and tilt gives the Born field to 1e-4; 2% noise stays below 0.02 rms; sub-pixel carrier within 0.05 px |
| `Odt.*` | 3 µm sphere, 49 angles: mean inner index within 4e-3 of truth, GP reduces RMSE by at least 30%; R2C and C2C GP agree to 1e-4 of the contrast; TV has lower RMSE than GP |
| `Cuda.*` | field extraction (also with a crop across k_x = 0), Rytov and reconstruction (gather, scatter, R2C, C2C, TV) match CPU to 1e-4 relative |
| `Pipeline.*` | CPU Pipeline matches the per-call chain to 1e-6; CUDA Pipeline matches it to 1e-4 with 1 and 4 upload parts and gives bitwise the same result on a second call |

The ODT test was written first with a k-space linear-interpolation gridding. It failed, and a
comparison of single k-space voxels against the FFT of the rendered phantom showed which
voxels were wrong (see the README, "Gridding"). Nearest-bin gridding passes.

The forward model evaluates the analytic sphere transform at exact Ewald-cap frequencies, so
it does not share the reconstruction's gridding. It does share the first-order scattering
model: the round trip checks the implementation of the Fourier diffraction theorem, not the
validity of the Born or Rytov approximation for a real cell.
