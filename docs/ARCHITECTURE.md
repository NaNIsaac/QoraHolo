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

Every public function takes and returns host arrays (`Array3<T>`: shape plus a
`std::vector`). The CUDA path uploads, computes, and downloads inside each call. That keeps
the API identical across backends and makes CPU-versus-GPU tests a one-line change; the cost
is a host round trip between stages, listed in the benchmark. A device-resident pipeline
would remove it.

## Source layout

| file | contents |
|---|---|
| `src/api.cpp` | argument checks, crop-plan resolution, backend dispatch, `make_grid` |
| `src/common.hpp` | `QH_HD` helpers compiled for host and device: DFT index maps, Makhoul reordering, Neumann eigenvalues, `ewald_target`, `index_from_potential` |
| `src/cpu/*.cpp` | pocketfft (C2C over chosen axes, pocketfft's own thread pool) and OpenMP loops |
| `src/cuda/*.cu` | cuFFT plans and kernels; `util.cuh` holds RAII buffers, plans and error checks |
| `src/phantom.cpp` | sphere rendering, analytic forward model, hologram synthesis (CPU only) |

The two backends call the same `common.hpp` functions for every index or physics expression,
so a disagreement between them can only come from FFT rounding or summation order.

## Centre-origin bookkeeping

Real-space arrays have r = 0 at index n/2. For even n, moving the origin is a multiplication
by (−1)^k in the other domain, so no array is ever shifted in memory:

- `extract_fields` multiplies the hologram by (−1)^(x+y) before the FFT (spectrum comes out
  centred), multiplies the cropped window by one sign per image (−1)^(c_y+c_x), and
  multiplies the output by (−1)^(x+y) after the inverse FFT.
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

Measured on the RTX 5070: 0.07 ms against 0.10 ms at 256³, 0.62 ms against 0.84 ms at
512³ (49 illuminations). Either way the mapping is under 0.3% of the GPU reconstruction
time, which the 3D FFTs of the Gerchberg-Papoulis loop dominate. The results are
identical to the printed precision; float summation order differs, so they are not bitwise
equal in general.

## Phase unwrapping on the GPU

The DCT-II along an axis is a reorder kernel, a batched 1D C2C FFT, and a twiddle kernel
(`sincospif`). Along x the transforms are contiguous and one cuFFT plan covers the whole
stack. Along y they are strided by n within an image but not across images, which a single
`cufftPlanMany` cannot express, so the same plan is executed once per image. The inverse is
the exact inverse of the forward transform (V_k = e^{iπk/2N}(X_k − iX_{N−k}), inverse FFT,
reorder back), so DCT → divide by the Neumann eigenvalues → inverse DCT solves the Poisson
problem without extra scaling. The congruence constant and the border mean are block
reductions with double-precision atomics.

## Memory

For an n³ volume the CUDA reconstruction holds the complex potential (8 n³ bytes), the hit
counts (4 n³), the measured spectrum while GP runs (8 n³) and the output (4 n³): 24 n³
bytes, 3 GiB at 512³, plus the field spectra. 512³ ran on the 12 GB RTX 5070; 1024³ would
not fit.

## Validation

| test | checks |
|---|---|
| `Dct.*` | DCT-II via FFT equals pocketfft's DCT-II; inverse round-trips |
| `Unwrap.*` | ~40 rad smooth phase recovered to 2e-3 rad up to a 2π multiple; Rytov border rule |
| `Field.*` | hologram with on-grid carrier and tilt gives the Born field to 1e-4; 2% noise stays below 0.02 rms; sub-pixel carrier within 0.05 px |
| `Odt.*` | 3 µm sphere, 49 angles: mean inner index within 4e-3 of truth, GP reduces RMSE by at least 30% |
| `Cuda.*` | field extraction, Rytov and reconstruction (gather and scatter) match CPU to 1e-4 relative |

The ODT test was written first with a k-space linear-interpolation gridding. It failed, and a
comparison of single k-space voxels against the FFT of the rendered phantom showed which
voxels were wrong (see the README, "Gridding"). Nearest-bin gridding passes.

The forward model evaluates the analytic sphere transform at exact Ewald-cap frequencies, so
it does not share the reconstruction's gridding. It does share the first-order scattering
model: the round trip checks the implementation of the Fourier diffraction theorem, not the
validity of the Born or Rytov approximation for a real cell.
