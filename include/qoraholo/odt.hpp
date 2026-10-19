#pragma once

#include <span>
#include <vector>

#include "qoraholo/array.hpp"
#include "qoraholo/backend.hpp"

namespace qoraholo {

// Lengths in micrometres, spatial frequencies in cycles per micrometre.
struct Geometry {
  float wavelength = 0.532f;  // vacuum
  float n_medium = 1.337f;
  float na = 1.2f;     // detection pupil
  float pixel = 0.1f;  // field pixel in the sample plane; also the voxel side
};

// Lateral spatial frequency of an incident plane wave, (n_m / lambda) * sin(theta) * (sin phi,
// cos phi). Its magnitude must be below n_m / lambda.
struct Illumination {
  float fy = 0;
  float fx = 0;
};

// How spectra are written into k-space. gather: one owner per lateral k column, no atomics.
// scatter: one thread per (angle, pixel) with atomic adds. Identical results up to float
// summation order; the choice only matters on CUDA.
enum class Mapping { gather, scatter };

// Missing-cone solver run for OdtOptions::iterations steps after the Ewald mapping.
enum class Solver {
  // Gerchberg-Papoulis: n >= n_medium in real space, measured voxels reset in k-space.
  gerchberg_papoulis,
  // FISTA on 0.5 |M F x - d|^2 + tv_weight * k0^2 * TV(x) with x >= 0, x the scattering
  // potential, M the measured voxels (Beck & Teboulle 2009; Lim et al. 2015 for ODT).
  tv,
};

// 3D transforms of the Gerchberg-Papoulis loop. r2c: the constrained potential is real, so the
// loop runs on half spectra; the iterates equal those of c2c, but the returned volume is the
// real part only (no absorption). c2c: full complex spectra, the imaginary part of the last
// inverse transform enters the index as it does without iterations.
enum class GpTransform { r2c, c2c };

struct OdtOptions {
  int nz = 0;          // axial voxels; 0: field side
  int iterations = 0;  // missing-cone iterations; 0: plain Rytov ODT
  Solver solver = Solver::gerchberg_papoulis;
  GpTransform gp_transform = GpTransform::r2c;  // TV always works on half spectra
  // TV weight for the potential divided by k0^2, i.e. for n^2 - n_medium^2, so it does not
  // depend on the wavelength.
  float tv_weight = 1e-2f;
  int tv_inner = 10;  // dual iterations of the TV proximal step per outer iteration
  Mapping mapping = Mapping::gather;
  Backend backend = Backend::cpu;
};

struct OdtResult {
  Array3<float> ri;   // (nz, n, n), centre-origin, real part of the refractive index
  double filled = 0;  // fraction of k-space voxels that received data
};

// Rytov fields (n x n each, centre-origin, pixel = geometry.pixel) for each illumination ->
// refractive index by the Fourier diffraction theorem. See docs/ARCHITECTURE.md for the
// convention.
[[nodiscard]] OdtResult reconstruct(const Array3<cfloat>& rytov_fields,
                                    std::span<const Illumination> illuminations,
                                    const Geometry& geometry, const OdtOptions& options = {});

// Normal incidence followed by count - 1 directions evenly spaced on a circle at na_illumination.
[[nodiscard]] std::vector<Illumination> circular_scan(int count, float na_illumination,
                                                      const Geometry& geometry);

}  // namespace qoraholo
