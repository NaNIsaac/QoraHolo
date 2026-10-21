#pragma once

#include <span>
#include <vector>

#include "qoraholo/array.hpp"
#include "qoraholo/backend.hpp"
#include "qoraholo/odt.hpp"

namespace qoraholo {

// Multi-slice beam propagation (split step). The volume (nz, n, n), voxel = geometry.pixel, is
// cut into nz slices of one voxel. The envelope w = u / u_in is propagated through them: each
// slice multiplies it by the thin phase screen exp(i k0 (n - n_medium) pixel), then the angular
// spectrum propagator of the medium carries it one voxel along z. After the last slice the field
// is refocused to z = 0 at the volume centre and band-limited to the detection NA, which is the
// convention of forward_fields() and reconstruct(). Lateral boundaries are periodic.
struct BpmOptions {
  // Weight the scattered part of each slice by n_medium / (lambda nu_z), the obliquity of the
  // scattered wave, so that the weak-object limit is the first Born approximation at every
  // angle. Without it the plain phase screen is accurate for near-axial scattering only.
  // Capped at its value at the detection NA.
  bool obliquity = true;
  Backend backend = Backend::cpu;
};

// Refractive index (nz, n, n) -> field ratios u / u_in at z = 0, (count, n, n).
[[nodiscard]] Array3<cfloat> bpm_forward(const Array3<float>& ri,
                                         std::span<const Illumination> illuminations,
                                         const Geometry& geometry, const BpmOptions& options = {});

struct BpmSolveOptions {
  int iterations = 20;
  // Illuminations per gradient step; 0 or count: all of them, with FISTA momentum. Otherwise
  // each iteration is one pass of proximal gradient steps over consecutive batches.
  int batch = 0;
  // Step as a multiple of 1 / L, L the Lipschitz constant of the data-term gradient for a
  // weak object (the constant volume is its largest eigenvector).
  float step = 1.0f;
  // Weight of TV(n - n_medium); 0: non-negativity n >= n_medium only.
  float tv_weight = 0.0f;
  int tv_inner = 10;
  bool obliquity = true;
  Backend backend = Backend::cpu;
};

struct BpmResult {
  Array3<float> ri;
  // Data term 0.5 / count * sum |bpm(x) - fields|^2 per iteration, at the point where the
  // gradient was taken (summed over the batches of the pass with batch > 0).
  std::vector<double> loss;
};

// Field ratios (count, n, n) -> refractive index by gradient descent on the BPM (learning
// tomography, Kamilov et al. 2016): gradient by back-propagation of the residual, then the
// proximal step of n >= n_medium and the optional TV term. initial (nz, n, n) is the starting
// volume, e.g. a Rytov reconstruction; nz is taken from it.
[[nodiscard]] BpmResult bpm_reconstruct(const Array3<cfloat>& fields,
                                        std::span<const Illumination> illuminations,
                                        const Geometry& geometry, const Array3<float>& initial,
                                        const BpmSolveOptions& options = {});

}  // namespace qoraholo
