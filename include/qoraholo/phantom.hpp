#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "qoraholo/array.hpp"
#include "qoraholo/odt.hpp"

namespace qoraholo {

// Uniform sphere, position in micrometres from the volume centre. Spheres are painted in
// order; each must lie entirely inside or entirely outside every earlier sphere, which keeps
// the forward model analytic (a nucleus inside a cell is fine, partial overlap is not).
struct Sphere {
  float z = 0;
  float y = 0;
  float x = 0;
  float radius = 1;
  float n = 1.38f;
};

// born: first Born. rytov: the first Born phase taken as a Rytov phase. exact: the scalar
// Helmholtz equation solved exactly by partial waves (the scalar analogue of Mie theory),
// multiple scattering included; one sphere only.
enum class ScatteringModel { born, rytov, exact };

// Refractive index sampled at voxel centres, (nz, n, n), centre-origin, voxel = geometry.pixel.
[[nodiscard]] Array3<float> render_phantom(std::span<const Sphere> spheres,
                                           const Geometry& geometry, int nz, int n);

// Field ratios u / u_in (n x n per illumination) at the plane z = 0, band-limited to the
// detection NA. born / rytov: the analytic 3D spectrum of the spheres placed on each Ewald cap,
// i.e. the same Fourier diffraction theorem as reconstruct(), evaluated at exact frequencies
// rather than on the voxel grid. exact: the forward-scattered plane waves of the partial-wave
// solution, refocused to z = 0 the same way.
[[nodiscard]] Array3<cfloat> forward_fields(std::span<const Sphere> spheres,
                                            std::span<const Illumination> illuminations,
                                            const Geometry& geometry, int n,
                                            ScatteringModel model = ScatteringModel::rytov);

struct HologramOptions {
  std::array<float, 2> carrier{5.0f, 5.0f};  // reference tilt {fy, fx}, cycles per micrometre
  float reference_amplitude = 1;
  float noise = 0;  // standard deviation of additive Gaussian noise, intensity units
  std::uint64_t seed = 1;
};

// |R + u|^2 + noise with u = u_in * ratio and R = a exp(-2 pi i carrier . r), so the field term
// u R* lies at illumination + carrier in the spectrum (the +1 sideband for carrier x > 0).
// Pass a ratio of ones for background holograms.
[[nodiscard]] Array3<float> synthesize_holograms(const Array3<cfloat>& ratio,
                                                 std::span<const Illumination> illuminations,
                                                 const Geometry& geometry,
                                                 const HologramOptions& options = {});

}  // namespace qoraholo
