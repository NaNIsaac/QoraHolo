#include <algorithm>
#include <cmath>
#include <numbers>

#include "backends.hpp"
#include "cpu/fft.hpp"

namespace qoraholo::cpu {

using detail::EwaldTarget;
using detail::parity;

OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const detail::OdtGrid& g, const Geometry& geometry, int gp_iterations) {
  const int n = g.n;
  const int nz = g.nz;
  const int count = rytov_fields.nz;
  const std::size_t plane = static_cast<std::size_t>(n) * n;

  // Lateral centre-origin phases of the fields and the volume cancel, so a plain FFT suffices.
  Array3<cfloat> spectra = rytov_fields;
  fft(spectra.data.data(), {std::size_t(count), std::size_t(n), std::size_t(n)}, {1, 2}, true);

  std::vector<float> fiz(count);
  for (int a = 0; a < count; ++a) {
    fiz[a] = std::sqrt(g.fm2 - illum[a].fy * illum[a].fy - illum[a].fx * illum[a].fx);
  }

  // Each thread owns whole k_z columns, so the accumulation needs no synchronisation.
  Array3<cfloat> potential(nz, n, n);
  std::vector<float> weight(potential.size(), 0.0f);
#pragma omp parallel for schedule(dynamic, 64)
  for (int col = 0; col < static_cast<int>(plane); ++col) {
    const int ky = col / n;
    const int kx = col % n;
    for (int a = 0; a < count; ++a) {
      EwaldTarget t{};
      if (!detail::ewald_target(g, ky, kx, illum[a].fy, illum[a].fx, fiz[a], t)) continue;
      const cfloat v = cfloat(0.0f, -g.gain * t.nuz) * spectra.slice(a)[col];
      const std::size_t i = t.kz * plane + col;
      potential.data[i] += parity(t.kz) * v;
      weight[i] += 1.0f;
    }
  }

  std::size_t filled = 0;
  for (std::size_t i = 0; i < potential.size(); ++i) {
    if (weight[i] > 0) {
      potential.data[i] /= weight[i];
      ++filled;
    }
  }

  const std::vector<std::size_t> shape{std::size_t(nz), std::size_t(n), std::size_t(n)};
  const std::vector<std::size_t> axes{0, 1, 2};
  const float inv = 1.0f / static_cast<float>(potential.size());
  if (gp_iterations > 0) {
    const Array3<cfloat> measured = potential;
    fft(potential.data.data(), shape, axes, false, inv);
    for (int it = 0; it < gp_iterations; ++it) {
      for (auto& v : potential.data) v = {std::max(v.real(), 0.0f), 0.0f};
      fft(potential.data.data(), shape, axes, true);
      for (std::size_t i = 0; i < potential.size(); ++i) {
        if (weight[i] > 0) potential.data[i] = measured.data[i];
      }
      fft(potential.data.data(), shape, axes, false, inv);
    }
  } else {
    fft(potential.data.data(), shape, axes, false, inv);
  }

  const float k0 = 2.0f * std::numbers::pi_v<float> / geometry.wavelength;
  const float nm2 = geometry.n_medium * geometry.n_medium;
  OdtResult result;
  result.ri = Array3<float>(nz, n, n);
  for (std::size_t i = 0; i < potential.size(); ++i) {
    result.ri.data[i] = std::sqrt(nm2 + potential.data[i] / (k0 * k0)).real();
  }
  result.filled = static_cast<double>(filled) / static_cast<double>(potential.size());
  return result;
}

}  // namespace qoraholo::cpu
