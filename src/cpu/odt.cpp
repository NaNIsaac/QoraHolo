#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>

#include "backends.hpp"
#include "cpu/fft.hpp"

namespace qoraholo::cpu {

using detail::EwaldTarget;
using detail::parity;

namespace {

class Stopwatch {
 public:
  double lap() {
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - last_).count();
    last_ = now;
    return ms;
  }

 private:
  std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

}  // namespace

OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const detail::OdtGrid& g, const Geometry& geometry, int gp_iterations,
                      OdtTimings* timings) {
  OdtTimings t_local;
  OdtTimings& tm = timings ? *timings : t_local;
  Stopwatch watch;
  const int n = g.n;
  const int nz = g.nz;
  const int count = rytov_fields.nz;
  const std::size_t plane = static_cast<std::size_t>(n) * n;

  // Lateral centre-origin phases of the fields and the volume cancel, so a plain FFT suffices.
  Array3<cfloat> spectra = rytov_fields;
  fft(spectra.data.data(), {std::size_t(count), std::size_t(n), std::size_t(n)}, {1, 2}, true);
  tm.fft2 = watch.lap();

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
  tm.mapping = watch.lap();

  const auto total = static_cast<std::ptrdiff_t>(potential.size());
  std::ptrdiff_t filled = 0;
#pragma omp parallel for reduction(+ : filled)
  for (std::ptrdiff_t i = 0; i < total; ++i) {
    if (weight[i] > 0) {
      potential.data[i] /= weight[i];
      ++filled;
    }
  }
  tm.normalize = watch.lap();

  const std::vector<std::size_t> shape{std::size_t(nz), std::size_t(n), std::size_t(n)};
  const std::vector<std::size_t> axes{0, 1, 2};
  const float inv = 1.0f / static_cast<float>(potential.size());
  if (gp_iterations > 0) {
    const Array3<cfloat> measured = potential;
    fft(potential.data.data(), shape, axes, false, inv);
    for (int it = 0; it < gp_iterations; ++it) {
      // n >= n_m and a non-absorbing sample: the potential is real and non-negative.
#pragma omp parallel for
      for (std::ptrdiff_t i = 0; i < total; ++i) {
        potential.data[i] = {std::max(potential.data[i].real(), 0.0f), 0.0f};
      }
      fft(potential.data.data(), shape, axes, true);
#pragma omp parallel for
      for (std::ptrdiff_t i = 0; i < total; ++i) {
        if (weight[i] > 0) potential.data[i] = measured.data[i];
      }
      fft(potential.data.data(), shape, axes, false, inv);
    }
  } else {
    fft(potential.data.data(), shape, axes, false, inv);
  }
  tm.gp = watch.lap();

  const float k0 = 2.0f * std::numbers::pi_v<float> / geometry.wavelength;
  const float nm2 = geometry.n_medium * geometry.n_medium;
  const float inv_k02 = 1.0f / (k0 * k0);
  OdtResult result;
  result.ri = Array3<float>(nz, n, n);
#pragma omp parallel for
  for (std::ptrdiff_t i = 0; i < total; ++i) {
    result.ri.data[i] = detail::index_from_potential(potential.data[i].real(),
                                                     potential.data[i].imag(), nm2, inv_k02);
  }
  result.filled = static_cast<double>(filled) / static_cast<double>(potential.size());
  tm.finish = watch.lap();
  return result;
}

}  // namespace qoraholo::cpu
