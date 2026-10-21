#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>

#include "backends.hpp"
#include "cpu/fft.hpp"
#include "cpu/tv.hpp"

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

// Measured data on the half spectrum (nz, n, n / 2 + 1) of a real potential.
struct HalfData {
  std::vector<cfloat> a;
  std::vector<std::uint8_t> hits;
};

HalfData hermitian_data(const Array3<cfloat>& p, const std::vector<float>& weight) {
  const int nz = p.nz;
  const int n = p.nx;
  const int h = n / 2 + 1;
  HalfData d{std::vector<cfloat>(std::size_t(nz) * n * h),
             std::vector<std::uint8_t>(std::size_t(nz) * n * h)};
#pragma omp parallel for
  for (int kz = 0; kz < nz; ++kz) {
    const int mz = detail::negative_bin(kz, nz);
    for (int ky = 0; ky < n; ++ky) {
      const int my = detail::negative_bin(ky, n);
      for (int kx = 0; kx < h; ++kx) {
        const std::size_t i = (std::size_t(kz) * n + ky) * n + kx;
        const std::size_t j = (std::size_t(mz) * n + my) * n + detail::negative_bin(kx, n);
        const std::size_t o = (std::size_t(kz) * n + ky) * h + kx;
        float re = 0;
        float im = 0;
        detail::hermitian_data(p.data[i].real(), p.data[i].imag(), p.data[j].real(),
                               p.data[j].imag(), re, im);
        d.a[o] = {re, im};
        d.hits[o] = static_cast<std::uint8_t>((weight[i] > 0) + (weight[j] > 0));
      }
    }
  }
  return d;
}

void restore(std::vector<cfloat>& e, const HalfData& d) {
  const auto total = static_cast<std::ptrdiff_t>(e.size());
#pragma omp parallel for
  for (std::ptrdiff_t i = 0; i < total; ++i) {
    float re = e[i].real();
    float im = e[i].imag();
    detail::hermitian_restore(d.a[i].real(), d.a[i].imag(), d.hits[i], re, im);
    e[i] = {re, im};
  }
}

// Gerchberg-Papoulis on half spectra. Returns the real potential, (nz, n, n).
std::vector<float> gp_real(const HalfData& d, int nz, int n, int iterations) {
  const float inv = 1.0f / (static_cast<float>(nz) * static_cast<float>(n) * static_cast<float>(n));
  std::vector<float> x(std::size_t(nz) * n * n);
  std::vector<cfloat> e = d.a;
  irfft3(e.data(), x.data(), nz, n, n, inv);
  const auto total = static_cast<std::ptrdiff_t>(x.size());
  for (int it = 0; it < iterations; ++it) {
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < total; ++i) x[i] = std::max(x[i], 0.0f);
    rfft3(x.data(), e.data(), nz, n, n);
    restore(e, d);
    irfft3(e.data(), x.data(), nz, n, n, inv);
  }
  return x;
}

// FISTA (Beck & Teboulle 2009) with step 1 on 0.5 |M F x - d|^2 + lambda TV(x), x >= 0. The
// gradient step y - grad f(y) is the Gerchberg-Papoulis k-space step on the half spectrum.
std::vector<float> tv_real(const HalfData& d, int nz, int n, int iterations, float lambda,
                           int inner) {
  const float inv = 1.0f / (static_cast<float>(nz) * static_cast<float>(n) * static_cast<float>(n));
  const detail::Volume v{nz, n, n, n};
  const std::size_t size = std::size_t(nz) * n * n;
  std::vector<float> y(size);
  std::vector<cfloat> e = d.a;
  irfft3(e.data(), y.data(), nz, n, n, inv);
  std::vector<float> x_prev = y;
  std::vector<float> x(size);
  Dual dual(size);
  const auto total = static_cast<std::ptrdiff_t>(size);
  float t = 1.0f;
  for (int it = 0; it < iterations; ++it) {
    rfft3(y.data(), e.data(), nz, n, n);
    restore(e, d);
    irfft3(e.data(), y.data(), nz, n, n, inv);
    tv_prox(y.data(), x.data(), dual, v, lambda, inner);
    const float t_next = detail::fista_t(t);
    const float momentum = (t - 1.0f) / t_next;
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < total; ++i) y[i] = x[i] + momentum * (x[i] - x_prev[i]);
    std::swap(x, x_prev);
    t = t_next;
  }
  return x_prev;
}

}  // namespace

OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const detail::OdtGrid& g, const Geometry& geometry, const OdtOptions& options,
                      OdtTimings* timings) {
  const int iterations = options.iterations;
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

  const float k0 = 2.0f * std::numbers::pi_v<float> / geometry.wavelength;
  const float nm2 = geometry.n_medium * geometry.n_medium;
  const float inv_k02 = 1.0f / (k0 * k0);
  OdtResult result;
  result.filled = static_cast<double>(filled) / static_cast<double>(potential.size());

  const bool tv = options.solver == Solver::tv;
  if (iterations > 0 && (tv || options.gp_transform == GpTransform::r2c)) {
    const HalfData d = hermitian_data(potential, weight);
    const std::size_t peak = potential.size() * (sizeof(cfloat) + sizeof(float)) +
                             d.a.size() * (sizeof(cfloat) + sizeof(std::uint8_t));
    potential = {};
    weight = {};
    std::vector<float> x =
        tv ? tv_real(d, nz, n, iterations, options.tv_weight * k0 * k0, options.tv_inner)
           : gp_real(d, nz, n, iterations);
    // Loop phase: data, spectrum and x; TV adds y, x_prev and a dual of six volumes.
    tm.bytes = std::max(peak, d.a.size() * (2 * sizeof(cfloat) + sizeof(std::uint8_t)) +
                                  x.size() * sizeof(float) * (tv ? 9 : 1));
    tm.solve = watch.lap();
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < total; ++i) {
      x[i] = detail::index_from_potential(x[i], 0.0f, nm2, inv_k02);
    }
    result.ri = Array3<float>();
    result.ri.nz = nz;
    result.ri.ny = n;
    result.ri.nx = n;
    result.ri.data = std::move(x);
    tm.finish = watch.lap();
    return result;
  }

  const std::vector<std::size_t> shape{std::size_t(nz), std::size_t(n), std::size_t(n)};
  const std::vector<std::size_t> axes{0, 1, 2};
  const float inv = 1.0f / static_cast<float>(potential.size());
  tm.bytes = potential.size() * (sizeof(cfloat) + sizeof(float));
  if (iterations > 0) {
    const Array3<cfloat> measured = potential;
    tm.bytes += measured.size() * sizeof(cfloat);
    fft(potential.data.data(), shape, axes, false, inv);
    for (int it = 0; it < iterations; ++it) {
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
  tm.solve = watch.lap();

  result.ri = Array3<float>(nz, n, n);
#pragma omp parallel for
  for (std::ptrdiff_t i = 0; i < total; ++i) {
    result.ri.data[i] = detail::index_from_potential(potential.data[i].real(),
                                                     potential.data[i].imag(), nm2, inv_k02);
  }
  tm.finish = watch.lap();
  return result;
}

}  // namespace qoraholo::cpu
