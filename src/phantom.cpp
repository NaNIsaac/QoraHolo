#include "qoraholo/phantom.hpp"

#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

#include "common.hpp"
#include "cpu/fft.hpp"

namespace qoraholo {

namespace {

constexpr double kPi = std::numbers::pi;

bool inside(const Sphere& s, double z, double y, double x) {
  const double dz = z - s.z;
  const double dy = y - s.y;
  const double dx = x - s.x;
  return dz * dz + dy * dy + dx * dx <= double(s.radius) * s.radius;
}

// Refractive index at a point under painter's order.
double index_at(std::span<const Sphere> spheres, std::size_t count, double n_medium, double z,
                double y, double x) {
  double n = n_medium;
  for (std::size_t i = 0; i < count; ++i) {
    if (inside(spheres[i], z, y, x)) n = spheres[i].n;
  }
  return n;
}

// 3 (sin x - x cos x) / x^3, the normalised transform of a uniform ball.
double ball(double x) {
  if (x < 1e-3) return 1.0 - x * x / 10.0;
  return 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
}

// Partial-wave coefficients c_l of the scattered wave of a homogeneous sphere, scalar Helmholtz
// equation: outside u = sum i^l (2l+1) (j_l(kr) + c_l h_l(kr)) P_l, inside b_l j_l(mkr), with u
// and du/dr continuous at r = R. x = kR, m = n_sphere / n_medium.
std::vector<std::complex<double>> partial_waves(double x, double m) {
  const int lmax = static_cast<int>(x + 4.0 * std::cbrt(x) + 10.0);
  // j_l / j_{l-1} by downward recurrence (upward is unstable for l > x), started well above lmax.
  const auto ratios = [&](double z) {
    std::vector<double> rho(lmax + 1);
    double r = 0;
    for (int l = lmax + 60; l >= 1; --l) {
      r = 1.0 / ((2.0 * l + 1.0) / z - r);
      if (l <= lmax) rho[l] = r;
    }
    return rho;
  };
  const std::vector<double> rx = ratios(x);
  const std::vector<double> rm = ratios(m * x);
  std::vector<std::complex<double>> c(lmax + 1);
  double j_prev = std::cos(x) / x;  // j_{-1}
  double j = std::sin(x) / x;
  double y_prev = std::sin(x) / x;  // y_{-1}
  double y = -std::cos(x) / x;
  for (int l = 0; l <= lmax; ++l) {
    if (l > 0) {
      const double y_next = (2.0 * l - 1.0) / x * y - y_prev;
      y_prev = y;
      y = y_next;
      j_prev = j;
      j = rx[l] * j;
    }
    const std::complex<double> h(j, y);
    const std::complex<double> h_prev(j_prev, y_prev);
    const double dj = j_prev - (l + 1.0) / x * j;
    const std::complex<double> dh = h_prev - (l + 1.0) / x * h;
    // j_l'(mx) / j_l(mx) = j_{l-1} / j_l - (l + 1) / (mx); for l = 0, j_{-1}/j_0 = cot(mx).
    const double d = (l == 0 ? 1.0 / std::tan(m * x) : 1.0 / rm[l]) - (l + 1.0) / (m * x);
    c[l] = (m * d * j - dj) / (dh - m * d * h);
  }
  return c;
}

}  // namespace

Array3<float> render_phantom(std::span<const Sphere> spheres, const Geometry& geometry, int nz,
                             int n) {
  if (nz < 1 || n < 1) throw std::invalid_argument("render_phantom: empty shape");
  Array3<float> out(nz, n, n);
  const double d = geometry.pixel;
#pragma omp parallel for
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        out(k, j, i) =
            static_cast<float>(index_at(spheres, spheres.size(), geometry.n_medium,
                                        (k - nz / 2) * d, (j - n / 2) * d, (i - n / 2) * d));
      }
    }
  }
  return out;
}

Array3<cfloat> forward_fields(std::span<const Sphere> spheres,
                              std::span<const Illumination> illuminations, const Geometry& geometry,
                              int n, ScatteringModel model) {
  if (n < 4 || n % 2 != 0) throw std::invalid_argument("forward_fields: n must be even");
  const double k0 = 2.0 * kPi / geometry.wavelength;
  const double nm2 = double(geometry.n_medium) * geometry.n_medium;
  const double fm2 = nm2 / (double(geometry.wavelength) * geometry.wavelength);
  const double fna2 =
      double(geometry.na) * geometry.na / (double(geometry.wavelength) * geometry.wavelength);
  const double d = geometry.pixel;
  const double df = 1.0 / (n * d);

  if (model == ScatteringModel::exact && spheres.size() != 1) {
    throw std::invalid_argument("forward_fields: the exact model takes exactly one sphere");
  }
  // exact: scattering amplitude f(theta) = -i / k sum (2l+1) c_l P_l(cos theta).
  const double km = k0 * geometry.n_medium;
  const auto waves = model == ScatteringModel::exact
                         ? partial_waves(km * spheres[0].radius, spheres[0].n / geometry.n_medium)
                         : std::vector<std::complex<double>>{};

  // Each sphere contributes its potential step over the region it is painted on.
  std::vector<double> step(spheres.size());
  for (std::size_t i = 0; i < spheres.size(); ++i) {
    const auto& s = spheres[i];
    const double parent = index_at(spheres, i, geometry.n_medium, s.z, s.y, s.x);
    const double volume = 4.0 / 3.0 * kPi * double(s.radius) * s.radius * s.radius;
    step[i] = k0 * k0 * (double(s.n) * s.n - parent * parent) * volume;
  }

  const int count = static_cast<int>(illuminations.size());
  Array3<cfloat> out(count, n, n);
  for (int a = 0; a < count; ++a) {
    const double fiy = illuminations[a].fy;
    const double fix = illuminations[a].fx;
    if (fiy * fiy + fix * fix >= fm2) {
      throw std::invalid_argument("forward_fields: illumination beyond n_medium / wavelength");
    }
    const double fiz = std::sqrt(fm2 - fiy * fiy - fix * fix);
    cfloat* f = out.slice(a);
#pragma omp parallel for
    for (int ky = 0; ky < n; ++ky) {
      for (int kx = 0; kx < n; ++kx) {
        const double my = detail::signed_bin(ky, n) * df;
        const double mx = detail::signed_bin(kx, n) * df;
        const double fy = my + fiy;
        const double fx = mx + fix;
        const double rho2 = fy * fy + fx * fx;
        std::complex<double> u = 0;
        if (rho2 <= fna2 && rho2 < fm2) {
          const double nuz = std::sqrt(fm2 - rho2);
          const double qz = nuz - fiz;
          const double q = 2.0 * kPi * std::sqrt(qz * qz + my * my + mx * mx);
          // 4 pi times the scattering amplitude: U_s(nu) = i / (4 pi nu_z) * o, which is the
          // Born relation with o = O_hat; the centre-origin DFT carries (-1)^k / d^2.
          std::complex<double> o = 0;
          if (model == ScatteringModel::exact) {
            const auto& s = spheres[0];
            const double cos_t = (fy * fiy + fx * fix + nuz * fiz) / fm2;
            std::complex<double> sum = 0;
            double p_prev = 0;  // P_{l-1}
            double p = 1;       // P_l
            for (std::size_t l = 0; l < waves.size(); ++l) {
              const double w = 2.0 * double(l) + 1.0;
              sum += w * waves[l] * p;
              const double p_next = (w * cos_t * p - double(l) * p_prev) / (double(l) + 1.0);
              p_prev = p;
              p = p_next;
            }
            const double ph = -2.0 * kPi * (qz * s.z + my * s.y + mx * s.x);
            o = 4.0 * kPi * std::complex<double>(0.0, -1.0 / km) * sum * std::polar(1.0, ph);
          } else {
            for (std::size_t i = 0; i < spheres.size(); ++i) {
              const auto& s = spheres[i];
              const double ph = -2.0 * kPi * (qz * s.z + my * s.y + mx * s.x);
              o += step[i] * ball(q * s.radius) * std::polar(1.0, ph);
            }
          }
          u = std::complex<double>(0.0, 1.0 / (4.0 * kPi * nuz)) * o *
              double(detail::parity(ky + kx)) / (d * d);
        }
        f[ky * n + kx] = cfloat(u);
      }
    }
  }
  cpu::fft(out.data.data(), {std::size_t(count), std::size_t(n), std::size_t(n)}, {1, 2}, false,
           1.0f / static_cast<float>(n * n));
  for (auto& v : out.data) v = model == ScatteringModel::rytov ? std::exp(v) : 1.0f + v;
  return out;
}

Array3<float> synthesize_holograms(const Array3<cfloat>& ratio,
                                   std::span<const Illumination> illuminations,
                                   const Geometry& geometry, const HologramOptions& options) {
  if (static_cast<std::size_t>(ratio.nz) != illuminations.size()) {
    throw std::invalid_argument("synthesize_holograms: need one illumination per field");
  }
  Array3<float> out(ratio.nz, ratio.ny, ratio.nx);
  std::mt19937_64 rng(options.seed);
  std::normal_distribution<float> noise(0.0f, options.noise);
  const double d = geometry.pixel;
  for (int a = 0; a < ratio.nz; ++a) {
    for (int y = 0; y < ratio.ny; ++y) {
      for (int x = 0; x < ratio.nx; ++x) {
        const double ry = (y - ratio.ny / 2) * d;
        const double rx = (x - ratio.nx / 2) * d;
        const auto ui =
            std::polar(1.0, 2.0 * kPi * (illuminations[a].fy * ry + illuminations[a].fx * rx));
        const auto ref =
            std::polar(double(options.reference_amplitude),
                       -2.0 * kPi * (options.carrier[0] * ry + options.carrier[1] * rx));
        const auto u = ui * std::complex<double>(ratio(a, y, x));
        float v = static_cast<float>(std::norm(ref + u));
        if (options.noise > 0) v += noise(rng);
        out(a, y, x) = v;
      }
    }
  }
  return out;
}

}  // namespace qoraholo
