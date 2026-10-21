#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>

#include "backends.hpp"

namespace qoraholo {

BpmTables bpm_tables(int n, int nz, std::span<const Illumination> illuminations,
                     const Geometry& geometry, bool obliquity) {
  constexpr double kPi = std::numbers::pi;
  const double lambda = geometry.wavelength;
  const double fm = geometry.n_medium / lambda;
  const double fna = geometry.na / lambda;
  const double d = geometry.pixel;
  const double df = 1.0 / (n * d);
  // Obliquity cap: nu_z at the edge of the detection pupil (or 0.1 fm if the NA reaches n_m).
  const double nuz_min = std::sqrt(std::max(fm * fm - fna * fna, 0.01 * fm * fm));
  // Distance from the last slice centre back to z = 0.
  const double back = -(nz - 1 - nz / 2) * d;

  BpmTables t;
  t.n = n;
  t.nz = nz;
  t.count = static_cast<int>(illuminations.size());
  t.phase = static_cast<float>(2.0 * kPi / lambda * d);
  const std::size_t plane = std::size_t(n) * n;
  t.step.resize(plane * t.count);
  t.last.resize(plane * t.count);
  t.q.resize(plane * t.count);
  double q0 = 0;
  for (int a = 0; a < t.count; ++a) {
    const double fiy = illuminations[a].fy;
    const double fix = illuminations[a].fx;
    const double fiz = std::sqrt(fm * fm - fiy * fiy - fix * fix);
    for (int ky = 0; ky < n; ++ky) {
      for (int kx = 0; kx < n; ++kx) {
        const std::size_t i = a * plane + std::size_t(ky) * n + kx;
        const double fy = detail::signed_bin(ky, n) * df + fiy;
        const double fx = detail::signed_bin(kx, n) * df + fix;
        const double rho2 = fy * fy + fx * fx;
        double q = 1;
        if (rho2 < fm * fm) {
          const double nuz = std::sqrt(fm * fm - rho2);
          t.step[i] = cfloat(std::polar(1.0, 2.0 * kPi * (nuz - fiz) * d));
          if (rho2 <= fna * fna)
            t.last[i] = cfloat(std::polar(1.0, 2.0 * kPi * (nuz - fiz) * back));
          if (obliquity) q = fm / std::max(nuz, nuz_min);
        }
        t.q[i] = static_cast<float>(q);
      }
    }
    q0 += double(t.q[a * plane]) * t.q[a * plane];
  }
  // Data term 0.5 / count sum_a |J_a x|^2; for a constant x every slice adds i phase q_a(0) x
  // to the zero-frequency bin of every angle.
  t.lipschitz = static_cast<float>(nz * double(t.phase) * t.phase * q0 / t.count);
  return t;
}

}  // namespace qoraholo
