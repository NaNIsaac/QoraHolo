#include <algorithm>
#include <cmath>
#include <numbers>

#include "backends.hpp"
#include "cpu/fft.hpp"

namespace qoraholo {

using detail::parity;

namespace {

// fftshifted spectrum of one hologram with the origin at pixel 0.
std::vector<cfloat> shifted_spectrum(const float* h, int n) {
  std::vector<cfloat> s(static_cast<std::size_t>(n) * n);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) s[y * n + x] = h[y * n + x] * parity(x + y);
  }
  cpu::fft(s.data(), {std::size_t(n), std::size_t(n)}, {0, 1}, true);
  return s;
}

// Jacobsen's three-bin estimator with Candan's bias correction for a rectangular window.
float refine(cfloat lo, cfloat mid, cfloat hi, int n) {
  const cfloat den = 2.0f * mid - lo - hi;
  if (std::abs(den) == 0.0f) return 0.0f;
  const float c = std::numbers::pi_v<float> / static_cast<float>(n);
  const float d = ((lo - hi) / den).real() * std::tan(c) / c;
  return std::clamp(d, -0.5f, 0.5f);
}

}  // namespace

Sideband find_sideband(const Array3<float>& holograms, float dc_exclusion) {
  const int n = holograms.nx;
  if (holograms.nz < 1 || n != holograms.ny || n % 2 != 0 || n < 4) {
    throw std::invalid_argument("find_sideband: holograms must be square with an even side");
  }
  if (dc_exclusion <= 0) dc_exclusion = static_cast<float>(n) / 8.0f;
  const auto s = shifted_spectrum(holograms.slice(0), n);
  const int c = n / 2;
  Sideband best;
  for (int y = 1; y < n - 1; ++y) {
    for (int x = c; x < n - 1; ++x) {
      const int dy = y - c;
      const int dx = x - c;
      if (dx == 0 && dy <= 0) continue;
      if (static_cast<float>(dy * dy + dx * dx) <= dc_exclusion * dc_exclusion) continue;
      const float m = std::abs(s[y * n + x]);
      if (m > best.magnitude) best = {dy, dx, float(dy), float(dx), m};
    }
  }
  if (best.magnitude == 0) throw std::runtime_error("find_sideband: no peak outside the DC region");
  const int y = best.iy + c;
  const int x = best.ix + c;
  best.y += refine(s[(y - 1) * n + x], s[y * n + x], s[(y + 1) * n + x], n);
  best.x += refine(s[y * n + x - 1], s[y * n + x], s[y * n + x + 1], n);
  return best;
}

Array3<cfloat> field_ratio(const Array3<cfloat>& sample, const Array3<cfloat>& background) {
  if (sample.ny != background.ny || sample.nx != background.nx ||
      (background.nz != 1 && background.nz != sample.nz)) {
    throw std::invalid_argument("field_ratio: background shape does not match");
  }
  Array3<cfloat> out(sample.nz, sample.ny, sample.nx);
  const std::size_t plane = sample.plane();
  for (int z = 0; z < sample.nz; ++z) {
    const cfloat* b = background.slice(background.nz == 1 ? 0 : z);
    for (std::size_t i = 0; i < plane; ++i) out.slice(z)[i] = sample.slice(z)[i] / b[i];
  }
  return out;
}

namespace cpu {

Array3<cfloat> extract_fields(const Array3<float>& holograms, const CropPlan& plan) {
  const int n = holograms.nx;
  const int m = plan.out;
  const std::size_t nn = holograms.plane();
  std::vector<cfloat> spec(holograms.size());
  for (int z = 0; z < holograms.nz; ++z) {
    const float* h = holograms.slice(z);
    cfloat* s = spec.data() + z * nn;
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) s[y * n + x] = h[y * n + x] * parity(x + y);
    }
  }
  fft(spec.data(), {std::size_t(holograms.nz), std::size_t(n), std::size_t(n)}, {1, 2}, true);

  // The crop window is centred so that output bin m/2 holds the sideband centre. Origin
  // bookkeeping for centre-origin input and output reduces to one sign per image.
  Array3<cfloat> out(holograms.nz, m, m);
  const float sign = parity(plan.cy + plan.cx);
  const float r2 = plan.radius * plan.radius;
  for (int z = 0; z < holograms.nz; ++z) {
    const cfloat* s = spec.data() + z * nn;
    cfloat* o = out.slice(z);
    for (int j = 0; j < m; ++j) {
      const int dy = j - m / 2;
      const int sy = n / 2 + plan.cy + dy;
      for (int i = 0; i < m; ++i) {
        const int dx = i - m / 2;
        const int sx = n / 2 + plan.cx + dx;
        const bool inside =
            static_cast<float>(dy * dy + dx * dx) <= r2 && sy >= 0 && sy < n && sx >= 0 && sx < n;
        o[j * m + i] = inside ? s[sy * n + sx] * sign : cfloat{};
      }
    }
  }
  const float scale = 1.0f / static_cast<float>(nn);
  fft(out.data.data(), {std::size_t(out.nz), std::size_t(m), std::size_t(m)}, {1, 2}, false, scale);
  for (int z = 0; z < out.nz; ++z) {
    cfloat* o = out.slice(z);
    for (int y = 0; y < m; ++y) {
      for (int x = 0; x < m; ++x) o[y * m + x] *= parity(x + y);
    }
  }
  return out;
}

}  // namespace cpu
}  // namespace qoraholo
