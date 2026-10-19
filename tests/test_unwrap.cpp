#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include "common.hpp"
#include "qoraholo/unwrap.hpp"

using qoraholo::Array3;
using qoraholo::cfloat;

namespace {

// Two Gaussian bumps and a tilt: about 40 rad peak to valley, gradients well below pi per pixel.
Array3<float> smooth_phase(int nz, int ny, int nx) {
  Array3<float> p(nz, ny, nx);
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const float u = (x - nx * 0.4f) / (nx * 0.15f);
        const float v = (y - ny * 0.5f) / (ny * 0.2f);
        const float w = (x - nx * 0.7f) / (nx * 0.1f);
        const float s = (y - ny * 0.3f) / (ny * 0.1f);
        p(z, y, x) = 30.0f * std::exp(-(u * u + v * v)) - (12.0f + z) * std::exp(-(w * w + s * s)) +
                     0.05f * x;
      }
    }
  }
  return p;
}

}  // namespace

TEST(Unwrap, RecoversSmoothPhaseUpToConstant) {
  const auto truth = smooth_phase(2, 96, 128);
  Array3<float> wrapped = truth;
  for (auto& v : wrapped.data) v = qoraholo::detail::wrap_phase(v);

  const auto got = qoraholo::unwrap(wrapped);
  for (int z = 0; z < truth.nz; ++z) {
    const float offset = got(z, 0, 0) - truth(z, 0, 0);
    // Congruent to the input: the offset is a multiple of 2 pi.
    EXPECT_NEAR(std::remainder(offset, 2.0f * std::numbers::pi_v<float>), 0.0f, 1e-3f);
    float worst = 0;
    for (int y = 0; y < truth.ny; ++y) {
      for (int x = 0; x < truth.nx; ++x) {
        worst = std::max(worst, std::abs(got(z, y, x) - truth(z, y, x) - offset));
      }
    }
    EXPECT_LT(worst, 2e-3f) << "image " << z;
  }
}

TEST(Unwrap, RytovPutsBorderNearZero) {
  auto phase = smooth_phase(1, 64, 64);
  for (auto& v : phase.data) v += 4.0f * std::numbers::pi_v<float>;
  Array3<cfloat> ratio(1, 64, 64);
  for (std::size_t i = 0; i < ratio.size(); ++i) ratio.data[i] = std::polar(0.8f, phase.data[i]);

  const auto r = qoraholo::rytov(ratio);
  for (std::size_t i = 0; i < ratio.size(); ++i) {
    EXPECT_NEAR(r.data[i].real(), std::log(0.8f), 1e-5f);
    EXPECT_NEAR(r.data[i].imag(), phase.data[i] - 4.0f * std::numbers::pi_v<float>, 2e-3f);
  }
}
