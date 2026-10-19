#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "qoraholo/qoraholo.hpp"

using namespace qoraholo;

namespace {

constexpr int kN = 64;
const Geometry kGeom{0.532f, 1.337f, 1.2f, 0.2f};
const std::vector<Sphere> kSphere{{0.0f, 0.0f, 0.0f, 3.0f, 1.36f}};

struct Metrics {
  double inner_mean;  // mean RI inside 0.6 R
  double rmse;        // against the rendered phantom
};

Metrics measure(const Array3<float>& ri, const Array3<float>& truth, const Sphere& s) {
  double inner = 0;
  int count = 0;
  double sq = 0;
  for (int z = 0; z < ri.nz; ++z) {
    for (int y = 0; y < ri.ny; ++y) {
      for (int x = 0; x < ri.nx; ++x) {
        const double dz = (z - ri.nz / 2) * kGeom.pixel - s.z;
        const double dy = (y - ri.ny / 2) * kGeom.pixel - s.y;
        const double dx = (x - ri.nx / 2) * kGeom.pixel - s.x;
        if (dz * dz + dy * dy + dx * dx < 0.36 * s.radius * s.radius) {
          inner += ri(z, y, x);
          ++count;
        }
        const double e = ri(z, y, x) - truth(z, y, x);
        sq += e * e;
      }
    }
  }
  return {inner / count, std::sqrt(sq / double(ri.size()))};
}

}  // namespace

TEST(Odt, RecoversSphereIndexAndGpHelps) {
  const auto illum = circular_scan(49, 0.9f, kGeom);
  const auto ratio = forward_fields(kSphere, illum, kGeom, kN, ScatteringModel::rytov);
  const auto fields = rytov(ratio);
  const auto truth = render_phantom(kSphere, kGeom, kN, kN);

  const auto plain = reconstruct(fields, illum, kGeom);
  OdtOptions gp;
  gp.gp_iterations = 20;
  const auto refined = reconstruct(fields, illum, kGeom, gp);

  const auto m0 = measure(plain.ri, truth, kSphere[0]);
  const auto m1 = measure(refined.ri, truth, kSphere[0]);
  // Measured when written: inner mean 1.3572 -> 1.3623, rmse 2.4e-3 -> 1.2e-3.
  EXPECT_NEAR(m0.inner_mean, 1.36, 4e-3);
  EXPECT_NEAR(m1.inner_mean, 1.36, 4e-3);
  EXPECT_LT(m1.rmse, 0.7 * m0.rmse);
}
