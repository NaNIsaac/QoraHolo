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
  gp.iterations = 20;
  const auto refined = reconstruct(fields, illum, kGeom, gp);

  const auto m0 = measure(plain.ri, truth, kSphere[0]);
  const auto m1 = measure(refined.ri, truth, kSphere[0]);
  // Measured when written: inner mean 1.3572 -> 1.3623, rmse 2.4e-3 -> 1.2e-3.
  EXPECT_NEAR(m0.inner_mean, 1.36, 4e-3);
  EXPECT_NEAR(m1.inner_mean, 1.36, 4e-3);
  EXPECT_LT(m1.rmse, 0.7 * m0.rmse);
}

// The R2C loop reproduces the C2C iterates; only the imaginary part of the last inverse
// transform, which C2C passes into the index, is dropped. That is second order in the contrast.
TEST(Odt, RealGpMatchesComplexGp) {
  const auto illum = circular_scan(49, 0.9f, kGeom);
  const auto fields = rytov(forward_fields(kSphere, illum, kGeom, kN, ScatteringModel::rytov));
  OdtOptions real;
  real.iterations = 10;
  OdtOptions complex = real;
  complex.gp_transform = GpTransform::c2c;
  const auto a = reconstruct(fields, illum, kGeom, real);
  const auto b = reconstruct(fields, illum, kGeom, complex);
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < a.ri.size(); ++i) {
    const double d = double(a.ri.data[i]) - b.ri.data[i];
    const double c = double(b.ri.data[i]) - kGeom.n_medium;
    num += d * d;
    den += c * c;
  }
  // Measured when written: 5e-6 relative.
  EXPECT_LT(std::sqrt(num / den), 1e-4);
  EXPECT_DOUBLE_EQ(a.filled, b.filled);
}

TEST(Odt, TvReducesErrorAgainstGp) {
  const auto illum = circular_scan(49, 0.9f, kGeom);
  const auto fields = rytov(forward_fields(kSphere, illum, kGeom, kN, ScatteringModel::rytov));
  const auto truth = render_phantom(kSphere, kGeom, kN, kN);
  OdtOptions gp;
  gp.iterations = 20;
  OdtOptions tv = gp;
  tv.solver = Solver::tv;
  const auto m_gp = measure(reconstruct(fields, illum, kGeom, gp).ri, truth, kSphere[0]);
  const auto m_tv = measure(reconstruct(fields, illum, kGeom, tv).ri, truth, kSphere[0]);
  // Measured when written: rmse 1.17e-3 (GP) and 1.01e-3 (TV), TV inner mean 1.3600.
  EXPECT_NEAR(m_tv.inner_mean, 1.36, 4e-3);
  EXPECT_LT(m_tv.rmse, m_gp.rmse);
}
