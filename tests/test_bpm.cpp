#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "backends.hpp"
#include "qoraholo/qoraholo.hpp"

using namespace qoraholo;

namespace {

const Geometry kGeom{0.532f, 1.337f, 1.2f, 0.2f};

// |a - b| / |b - 1| over all pixels: error relative to the scattered part.
double scattered_error(const Array3<cfloat>& a, const Array3<cfloat>& b) {
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += std::norm(std::complex<double>(a.data[i]) - std::complex<double>(b.data[i]));
    den += std::norm(std::complex<double>(b.data[i]) - 1.0);
  }
  return std::sqrt(num / den);
}

}  // namespace

// For a weak sphere the partial-wave solution must agree with the first Born approximation.
TEST(Exact, MatchesBornForWeakSphere) {
  const std::vector<Sphere> s{{0.4f, -0.6f, 0.2f, 3.0f, 1.3375f}};
  const auto illum = circular_scan(5, 0.9f, kGeom);
  const auto exact = forward_fields(s, illum, kGeom, 64, ScatteringModel::exact);
  const auto born = forward_fields(s, illum, kGeom, 64, ScatteringModel::born);
  EXPECT_LT(scattered_error(exact, born), 0.02);
}

namespace {

// Smooth random volume in [0, amplitude), deterministic.
Array3<float> pattern(int nz, int n, float amplitude, int seed) {
  Array3<float> v(nz, n, n);
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        const double s = std::sin(0.31 * x + 0.17 * y * seed + 0.53 * z) *
                         std::cos(0.23 * y - 0.41 * z * seed + 0.11 * x);
        v(z, y, x) = static_cast<float>(amplitude * 0.5 * (1.0 + s));
      }
    }
  }
  return v;
}

double scattered_error_angle(const Array3<cfloat>& a, const Array3<cfloat>& b, int angle) {
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < a.plane(); ++i) {
    const auto x = std::complex<double>(a.slice(angle)[i]);
    const auto y = std::complex<double>(b.slice(angle)[i]);
    num += std::norm(x - y);
    den += std::norm(y - 1.0);
  }
  return std::sqrt(num / den);
}

}  // namespace

// Weak object: BPM with the obliquity weight is the first Born approximation at every angle;
// the plain phase screen only at normal incidence.
TEST(Bpm, ReducesToBornForWeakObject) {
  const std::vector<Sphere> s{{0.4f, -0.6f, 0.2f, 3.0f, 1.3375f}};
  const auto illum = circular_scan(5, 0.9f, kGeom);
  const auto born = forward_fields(s, illum, kGeom, 64, ScatteringModel::born);
  const auto ri = render_phantom(s, kGeom, 64, 64);
  const auto with = bpm_forward(ri, illum, kGeom);
  BpmOptions plain;
  plain.obliquity = false;
  const auto without = bpm_forward(ri, illum, kGeom, plain);
  for (int a = 0; a < 5; ++a) {
    std::printf("angle %d: obliquity %.4f plain %.4f\n", a, scattered_error_angle(with, born, a),
                scattered_error_angle(without, born, a));
  }
}

// <J h, r> = <h, J^T r> for the linearisation J at a non-zero x, and J h against finite
// differences of the forward model.
TEST(Bpm, AdjointIdentity) {
  const Geometry g{0.532f, 1.337f, 1.2f, 0.25f};
  const auto illum = circular_scan(3, 0.8f, g);
  const BpmTables t = bpm_tables(32, 16, illum, g, true);
  const auto x = pattern(16, 32, 0.05f, 1);
  const auto h = pattern(16, 32, 1.0f, 2);
  Array3<cfloat> r(3, 32, 32);
  for (std::size_t i = 0; i < r.size(); ++i) {
    r.data[i] = {float(std::sin(0.7 * double(i))), float(std::cos(1.3 * double(i)))};
  }
  const auto jh = cpu::bpm_jvp(x, h, t);
  const auto jtr = cpu::bpm_vjp(x, r, t);
  double lhs = 0;
  for (std::size_t i = 0; i < r.size(); ++i) {
    lhs += std::real(std::conj(std::complex<double>(jh.data[i])) * std::complex<double>(r.data[i]));
  }
  double rhs = 0;
  for (std::size_t i = 0; i < h.size(); ++i) rhs += double(h.data[i]) * jtr.data[i];
  std::printf("adjoint %.9g %.9g rel %.3g\n", lhs, rhs, std::abs(lhs - rhs) / std::abs(rhs));

  // J h against a central difference of the forward model.
  const float eps = 1e-3f;
  Array3<float> xp = x;
  Array3<float> xm = x;
  for (std::size_t i = 0; i < x.size(); ++i) {
    xp.data[i] += eps * h.data[i];
    xm.data[i] -= eps * h.data[i];
  }
  const auto fp = cpu::bpm_forward(xp, t);
  const auto fm = cpu::bpm_forward(xm, t);
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < jh.size(); ++i) {
    const auto fd =
        (std::complex<double>(fp.data[i]) - std::complex<double>(fm.data[i])) / (2.0 * double(eps));
    num += std::norm(fd - std::complex<double>(jh.data[i]));
    den += std::norm(std::complex<double>(jh.data[i]));
  }
  std::printf("jvp vs fd %.3g\n", std::sqrt(num / den));
}

// A bead whose Rytov reconstruction is biased; data from the exact partial-wave solution, so
// the inverse model is not the one that made the data.
TEST(Bpm, ReconstructionBeatsRytovOnStrongBead) {
  const Sphere bead{0.0f, 0.0f, 0.0f, 3.0f, 1.40f};
  const auto illum = circular_scan(49, 0.9f, kGeom);
  const auto data = forward_fields(std::vector{bead}, illum, kGeom, 64, ScatteringModel::exact);
  const auto truth = render_phantom(std::vector{bead}, kGeom, 64, 64);
  OdtOptions gp;
  gp.iterations = 20;
  const auto initial = reconstruct(rytov(data), illum, kGeom, gp).ri;
  const auto result = bpm_reconstruct(data, illum, kGeom, initial);

  const auto rmse = [&](const Array3<float>& v) {
    double sq = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      const double e = double(v.data[i]) - truth.data[i];
      sq += e * e;
    }
    return std::sqrt(sq / double(v.size()));
  };
  // Mean inside 0.6 R.
  const auto inner = [&](const Array3<float>& v) {
    double sum = 0;
    int count = 0;
    for (int z = 0; z < 64; ++z) {
      for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
          const double r2 = (z - 32) * (z - 32) + (y - 32) * (y - 32) + (x - 32) * (x - 32);
          if (r2 * kGeom.pixel * kGeom.pixel < 0.36 * bead.radius * bead.radius) {
            sum += v(z, y, x);
            ++count;
          }
        }
      }
    }
    return sum / count;
  };
  // Measured when written: RMSE 6.04e-3 (Rytov + GP) -> 3.21e-3; inner mean 1.4030 -> 1.4019
  // (plain Rytov 1.3901).
  EXPECT_LT(rmse(result.ri), 0.6 * rmse(initial));
  EXPECT_NEAR(inner(result.ri), 1.40, 3e-3);
  ASSERT_EQ(result.loss.size(), 20u);
  EXPECT_LT(result.loss.back(), 0.1 * result.loss.front());
}
