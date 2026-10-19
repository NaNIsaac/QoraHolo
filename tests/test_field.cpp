#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "qoraholo/qoraholo.hpp"

using namespace qoraholo;

namespace {

// Hologram grid: 128 px at 55 nm; fields come out at 64 px, 110 nm.
constexpr int kHolo = 128;
constexpr int kField = 64;

Geometry holo_geometry() { return {0.532f, 1.337f, 1.2f, 0.055f}; }

Geometry field_geometry() {
  auto g = holo_geometry();
  g.pixel *= static_cast<float>(kHolo) / kField;
  return g;
}

const std::vector<Sphere> kSpheres{{0.0f, 0.3f, -0.2f, 2.0f, 1.36f}};
const std::vector<Illumination> kIllum{{0.0f, 0.0f}, {0.4f, -0.7f}};

// Frequency of a whole number of cycles across the hologram, so nothing leaks between bins.
float on_grid(float bins) { return bins / (kHolo * holo_geometry().pixel); }

double max_error(const Array3<cfloat>& a, const Array3<cfloat>& b) {
  double worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, double(std::abs(a.data[i] - b.data[i])));
  return worst;
}

}  // namespace

TEST(Field, FindsCarrierSideband) {
  const HologramOptions opt{{5.0f, 5.0f}, 1.0f, 0.0f, 1};
  const std::vector<Illumination> normal{{}};
  const auto bg =
      synthesize_holograms(Array3<cfloat>(1, kHolo, kHolo, 1.0f), normal, holo_geometry(), opt);
  const auto s = find_sideband(bg);
  const float expected = 5.0f * kHolo * holo_geometry().pixel;  // 35.2 px
  EXPECT_EQ(s.iy, 35);
  EXPECT_EQ(s.ix, 35);
  EXPECT_NEAR(s.y, expected, 0.05f);
  EXPECT_NEAR(s.x, expected, 0.05f);
}

TEST(Field, SyntheticHologramGivesForwardField) {
  // Born fields are band limited to the pupil, so a crop that covers the pupil loses nothing.
  // With carrier and illumination on the DFT grid the recovery is exact up to rounding.
  const std::vector<Illumination> illum{{0.0f, 0.0f}, {on_grid(3), on_grid(-5)}};
  const auto truth_holo =
      forward_fields(kSpheres, illum, holo_geometry(), kHolo, ScatteringModel::born);
  const auto truth =
      forward_fields(kSpheres, illum, field_geometry(), kField, ScatteringModel::born);
  const HologramOptions opt{{on_grid(35), on_grid(35)}, 1.0f, 0.0f, 1};
  const auto holo = synthesize_holograms(truth_holo, illum, holo_geometry(), opt);
  const auto bg =
      synthesize_holograms(Array3<cfloat>(2, kHolo, kHolo, 1.0f), illum, holo_geometry(), opt);

  FieldOptions fo;
  fo.center = {35.0f, 35.0f};
  fo.radius = 17.0f;
  fo.output_size = kField;
  const auto ratio = field_ratio(extract_fields(holo, fo), extract_fields(bg, fo));
  EXPECT_LT(max_error(ratio, truth), 1e-4);
}

// Off-grid carrier and tilt leak across the spectrum; the error concentrates at the borders.
double interior_rms(const Array3<cfloat>& a, const Array3<cfloat>& b, int margin) {
  double sq = 0;
  int count = 0;
  for (int z = 0; z < a.nz; ++z) {
    for (int y = margin; y < a.ny - margin; ++y) {
      for (int x = margin; x < a.nx - margin; ++x) {
        sq += std::norm(a(z, y, x) - b(z, y, x));
        ++count;
      }
    }
  }
  return std::sqrt(sq / count);
}

TEST(Field, NoisyHologramStaysClose) {
  const auto truth_holo =
      forward_fields(kSpheres, kIllum, holo_geometry(), kHolo, ScatteringModel::born);
  const auto truth =
      forward_fields(kSpheres, kIllum, field_geometry(), kField, ScatteringModel::born);
  // Noise of 2% of the mean intensity (2.0) per pixel.
  const HologramOptions opt{{5.0f, 5.0f}, 1.0f, 0.04f, 3};
  const auto holo = synthesize_holograms(truth_holo, kIllum, holo_geometry(), opt);
  auto bg_opt = opt;
  bg_opt.seed = 4;
  const auto bg =
      synthesize_holograms(Array3<cfloat>(2, kHolo, kHolo, 1.0f), kIllum, holo_geometry(), bg_opt);
  FieldOptions fo;
  fo.center = {35.2f, 35.2f};
  fo.radius = 17.0f;
  fo.output_size = kField;
  const auto ratio = field_ratio(extract_fields(holo, fo), extract_fields(bg, fo));
  EXPECT_LT(interior_rms(ratio, truth, 8), 0.02);
}

TEST(Field, AutoDetectionMatchesExplicitCentre) {
  const std::vector<Illumination> normal{{}};
  const auto truth =
      forward_fields(kSpheres, normal, holo_geometry(), kHolo, ScatteringModel::born);
  const auto holo = synthesize_holograms(truth, normal, holo_geometry());
  FieldOptions fo;
  fo.radius = 17.0f;
  const auto automatic = extract_fields(holo, fo);
  fo.center = {35.0f, 35.0f};
  const auto given = extract_fields(holo, fo);
  EXPECT_LT(max_error(automatic, given), 1e-6);
}
