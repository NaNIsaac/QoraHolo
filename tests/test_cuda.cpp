#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "qoraholo/qoraholo.hpp"

using namespace qoraholo;

namespace {

template <class T>
double relative_error(const Array3<T>& a, const Array3<T>& b) {
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += std::norm(std::complex<double>(a.data[i]) - std::complex<double>(b.data[i]));
    den += std::norm(std::complex<double>(b.data[i]));
  }
  return std::sqrt(num / den);
}

class Cuda : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!cuda_available()) GTEST_SKIP() << "no CUDA device";
  }
  const Geometry geom{0.532f, 1.337f, 1.2f, 0.2f};
  const std::vector<Sphere> spheres{{0.0f, 0.4f, -0.3f, 2.5f, 1.36f},
                                    {0.0f, 0.4f, -0.3f, 1.0f, 1.38f}};
};

}  // namespace

TEST_F(Cuda, FieldExtractionMatchesCpu) {
  const auto illum = circular_scan(5, 0.8f, geom);
  Geometry hg = geom;
  hg.pixel = 0.05f;
  const auto ratio = forward_fields(spheres, illum, hg, 128, ScatteringModel::rytov);
  const auto holo = synthesize_holograms(ratio, illum, hg, {{5.0f, 5.0f}, 1.0f, 0.02f, 1});
  FieldOptions fo;
  fo.center = {32.0f, 32.0f};
  fo.radius = 16.0f;
  fo.output_size = 64;
  EXPECT_LT(relative_error(extract_fields(holo, fo, Backend::cuda), extract_fields(holo, fo)),
            1e-4);
}

// The GPU reads R2C half spectra; a crop that straddles k_x = 0 needs the conjugate bins.
TEST_F(Cuda, FieldExtractionAcrossZeroKx) {
  const auto illum = circular_scan(3, 0.5f, geom);
  Geometry hg = geom;
  hg.pixel = 0.05f;
  const float df = 1.0f / (128 * hg.pixel);
  const auto ratio = forward_fields(spheres, illum, hg, 128, ScatteringModel::rytov);
  const auto holo =
      synthesize_holograms(ratio, illum, hg, {{-30.0f * df, 4.0f * df}, 1.0f, 0.02f, 1});
  FieldOptions fo;
  fo.center = {-30.0f, 4.0f};
  fo.radius = 12.0f;
  fo.output_size = 32;
  EXPECT_LT(relative_error(extract_fields(holo, fo, Backend::cuda), extract_fields(holo, fo)),
            1e-4);
}

TEST_F(Cuda, UnwrapAndRytovMatchCpu) {
  const auto illum = circular_scan(5, 0.8f, geom);
  const auto ratio = forward_fields(spheres, illum, geom, 64, ScatteringModel::rytov);
  EXPECT_LT(relative_error(rytov(ratio, Backend::cuda), rytov(ratio)), 1e-4);
}

TEST_F(Cuda, ReconstructionMatchesCpu) {
  const auto illum = circular_scan(25, 0.9f, geom);
  const auto fields = rytov(forward_fields(spheres, illum, geom, 64, ScatteringModel::rytov));
  for (Mapping mapping : {Mapping::gather, Mapping::scatter}) {
    OdtOptions cpu_opt;
    cpu_opt.gp_iterations = 5;
    OdtOptions gpu_opt = cpu_opt;
    gpu_opt.backend = Backend::cuda;
    gpu_opt.mapping = mapping;
    const auto a = reconstruct(fields, illum, geom, cpu_opt);
    const auto b = reconstruct(fields, illum, geom, gpu_opt);
    // Compare the index contrast; the medium index would hide differences.
    Array3<float> da = a.ri;
    Array3<float> db = b.ri;
    for (auto& v : da.data) v -= geom.n_medium;
    for (auto& v : db.data) v -= geom.n_medium;
    EXPECT_LT(relative_error(db, da), 1e-4);
    EXPECT_DOUBLE_EQ(a.filled, b.filled);
  }
}
