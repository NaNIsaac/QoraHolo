#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "backends.hpp"
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
  struct Case {
    Mapping mapping;
    GpTransform transform;
  };
  for (const Case c :
       {Case{Mapping::gather, GpTransform::r2c}, Case{Mapping::scatter, GpTransform::r2c},
        Case{Mapping::gather, GpTransform::c2c}}) {
    OdtOptions cpu_opt;
    cpu_opt.iterations = 5;
    cpu_opt.gp_transform = c.transform;
    OdtOptions gpu_opt = cpu_opt;
    gpu_opt.backend = Backend::cuda;
    gpu_opt.mapping = c.mapping;
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

TEST_F(Cuda, TvMatchesCpu) {
  const auto illum = circular_scan(25, 0.9f, geom);
  const auto fields = rytov(forward_fields(spheres, illum, geom, 64, ScatteringModel::rytov));
  OdtOptions cpu_opt;
  cpu_opt.iterations = 10;
  cpu_opt.solver = Solver::tv;
  OdtOptions gpu_opt = cpu_opt;
  gpu_opt.backend = Backend::cuda;
  Array3<float> da = reconstruct(fields, illum, geom, cpu_opt).ri;
  Array3<float> db = reconstruct(fields, illum, geom, gpu_opt).ri;
  for (auto& v : da.data) v -= geom.n_medium;
  for (auto& v : db.data) v -= geom.n_medium;
  EXPECT_LT(relative_error(db, da), 1e-4);
}

#ifdef QORAHOLO_HAS_CUDA
// The fused and tiled FGP kernels do the arithmetic of the two-kernel version in the same
// order. 60 x 60 x 62 leaves partial tiles along every axis.
TEST_F(Cuda, TvKernelVariantsAgree) {
  const auto illum = circular_scan(25, 0.9f, geom);
  const auto fields = rytov(forward_fields(spheres, illum, geom, 60, ScatteringModel::rytov));
  OdtOptions opt;
  opt.nz = 62;
  opt.iterations = 5;
  opt.solver = Solver::tv;
  const auto grid = make_grid(60, 62, geom);
  const auto ref = cuda::reconstruct(fields, illum, grid, geom, opt, nullptr, TvKernel::separate);
  for (TvKernel k : {TvKernel::fused, TvKernel::tiled, TvKernel::marching}) {
    const auto got = cuda::reconstruct(fields, illum, grid, geom, opt, nullptr, k);
    double worst = 0;
    for (std::size_t i = 0; i < got.ri.size(); ++i) {
      worst = std::max(worst, std::abs(double(got.ri.data[i]) - ref.ri.data[i]));
    }
    EXPECT_LT(worst, 1e-6);
  }
}
#endif
