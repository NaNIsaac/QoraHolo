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

TEST_F(Cuda, BpmMatchesCpu) {
  const auto illum = circular_scan(7, 0.9f, geom);
  const auto ri = render_phantom(spheres, geom, 32, 48);
  BpmOptions gpu;
  gpu.backend = Backend::cuda;
  EXPECT_LT(relative_error(bpm_forward(ri, illum, geom, gpu), bpm_forward(ri, illum, geom)), 1e-4);
}

#ifdef QORAHOLO_HAS_CUDA
TEST_F(Cuda, BpmAdjointMatchesCpu) {
  const auto illum = circular_scan(7, 0.9f, geom);
  const auto ri = render_phantom(spheres, geom, 32, 48);
  const BpmTables t = bpm_tables(48, 32, illum, geom, true);
  Array3<float> x = ri;
  for (float& v : x.data) v -= geom.n_medium;
  Array3<cfloat> r(7, 48, 48);
  for (std::size_t i = 0; i < r.size(); ++i) {
    r.data[i] = {float(std::sin(0.7 * double(i))), float(std::cos(1.3 * double(i)))};
  }
  EXPECT_LT(relative_error(cuda::bpm_vjp(x, r, t), cpu::bpm_vjp(x, r, t)), 1e-4);
}
#endif

TEST_F(Cuda, BpmReconstructionMatchesCpu) {
  const auto illum = circular_scan(7, 0.9f, geom);
  const auto data = bpm_forward(render_phantom(spheres, geom, 32, 48), illum, geom);
  const Array3<float> initial(32, 48, 48, geom.n_medium);
  for (int batch : {0, 3}) {
    BpmSolveOptions o;
    o.iterations = 5;
    o.batch = batch;
    o.tv_weight = 1e-2f;
    const auto a = bpm_reconstruct(data, illum, geom, initial, o);
    o.backend = Backend::cuda;
    const auto b = bpm_reconstruct(data, illum, geom, initial, o);
    Array3<float> da = a.ri;
    Array3<float> db = b.ri;
    for (float& v : da.data) v -= geom.n_medium;
    for (float& v : db.data) v -= geom.n_medium;
    EXPECT_LT(relative_error(db, da), 1e-3) << batch;
    EXPECT_NEAR(b.loss.back(), a.loss.back(), 1e-3 * a.loss.back()) << batch;
  }
}

#ifdef QORAHOLO_HAS_CUDA
// The gradient over all illuminations computed in device passes of 2 (memory bound) is the same.
TEST_F(Cuda, BpmChunkedGradient) {
  const auto illum = circular_scan(7, 0.9f, geom);
  const auto data = bpm_forward(render_phantom(spheres, geom, 32, 48), illum, geom);
  const BpmTables t = bpm_tables(48, 32, illum, geom, true);
  const Array3<float> x(32, 48, 48);
  BpmSolveOptions o;
  o.iterations = 3;
  const auto a = cuda::bpm_reconstruct(data, x, t, o);
  const auto b = cuda::bpm_reconstruct(data, x, t, o, 2);
  EXPECT_LT(relative_error(b.ri, a.ri), 1e-5);
}
#endif
