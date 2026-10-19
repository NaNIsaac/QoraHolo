#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "qoraholo/qoraholo.hpp"

using namespace qoraholo;

namespace {

constexpr int kHolo = 128;
constexpr int kAngles = 9;

struct Data {
  Geometry holo_geom{0.532f, 1.337f, 1.2f, 0.1f};
  Geometry geom{0.532f, 1.337f, 1.2f, 0.2f};
  std::vector<Illumination> illum = circular_scan(kAngles, 0.9f, geom);
  Array3<float> holo;
  Array3<float> bg;
  PipelineOptions options;

  Data() {
    const std::vector<Sphere> cell{{0.0f, 0.4f, -0.3f, 2.5f, 1.36f},
                                   {0.0f, 0.4f, -0.3f, 1.0f, 1.38f}};
    const auto ratio = forward_fields(cell, illum, holo_geom, kHolo, ScatteringModel::rytov);
    const float df = 1.0f / (kHolo * holo_geom.pixel);
    const HologramOptions hopt{{36.0f * df, 36.0f * df}, 1.0f, 0.01f, 3};
    holo = synthesize_holograms(ratio, illum, holo_geom, hopt);
    bg = synthesize_holograms(Array3<cfloat>(kAngles, kHolo, kHolo, 1.0f), illum, holo_geom, hopt);
    options.count = kAngles;
    options.hologram_size = kHolo;
    options.field.center = {36.0f, 36.0f};
    options.field.radius = 30.0f;
    options.field.output_size = kHolo / 2;
    options.illuminations = illum;
    options.geometry = geom;
    options.odt.gp_iterations = 5;
  }

  // The same stages through the per-call functions on the CPU.
  [[nodiscard]] OdtResult reference() const {
    const auto ratio =
        field_ratio(extract_fields(holo, options.field), extract_fields(bg, options.field));
    return reconstruct(rytov(ratio), illum, geom, options.odt);
  }
};

double contrast_error(const Array3<float>& a, const Array3<float>& b, float nm) {
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (double(a.data[i]) - b.data[i]) * (double(a.data[i]) - b.data[i]);
    den += (double(b.data[i]) - nm) * (double(b.data[i]) - nm);
  }
  return std::sqrt(num / den);
}

}  // namespace

TEST(Pipeline, CpuMatchesPerCallFunctions) {
  const Data d;
  Pipeline p(d.options);
  p.set_background(d.bg);
  const OdtResult r = p.run(d.holo);
  const OdtResult ref = d.reference();
  EXPECT_LT(contrast_error(r.ri, ref.ri, d.geom.n_medium), 1e-6);
  EXPECT_DOUBLE_EQ(r.filled, ref.filled);
}

TEST(Pipeline, NeedsBackgroundAndCarrier) {
  Data d;
  Pipeline p(d.options);
  EXPECT_THROW(p.process(), std::logic_error);
  d.options.field.center.reset();
  EXPECT_THROW(Pipeline{d.options}, std::invalid_argument);
}

TEST(Pipeline, CudaMatchesCpuAndReusesPlans) {
  if (!cuda_available()) GTEST_SKIP() << "no CUDA device";
  Data d;
  const OdtResult ref = d.reference();
  for (int chunks : {1, 4}) {
    d.options.odt.backend = Backend::cuda;
    d.options.upload_chunks = chunks;
    Pipeline p(d.options);
    p.set_background(d.bg);
    const OdtResult first = p.run(d.holo);
    EXPECT_LT(contrast_error(first.ri, ref.ri, d.geom.n_medium), 1e-4);
    EXPECT_DOUBLE_EQ(first.filled, ref.filled);
    // Second call on the same plans and buffers: nothing carried over from the first.
    const OdtResult second = p.run(d.holo);
    EXPECT_EQ(first.ri.data, second.ri.data);
    EXPECT_GT(p.device_bytes(), 0u);
  }
}
