#include "qoraholo/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

#include "backends.hpp"

namespace qoraholo {

namespace {

double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

class CpuPipeline final : public Pipeline::Impl {
 public:
  CpuPipeline(const PipelineOptions& options, const CropPlan& crop, const detail::OdtGrid& grid)
      : options_(options),
        crop_(crop),
        grid_(grid),
        input_(options.count, options.hologram_size, options.hologram_size) {
    shape = {grid.nz, grid.n, grid.n};
  }

  void set_background(const Array3<float>& backgrounds) override {
    background_ = cpu::extract_fields(backgrounds, crop_);
  }

  std::span<float> input() override { return input_.data; }

  void process() override {
    const auto t0 = std::chrono::steady_clock::now();
    auto t = t0;
    const Array3<cfloat> ratio = field_ratio(cpu::extract_fields(input_, crop_), background_);
    timings.fields = since(t);
    t = std::chrono::steady_clock::now();
    const Array3<cfloat> phase = cpu::rytov(ratio);
    timings.rytov = since(t);
    OdtTimings odt;
    OdtResult r = cpu::reconstruct(phase, options_.illuminations, grid_, options_.geometry,
                                   options_.odt.gp_iterations, &odt);
    output_ = std::move(r.ri.data);
    filled = r.filled;
    timings.fft2 = odt.fft2;
    timings.mapping = odt.mapping;
    timings.normalize = odt.normalize;
    timings.solve = odt.gp;
    timings.finish = odt.finish;
    timings.total = since(t0);
  }

  [[nodiscard]] std::span<const float> output() const override { return output_; }

 private:
  PipelineOptions options_;
  CropPlan crop_;
  detail::OdtGrid grid_;
  Array3<float> input_;
  Array3<cfloat> background_;
  std::vector<float> output_;
};

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}

}  // namespace

Pipeline::Pipeline(PipelineOptions options) {
  const auto t0 = std::chrono::steady_clock::now();
  const int n = options.hologram_size;
  require(options.count >= 1, "Pipeline: count must be positive");
  require(n >= 4 && n % 2 == 0, "Pipeline: hologram_size must be even");
  require(options.field.center.has_value(), "Pipeline: field.center (the carrier) is required");
  require(options.upload_chunks >= 1 && options.upload_chunks <= options.count,
          "Pipeline: upload_chunks must be between 1 and count");
  const CropPlan crop = crop_plan(options.field, *options.field.center, n);
  const detail::OdtGrid grid =
      odt_grid(crop.out, options.count, options.illuminations, options.geometry, options.odt);
  if (options.odt.backend == Backend::cuda) {
#ifdef QORAHOLO_HAS_CUDA
    impl_ = cuda::make_pipeline(options, crop, grid);
#endif
  } else {
    impl_ = std::make_unique<CpuPipeline>(options, crop, grid);
  }
  impl_->setup_ms = since(t0);
}

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

void Pipeline::set_background(const Array3<float>& backgrounds) {
  const auto in = impl_->input();
  require(backgrounds.size() == in.size() && backgrounds.ny == backgrounds.nx,
          "Pipeline::set_background: shape must be (count, N, N)");
  impl_->set_background(backgrounds);
  impl_->has_background = true;
}

std::span<float> Pipeline::input() { return impl_->input(); }

void Pipeline::process() {
  if (!impl_->has_background) throw std::logic_error("Pipeline: set_background() first");
  impl_->process();
}

std::span<const float> Pipeline::output() const { return impl_->output(); }

double Pipeline::filled() const { return impl_->filled; }

OdtResult Pipeline::run(const Array3<float>& holograms) {
  const auto in = input();
  require(holograms.size() == in.size() && holograms.ny == holograms.nx,
          "Pipeline::run: shape must be (count, N, N)");
  std::copy(holograms.data.begin(), holograms.data.end(), in.begin());
  process();
  OdtResult r;
  const auto [nz, ny, nx] = impl_->shape;
  r.ri = Array3<float>(nz, ny, nx);
  const auto out = output();
  std::copy(out.begin(), out.end(), r.ri.data.begin());
  r.filled = filled();
  return r;
}

const PipelineTimings& Pipeline::timings() const { return impl_->timings; }
double Pipeline::setup_ms() const { return impl_->setup_ms; }
std::size_t Pipeline::device_bytes() const { return impl_->device_bytes; }
std::array<int, 3> Pipeline::output_shape() const { return impl_->shape; }

}  // namespace qoraholo
