#include <algorithm>
#include <chrono>
#include <vector>

#include "backends.hpp"
#include "cuda/stages.cuh"

namespace qoraholo::cuda {

namespace {

class CudaPipeline final : public Pipeline::Impl {
 public:
  CudaPipeline(const PipelineOptions& o, const CropPlan& crop, const detail::OdtGrid& g)
      : count_(o.count),
        n_(o.hologram_size),
        m_(crop.out),
        chunk_((o.count + o.upload_chunks - 1) / o.upload_chunks),
        chunks_((o.count + chunk_ - 1) / chunk_),
        field_(n_, chunk_, count_, crop),
        unwrapper_(count_, m_, m_),
        solver_(g, count_, o.geometry, o.odt),
        input_(std::size_t(count_) * n_ * n_),
        output_(std::size_t(g.nz) * g.n * g.n),
        filled_(1),
        holo_(std::size_t(chunks_) * chunk_ * n_ * n_),
        background_(std::size_t(count_) * m_ * m_),
        fields_(background_.size()),
        psi_(background_.size()),
        logamp_(background_.size()),
        phi_(background_.size()),
        chunk_done_(chunks_),
        clock_(marks) {
    std::vector<FftPlan*> plans;
    for (const auto stage : {field_.plans(), unwrapper_.plans(), solver_.plans()}) {
      plans.insert(plans.end(), stage.begin(), stage.end());
    }
    work_ = FftWorkArea(plans);
    field_.set_stream(compute_.get());
    unwrapper_.set_stream(compute_.get());
    solver_.set_stream(compute_.get());
    solver_.set_illuminations(o.illuminations);
    shape = {g.nz, g.n, g.n};
  }

  void set_background(const Array3<float>& backgrounds) override {
    const cudaStream_t s = compute_.get();
    const std::size_t image = std::size_t(n_) * n_;
    for (int first = 0; first < count_; first += chunk_) {
      const int valid = std::min(chunk_, count_ - first);
      check(cudaMemcpyAsync(holo_.get(), backgrounds.data.data() + first * image,
                            valid * image * sizeof(float), cudaMemcpyHostToDevice, s),
            "upload backgrounds");
      field_.crop(holo_.get(), valid, background_.get(), first);
    }
    field_.inverse(background_.get());
    check(cudaStreamSynchronize(s), "set_background");
  }

  std::span<float> input() override { return {input_.get(), input_.size()}; }

  void process() override {
    const auto t0 = std::chrono::steady_clock::now();
    const cudaStream_t s = compute_.get();
    const cudaStream_t c = copy_.get();
    const std::size_t image = std::size_t(n_) * n_;
    const std::size_t total = background_.size();

    clock_.mark(start, s);
    check(cudaStreamWaitEvent(c, clock_.event(start)), "cudaStreamWaitEvent");
    for (int k = 0; k < chunks_; ++k) {
      const int first = k * chunk_;
      const int valid = std::min(chunk_, count_ - first);
      float* dst = holo_.get() + first * image;
      check(cudaMemcpyAsync(dst, input_.get() + first * image, valid * image * sizeof(float),
                            cudaMemcpyHostToDevice, c),
            "upload holograms");
      chunk_done_[k].record(c);
      check(cudaStreamWaitEvent(s, chunk_done_[k].get()), "cudaStreamWaitEvent");
      field_.crop(dst, valid, fields_.get(), first);
    }
    field_.inverse(fields_.get());
    clock_.mark(fields, s);

    ratio_phase(fields_.get(), background_.get(), total, psi_.get(), logamp_.get(), s);
    unwrapper_.unwrap(psi_.get(), phi_.get());
    unwrapper_.assemble_rytov(logamp_.get(), phi_.get(), fields_.get());
    clock_.mark(rytov, s);

    solver_.run(fields_.get(), &clock_, rytov + 1);
    const auto [nz, ny, nx] = shape;
    check(cudaMemcpy2DAsync(output_.get(), nx * sizeof(float), solver_.index(),
                            solver_.pitch() * sizeof(float), nx * sizeof(float), nz * ny,
                            cudaMemcpyDeviceToHost, s),
          "download index");
    check(cudaMemcpyAsync(filled_.get(), solver_.filled(), sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost, s),
          "download filled");
    clock_.mark(downloaded, s);
    check(cudaStreamSynchronize(s), "process");

    filled = static_cast<double>(*filled_.get()) / static_cast<double>(output_.size());
    const int o = rytov + 1;
    timings.upload = clock_.ms_to(start, chunk_done_.back());
    timings.fields = clock_.ms(start, fields);
    timings.rytov = clock_.ms(fields, rytov);
    timings.fft2 = clock_.ms(rytov, o + OdtSolver::fft2);
    timings.mapping = clock_.ms(o + OdtSolver::fft2, o + OdtSolver::mapping);
    timings.normalize = clock_.ms(o + OdtSolver::mapping, o + OdtSolver::normalize);
    timings.solve = clock_.ms(o + OdtSolver::normalize, o + OdtSolver::solve);
    timings.finish = clock_.ms(o + OdtSolver::solve, o + OdtSolver::finish);
    timings.download = clock_.ms(o + OdtSolver::finish, downloaded);
    timings.total =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }

  [[nodiscard]] std::span<const float> output() const override {
    return {output_.get(), output_.size()};
  }

 private:
  enum { start, fields, rytov, downloaded = rytov + 1 + OdtSolver::marks, marks };

  int count_;
  int n_;
  int m_;
  int chunk_;
  int chunks_;
  Stream compute_;
  Stream copy_;
  FieldStage field_;
  Unwrapper unwrapper_;
  OdtSolver solver_;
  FftWorkArea work_;
  PinnedBuffer<float> input_;
  PinnedBuffer<float> output_;
  PinnedBuffer<unsigned long long> filled_;
  DeviceBuffer<float> holo_;
  DeviceBuffer<float2> background_;
  DeviceBuffer<float2> fields_;
  DeviceBuffer<float> psi_;
  DeviceBuffer<float> logamp_;
  DeviceBuffer<float> phi_;
  std::vector<Event> chunk_done_;
  StageClock clock_;
};

}  // namespace

std::unique_ptr<Pipeline::Impl> make_pipeline(const PipelineOptions& options, const CropPlan& crop,
                                              const detail::OdtGrid& grid) {
  std::size_t free_before = 0;
  std::size_t free_after = 0;
  std::size_t device_total = 0;
  check(cudaMemGetInfo(&free_before, &device_total), "cudaMemGetInfo");
  auto p = std::make_unique<CudaPipeline>(options, crop, grid);
  check(cudaMemGetInfo(&free_after, &device_total), "cudaMemGetInfo");
  p->device_bytes = free_before > free_after ? free_before - free_after : 0;
  return p;
}

}  // namespace qoraholo::cuda
