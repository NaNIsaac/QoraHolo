#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qoraholo::cuda {

inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

inline void check(cufftResult r, const char* what) {
  if (r != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cuFFT error " + std::to_string(int(r)));
  }
}

inline void check_launch(const char* what) { check(cudaGetLastError(), what); }

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) : n_(n) {
    if (n > 0) check(cudaMalloc(&p_, n * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(p_); }
  DeviceBuffer(DeviceBuffer&& o) noexcept
      : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    std::swap(p_, o.p_);
    std::swap(n_, o.n_);
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() const { return p_; }
  [[nodiscard]] std::size_t size() const { return n_; }
  [[nodiscard]] std::size_t bytes() const { return n_ * sizeof(T); }
  void upload(const void* host, cudaStream_t s = nullptr) {
    check(cudaMemcpyAsync(p_, host, bytes(), cudaMemcpyHostToDevice, s), "upload");
  }
  void download(void* host, cudaStream_t s = nullptr) const {
    check(cudaMemcpyAsync(host, p_, bytes(), cudaMemcpyDeviceToHost, s), "download");
    check(cudaStreamSynchronize(s), "download");
  }
  void zero(cudaStream_t s = nullptr) { check(cudaMemsetAsync(p_, 0, bytes(), s), "cudaMemset"); }

 private:
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

// Page-locked host memory: the only kind the copy engines can read without a staging copy.
template <class T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(std::size_t n) : n_(n) {
    if (n > 0) check(cudaMallocHost(&p_, n * sizeof(T)), "cudaMallocHost");
  }
  ~PinnedBuffer() { cudaFreeHost(p_); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  [[nodiscard]] T* get() const { return p_; }
  [[nodiscard]] std::size_t size() const { return n_; }

 private:
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

class Stream {
 public:
  Stream() { check(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), "cudaStreamCreate"); }
  ~Stream() { cudaStreamDestroy(s_); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  [[nodiscard]] cudaStream_t get() const { return s_; }

 private:
  cudaStream_t s_{};
};

class Event {
 public:
  Event() { check(cudaEventCreate(&e_), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(e_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  [[nodiscard]] cudaEvent_t get() const { return e_; }
  void record(cudaStream_t s) { check(cudaEventRecord(e_, s), "cudaEventRecord"); }

 private:
  cudaEvent_t e_{};
};

// Events recorded at stage boundaries on one stream; read after the stream has finished, so
// timing adds no synchronisation inside the pipeline.
class StageClock {
 public:
  explicit StageClock(int marks) : events_(marks) {}
  void mark(int i, cudaStream_t s) { events_.at(i).record(s); }
  [[nodiscard]] cudaEvent_t event(int i) const { return events_.at(i).get(); }
  [[nodiscard]] double ms(int from, int to) const { return ms_to(from, events_.at(to)); }
  [[nodiscard]] double ms_to(int from, const Event& to) const {
    float t = 0;
    check(cudaEventElapsedTime(&t, events_.at(from).get(), to.get()), "cudaEventElapsedTime");
    return t;
  }

 private:
  std::vector<Event> events_;
};

// Advanced data layout of one side of a cuFFT plan (cufftMakePlanMany's embed, stride, dist).
// At namespace scope because gcc rejects a nested struct with member initializers as a
// default argument inside its enclosing class.
struct FftLayout {
  const int* embed = nullptr;  // nullptr: dense
  int stride = 1;
  int dist = 0;
};

// cuFFT plan created without its own work area. Plans that run one after another on the same
// stream can then share a single area sized for the largest of them (FftWorkArea).
class FftPlan {
 public:
  using Layout = FftLayout;

  FftPlan(int rank, const int* dims, int batch, cufftType type, Layout in = {}, Layout out = {}) {
    check(cufftCreate(&h_), "cufftCreate");
    check(cufftSetAutoAllocation(h_, 0), "cufftSetAutoAllocation");
    check(cufftMakePlanMany(h_, rank, const_cast<int*>(dims), const_cast<int*>(in.embed), in.stride,
                            in.dist, const_cast<int*>(out.embed), out.stride, out.dist, type, batch,
                            &work_),
          "cufftMakePlanMany");
  }
  ~FftPlan() { cufftDestroy(h_); }
  FftPlan(const FftPlan&) = delete;
  FftPlan& operator=(const FftPlan&) = delete;

  [[nodiscard]] std::size_t work_size() const { return work_; }
  void set_work_area(void* p) { check(cufftSetWorkArea(h_, p), "cufftSetWorkArea"); }
  void set_stream(cudaStream_t s) { check(cufftSetStream(h_, s), "cufftSetStream"); }

  void forward(float2* data) const { c2c(data, data, CUFFT_FORWARD); }
  void inverse(float2* data) const { c2c(data, data, CUFFT_INVERSE); }
  void c2c(float2* in, float2* out, int direction) const {
    check(cufftExecC2C(h_, reinterpret_cast<cufftComplex*>(in),
                       reinterpret_cast<cufftComplex*>(out), direction),
          "cufftExecC2C");
  }
  void r2c(float* in, float2* out) const {
    check(cufftExecR2C(h_, in, reinterpret_cast<cufftComplex*>(out)), "cufftExecR2C");
  }
  void c2r(float2* in, float* out) const {
    check(cufftExecC2R(h_, reinterpret_cast<cufftComplex*>(in), out), "cufftExecC2R");
  }

 private:
  cufftHandle h_{};
  std::size_t work_ = 0;
};

// One device work area shared by plans that never run concurrently.
class FftWorkArea {
 public:
  FftWorkArea() = default;
  explicit FftWorkArea(std::span<FftPlan* const> plans) {
    std::size_t need = 0;
    for (const FftPlan* p : plans) need = std::max(need, p->work_size());
    area_ = DeviceBuffer<std::byte>(need);
    for (FftPlan* p : plans) p->set_work_area(area_.get());
  }
  [[nodiscard]] std::size_t bytes() const { return area_.size(); }

 private:
  DeviceBuffer<std::byte> area_;
};

inline unsigned blocks_for(std::size_t n, unsigned threads = 256) {
  return static_cast<unsigned>((n + threads - 1) / threads);
}

__device__ __forceinline__ std::size_t grid_index() {
  return blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__device__ __forceinline__ float2 cscale(float2 a, float s) { return {a.x * s, a.y * s}; }

}  // namespace qoraholo::cuda
