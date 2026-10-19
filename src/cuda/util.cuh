#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

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
  explicit DeviceBuffer(std::size_t n) : n_(n) {
    check(cudaMalloc(&p_, n * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(p_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() const { return p_; }
  [[nodiscard]] std::size_t size() const { return n_; }
  void upload(const void* host) {
    check(cudaMemcpy(p_, host, n_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  void download(void* host) const {
    check(cudaMemcpy(host, p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
  }
  void zero() { check(cudaMemset(p_, 0, n_ * sizeof(T)), "cudaMemset"); }

 private:
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

// C2C plan over contiguous, densely packed transforms.
class FftPlan {
 public:
  FftPlan(int rank, int* dims, int batch) {
    long long dist = 1;
    for (int i = 0; i < rank; ++i) dist *= dims[i];
    check(cufftPlanMany(&h_, rank, dims, nullptr, 1, static_cast<int>(dist), nullptr, 1,
                        static_cast<int>(dist), CUFFT_C2C, batch),
          "cufftPlanMany");
  }
  // 1D transforms of length n with the given element stride and distance between transforms.
  FftPlan(int n, int stride, int dist, int batch) {
    check(cufftPlanMany(&h_, 1, &n, &n, stride, dist, &n, stride, dist, CUFFT_C2C, batch),
          "cufftPlanMany");
  }
  ~FftPlan() { cufftDestroy(h_); }
  FftPlan(const FftPlan&) = delete;
  FftPlan& operator=(const FftPlan&) = delete;

  void forward(float2* data) const {
    check(cufftExecC2C(h_, reinterpret_cast<cufftComplex*>(data),
                       reinterpret_cast<cufftComplex*>(data), CUFFT_FORWARD),
          "cufftExecC2C");
  }
  void inverse(float2* data) const {
    check(cufftExecC2C(h_, reinterpret_cast<cufftComplex*>(data),
                       reinterpret_cast<cufftComplex*>(data), CUFFT_INVERSE),
          "cufftExecC2C");
  }

 private:
  cufftHandle h_{};
};

inline unsigned blocks_for(std::size_t n, unsigned threads = 256) {
  return static_cast<unsigned>((n + threads - 1) / threads);
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__device__ __forceinline__ float2 cscale(float2 a, float s) { return {a.x * s, a.y * s}; }

// Device-resident unwrap shared by unwrap() and rytov(): psi and phi are (nz, ny, nx).
void unwrap_device(const float* psi, float* phi, int nz, int ny, int nx);

}  // namespace qoraholo::cuda
