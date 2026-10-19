#include <cmath>
#include <numbers>
#include <vector>

#include "backends.hpp"
#include "cuda/util.cuh"

namespace qoraholo::cuda {

using detail::EwaldTarget;
using detail::OdtGrid;
using detail::parity;

namespace {

__device__ __forceinline__ std::size_t grid_index() {
  return blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
}

// One thread per lateral column (ky, kx): it alone writes that column, so plain adds suffice.
// Neighbouring threads tend to hit the same k_z, which keeps the writes mostly coalesced.
__global__ void map_gather(const float2* spectra, const float3* illum, int count, OdtGrid g,
                           float2* potential, float* weight) {
  const std::size_t plane = std::size_t(g.n) * g.n;
  const std::size_t col = grid_index();
  if (col >= plane) return;
  const int ky = static_cast<int>(col / g.n);
  const int kx = static_cast<int>(col % g.n);
  for (int a = 0; a < count; ++a) {
    const float3 il = illum[a];
    EwaldTarget t{};
    if (!detail::ewald_target(g, ky, kx, il.x, il.y, il.z, t)) continue;
    const float2 s = spectra[a * plane + col];
    const float f = -g.gain * t.nuz * parity(t.kz);
    const std::size_t i = t.kz * plane + col;
    float2 p = potential[i];
    // (0, f) * s
    p.x += -f * s.y;
    p.y += f * s.x;
    potential[i] = p;
    weight[i] += 1.0f;
  }
}

// One thread per (illumination, pixel); collisions between illuminations need atomics.
__global__ void map_scatter(const float2* spectra, const float3* illum, int count, OdtGrid g,
                            float2* potential, float* weight) {
  const std::size_t plane = std::size_t(g.n) * g.n;
  const std::size_t i = grid_index();
  if (i >= plane * count) return;
  const int a = static_cast<int>(i / plane);
  const std::size_t col = i % plane;
  const int ky = static_cast<int>(col / g.n);
  const int kx = static_cast<int>(col % g.n);
  const float3 il = illum[a];
  EwaldTarget t{};
  if (!detail::ewald_target(g, ky, kx, il.x, il.y, il.z, t)) return;
  const float2 s = spectra[i];
  const float f = -g.gain * t.nuz * parity(t.kz);
  const std::size_t j = t.kz * plane + col;
  atomicAdd(&potential[j].x, -f * s.y);
  atomicAdd(&potential[j].y, f * s.x);
  atomicAdd(&weight[j], 1.0f);
}

__global__ void normalize(float2* potential, const float* weight, std::size_t total,
                          unsigned long long* filled) {
  __shared__ unsigned int block_count;
  if (threadIdx.x == 0) block_count = 0;
  __syncthreads();
  const std::size_t i = grid_index();
  if (i < total && weight[i] > 0) {
    potential[i] = cscale(potential[i], 1.0f / weight[i]);
    atomicAdd(&block_count, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0 && block_count > 0) atomicAdd(filled, block_count);
}

__global__ void nonnegative(float2* potential, std::size_t total, float scale) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  potential[i] = {fmaxf(potential[i].x * scale, 0.0f), 0.0f};
}

__global__ void restore_measured(float2* potential, const float2* measured, const float* weight,
                                 std::size_t total) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  if (weight[i] > 0) potential[i] = measured[i];
}

__global__ void to_index(const float2* potential, float* ri, std::size_t total, float scale,
                         float nm2, float inv_k02) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  ri[i] =
      detail::index_from_potential(potential[i].x * scale, potential[i].y * scale, nm2, inv_k02);
}

class EventTimer {
 public:
  EventTimer() {
    check(cudaEventCreate(&start_), "cudaEventCreate");
    check(cudaEventCreate(&stop_), "cudaEventCreate");
    check(cudaEventRecord(start_), "cudaEventRecord");
  }
  ~EventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  double lap() {
    check(cudaEventRecord(stop_), "cudaEventRecord");
    check(cudaEventSynchronize(stop_), "cudaEventSynchronize");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
    std::swap(start_, stop_);
    return ms;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

}  // namespace

OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const OdtGrid& g, const Geometry& geometry, int gp_iterations,
                      Mapping mapping, OdtTimings* timings) {
  OdtTimings t_local;
  OdtTimings& tm = timings ? *timings : t_local;
  const int count = rytov_fields.nz;
  const std::size_t plane = std::size_t(g.n) * g.n;
  const std::size_t total = plane * g.nz;
  EventTimer timer;

  std::vector<float3> il(count);
  for (int a = 0; a < count; ++a) {
    il[a] = {illum[a].fy, illum[a].fx,
             std::sqrt(g.fm2 - illum[a].fy * illum[a].fy - illum[a].fx * illum[a].fx)};
  }
  DeviceBuffer<float3> d_illum(count);
  d_illum.upload(il.data());
  DeviceBuffer<float2> spectra(rytov_fields.size());
  spectra.upload(rytov_fields.data.data());
  DeviceBuffer<float2> potential(total);
  DeviceBuffer<float> weight(total);
  potential.zero();
  weight.zero();
  tm.upload = timer.lap();

  int dims2[2] = {g.n, g.n};
  FftPlan(2, dims2, count).forward(spectra.get());
  tm.fft2 = timer.lap();

  if (mapping == Mapping::gather) {
    map_gather<<<blocks_for(plane, 128), 128>>>(spectra.get(), d_illum.get(), count, g,
                                                potential.get(), weight.get());
  } else {
    map_scatter<<<blocks_for(plane * count), 256>>>(spectra.get(), d_illum.get(), count, g,
                                                    potential.get(), weight.get());
  }
  check_launch("ewald mapping");
  tm.mapping = timer.lap();

  DeviceBuffer<unsigned long long> filled(1);
  filled.zero();
  const unsigned grid = blocks_for(total);
  normalize<<<grid, 256>>>(potential.get(), weight.get(), total, filled.get());
  check_launch("normalize");
  tm.normalize = timer.lap();

  int dims3[3] = {g.nz, g.n, g.n};
  const FftPlan plan3(3, dims3, 1);
  const float inv = 1.0f / static_cast<float>(total);
  if (gp_iterations > 0) {
    DeviceBuffer<float2> measured(total);
    check(cudaMemcpy(measured.get(), potential.get(), total * sizeof(float2),
                     cudaMemcpyDeviceToDevice),
          "copy measured");
    for (int it = 0; it < gp_iterations; ++it) {
      plan3.inverse(potential.get());
      nonnegative<<<grid, 256>>>(potential.get(), total, inv);
      plan3.forward(potential.get());
      restore_measured<<<grid, 256>>>(potential.get(), measured.get(), weight.get(), total);
    }
    check_launch("gerchberg-papoulis");
  }
  plan3.inverse(potential.get());
  tm.gp = timer.lap();

  const float k0 = 2.0f * std::numbers::pi_v<float> / geometry.wavelength;
  DeviceBuffer<float> ri(total);
  to_index<<<grid, 256>>>(potential.get(), ri.get(), total, inv,
                          geometry.n_medium * geometry.n_medium, 1.0f / (k0 * k0));
  check_launch("to_index");
  tm.finish = timer.lap();

  OdtResult result;
  result.ri = Array3<float>(g.nz, g.n, g.n);
  ri.download(result.ri.data.data());
  unsigned long long host_filled = 0;
  filled.download(&host_filled);
  result.filled = static_cast<double>(host_filled) / static_cast<double>(total);
  tm.download = timer.lap();
  return result;
}

}  // namespace qoraholo::cuda
