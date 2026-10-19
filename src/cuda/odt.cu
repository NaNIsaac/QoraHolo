#include <array>
#include <chrono>
#include <cmath>
#include <numbers>
#include <vector>

#include "backends.hpp"
#include "cuda/stages.cuh"

namespace qoraholo::cuda {

using detail::EwaldTarget;
using detail::OdtGrid;
using detail::parity;

namespace {

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

__global__ void normalize_hits(float2* potential, const float* weight, std::size_t total,
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

}  // namespace

OdtSolver::OdtSolver(const OdtGrid& g, int count, const Geometry& geometry, int gp_iterations,
                     Mapping mapping)
    : g_(g),
      count_(count),
      gp_iterations_(gp_iterations),
      mapping_(mapping),
      nm2_(geometry.n_medium * geometry.n_medium),
      illum_(count),
      potential_(std::size_t(g.nz) * g.n * g.n),
      weight_(potential_.size()),
      measured_(gp_iterations > 0 ? potential_.size() : 0),
      filled_(1),
      plan2_(2, std::array{g.n, g.n}.data(), count, CUFFT_C2C),
      plan3_(3, std::array{g.nz, g.n, g.n}.data(), 1, CUFFT_C2C),
      plan_list_{&plan2_, &plan3_} {
  const float k0 = 2.0f * std::numbers::pi_v<float> / geometry.wavelength;
  inv_k02_ = 1.0f / (k0 * k0);
}

void OdtSolver::set_stream(cudaStream_t s) {
  stream_ = s;
  plan2_.set_stream(s);
  plan3_.set_stream(s);
}

void OdtSolver::set_illuminations(std::span<const Illumination> illum) {
  std::vector<float3> il(count_);
  for (int a = 0; a < count_; ++a) {
    il[a] = {illum[a].fy, illum[a].fx,
             std::sqrt(g_.fm2 - illum[a].fy * illum[a].fy - illum[a].fx * illum[a].fx)};
  }
  check(cudaMemcpy(illum_.get(), il.data(), illum_.bytes(), cudaMemcpyHostToDevice),
        "upload illuminations");
}

std::size_t OdtSolver::bytes() const {
  return illum_.bytes() + potential_.bytes() + weight_.bytes() + measured_.bytes() +
         filled_.bytes();
}

void OdtSolver::run(float2* fields, StageClock* clock, int first) {
  const auto mark = [&](Mark m) {
    if (clock) clock->mark(first + m, stream_);
  };
  const std::size_t plane = std::size_t(g_.n) * g_.n;
  const std::size_t total = plane * g_.nz;
  const unsigned grid = blocks_for(total);

  plan2_.forward(fields);
  mark(Mark::fft2);

  potential_.zero(stream_);
  weight_.zero(stream_);
  if (mapping_ == Mapping::gather) {
    map_gather<<<blocks_for(plane, 128), 128, 0, stream_>>>(fields, illum_.get(), count_, g_,
                                                            potential_.get(), weight_.get());
  } else {
    map_scatter<<<blocks_for(plane * count_), 256, 0, stream_>>>(fields, illum_.get(), count_, g_,
                                                                 potential_.get(), weight_.get());
  }
  check_launch("ewald mapping");
  mark(Mark::mapping);

  filled_.zero(stream_);
  normalize_hits<<<grid, 256, 0, stream_>>>(potential_.get(), weight_.get(), total, filled_.get());
  check_launch("normalize");
  mark(Mark::normalize);

  const float inv = 1.0f / static_cast<float>(total);
  if (gp_iterations_ > 0) {
    check(cudaMemcpyAsync(measured_.get(), potential_.get(), potential_.bytes(),
                          cudaMemcpyDeviceToDevice, stream_),
          "copy measured");
    for (int it = 0; it < gp_iterations_; ++it) {
      plan3_.inverse(potential_.get());
      nonnegative<<<grid, 256, 0, stream_>>>(potential_.get(), total, inv);
      plan3_.forward(potential_.get());
      restore_measured<<<grid, 256, 0, stream_>>>(potential_.get(), measured_.get(), weight_.get(),
                                                  total);
    }
    check_launch("gerchberg-papoulis");
  }
  plan3_.inverse(potential_.get());
  mark(Mark::gp);

  // The hit counts are no longer needed; the index goes into their buffer.
  to_index<<<grid, 256, 0, stream_>>>(potential_.get(), weight_.get(), total, inv, nm2_, inv_k02_);
  check_launch("to_index");
  index_ = weight_.get();
  pitch_ = g_.n;
  mark(Mark::finish);
}

void download_rows(const float* src, std::size_t pitch, int rows, int n, float* host,
                   cudaStream_t s) {
  check(cudaMemcpy2DAsync(host, n * sizeof(float), src, pitch * sizeof(float), n * sizeof(float),
                          rows, cudaMemcpyDeviceToHost, s),
        "download index");
  check(cudaStreamSynchronize(s), "download index");
}

OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const OdtGrid& g, const Geometry& geometry, int gp_iterations,
                      Mapping mapping, OdtTimings* timings) {
  enum { start, uploaded, downloaded = uploaded + OdtSolver::marks + 1 };
  const auto t0 = std::chrono::steady_clock::now();
  OdtSolver solver(g, rytov_fields.nz, geometry, gp_iterations, mapping);
  const FftWorkArea work(solver.plans());
  solver.set_illuminations(illum);
  const double setup =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  StageClock clock(downloaded + 1);
  clock.mark(start, nullptr);
  DeviceBuffer<float2> fields(rytov_fields.size());
  fields.upload(rytov_fields.data.data());
  clock.mark(uploaded, nullptr);
  solver.run(fields.get(), &clock, uploaded + 1);

  OdtResult result;
  result.ri = Array3<float>(g.nz, g.n, g.n);
  download_rows(solver.index(), solver.pitch(), g.nz * g.n, g.n, result.ri.data.data(), nullptr);
  unsigned long long filled = 0;
  check(cudaMemcpy(&filled, solver.filled(), sizeof filled, cudaMemcpyDeviceToHost),
        "download filled");
  clock.mark(downloaded, nullptr);
  check(cudaDeviceSynchronize(), "reconstruct");
  result.filled = static_cast<double>(filled) / (static_cast<double>(g.n) * g.n * g.nz);

  if (timings) {
    const int s = uploaded + 1;
    timings->setup = setup;
    timings->upload = clock.ms(start, uploaded);
    timings->fft2 = clock.ms(uploaded, s + OdtSolver::fft2);
    timings->mapping = clock.ms(s + OdtSolver::fft2, s + OdtSolver::mapping);
    timings->normalize = clock.ms(s + OdtSolver::mapping, s + OdtSolver::normalize);
    timings->gp = clock.ms(s + OdtSolver::normalize, s + OdtSolver::gp);
    timings->finish = clock.ms(s + OdtSolver::gp, s + OdtSolver::finish);
    timings->download = clock.ms(s + OdtSolver::finish, downloaded);
  }
  return result;
}

}  // namespace qoraholo::cuda
