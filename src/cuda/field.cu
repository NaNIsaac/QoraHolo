#include "backends.hpp"
#include "cuda/util.cuh"

namespace qoraholo::cuda {

using detail::parity;

namespace {

__global__ void modulate_real(const float* in, float2* out, std::size_t total, int n) {
  const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (i >= total) return;
  const int x = static_cast<int>(i % n);
  const int y = static_cast<int>((i / n) % n);
  out[i] = {in[i] * parity(x + y), 0.0f};
}

// Copies the disc around the sideband centre into an m x m spectrum with the centre at m / 2.
__global__ void crop_sideband(const float2* spec, float2* out, int count, int n, int m, int cy,
                              int cx, float r2, float sign) {
  const std::size_t total = std::size_t(count) * m * m;
  const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (i >= total) return;
  const int x = static_cast<int>(i % m);
  const int y = static_cast<int>((i / m) % m);
  const int z = static_cast<int>(i / (std::size_t(m) * m));
  const int dy = y - m / 2;
  const int dx = x - m / 2;
  const int sy = n / 2 + cy + dy;
  const int sx = n / 2 + cx + dx;
  float2 v{0.0f, 0.0f};
  if (static_cast<float>(dy * dy + dx * dx) <= r2 && sy >= 0 && sy < n && sx >= 0 && sx < n) {
    v = cscale(spec[(std::size_t(z) * n + sy) * n + sx], sign);
  }
  out[i] = v;
}

__global__ void demodulate(float2* data, std::size_t total, int m, float scale) {
  const std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  if (i >= total) return;
  const int x = static_cast<int>(i % m);
  const int y = static_cast<int>((i / m) % m);
  data[i] = cscale(data[i], scale * parity(x + y));
}

}  // namespace

Array3<cfloat> extract_fields(const Array3<float>& holograms, const CropPlan& plan) {
  const int count = holograms.nz;
  const int n = holograms.nx;
  const int m = plan.out;
  DeviceBuffer<float> holo(holograms.size());
  holo.upload(holograms.data.data());
  DeviceBuffer<float2> spec(holograms.size());
  modulate_real<<<blocks_for(spec.size()), 256>>>(holo.get(), spec.get(), spec.size(), n);
  check_launch("modulate_real");
  int dims[2] = {n, n};
  FftPlan(2, dims, count).forward(spec.get());

  Array3<cfloat> out(count, m, m);
  DeviceBuffer<float2> field(out.size());
  crop_sideband<<<blocks_for(field.size()), 256>>>(spec.get(), field.get(), count, n, m, plan.cy,
                                                   plan.cx, plan.radius * plan.radius,
                                                   parity(plan.cy + plan.cx));
  check_launch("crop_sideband");
  int out_dims[2] = {m, m};
  FftPlan(2, out_dims, count).inverse(field.get());
  demodulate<<<blocks_for(field.size()), 256>>>(field.get(), field.size(), m,
                                                1.0f / (static_cast<float>(n) * n));
  check_launch("demodulate");
  field.download(out.data.data());
  return out;
}

}  // namespace qoraholo::cuda
