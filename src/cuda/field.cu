#include <array>

#include "backends.hpp"
#include "cuda/stages.cuh"

namespace qoraholo::cuda {

using detail::parity;

namespace {

// Copies the disc around the sideband centre into an m x m spectrum with the centre at m / 2.
// half holds R2C spectra (n x (n/2 + 1)) of holograms indexed from pixel 0; bin (ky, kx) with
// kx < 0 is conj(H(-ky, -kx)). For centre-origin holograms and fields the origin bookkeeping
// reduces to one sign per image, as on the CPU.
__global__ void crop_half(const float2* half, int valid, int n, int m, int cy, int cx, float r2,
                          float sign, float2* out) {
  const std::size_t total = std::size_t(valid) * m * m;
  const std::size_t i = grid_index();
  if (i >= total) return;
  const int x = static_cast<int>(i % m);
  const int y = static_cast<int>((i / m) % m);
  const int z = static_cast<int>(i / (std::size_t(m) * m));
  const int dy = y - m / 2;
  const int dx = x - m / 2;
  const int fy = cy + dy;
  const int fx = cx + dx;
  float2 v{0.0f, 0.0f};
  if (static_cast<float>(dy * dy + dx * dx) <= r2 && fy >= -n / 2 && fy < n / 2 && fx >= -n / 2 &&
      fx < n / 2) {
    const int hw = n / 2 + 1;
    const float2* h = half + std::size_t(z) * n * hw;
    if (fx >= 0) {
      v = h[std::size_t((fy + n) % n) * hw + fx];
    } else {
      const float2 c = h[std::size_t((n - fy) % n) * hw - fx];
      v = {c.x, -c.y};
    }
    v = cscale(v, sign);
  }
  out[i] = v;
}

__global__ void demodulate_kernel(float2* data, std::size_t total, int m, float scale) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  const int x = static_cast<int>(i % m);
  const int y = static_cast<int>((i / m) % m);
  data[i] = cscale(data[i], scale * parity(x + y));
}

__global__ void ratio_phase_kernel(const float2* s, const float2* b, std::size_t total, float* psi,
                                   float* logamp) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  const float2 u = s[i];
  const float2 v = b[i];
  const float d = 1.0f / (v.x * v.x + v.y * v.y);
  const float re = (u.x * v.x + u.y * v.y) * d;
  const float im = (u.y * v.x - u.x * v.y) * d;
  psi[i] = atan2f(im, re);
  logamp[i] = logf(hypotf(re, im));
}

__global__ void split_ratio_kernel(const float2* ratio, std::size_t total, float* psi,
                                   float* logamp) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  psi[i] = atan2f(ratio[i].y, ratio[i].x);
  logamp[i] = logf(hypotf(ratio[i].x, ratio[i].y));
}

}  // namespace

FieldStage::FieldStage(int n, int batch, int count, const CropPlan& plan)
    : n_(n),
      batch_(batch),
      plan_(plan),
      spectra_(std::size_t(batch) * n * (n / 2 + 1)),
      r2c_(2, std::array{n, n}.data(), batch, CUFFT_R2C),
      inverse_(2, std::array{plan.out, plan.out}.data(), count, CUFFT_C2C),
      plan_list_{&r2c_, &inverse_} {}

void FieldStage::set_stream(cudaStream_t s) {
  stream_ = s;
  r2c_.set_stream(s);
  inverse_.set_stream(s);
}

void FieldStage::crop(const float* holograms, int valid, float2* fields, int first) {
  r2c_.r2c(const_cast<float*>(holograms), spectra_.get());
  const int m = plan_.out;
  const std::size_t total = std::size_t(valid) * m * m;
  crop_half<<<blocks_for(total), 256, 0, stream_>>>(
      spectra_.get(), valid, n_, m, plan_.cy, plan_.cx, plan_.radius * plan_.radius,
      parity(plan_.cy + plan_.cx), fields + std::size_t(first) * m * m);
  check_launch("crop_half");
}

void FieldStage::inverse(float2* fields) { inverse_.inverse(fields); }

void FieldStage::demodulate(float2* fields, int count) {
  const std::size_t total = std::size_t(count) * plan_.out * plan_.out;
  demodulate_kernel<<<blocks_for(total), 256, 0, stream_>>>(
      fields, total, plan_.out, 1.0f / (static_cast<float>(n_) * static_cast<float>(n_)));
  check_launch("demodulate");
}

void ratio_phase(const float2* sample, const float2* background, std::size_t total, float* psi,
                 float* logamp, cudaStream_t s) {
  ratio_phase_kernel<<<blocks_for(total), 256, 0, s>>>(sample, background, total, psi, logamp);
  check_launch("ratio_phase");
}

void split_ratio(const float2* ratio, std::size_t total, float* psi, float* logamp,
                 cudaStream_t s) {
  split_ratio_kernel<<<blocks_for(total), 256, 0, s>>>(ratio, total, psi, logamp);
  check_launch("split_ratio");
}

Array3<cfloat> extract_fields(const Array3<float>& holograms, const CropPlan& plan) {
  const int count = holograms.nz;
  const int m = plan.out;
  FieldStage stage(holograms.nx, count, count, plan);
  const FftWorkArea work(stage.plans());
  DeviceBuffer<float> holo(holograms.size());
  holo.upload(holograms.data.data());
  Array3<cfloat> out(count, m, m);
  DeviceBuffer<float2> field(out.size());
  stage.crop(holo.get(), count, field.get(), 0);
  stage.inverse(field.get());
  stage.demodulate(field.get(), count);
  field.download(out.data.data());
  return out;
}

}  // namespace qoraholo::cuda
