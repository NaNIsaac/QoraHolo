#include <algorithm>

#include "backends.hpp"
#include "cuda/util.cuh"

namespace qoraholo::cuda {

using detail::dct_position;
using detail::wrap_phase;

namespace {

struct Shape {
  int nz, ny, nx;
  __device__ std::size_t size() const { return std::size_t(nz) * ny * nx; }
};

__device__ __forceinline__ std::size_t grid_index() {
  return blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
}

__global__ void divergence(const float* psi, float* rho, Shape s) {
  const std::size_t i = grid_index();
  if (i >= s.size()) return;
  const int x = static_cast<int>(i % s.nx);
  const int y = static_cast<int>((i / s.nx) % s.ny);
  const float p = psi[i];
  float r = 0;
  if (x + 1 < s.nx) r += wrap_phase(psi[i + 1] - p);
  if (x > 0) r -= wrap_phase(p - psi[i - 1]);
  if (y + 1 < s.ny) r += wrap_phase(psi[i + s.nx] - p);
  if (y > 0) r -= wrap_phase(p - psi[i - s.nx]);
  rho[i] = r;
}

// Coordinate along the transformed axis (2: x, 1: y) and the index with that coordinate replaced.
__device__ __forceinline__ int axis_coord(std::size_t i, Shape s, int axis) {
  return axis == 2 ? static_cast<int>(i % s.nx) : static_cast<int>((i / s.nx) % s.ny);
}
__device__ __forceinline__ std::size_t with_coord(std::size_t i, Shape s, int axis, int c) {
  const int old = axis_coord(i, s, axis);
  const std::ptrdiff_t step = axis == 2 ? 1 : s.nx;
  return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) + (c - old) * step);
}

__global__ void dct_reorder(const float* a, float2* buf, Shape s, int axis) {
  const std::size_t i = grid_index();
  if (i >= s.size()) return;
  const int len = axis == 2 ? s.nx : s.ny;
  buf[with_coord(i, s, axis, dct_position(axis_coord(i, s, axis), len))] = {a[i], 0.0f};
}

__global__ void dct_twiddle(const float2* buf, float* a, Shape s, int axis) {
  const std::size_t i = grid_index();
  if (i >= s.size()) return;
  const int len = axis == 2 ? s.nx : s.ny;
  const int k = axis_coord(i, s, axis);
  float sn, cs;
  sincospif(-static_cast<float>(k) / (2.0f * len), &sn, &cs);
  a[i] = cs * buf[i].x - sn * buf[i].y;
}

__global__ void idct_twiddle(const float* a, float2* buf, Shape s, int axis) {
  const std::size_t i = grid_index();
  if (i >= s.size()) return;
  const int len = axis == 2 ? s.nx : s.ny;
  const int k = axis_coord(i, s, axis);
  const float mirror = k == 0 ? 0.0f : a[with_coord(i, s, axis, len - k)];
  float sn, cs;
  sincospif(static_cast<float>(k) / (2.0f * len), &sn, &cs);
  buf[i] = cmul({cs, sn}, {a[i], -mirror});
}

__global__ void idct_reorder(const float2* buf, float* a, Shape s, int axis) {
  const std::size_t i = grid_index();
  if (i >= s.size()) return;
  const int len = axis == 2 ? s.nx : s.ny;
  a[i] = buf[with_coord(i, s, axis, dct_position(axis_coord(i, s, axis), len))].x /
         static_cast<float>(len);
}

__global__ void poisson_divide(float* a, Shape s) {
  const std::size_t i = grid_index();
  if (i >= s.size()) return;
  const int kx = static_cast<int>(i % s.nx);
  const int ky = static_cast<int>((i / s.nx) % s.ny);
  a[i] = (kx == 0 && ky == 0) ? 0.0f : a[i] / detail::neumann_eigenvalue(ky, kx, s.ny, s.nx);
}

// Per-image sums of exp(i (psi - phi)), blockIdx.y is the image.
__global__ void congruence_sums(const float* psi, const float* phi, int plane, double* sums) {
  __shared__ float sc[256];
  __shared__ float ss[256];
  const int z = static_cast<int>(blockIdx.y);
  float c = 0;
  float sn = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < plane; i += gridDim.x * blockDim.x) {
    const std::size_t j = std::size_t(z) * plane + i;
    float a, b;
    sincosf(psi[j] - phi[j], &b, &a);
    c += a;
    sn += b;
  }
  sc[threadIdx.x] = c;
  ss[threadIdx.x] = sn;
  __syncthreads();
  for (int h = blockDim.x / 2; h > 0; h >>= 1) {
    if (threadIdx.x < h) {
      sc[threadIdx.x] += sc[threadIdx.x + h];
      ss[threadIdx.x] += ss[threadIdx.x + h];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&sums[2 * z], double(sc[0]));
    atomicAdd(&sums[2 * z + 1], double(ss[0]));
  }
}

__global__ void add_congruence(float* phi, const double* sums, int plane, std::size_t total) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  const std::size_t z = i / plane;
  phi[i] += static_cast<float>(atan2(sums[2 * z + 1], sums[2 * z]));
}

void dct_axis(float* a, float2* buf, Shape s, int axis, bool forward) {
  const std::size_t total = std::size_t(s.nz) * s.ny * s.nx;
  const unsigned grid = blocks_for(total);
  if (forward) {
    dct_reorder<<<grid, 256>>>(a, buf, s, axis);
  } else {
    idct_twiddle<<<grid, 256>>>(a, buf, s, axis);
  }
  check_launch("dct prepare");
  if (axis == 2) {
    FftPlan plan(s.nx, 1, s.nx, s.nz * s.ny);
    forward ? plan.forward(buf) : plan.inverse(buf);
  } else {
    // Columns of one image are a strided batch; images are not, so step through them.
    FftPlan plan(s.ny, s.nx, 1, s.nx);
    for (int z = 0; z < s.nz; ++z) {
      float2* p = buf + std::size_t(z) * s.ny * s.nx;
      forward ? plan.forward(p) : plan.inverse(p);
    }
  }
  if (forward) {
    dct_twiddle<<<grid, 256>>>(buf, a, s, axis);
  } else {
    idct_reorder<<<grid, 256>>>(buf, a, s, axis);
  }
  check_launch("dct finish");
}

// Border sums per image, blockIdx.y is the image.
__global__ void border_sums(const float* phi, Shape s, double* sums) {
  const int z = static_cast<int>(blockIdx.y);
  const int perimeter = 2 * s.nx + 2 * (s.ny - 2);
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= perimeter) return;
  int y, x;
  if (t < s.nx) {
    y = 0;
    x = t;
  } else if (t < 2 * s.nx) {
    y = s.ny - 1;
    x = t - s.nx;
  } else {
    const int r = t - 2 * s.nx;
    y = 1 + r / 2;
    x = (r & 1) ? s.nx - 1 : 0;
  }
  atomicAdd(&sums[z], double(phi[(std::size_t(z) * s.ny + y) * s.nx + x]));
}

__global__ void split_ratio(const float2* ratio, float* psi, float* logamp, std::size_t total) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  psi[i] = atan2f(ratio[i].y, ratio[i].x);
  logamp[i] = logf(hypotf(ratio[i].x, ratio[i].y));
}

__global__ void assemble_rytov(const float* logamp, const float* phi, const double* border,
                               double perimeter, int plane, float2* out, std::size_t total) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  const std::size_t z = i / plane;
  const double two_pi = 6.283185307179586;
  const float shift = static_cast<float>(two_pi * round(border[z] / perimeter / two_pi));
  out[i] = {logamp[i], phi[i] - shift};
}

}  // namespace

void unwrap_device(const float* psi, float* phi, int nz, int ny, int nx) {
  const Shape s{nz, ny, nx};
  const std::size_t total = std::size_t(nz) * ny * nx;
  const unsigned grid = blocks_for(total);
  divergence<<<grid, 256>>>(psi, phi, s);
  check_launch("divergence");
  DeviceBuffer<float2> buf(total);
  dct_axis(phi, buf.get(), s, 2, true);
  dct_axis(phi, buf.get(), s, 1, true);
  poisson_divide<<<grid, 256>>>(phi, s);
  check_launch("poisson_divide");
  dct_axis(phi, buf.get(), s, 1, false);
  dct_axis(phi, buf.get(), s, 2, false);

  DeviceBuffer<double> sums(2 * std::size_t(nz));
  sums.zero();
  const int plane = ny * nx;
  const dim3 rgrid(std::min(64u, blocks_for(plane)), nz);
  congruence_sums<<<rgrid, 256>>>(psi, phi, plane, sums.get());
  check_launch("congruence_sums");
  add_congruence<<<grid, 256>>>(phi, sums.get(), plane, total);
  check_launch("add_congruence");
}

Array3<float> unwrap(const Array3<float>& wrapped) {
  DeviceBuffer<float> psi(wrapped.size());
  DeviceBuffer<float> phi(wrapped.size());
  psi.upload(wrapped.data.data());
  unwrap_device(psi.get(), phi.get(), wrapped.nz, wrapped.ny, wrapped.nx);
  Array3<float> out(wrapped.nz, wrapped.ny, wrapped.nx);
  phi.download(out.data.data());
  return out;
}

Array3<cfloat> rytov(const Array3<cfloat>& ratio) {
  const std::size_t total = ratio.size();
  const unsigned grid = blocks_for(total);
  DeviceBuffer<float2> r(total);
  r.upload(ratio.data.data());
  DeviceBuffer<float> psi(total);
  DeviceBuffer<float> logamp(total);
  split_ratio<<<grid, 256>>>(r.get(), psi.get(), logamp.get(), total);
  check_launch("split_ratio");
  DeviceBuffer<float> phi(total);
  unwrap_device(psi.get(), phi.get(), ratio.nz, ratio.ny, ratio.nx);

  const Shape s{ratio.nz, ratio.ny, ratio.nx};
  const int perimeter = 2 * s.nx + 2 * (s.ny - 2);
  DeviceBuffer<double> border(std::size_t(ratio.nz));
  border.zero();
  border_sums<<<dim3(blocks_for(perimeter), ratio.nz), 256>>>(phi.get(), s, border.get());
  check_launch("border_sums");
  assemble_rytov<<<grid, 256>>>(logamp.get(), phi.get(), border.get(), double(perimeter),
                                ratio.ny * ratio.nx, r.get(), total);
  check_launch("assemble_rytov");
  Array3<cfloat> out(ratio.nz, ratio.ny, ratio.nx);
  r.download(out.data.data());
  return out;
}

}  // namespace qoraholo::cuda
