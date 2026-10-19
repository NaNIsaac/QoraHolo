#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
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

// Half-spectrum element o of an (nz, n, n) grid: its full index and that of its mirror -k.
struct HalfIndex {
  std::size_t k;
  std::size_t minus_k;
};
__device__ __forceinline__ HalfIndex half_index(std::size_t o, int nz, int n) {
  const int h = n / 2 + 1;
  const int kx = static_cast<int>(o % h);
  const int ky = static_cast<int>((o / h) % n);
  const int kz = static_cast<int>(o / (std::size_t(h) * n));
  return {(std::size_t(kz) * n + ky) * n + kx,
          (std::size_t(detail::negative_bin(kz, nz)) * n + detail::negative_bin(ky, n)) * n +
              detail::negative_bin(kx, n)};
}

__global__ void hermitian_hits(const float* weight, std::uint8_t* hits, int nz, int n,
                               std::size_t half) {
  const std::size_t o = grid_index();
  if (o >= half) return;
  const HalfIndex h = half_index(o, nz, n);
  hits[o] = static_cast<std::uint8_t>((weight[h.k] > 0) + (weight[h.minus_k] > 0));
}

__global__ void hermitian_data(const float2* p, float2* a, int nz, int n, std::size_t half) {
  const std::size_t o = grid_index();
  if (o >= half) return;
  const HalfIndex h = half_index(o, nz, n);
  float2 v;
  detail::hermitian_data(p[h.k].x, p[h.k].y, p[h.minus_k].x, p[h.minus_k].y, v.x, v.y);
  a[o] = v;
}

// Runs over the padded rows too; R2C never reads the padding.
__global__ void nonnegative_real(float* x, std::size_t total, float scale) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  x[i] = fmaxf(x[i] * scale, 0.0f);
}

__global__ void restore_half(float2* e, const float2* a, const std::uint8_t* hits,
                             std::size_t half) {
  const std::size_t o = grid_index();
  if (o >= half) return;
  float2 v = e[o];
  detail::hermitian_restore(a[o].x, a[o].y, hits[o], v.x, v.y);
  e[o] = v;
}

__global__ void to_index_real(float* x, std::size_t total, float scale, float nm2, float inv_k02) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  x[i] = detail::index_from_potential(x[i] * scale, 0.0f, nm2, inv_k02);
}

__global__ void scale_real(float* x, std::size_t total, float scale) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  x[i] *= scale;
}

// Voxel i of the valid nz * ny * nx region and its offset in the padded layout.
struct Voxel {
  int z, y, x;
  long long at;
};
__device__ __forceinline__ bool voxel(const detail::Volume& v, Voxel& out) {
  const std::size_t i = grid_index();
  if (i >= std::size_t(v.nz) * v.ny * v.nx) return false;
  out.x = static_cast<int>(i % v.nx);
  out.y = static_cast<int>((i / v.nx) % v.ny);
  out.z = static_cast<int>(i / (std::size_t(v.nx) * v.ny));
  out.at = v.at(out.z, out.y, out.x);
  return true;
}

// out = max(scale * b - lambda L(q), 0); q holds three components `stride` floats apart.
__global__ void tv_primal_kernel(const float* b, float scale, const float* q, std::size_t stride,
                                 float lambda, float* out, detail::Volume v) {
  Voxel w{};
  if (!voxel(v, w)) return;
  const float adj = detail::tv_adjoint(q, q + stride, q + 2 * stride, v, w.z, w.y, w.x);
  out[w.at] = detail::tv_primal(scale * b[w.at], lambda, adj);
}

__global__ void tv_dual_kernel(const float* z, float step, float momentum, float* p, float* r,
                               std::size_t stride, detail::Volume v) {
  Voxel w{};
  if (!voxel(v, w)) return;
  float g[3];
  detail::tv_gradient(z, v, w.z, w.y, w.x, g);
  float pv[3];
  float rv[3];
  for (int a = 0; a < 3; ++a) {
    pv[a] = p[a * stride + w.at];
    rv[a] = r[a * stride + w.at];
  }
  detail::tv_dual_step(g, step, momentum, pv, rv);
  for (int a = 0; a < 3; ++a) {
    p[a * stride + w.at] = pv[a];
    r[a * stride + w.at] = rv[a];
  }
}

// One FGP step in one kernel. z, the primal point, is recomputed from r at the voxel and at its
// three upper neighbours instead of being written and read back; r is read from r_in and the
// update written to r_out, because neighbouring threads still need the old values.
__global__ void tv_fgp_fused(const float* b, float scale, const float* r_in, float* r_out, float* p,
                             std::size_t stride, float lambda, float step, float momentum,
                             detail::Volume v) {
  Voxel w{};
  if (!voxel(v, w)) return;
  // Own p and r first, so their latency overlaps the primal points below.
  float pv[3];
  float rv[3];
  for (int a = 0; a < 3; ++a) {
    pv[a] = p[a * stride + w.at];
    rv[a] = r_in[a * stride + w.at];
  }
  const auto primal = [&](int z, int y, int x) {
    const float adj = detail::tv_adjoint(r_in, r_in + stride, r_in + 2 * stride, v, z, y, x);
    return detail::tv_primal(scale * b[v.at(z, y, x)], lambda, adj);
  };
  const float c = primal(w.z, w.y, w.x);
  float g[3];
  g[0] = w.z < v.nz - 1 ? c - primal(w.z + 1, w.y, w.x) : 0.0f;
  g[1] = w.y < v.ny - 1 ? c - primal(w.z, w.y + 1, w.x) : 0.0f;
  g[2] = w.x < v.nx - 1 ? c - primal(w.z, w.y, w.x + 1) : 0.0f;
  detail::tv_dual_step(g, step, momentum, pv, rv);
  for (int a = 0; a < 3; ++a) {
    p[a * stride + w.at] = pv[a];
    r_out[a * stride + w.at] = rv[a];
  }
}

// The fused step with a tile of kTx x kTy x kTz voxels per block. b and the three components of
// r are staged in shared memory for the tile plus one voxel on every side, z is computed once
// per voxel of the tile and its upper faces, then each thread updates its own voxel.
constexpr int kTx = 16;
constexpr int kTy = 8;
constexpr int kTz = 4;

__global__ void __launch_bounds__(kTx * kTy * kTz)
    tv_fgp_tiled(const float* b, float scale, const float* r_in, float* r_out, float* p,
                 std::size_t stride, float lambda, float step, float momentum, detail::Volume v) {
  constexpr int bx = kTx + 2;
  constexpr int by = kTy + 2;
  constexpr int bz = kTz + 2;
  __shared__ float sb[bz][by][bx];
  __shared__ float sr[3][bz][by][bx];
  __shared__ float sz[kTz + 1][kTy + 1][kTx + 1];
  const int x0 = static_cast<int>(blockIdx.x) * kTx;
  const int y0 = static_cast<int>(blockIdx.y) * kTy;
  const int z0 = static_cast<int>(blockIdx.z) * kTz;
  const int tid = static_cast<int>(threadIdx.x + kTx * (threadIdx.y + kTy * threadIdx.z));
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int tz = static_cast<int>(threadIdx.z);
  const int x = x0 + tx;
  const int y = y0 + ty;
  const int z = z0 + tz;
  const bool inside = x < v.nx && y < v.ny && z < v.nz;
  // Issued before the barriers so the loads overlap the staging.
  const long long at = inside ? v.at(z, y, x) : 0;
  float pv[3];
  for (int a = 0; a < 3; ++a) pv[a] = inside ? p[a * stride + at] : 0.0f;

  // Box [-1, kT] around the tile; outside the volume r and b are never used, store zeros.
  for (int i = tid; i < bx * by * bz; i += kTx * kTy * kTz) {
    const int lx = i % bx;
    const int ly = (i / bx) % by;
    const int lz = i / (bx * by);
    const int gx = x0 - 1 + lx;
    const int gy = y0 - 1 + ly;
    const int gz = z0 - 1 + lz;
    const bool in = gx >= 0 && gx < v.nx && gy >= 0 && gy < v.ny && gz >= 0 && gz < v.nz;
    const long long at = in ? v.at(gz, gy, gx) : 0;
    sb[lz][ly][lx] = in ? scale * b[at] : 0.0f;
    for (int a = 0; a < 3; ++a) sr[a][lz][ly][lx] = in ? r_in[a * stride + at] : 0.0f;
  }
  __syncthreads();

  // z on [0, kT] along each axis, the same arithmetic in the same order as tv_adjoint.
  for (int i = tid; i < (kTx + 1) * (kTy + 1) * (kTz + 1); i += kTx * kTy * kTz) {
    const int lx = i % (kTx + 1);
    const int ly = (i / (kTx + 1)) % (kTy + 1);
    const int lz = i / ((kTx + 1) * (kTy + 1));
    const int gx = x0 + lx;
    const int gy = y0 + ly;
    const int gz = z0 + lz;
    float zv = 0.0f;
    if (gx < v.nx && gy < v.ny && gz < v.nz) {
      const int cx = lx + 1;
      const int cy = ly + 1;
      const int cz = lz + 1;
      float s = 0;
      if (gz < v.nz - 1) s += sr[0][cz][cy][cx];
      if (gz > 0) s -= sr[0][cz - 1][cy][cx];
      if (gy < v.ny - 1) s += sr[1][cz][cy][cx];
      if (gy > 0) s -= sr[1][cz][cy - 1][cx];
      if (gx < v.nx - 1) s += sr[2][cz][cy][cx];
      if (gx > 0) s -= sr[2][cz][cy][cx - 1];
      zv = detail::tv_primal(sb[cz][cy][cx], lambda, s);
    }
    sz[lz][ly][lx] = zv;
  }
  __syncthreads();

  if (!inside) return;
  const float c = sz[tz][ty][tx];
  float g[3];
  g[0] = z < v.nz - 1 ? c - sz[tz + 1][ty][tx] : 0.0f;
  g[1] = y < v.ny - 1 ? c - sz[tz][ty + 1][tx] : 0.0f;
  g[2] = x < v.nx - 1 ? c - sz[tz][ty][tx + 1] : 0.0f;
  float rv[3];
  for (int a = 0; a < 3; ++a) rv[a] = sr[a][tz + 1][ty + 1][tx + 1];
  detail::tv_dual_step(g, step, momentum, pv, rv);
  for (int a = 0; a < 3; ++a) {
    p[a * stride + at] = pv[a];
    r_out[a * stride + at] = rv[a];
  }
}

// The fused step on x-y tiles of kMx x kMy voxels that walk along z through kMz planes. Shared
// memory holds r for two planes (with a one-voxel x-y halo) and z for two planes, so z
// neighbours come from the previous plane and only the x-y halo is loaded twice.
constexpr int kMx = 32;
constexpr int kMy = 8;
constexpr int kMz = 8;

__global__ void __launch_bounds__(kMx* kMy)
    tv_fgp_marching(const float* b, float scale, const float* r_in, float* r_out, float* p,
                    std::size_t stride, float lambda, float step, float momentum,
                    detail::Volume v) {
  constexpr int hx = kMx + 2;
  constexpr int hy = kMy + 2;
  __shared__ float sr[2][3][hy][hx];         // r on planes, local x, y in [-1, kM]
  __shared__ float sz[2][kMy + 1][kMx + 1];  // z on planes, local x, y in [0, kM]
  const int x0 = static_cast<int>(blockIdx.x) * kMx;
  const int y0 = static_cast<int>(blockIdx.y) * kMy;
  const int z0 = static_cast<int>(blockIdx.z) * kMz;
  const int z1 = min(z0 + kMz, v.nz);
  const int tid = static_cast<int>(threadIdx.x + kMx * threadIdx.y);

  const auto load = [&](int slot, int z) {
    for (int i = tid; i < hx * hy; i += kMx * kMy) {
      const int lx = i % hx;
      const int ly = i / hx;
      const int x = x0 - 1 + lx;
      const int y = y0 - 1 + ly;
      const bool in = x >= 0 && x < v.nx && y >= 0 && y < v.ny && z >= 0 && z < v.nz;
      const long long at = in ? v.at(z, y, x) : 0;
      for (int a = 0; a < 3; ++a) sr[slot][a][ly][lx] = in ? r_in[a * stride + at] : 0.0f;
    }
  };
  // z on plane z from r on that plane (slot) and the plane below (below), the arithmetic and
  // order of tv_adjoint and tv_primal.
  const auto primal = [&](int out, int slot, int below, int z) {
    for (int i = tid; i < (kMx + 1) * (kMy + 1); i += kMx * kMy) {
      const int lx = i % (kMx + 1);
      const int ly = i / (kMx + 1);
      const int x = x0 + lx;
      const int y = y0 + ly;
      float zv = 0.0f;
      if (x < v.nx && y < v.ny && z < v.nz) {
        const int cx = lx + 1;
        const int cy = ly + 1;
        float s = 0;
        if (z < v.nz - 1) s += sr[slot][0][cy][cx];
        if (z > 0) s -= sr[below][0][cy][cx];
        if (y < v.ny - 1) s += sr[slot][1][cy][cx];
        if (y > 0) s -= sr[slot][1][cy - 1][cx];
        if (x < v.nx - 1) s += sr[slot][2][cy][cx];
        if (x > 0) s -= sr[slot][2][cy][cx - 1];
        zv = detail::tv_primal(scale * b[v.at(z, y, x)], lambda, s);
      }
      sz[out][ly][lx] = zv;
    }
  };

  int cur = 0;  // r slot of plane z, the other holds z - 1 or z + 1
  int zc = 0;   // z slot of plane z
  load(1, z0 - 1);
  load(cur, z0);
  __syncthreads();
  primal(zc, cur, 1, z0);
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int x = x0 + tx;
  const int y = y0 + ty;
  const bool inside = x < v.nx && y < v.ny;
  for (int z = z0; z < z1; ++z) {
    const int next = 1 - cur;
    // Issued before the barriers so the loads overlap the plane update.
    const long long at = inside ? v.at(z, y, x) : 0;
    float pv[3];
    for (int a = 0; a < 3; ++a) pv[a] = inside ? p[a * stride + at] : 0.0f;
    __syncthreads();  // plane z - 1 in `next` and z slot 1 - zc are no longer read
    load(next, z + 1);
    __syncthreads();
    primal(1 - zc, next, cur, z + 1);
    __syncthreads();
    if (inside) {
      const float c = sz[zc][ty][tx];
      float g[3];
      g[0] = z < v.nz - 1 ? c - sz[1 - zc][ty][tx] : 0.0f;
      g[1] = y < v.ny - 1 ? c - sz[zc][ty + 1][tx] : 0.0f;
      g[2] = x < v.nx - 1 ? c - sz[zc][ty][tx + 1] : 0.0f;
      float rv[3];
      for (int a = 0; a < 3; ++a) rv[a] = sr[cur][a][ty + 1][tx + 1];
      detail::tv_dual_step(g, step, momentum, pv, rv);
      for (int a = 0; a < 3; ++a) {
        p[a * stride + at] = pv[a];
        r_out[a * stride + at] = rv[a];
      }
    }
    cur = next;
    zc = 1 - zc;
  }
}

__global__ void fista_momentum(const float* x, const float* x_prev, float momentum, float* y,
                               detail::Volume v) {
  Voxel w{};
  if (!voxel(v, w)) return;
  y[w.at] = x[w.at] + momentum * (x[w.at] - x_prev[w.at]);
}

}  // namespace

OdtSolver::OdtSolver(const OdtGrid& g, int count, const Geometry& geometry,
                     const OdtOptions& options)
    : g_(g),
      count_(count),
      iterations_(options.iterations),
      mapping_(options.mapping),
      tv_(options.iterations > 0 && options.solver == Solver::tv),
      real_(tv_ || (options.iterations > 0 && options.gp_transform == GpTransform::r2c)),
      tv_inner_(options.tv_inner),
      half_(std::size_t(g.nz) * g.n * (g.n / 2 + 1)),
      nm2_(geometry.n_medium * geometry.n_medium),
      illum_(count),
      // The real loop keeps its padded volume in the potential buffer and the Hermitian data in
      // the hit-count buffer; both are free once the mapping is folded into a and hits.
      potential_(std::size_t(g.nz) * g.n * g.n),
      weight_(real_ ? std::max(potential_.size(), 2 * half_) : potential_.size()),
      measured_(iterations_ > 0 && !real_ ? potential_.size() : 0),
      hits_(real_ ? half_ : 0),
      // TV: z, x_prev and the dual p, r (three components each), all in the padded layout.
      tv_buffers_(tv_ ? 11 * 2 * half_ : 0),
      filled_(1),
      plan2_(2, std::array{g.n, g.n}.data(), count, CUFFT_C2C) {
  const float k0 = 2.0f * std::numbers::pi_v<float> / geometry.wavelength;
  inv_k02_ = 1.0f / (k0 * k0);
  lambda_ = options.tv_weight * k0 * k0;
  const std::array dims{g.nz, g.n, g.n};
  plan_list_.push_back(&plan2_);
  if (real_) {
    const int h = g.n / 2 + 1;
    const std::array padded{g.nz, g.n, 2 * h};
    const std::array half{g.nz, g.n, h};
    const int real_dist = g.nz * g.n * 2 * h;
    const int half_dist = g.nz * g.n * h;
    r2c_.emplace(3, dims.data(), 1, CUFFT_R2C, FftPlan::Layout{padded.data(), 1, real_dist},
                 FftPlan::Layout{half.data(), 1, half_dist});
    c2r_.emplace(3, dims.data(), 1, CUFFT_C2R, FftPlan::Layout{half.data(), 1, half_dist},
                 FftPlan::Layout{padded.data(), 1, real_dist});
    plan_list_.push_back(&*r2c_);
    plan_list_.push_back(&*c2r_);
  } else {
    plan3_.emplace(3, dims.data(), 1, CUFFT_C2C);
    plan_list_.push_back(&*plan3_);
  }
}

void OdtSolver::set_stream(cudaStream_t s) {
  stream_ = s;
  for (FftPlan* p : plan_list_) p->set_stream(s);
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
  return illum_.bytes() + potential_.bytes() + weight_.bytes() + measured_.bytes() + hits_.bytes() +
         tv_buffers_.bytes() + filled_.bytes();
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
  check(cudaMemsetAsync(weight_.get(), 0, total * sizeof(float), stream_), "cudaMemset");
  mark(Mark::clear);
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
  if (real_) {
    const unsigned hgrid = blocks_for(half_);
    hermitian_hits<<<hgrid, 256, 0, stream_>>>(weight_.get(), hits_.get(), g_.nz, g_.n, half_);
    auto* a = reinterpret_cast<float2*>(weight_.get());
    hermitian_data<<<hgrid, 256, 0, stream_>>>(potential_.get(), a, g_.nz, g_.n, half_);
    check_launch("hermitian data");
    float2* e = potential_.get();
    auto* x = reinterpret_cast<float*>(e);
    const std::size_t padded = 2 * half_;
    check(cudaMemcpyAsync(e, a, half_ * sizeof(float2), cudaMemcpyDeviceToDevice, stream_),
          "copy data");
    c2r_->c2r(e, x);
    if (tv_) {
      // FISTA with step 1: the gradient step is the k-space restore, the proximal step is FGP
      // on the dual (p, r), warm-started from the previous outer iteration.
      const detail::Volume v{g_.nz, g_.n, g_.n, static_cast<int>(2 * (g_.n / 2 + 1))};
      const unsigned vgrid = blocks_for(total);
      float* z = tv_buffers_.get();
      float* x_prev = z + padded;
      float* p = z + 2 * padded;
      float* r = z + 5 * padded;
      float* r_next = z + 8 * padded;
      const dim3 slabs((g_.n + kMx - 1) / kMx, (g_.n + kMy - 1) / kMy, (g_.nz + kMz - 1) / kMz);
      const dim3 tiles((g_.n + kTx - 1) / kTx, (g_.n + kTy - 1) / kTy, (g_.nz + kTz - 1) / kTz);
      check(cudaMemsetAsync(p, 0, 3 * padded * sizeof(float), stream_), "cudaMemset");
      scale_real<<<blocks_for(padded), 256, 0, stream_>>>(x, padded, inv);
      check(cudaMemcpyAsync(x_prev, x, padded * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
            "copy x");
      const float step = 1.0f / (12.0f * lambda_);
      float t = 1.0f;
      for (int it = 0; it < iterations_; ++it) {
        r2c_->r2c(x, e);
        restore_half<<<hgrid, 256, 0, stream_>>>(e, a, hits_.get(), half_);
        c2r_->c2r(e, x);
        check(cudaMemcpyAsync(r, p, 3 * padded * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
              "copy dual");
        float tk = 1.0f;
        for (int k = 0; k < tv_inner_; ++k) {
          const float tn = detail::fista_t(tk);
          const float m = (tk - 1.0f) / tn;
          switch (tv_kernel_) {
            case TvKernel::separate:
              tv_primal_kernel<<<vgrid, 256, 0, stream_>>>(x, inv, r, padded, lambda_, z, v);
              tv_dual_kernel<<<vgrid, 256, 0, stream_>>>(z, step, m, p, r, padded, v);
              break;
            case TvKernel::fused:
              tv_fgp_fused<<<vgrid, 256, 0, stream_>>>(x, inv, r, r_next, p, padded, lambda_, step,
                                                       m, v);
              std::swap(r, r_next);
              break;
            case TvKernel::tiled:
              tv_fgp_tiled<<<tiles, dim3(kTx, kTy, kTz), 0, stream_>>>(x, inv, r, r_next, p, padded,
                                                                       lambda_, step, m, v);
              std::swap(r, r_next);
              break;
            case TvKernel::marching:
              tv_fgp_marching<<<slabs, dim3(kMx, kMy), 0, stream_>>>(x, inv, r, r_next, p, padded,
                                                                     lambda_, step, m, v);
              std::swap(r, r_next);
              break;
          }
          tk = tn;
        }
        tv_primal_kernel<<<vgrid, 256, 0, stream_>>>(x, inv, p, padded, lambda_, z, v);
        const float tn = detail::fista_t(t);
        fista_momentum<<<vgrid, 256, 0, stream_>>>(z, x_prev, (t - 1.0f) / tn, x, v);
        std::swap(z, x_prev);
        t = tn;
      }
      check_launch("fista-tv");
      mark(Mark::solve);
      to_index_real<<<blocks_for(padded), 256, 0, stream_>>>(x_prev, padded, 1.0f, nm2_, inv_k02_);
      check_launch("to_index");
      index_ = x_prev;
      pitch_ = v.pitch;
      mark(Mark::finish);
      return;
    }
    for (int it = 0; it < iterations_; ++it) {
      nonnegative_real<<<blocks_for(padded), 256, 0, stream_>>>(x, padded, inv);
      r2c_->r2c(x, e);
      restore_half<<<hgrid, 256, 0, stream_>>>(e, a, hits_.get(), half_);
      c2r_->c2r(e, x);
    }
    check_launch("gerchberg-papoulis");
    mark(Mark::solve);
    to_index_real<<<blocks_for(padded), 256, 0, stream_>>>(x, padded, inv, nm2_, inv_k02_);
    check_launch("to_index");
    index_ = x;
    pitch_ = 2 * (g_.n / 2 + 1);
    mark(Mark::finish);
    return;
  }

  if (iterations_ > 0) {
    check(cudaMemcpyAsync(measured_.get(), potential_.get(), potential_.bytes(),
                          cudaMemcpyDeviceToDevice, stream_),
          "copy measured");
    for (int it = 0; it < iterations_; ++it) {
      plan3_->inverse(potential_.get());
      nonnegative<<<grid, 256, 0, stream_>>>(potential_.get(), total, inv);
      plan3_->forward(potential_.get());
      restore_measured<<<grid, 256, 0, stream_>>>(potential_.get(), measured_.get(), weight_.get(),
                                                  total);
    }
    check_launch("gerchberg-papoulis");
  }
  plan3_->inverse(potential_.get());
  mark(Mark::solve);

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
                      const OdtGrid& g, const Geometry& geometry, const OdtOptions& options,
                      OdtTimings* timings, TvKernel tv_kernel) {
  enum { start, uploaded, downloaded = uploaded + OdtSolver::marks + 1 };
  const auto t0 = std::chrono::steady_clock::now();
  std::size_t free_before = 0;
  std::size_t free_after = 0;
  std::size_t device_total = 0;
  check(cudaMemGetInfo(&free_before, &device_total), "cudaMemGetInfo");
  OdtSolver solver(g, rytov_fields.nz, geometry, options);
  solver.set_tv_kernel(tv_kernel);
  const FftWorkArea work(solver.plans());
  check(cudaMemGetInfo(&free_after, &device_total), "cudaMemGetInfo");
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
    timings->bytes = free_before > free_after ? free_before - free_after : 0;
    timings->upload = clock.ms(start, uploaded);
    timings->fft2 = clock.ms(uploaded, s + OdtSolver::fft2);
    timings->clear = clock.ms(s + OdtSolver::fft2, s + OdtSolver::clear);
    timings->mapping = clock.ms(s + OdtSolver::clear, s + OdtSolver::mapping);
    timings->normalize = clock.ms(s + OdtSolver::mapping, s + OdtSolver::normalize);
    timings->solve = clock.ms(s + OdtSolver::normalize, s + OdtSolver::solve);
    timings->finish = clock.ms(s + OdtSolver::solve, s + OdtSolver::finish);
    timings->download = clock.ms(s + OdtSolver::finish, downloaded);
  }
  return result;
}

}  // namespace qoraholo::cuda
