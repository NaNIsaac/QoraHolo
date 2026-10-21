#include <utility>

#include "cuda/tv.cuh"

namespace qoraholo::cuda {

namespace {

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

}  // namespace

void tv_prox(const float* b, float scale, float* out, float* p, float* r, float* r_next,
             std::size_t stride, float lambda, int inner, const detail::Volume& v, TvKernel kernel,
             cudaStream_t s) {
  const unsigned vgrid = blocks_for(std::size_t(v.nz) * v.ny * v.nx);
  const dim3 slabs((v.nx + kMx - 1) / kMx, (v.ny + kMy - 1) / kMy, (v.nz + kMz - 1) / kMz);
  const dim3 tiles((v.nx + kTx - 1) / kTx, (v.ny + kTy - 1) / kTy, (v.nz + kTz - 1) / kTz);
  const float step = 1.0f / (12.0f * lambda);
  check(cudaMemcpyAsync(r, p, 3 * stride * sizeof(float), cudaMemcpyDeviceToDevice, s),
        "copy dual");
  float tk = 1.0f;
  for (int k = 0; k < inner; ++k) {
    const float tn = detail::fista_t(tk);
    const float m = (tk - 1.0f) / tn;
    switch (kernel) {
      case TvKernel::separate:
        tv_primal_kernel<<<vgrid, 256, 0, s>>>(b, scale, r, stride, lambda, out, v);
        tv_dual_kernel<<<vgrid, 256, 0, s>>>(out, step, m, p, r, stride, v);
        break;
      case TvKernel::fused:
        tv_fgp_fused<<<vgrid, 256, 0, s>>>(b, scale, r, r_next, p, stride, lambda, step, m, v);
        std::swap(r, r_next);
        break;
      case TvKernel::tiled:
        tv_fgp_tiled<<<tiles, dim3(kTx, kTy, kTz), 0, s>>>(b, scale, r, r_next, p, stride, lambda,
                                                           step, m, v);
        std::swap(r, r_next);
        break;
      case TvKernel::marching:
        tv_fgp_marching<<<slabs, dim3(kMx, kMy), 0, s>>>(b, scale, r, r_next, p, stride, lambda,
                                                         step, m, v);
        std::swap(r, r_next);
        break;
    }
    tk = tn;
  }
  tv_primal_kernel<<<vgrid, 256, 0, s>>>(b, scale, p, stride, lambda, out, v);
  check_launch("tv prox");
}

}  // namespace qoraholo::cuda
