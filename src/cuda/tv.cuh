#pragma once

// Total-variation proximal step on the device, shared by the ODT solver and the BPM inversion.

#include <cstddef>

#include "backends.hpp"
#include "cuda/util.cuh"

namespace qoraholo::cuda {

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

// out = argmin_x 0.5 |x - scale * b|^2 + lambda TV(x), x >= 0, by `inner` FGP iterations started
// from the dual p, which is updated (warm start for the next call). r and r_next are scratch
// duals; all three hold three components `stride` floats apart. out doubles as scratch for the
// separate-kernel variant.
void tv_prox(const float* b, float scale, float* out, float* p, float* r, float* r_next,
             std::size_t stride, float lambda, int inner, const detail::Volume& v, TvKernel kernel,
             cudaStream_t s);

}  // namespace qoraholo::cuda
