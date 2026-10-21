#pragma once

#include <vector>

#include "common.hpp"

namespace qoraholo::cpu {

// Dual variable of the TV proximal step, one array per axis (z, y, x).
struct Dual {
  std::vector<float> p[3];
  std::vector<float> r[3];
  explicit Dual(std::size_t size) {
    for (int a = 0; a < 3; ++a) {
      p[a].assign(size, 0.0f);
      r[a].assign(size, 0.0f);
    }
  }
};

// x = argmin 0.5 |x - b|^2 + lambda TV(x), x >= 0, by `inner` FGP iterations started
// from the dual of the previous call.
void tv_prox(const float* b, float* x, Dual& d, const detail::Volume& v, float lambda, int inner);

}  // namespace qoraholo::cpu
