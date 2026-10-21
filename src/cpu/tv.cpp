#include "cpu/tv.hpp"

#include <cstddef>

namespace qoraholo::cpu {

void tv_prox(const float* b, float* x, Dual& d, const detail::Volume& v, float lambda, int inner) {
  const std::ptrdiff_t total = std::ptrdiff_t(v.nz) * v.ny * v.nx;
  const float step = 1.0f / (12.0f * lambda);
  for (int a = 0; a < 3; ++a) d.r[a] = d.p[a];
  const auto primal = [&](const std::vector<float>* q) {
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < total; ++i) {
      const int xx = static_cast<int>(i % v.nx);
      const int yy = static_cast<int>((i / v.nx) % v.ny);
      const int zz = static_cast<int>(i / (std::ptrdiff_t(v.nx) * v.ny));
      x[i] = detail::tv_primal(
          b[i], lambda, detail::tv_adjoint(q[0].data(), q[1].data(), q[2].data(), v, zz, yy, xx));
    }
  };
  float t = 1.0f;
  for (int k = 0; k < inner; ++k) {
    primal(d.r);
    const float t_next = detail::fista_t(t);
    const float momentum = (t - 1.0f) / t_next;
#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < total; ++i) {
      const int xx = static_cast<int>(i % v.nx);
      const int yy = static_cast<int>((i / v.nx) % v.ny);
      const int zz = static_cast<int>(i / (std::ptrdiff_t(v.nx) * v.ny));
      float g[3];
      detail::tv_gradient(x, v, zz, yy, xx, g);
      float p[3] = {d.p[0][i], d.p[1][i], d.p[2][i]};
      float r[3] = {d.r[0][i], d.r[1][i], d.r[2][i]};
      detail::tv_dual_step(g, step, momentum, p, r);
      for (int a = 0; a < 3; ++a) {
        d.p[a][i] = p[a];
        d.r[a][i] = r[a];
      }
    }
    t = t_next;
  }
  primal(d.p);
}

}  // namespace qoraholo::cpu
