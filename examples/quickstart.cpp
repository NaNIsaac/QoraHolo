// Synthetic bead -> 49 fields -> refractive-index volume, on CUDA when available.

#include <cstdio>
#include <vector>

#include "qoraholo/qoraholo.hpp"

int main() {
  using namespace qoraholo;
  const Geometry geom{0.532f, 1.337f, 1.2f, 0.2f};  // um; NA of the objective
  const std::vector<Sphere> bead{{0.0f, 0.0f, 0.0f, 3.0f, 1.36f}};
  const auto illum = circular_scan(49, 0.9f, geom);

  const auto ratio = forward_fields(bead, illum, geom, 64);  // stand-in for measured u / u_bg
  const Backend backend = cuda_available() ? Backend::cuda : Backend::cpu;

  OdtOptions opt;
  opt.gp_iterations = 20;
  opt.backend = backend;
  const OdtResult r = reconstruct(rytov(ratio, backend), illum, geom, opt);

  std::printf("%s: n at the centre %.4f (true 1.3600), k-space filled %.1f%%\n",
              backend == Backend::cuda ? "cuda" : "cpu", r.ri(32, 32, 32), 100.0 * r.filled);
}
