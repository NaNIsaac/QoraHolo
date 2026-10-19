#include <cmath>
#include <numbers>

#include "backends.hpp"
#include "cpu/fft.hpp"

namespace qoraholo::cpu {

using detail::wrap_phase;

Array3<float> unwrap(const Array3<float>& psi) {
  const int nz = psi.nz;
  const int ny = psi.ny;
  const int nx = psi.nx;
  Array3<float> phi(nz, ny, nx);
#pragma omp parallel for
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const float p = psi(z, y, x);
        float r = 0;
        if (x + 1 < nx) r += wrap_phase(psi(z, y, x + 1) - p);
        if (x > 0) r -= wrap_phase(p - psi(z, y, x - 1));
        if (y + 1 < ny) r += wrap_phase(psi(z, y + 1, x) - p);
        if (y > 0) r -= wrap_phase(p - psi(z, y - 1, x));
        phi(z, y, x) = r;
      }
    }
  }

  dct2(phi, 2);
  dct2(phi, 1);
#pragma omp parallel for
  for (int z = 0; z < nz; ++z) {
    for (int ky = 0; ky < ny; ++ky) {
      for (int kx = 0; kx < nx; ++kx) {
        phi(z, ky, kx) = (ky == 0 && kx == 0)
                             ? 0.0f
                             : phi(z, ky, kx) / detail::neumann_eigenvalue(ky, kx, ny, nx);
      }
    }
  }
  idct2(phi, 1);
  idct2(phi, 2);

  // The least-squares solution is defined up to a constant; pick the one that makes it
  // congruent to the wrapped input.
  const std::size_t plane = psi.plane();
#pragma omp parallel for
  for (int z = 0; z < nz; ++z) {
    std::complex<double> acc = 0;
    const float* w = psi.slice(z);
    float* u = phi.slice(z);
    for (std::size_t i = 0; i < plane; ++i) acc += std::polar(1.0, double(w[i] - u[i]));
    const auto c = static_cast<float>(std::arg(acc));
    for (std::size_t i = 0; i < plane; ++i) u[i] += c;
  }
  return phi;
}

Array3<cfloat> rytov(const Array3<cfloat>& ratio) {
  Array3<float> wrapped(ratio.nz, ratio.ny, ratio.nx);
  for (std::size_t i = 0; i < ratio.size(); ++i) wrapped.data[i] = std::arg(ratio.data[i]);
  Array3<float> phase = unwrap(wrapped);

  Array3<cfloat> out(ratio.nz, ratio.ny, ratio.nx);
  const int ny = ratio.ny;
  const int nx = ratio.nx;
  for (int z = 0; z < ratio.nz; ++z) {
    double sum = 0;
    for (int x = 0; x < nx; ++x) sum += phase(z, 0, x) + phase(z, ny - 1, x);
    for (int y = 1; y < ny - 1; ++y) sum += phase(z, y, 0) + phase(z, y, nx - 1);
    const double mean = sum / (2.0 * nx + 2.0 * (ny - 2));
    const auto shift =
        static_cast<float>(2.0 * std::numbers::pi * std::round(mean / (2.0 * std::numbers::pi)));
    const std::size_t off = z * ratio.plane();
    for (std::size_t i = off; i < off + ratio.plane(); ++i) {
      out.data[i] = {std::log(std::abs(ratio.data[i])), phase.data[i] - shift};
    }
  }
  return out;
}

}  // namespace qoraholo::cpu
