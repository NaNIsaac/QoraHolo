#include "cpu/fft.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <thread>

#include "common.hpp"
#include "pocketfft_hdronly.h"

namespace qoraholo::cpu {

int threads() {
  static const int n = std::max(1u, std::thread::hardware_concurrency());
  return n;
}

void fft(cfloat* data, const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
         bool forward, float scale) {
  pocketfft::stride_t stride(shape.size());
  std::ptrdiff_t s = sizeof(cfloat);
  for (std::size_t i = shape.size(); i-- > 0;) {
    stride[i] = s;
    s *= static_cast<std::ptrdiff_t>(shape[i]);
  }
  pocketfft::c2c(shape, stride, stride, axes, forward, data, data, scale,
                 static_cast<std::size_t>(threads()));
}

void fft2(cfloat* data, int n, bool forward, float scale) {
  const pocketfft::shape_t shape{std::size_t(n), std::size_t(n)};
  const pocketfft::stride_t stride{std::ptrdiff_t(n * sizeof(cfloat)), sizeof(cfloat)};
  pocketfft::c2c(shape, stride, stride, {0, 1}, forward, data, data, scale, std::size_t{1});
}

namespace {

pocketfft::stride_t dense_strides(const pocketfft::shape_t& shape, std::ptrdiff_t element) {
  pocketfft::stride_t stride(shape.size());
  for (std::size_t i = shape.size(); i-- > 0;) {
    stride[i] = element;
    element *= static_cast<std::ptrdiff_t>(shape[i]);
  }
  return stride;
}

}  // namespace

void rfft3(const float* in, cfloat* out, int nz, int ny, int nx) {
  const pocketfft::shape_t shape{std::size_t(nz), std::size_t(ny), std::size_t(nx)};
  const pocketfft::shape_t half{std::size_t(nz), std::size_t(ny), std::size_t(nx / 2 + 1)};
  pocketfft::r2c(shape, dense_strides(shape, sizeof(float)), dense_strides(half, sizeof(cfloat)),
                 {0, 1, 2}, true, in, out, 1.0f, static_cast<std::size_t>(threads()));
}

void irfft3(cfloat* in, float* out, int nz, int ny, int nx, float scale) {
  // pocketfft's multi-axis c2r copies the spectrum into a temporary; transforming the leading
  // axes in place instead saves that allocation on every iteration.
  const pocketfft::shape_t shape{std::size_t(nz), std::size_t(ny), std::size_t(nx)};
  const pocketfft::shape_t half{std::size_t(nz), std::size_t(ny), std::size_t(nx / 2 + 1)};
  const auto hs = dense_strides(half, sizeof(cfloat));
  const auto nt = static_cast<std::size_t>(threads());
  pocketfft::c2c(half, hs, hs, {0, 1}, false, in, in, 1.0f, nt);
  pocketfft::c2r(shape, hs, dense_strides(shape, sizeof(float)), std::size_t{2}, false, in, out,
                 scale, nt);
}

namespace {

std::vector<cfloat> twiddles(int len, double sign) {
  std::vector<cfloat> w(len);
  for (int k = 0; k < len; ++k) {
    const double a = sign * std::numbers::pi * k / (2.0 * len);
    w[k] = {static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a))};
  }
  return w;
}

// Visits every element as (line base offset, position along the axis, stride along the axis).
template <class F>
void for_each_line(const Array3<float>& a, int axis, F&& f) {
  const int len = axis == 2 ? a.nx : a.ny;
  const std::ptrdiff_t step = axis == 2 ? 1 : a.nx;
  const int lines = a.nz * (axis == 2 ? a.ny : a.nx);
#pragma omp parallel for
  for (int line = 0; line < lines; ++line) {
    std::ptrdiff_t base = 0;
    if (axis == 2) {
      base = static_cast<std::ptrdiff_t>(line) * a.nx;
    } else {
      const int z = line / a.nx;
      const int x = line % a.nx;
      base = static_cast<std::ptrdiff_t>(z) * a.ny * a.nx + x;
    }
    f(base, step, len);
  }
}

}  // namespace

void dct2(Array3<float>& a, int axis) {
  const int len = axis == 2 ? a.nx : a.ny;
  std::vector<cfloat> buf(a.size());
  for_each_line(a, axis, [&](std::ptrdiff_t base, std::ptrdiff_t step, int n) {
    for (int i = 0; i < n; ++i) {
      buf[base + detail::dct_position(i, n) * step] = a.data[base + i * step];
    }
  });
  fft(buf.data(), {std::size_t(a.nz), std::size_t(a.ny), std::size_t(a.nx)}, {std::size_t(axis)},
      true);
  const auto w = twiddles(len, -1.0);
  for_each_line(a, axis, [&](std::ptrdiff_t base, std::ptrdiff_t step, int n) {
    for (int k = 0; k < n; ++k) a.data[base + k * step] = (w[k] * buf[base + k * step]).real();
  });
}

void idct2(Array3<float>& a, int axis) {
  const int len = axis == 2 ? a.nx : a.ny;
  std::vector<cfloat> buf(a.size());
  const auto w = twiddles(len, 1.0);
  for_each_line(a, axis, [&](std::ptrdiff_t base, std::ptrdiff_t step, int n) {
    for (int k = 0; k < n; ++k) {
      const float mirror = k == 0 ? 0.0f : a.data[base + (n - k) * step];
      buf[base + k * step] = w[k] * cfloat(a.data[base + k * step], -mirror);
    }
  });
  fft(buf.data(), {std::size_t(a.nz), std::size_t(a.ny), std::size_t(a.nx)}, {std::size_t(axis)},
      false, 1.0f / static_cast<float>(len));
  for_each_line(a, axis, [&](std::ptrdiff_t base, std::ptrdiff_t step, int n) {
    for (int i = 0; i < n; ++i)
      a.data[base + i * step] = buf[base + detail::dct_position(i, n) * step].real();
  });
}

}  // namespace qoraholo::cpu
