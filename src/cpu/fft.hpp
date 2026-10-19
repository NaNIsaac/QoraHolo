#pragma once

#include <cstddef>
#include <vector>

#include "qoraholo/array.hpp"

namespace qoraholo::cpu {

[[nodiscard]] int threads();

// In-place complex FFT of a contiguous row-major array over the given axes, multiplied by scale.
void fft(cfloat* data, const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
         bool forward, float scale = 1.0f);

// Real-to-complex FFT over all three axes of a dense (nz, ny, nx) array into the half spectrum
// (nz, ny, nx / 2 + 1), and its inverse times scale. irfft3 overwrites its input.
void rfft3(const float* in, cfloat* out, int nz, int ny, int nx);
void irfft3(cfloat* in, float* out, int nz, int ny, int nx, float scale);

// Unnormalised DCT-II along one axis (1: rows of y, 2: x) of a (nz, ny, nx) stack, via FFT
// (Makhoul 1980), and its exact inverse.
void dct2(Array3<float>& a, int axis);
void idct2(Array3<float>& a, int axis);

}  // namespace qoraholo::cpu
