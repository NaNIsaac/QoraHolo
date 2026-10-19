#include <gtest/gtest.h>

#include <random>

#include "cpu/fft.hpp"
#include "pocketfft_hdronly.h"

using qoraholo::Array3;

namespace {

Array3<float> random_stack(int nz, int ny, int nx) {
  Array3<float> a(nz, ny, nx);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (auto& v : a.data) v = u(rng);
  return a;
}

}  // namespace

TEST(Dct, MatchesPocketfftTypeTwo) {
  for (int axis : {1, 2}) {
    const auto a = random_stack(3, 12, 10);
    auto mine = a;
    qoraholo::cpu::dct2(mine, axis);

    Array3<float> ref(a.nz, a.ny, a.nx);
    const pocketfft::shape_t shape{std::size_t(a.nz), std::size_t(a.ny), std::size_t(a.nx)};
    const pocketfft::stride_t stride{std::ptrdiff_t(a.ny * a.nx * sizeof(float)),
                                     std::ptrdiff_t(a.nx * sizeof(float)), sizeof(float)};
    // pocketfft follows FFTW's REDFT10, which is twice the unnormalised sum used here.
    pocketfft::dct(shape, stride, stride, {std::size_t(axis)}, 2, a.data.data(), ref.data.data(),
                   0.5f, false);
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(mine.data[i], ref.data[i], 1e-4f);
  }
}

TEST(Dct, InverseRoundTrips) {
  for (int axis : {1, 2}) {
    const auto a = random_stack(2, 16, 8);
    auto b = a;
    qoraholo::cpu::dct2(b, axis);
    qoraholo::cpu::idct2(b, axis);
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(b.data[i], a.data[i], 1e-5f);
  }
}
