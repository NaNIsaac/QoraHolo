#pragma once

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qoraholo {

using cfloat = std::complex<float>;

// Row-major stack of 2D images or a 3D volume: index (z, y, x), x fastest.
// A single image is a stack with nz == 1.
template <class T>
struct Array3 {
  int nz = 0;
  int ny = 0;
  int nx = 0;
  std::vector<T> data;

  Array3() = default;
  Array3(int nz_, int ny_, int nx_, T fill = T{})
      : nz(nz_), ny(ny_), nx(nx_), data(static_cast<std::size_t>(nz_) * ny_ * nx_, fill) {
    if (nz_ < 0 || ny_ < 0 || nx_ < 0) throw std::invalid_argument("negative array shape");
  }

  [[nodiscard]] std::size_t size() const { return data.size(); }
  [[nodiscard]] std::size_t plane() const { return static_cast<std::size_t>(ny) * nx; }
  [[nodiscard]] T* slice(int z) { return data.data() + z * plane(); }
  [[nodiscard]] const T* slice(int z) const { return data.data() + z * plane(); }

  T& operator()(int z, int y, int x) {
    return data[(z * plane()) + static_cast<std::size_t>(y) * nx + x];
  }
  const T& operator()(int z, int y, int x) const {
    return data[(z * plane()) + static_cast<std::size_t>(y) * nx + x];
  }
};

}  // namespace qoraholo
