#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "backends.hpp"
#include "qoraholo/qoraholo.hpp"

namespace qoraholo {

namespace {

void require(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument(what);
}

void require_cuda() {
#ifdef QORAHOLO_HAS_CUDA
  if (!cuda::device_present()) throw std::runtime_error("no CUDA device available");
#else
  throw std::runtime_error("qoraholo was built without CUDA");
#endif
}

void require_even_square(int ny, int nx, const char* who) {
  require(ny == nx && nx >= 4 && nx % 2 == 0,
          std::string(who) + ": images must be square with an even side");
}

}  // namespace

detail::OdtGrid make_grid(int n, int nz, const Geometry& geometry) {
  const float fm = geometry.n_medium / geometry.wavelength;
  const float fna = geometry.na / geometry.wavelength;
  return {n,
          nz,
          1.0f / (static_cast<float>(n) * geometry.pixel),
          1.0f / (static_cast<float>(nz) * geometry.pixel),
          fm * fm,
          fna * fna,
          4.0f * std::numbers::pi_v<float> / geometry.pixel};
}

bool cuda_available() {
#ifdef QORAHOLO_HAS_CUDA
  return cuda::device_present();
#else
  return false;
#endif
}

Array3<cfloat> extract_fields(const Array3<float>& holograms, const FieldOptions& options,
                              Backend backend) {
  require_even_square(holograms.ny, holograms.nx, "extract_fields");
  require(holograms.nz >= 1, "extract_fields: empty stack");
  const int n = holograms.nx;
  std::array<float, 2> c{};
  if (options.center) {
    c = *options.center;
  } else {
    const Sideband s = find_sideband(holograms, options.dc_exclusion);
    c = {static_cast<float>(s.iy), static_cast<float>(s.ix)};
  }
  CropPlan plan{static_cast<int>(std::lround(c[0])), static_cast<int>(std::lround(c[1])),
                options.radius, options.output_size > 0 ? options.output_size : n};
  if (plan.radius <= 0) plan.radius = std::hypot(c[0], c[1]) / 3.0f;
  require(plan.out % 2 == 0 && plan.out <= n, "extract_fields: output_size must be even and <= N");
  require(plan.radius > 0 && plan.radius <= static_cast<float>(plan.out) / 2.0f,
          "extract_fields: crop radius must fit in the output window");

  if (backend == Backend::cuda) {
    require_cuda();
#ifdef QORAHOLO_HAS_CUDA
    return cuda::extract_fields(holograms, plan);
#endif
  }
  return cpu::extract_fields(holograms, plan);
}

Array3<float> unwrap(const Array3<float>& wrapped, Backend backend) {
  require(wrapped.ny % 2 == 0 && wrapped.nx % 2 == 0 && wrapped.ny >= 2 && wrapped.nx >= 2,
          "unwrap: image sides must be even");
  if (backend == Backend::cuda) {
    require_cuda();
#ifdef QORAHOLO_HAS_CUDA
    return cuda::unwrap(wrapped);
#endif
  }
  return cpu::unwrap(wrapped);
}

Array3<cfloat> rytov(const Array3<cfloat>& ratio, Backend backend) {
  require(ratio.ny % 2 == 0 && ratio.nx % 2 == 0 && ratio.ny >= 4 && ratio.nx >= 4,
          "rytov: image sides must be even");
  if (backend == Backend::cuda) {
    require_cuda();
#ifdef QORAHOLO_HAS_CUDA
    return cuda::rytov(ratio);
#endif
  }
  return cpu::rytov(ratio);
}

OdtResult reconstruct(const Array3<cfloat>& rytov_fields,
                      std::span<const Illumination> illuminations, const Geometry& geometry,
                      const OdtOptions& options) {
  require_even_square(rytov_fields.ny, rytov_fields.nx, "reconstruct");
  require(rytov_fields.nz >= 1 && static_cast<std::size_t>(rytov_fields.nz) == illuminations.size(),
          "reconstruct: need one illumination per field");
  require(geometry.wavelength > 0 && geometry.pixel > 0 && geometry.n_medium > 0 && geometry.na > 0,
          "reconstruct: geometry values must be positive");
  const int n = rytov_fields.nx;
  const int nz = options.nz > 0 ? options.nz : n;
  require(nz % 2 == 0, "reconstruct: nz must be even");
  const float fm = geometry.n_medium / geometry.wavelength;
  for (const auto& il : illuminations) {
    require(il.fy * il.fy + il.fx * il.fx < fm * fm,
            "reconstruct: illumination frequency beyond n_medium / wavelength");
  }

  const detail::OdtGrid grid = make_grid(n, nz, geometry);
  if (options.backend == Backend::cuda) {
    require_cuda();
#ifdef QORAHOLO_HAS_CUDA
    return cuda::reconstruct(rytov_fields, illuminations, grid, geometry, options.gp_iterations,
                             options.mapping);
#endif
  }
  return cpu::reconstruct(rytov_fields, illuminations, grid, geometry, options.gp_iterations);
}

std::vector<Illumination> circular_scan(int count, float na_illumination,
                                        const Geometry& geometry) {
  require(count >= 1, "circular_scan: count must be positive");
  require(na_illumination >= 0 && na_illumination < geometry.n_medium,
          "circular_scan: na_illumination must be below n_medium");
  std::vector<Illumination> out(count);
  const float f = na_illumination / geometry.wavelength;
  for (int i = 1; i < count; ++i) {
    const float phi = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i - 1) /
                      static_cast<float>(count - 1);
    out[i] = {f * std::sin(phi), f * std::cos(phi)};
  }
  return out;
}

}  // namespace qoraholo
