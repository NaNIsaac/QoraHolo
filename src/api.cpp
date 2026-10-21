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

CropPlan crop_plan(const FieldOptions& options, std::array<float, 2> center, int n) {
  CropPlan plan{static_cast<int>(std::lround(center[0])), static_cast<int>(std::lround(center[1])),
                options.radius, options.output_size > 0 ? options.output_size : n};
  if (plan.radius <= 0) plan.radius = std::hypot(center[0], center[1]) / 3.0f;
  require(plan.out % 2 == 0 && plan.out <= n, "extract_fields: output_size must be even and <= N");
  require(plan.radius > 0 && plan.radius <= static_cast<float>(plan.out) / 2.0f,
          "extract_fields: crop radius must fit in the output window");
  return plan;
}

Array3<cfloat> extract_fields(const Array3<float>& holograms, const FieldOptions& options,
                              Backend backend) {
  require_even_square(holograms.ny, holograms.nx, "extract_fields");
  require(holograms.nz >= 1, "extract_fields: empty stack");
  std::array<float, 2> c{};
  if (options.center) {
    c = *options.center;
  } else {
    const Sideband s = find_sideband(holograms, options.dc_exclusion);
    c = {static_cast<float>(s.iy), static_cast<float>(s.ix)};
  }
  const CropPlan plan = crop_plan(options, c, holograms.nx);
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

detail::OdtGrid odt_grid(int n, int count, std::span<const Illumination> illuminations,
                         const Geometry& geometry, const OdtOptions& options) {
  require_even_square(n, n, "reconstruct");
  require(count >= 1 && static_cast<std::size_t>(count) == illuminations.size(),
          "reconstruct: need one illumination per field");
  require(geometry.wavelength > 0 && geometry.pixel > 0 && geometry.n_medium > 0 && geometry.na > 0,
          "reconstruct: geometry values must be positive");
  const int nz = options.nz > 0 ? options.nz : n;
  require(nz % 2 == 0, "reconstruct: nz must be even");
  const float fm = geometry.n_medium / geometry.wavelength;
  for (const auto& il : illuminations) {
    require(il.fy * il.fy + il.fx * il.fx < fm * fm,
            "reconstruct: illumination frequency beyond n_medium / wavelength");
  }
  if (options.backend == Backend::cuda) require_cuda();
  return make_grid(n, nz, geometry);
}

OdtResult reconstruct(const Array3<cfloat>& rytov_fields,
                      std::span<const Illumination> illuminations, const Geometry& geometry,
                      const OdtOptions& options) {
  require(rytov_fields.ny == rytov_fields.nx, "reconstruct: images must be square");
  const detail::OdtGrid grid =
      odt_grid(rytov_fields.nx, rytov_fields.nz, illuminations, geometry, options);
#ifdef QORAHOLO_HAS_CUDA
  if (options.backend == Backend::cuda) {
    return cuda::reconstruct(rytov_fields, illuminations, grid, geometry, options);
  }
#endif
  return cpu::reconstruct(rytov_fields, illuminations, grid, geometry, options);
}

namespace {

BpmTables checked_bpm_tables(int nz, int n, int count, std::span<const Illumination> illuminations,
                             const Geometry& geometry, bool obliquity, Backend backend,
                             const char* who) {
  const std::string w(who);
  require(n >= 4 && n % 2 == 0 && nz >= 1, w + ": volume must be (nz, n, n) with an even n");
  require(count >= 1 && static_cast<std::size_t>(count) == illuminations.size(),
          w + ": need one illumination per field");
  require(geometry.wavelength > 0 && geometry.pixel > 0 && geometry.n_medium > 0 && geometry.na > 0,
          w + ": geometry values must be positive");
  const float fm = geometry.n_medium / geometry.wavelength;
  for (const auto& il : illuminations) {
    require(il.fy * il.fy + il.fx * il.fx < fm * fm,
            w + ": illumination frequency beyond n_medium / wavelength");
  }
  if (backend == Backend::cuda) require_cuda();
  return bpm_tables(n, nz, illuminations, geometry, obliquity);
}

Array3<float> contrast(const Array3<float>& ri, float n_medium) {
  Array3<float> x = ri;
  for (float& v : x.data) v -= n_medium;
  return x;
}

}  // namespace

Array3<cfloat> bpm_forward(const Array3<float>& ri, std::span<const Illumination> illuminations,
                           const Geometry& geometry, const BpmOptions& options) {
  require(ri.ny == ri.nx, "bpm_forward: volume must be (nz, n, n)");
  const BpmTables t =
      checked_bpm_tables(ri.nz, ri.nx, static_cast<int>(illuminations.size()), illuminations,
                         geometry, options.obliquity, options.backend, "bpm_forward");
  const Array3<float> x = contrast(ri, geometry.n_medium);
#ifdef QORAHOLO_HAS_CUDA
  if (options.backend == Backend::cuda) return cuda::bpm_forward(x, t);
#endif
  return cpu::bpm_forward(x, t);
}

BpmResult bpm_reconstruct(const Array3<cfloat>& fields, std::span<const Illumination> illuminations,
                          const Geometry& geometry, const Array3<float>& initial,
                          const BpmSolveOptions& options) {
  require(initial.ny == initial.nx && fields.ny == initial.ny && fields.nx == initial.nx,
          "bpm_reconstruct: fields (count, n, n) and initial (nz, n, n) must share n");
  require(options.iterations >= 0 && options.step > 0 && options.tv_weight >= 0 &&
              options.tv_inner >= 1,
          "bpm_reconstruct: invalid solver options");
  const BpmTables t = checked_bpm_tables(initial.nz, initial.nx, fields.nz, illuminations, geometry,
                                         options.obliquity, options.backend, "bpm_reconstruct");
  BpmResult r;
#ifdef QORAHOLO_HAS_CUDA
  if (options.backend == Backend::cuda) {
    r = cuda::bpm_reconstruct(fields, contrast(initial, geometry.n_medium), t, options);
  }
#endif
  if (options.backend == Backend::cpu) {
    r = cpu::bpm_reconstruct(fields, contrast(initial, geometry.n_medium), t, options);
  }
  for (float& v : r.ri.data) v += geometry.n_medium;
  return r;
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
