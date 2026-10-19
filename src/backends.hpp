#pragma once

// Backend entry points behind the public dispatch in api.cpp. Arguments are validated there.

#include <span>

#include "common.hpp"
#include "qoraholo/field.hpp"
#include "qoraholo/odt.hpp"

namespace qoraholo {

// Integer crop centre, radius and output side resolved from FieldOptions.
struct CropPlan {
  int cy;
  int cx;
  float radius;
  int out;
};

namespace cpu {
Array3<cfloat> extract_fields(const Array3<float>& holograms, const CropPlan& plan);
Array3<float> unwrap(const Array3<float>& wrapped);
Array3<cfloat> rytov(const Array3<cfloat>& ratio);
OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const detail::OdtGrid& grid, const Geometry& geometry, int gp_iterations);
}  // namespace cpu

#ifdef QORAHOLO_HAS_CUDA
namespace cuda {
bool device_present();
Array3<cfloat> extract_fields(const Array3<float>& holograms, const CropPlan& plan);
Array3<float> unwrap(const Array3<float>& wrapped);
Array3<cfloat> rytov(const Array3<cfloat>& ratio);
OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const detail::OdtGrid& grid, const Geometry& geometry, int gp_iterations,
                      Mapping mapping);
}  // namespace cuda
#endif

}  // namespace qoraholo
