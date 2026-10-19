#pragma once

// Backend entry points behind the public dispatch in api.cpp. Arguments are validated there.

#include <array>
#include <memory>
#include <span>

#include "common.hpp"
#include "qoraholo/field.hpp"
#include "qoraholo/odt.hpp"
#include "qoraholo/pipeline.hpp"

namespace qoraholo {

// Integer crop centre, radius and output side resolved from FieldOptions.
struct CropPlan {
  int cy;
  int cx;
  float radius;
  int out;
};

// Stage times of reconstruct() in milliseconds, filled when a pointer is passed (benchmark).
// On CUDA the device stages are timed with events and exclude the transfers; setup is the
// host time for allocations and cuFFT plans.
struct OdtTimings {
  double setup = 0;
  double upload = 0;
  double fft2 = 0;
  double mapping = 0;
  double normalize = 0;
  double gp = 0;
  double finish = 0;
  double download = 0;
};

// k-space grid constants for fields of side n reconstructed into nz axial voxels.
detail::OdtGrid make_grid(int n, int nz, const Geometry& geometry);

// Validated crop for holograms of side n with the sideband at center (pixels from DC).
CropPlan crop_plan(const FieldOptions& options, std::array<float, 2> center, int n);
// Validates the reconstruct() arguments for count fields of side n and returns the grid.
detail::OdtGrid odt_grid(int n, int count, std::span<const Illumination> illuminations,
                         const Geometry& geometry, const OdtOptions& options);

class Pipeline::Impl {
 public:
  virtual ~Impl() = default;
  virtual void set_background(const Array3<float>& backgrounds) = 0;
  virtual std::span<float> input() = 0;
  virtual void process() = 0;
  [[nodiscard]] virtual std::span<const float> output() const = 0;

  PipelineTimings timings;
  double filled = 0;
  double setup_ms = 0;
  std::size_t device_bytes = 0;
  std::array<int, 3> shape{};
  bool has_background = false;
};

namespace cpu {
Array3<cfloat> extract_fields(const Array3<float>& holograms, const CropPlan& plan);
Array3<float> unwrap(const Array3<float>& wrapped);
Array3<cfloat> rytov(const Array3<cfloat>& ratio);
OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const detail::OdtGrid& grid, const Geometry& geometry, int gp_iterations,
                      OdtTimings* timings = nullptr);
}  // namespace cpu

#ifdef QORAHOLO_HAS_CUDA
namespace cuda {
bool device_present();
std::unique_ptr<Pipeline::Impl> make_pipeline(const PipelineOptions& options, const CropPlan& crop,
                                              const detail::OdtGrid& grid);
Array3<cfloat> extract_fields(const Array3<float>& holograms, const CropPlan& plan);
Array3<float> unwrap(const Array3<float>& wrapped);
Array3<cfloat> rytov(const Array3<cfloat>& ratio);
OdtResult reconstruct(const Array3<cfloat>& rytov_fields, std::span<const Illumination> illum,
                      const detail::OdtGrid& grid, const Geometry& geometry, int gp_iterations,
                      Mapping mapping, OdtTimings* timings = nullptr);
}  // namespace cuda
#endif

}  // namespace qoraholo
