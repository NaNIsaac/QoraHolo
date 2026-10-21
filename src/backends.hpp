#pragma once

// Backend entry points behind the public dispatch in api.cpp. Arguments are validated there.

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "common.hpp"
#include "qoraholo/bpm.hpp"
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
// host time for allocations and cuFFT plans. bytes: memory held by the solver (device memory
// on CUDA, measured; array sizes on CPU).
struct OdtTimings {
  std::size_t bytes = 0;
  double setup = 0;
  double upload = 0;
  double fft2 = 0;
  double clear = 0;  // CUDA: zeroing the volume and hit counts before mapping
  double mapping = 0;
  double normalize = 0;
  double solve = 0;
  double finish = 0;
  double download = 0;
};

// CUDA kernels for one FGP step of the TV solver: a primal and a dual kernel; both fused into
// one (the default, fastest when measured); the fused kernel on 3D shared-memory tiles; or on
// x-y tiles that walk along z with two planes in shared memory. Selectable for the benchmark
// and the tests; all do the same arithmetic.
enum class TvKernel { separate, fused, tiled, marching };

// k-space grid constants for fields of side n reconstructed into nz axial voxels.
detail::OdtGrid make_grid(int n, int nz, const Geometry& geometry);

// Validated crop for holograms of side n with the sideband at center (pixels from DC).
CropPlan crop_plan(const FieldOptions& options, std::array<float, 2> center, int n);
// Validates the reconstruct() arguments for count fields of side n and returns the grid.
detail::OdtGrid odt_grid(int n, int count, std::span<const Illumination> illuminations,
                         const Geometry& geometry, const OdtOptions& options);

// Per-illumination tables of the BPM, each count * n * n with lateral bins in FFT order,
// computed in double on the host so both backends use the same values.
struct BpmTables {
  int n = 0;
  int nz = 0;
  int count = 0;
  float phase = 0;           // k0 * pixel: screen phase per unit of n - n_medium
  std::vector<cfloat> step;  // envelope propagator over one voxel, zero outside the medium band
  std::vector<cfloat> last;  // from the last slice back to z = 0, times the detection pupil
  std::vector<float> q;      // weight of the scattered part of a slice (1 without obliquity)
  float lipschitz = 0;       // see BpmSolveOptions::step
};

BpmTables bpm_tables(int n, int nz, std::span<const Illumination> illuminations,
                     const Geometry& geometry, bool obliquity);

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
                      const detail::OdtGrid& grid, const Geometry& geometry,
                      const OdtOptions& options, OdtTimings* timings = nullptr);
// x: n - n_medium. fields: measured ratios.
Array3<cfloat> bpm_forward(const Array3<float>& x, const BpmTables& t);
BpmResult bpm_reconstruct(const Array3<cfloat>& fields, Array3<float> x, const BpmTables& t,
                          const BpmSolveOptions& options);
// Linearisation of bpm_forward at x: J h, and J^T r = Re(J^H r), for the adjoint test.
Array3<cfloat> bpm_jvp(const Array3<float>& x, const Array3<float>& h, const BpmTables& t);
Array3<float> bpm_vjp(const Array3<float>& x, const Array3<cfloat>& r, const BpmTables& t);
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
                      const detail::OdtGrid& grid, const Geometry& geometry,
                      const OdtOptions& options, OdtTimings* timings = nullptr,
                      TvKernel tv_kernel = TvKernel::fused);
Array3<cfloat> bpm_forward(const Array3<float>& x, const BpmTables& t);
// chunk: illuminations per device pass of the gradient; 0 picks the largest that keeps the
// stored envelopes within half the free device memory.
BpmResult bpm_reconstruct(const Array3<cfloat>& fields, Array3<float> x, const BpmTables& t,
                          const BpmSolveOptions& options, int chunk = 0);
Array3<float> bpm_vjp(const Array3<float>& x, const Array3<cfloat>& r, const BpmTables& t);
}  // namespace cuda
#endif

}  // namespace qoraholo
