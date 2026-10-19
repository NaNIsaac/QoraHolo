#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "qoraholo/array.hpp"
#include "qoraholo/field.hpp"
#include "qoraholo/odt.hpp"

namespace qoraholo {

struct PipelineOptions {
  int count = 0;          // holograms per stack, one per illumination
  int hologram_size = 0;  // side of the square holograms
  FieldOptions field;     // field.center is required: the reference carrier
  std::vector<Illumination> illuminations;
  Geometry geometry;  // pixel is the field pixel, which is also the voxel side
  OdtOptions odt;     // odt.backend selects CPU or CUDA
  // CUDA: the sample holograms are copied in this many parts on a second stream, so the field
  // extraction of one part runs while the next is still being copied. 1: one copy, then compute.
  int upload_chunks = 1;
};

// Milliseconds spent in each stage by the last process() call. On CUDA these are event
// intervals on the device; `upload` is the copy of the sample holograms, `fields` runs from
// the start of that copy to the end of field extraction, so with upload_chunks > 1 it
// contains the overlapped part of the copy.
struct PipelineTimings {
  double upload = 0;
  double fields = 0;
  double rytov = 0;
  double fft2 = 0;
  double clear = 0;  // CUDA: zeroing the volume before the mapping; on CPU inside mapping
  double mapping = 0;
  double normalize = 0;
  double solve = 0;
  double finish = 0;
  double download = 0;
  double total = 0;  // host wall time of process()
};

// Reconstruction from holograms to refractive index with everything allocated once. On CUDA
// the cuFFT plans and device buffers are created by the constructor, the background fields stay
// on the device, and process() does one host-to-device copy of the sample holograms and one
// device-to-host copy of the volume, both from page-locked buffers. The CPU backend runs the
// same stages with the per-call functions.
class Pipeline {
 public:
  explicit Pipeline(PipelineOptions options);
  ~Pipeline();
  Pipeline(Pipeline&&) noexcept;
  Pipeline& operator=(Pipeline&&) noexcept;

  // Background holograms (count, N, N). Their fields are extracted once and reused.
  void set_background(const Array3<float>& backgrounds);

  // count * N * N floats for the sample holograms, (count, N, N) row-major. Page-locked on
  // CUDA, so a camera or file reader can write into it directly.
  [[nodiscard]] std::span<float> input();
  // Reconstructs from input(). Needs set_background() first.
  void process();
  // (nz, n, n) refractive index from the last process(); page-locked on CUDA. Overwritten by
  // the next call.
  [[nodiscard]] std::span<const float> output() const;
  // Fraction of k-space voxels that received data in the last process().
  [[nodiscard]] double filled() const;

  // Copies the holograms into input(), runs process(), copies the result.
  [[nodiscard]] OdtResult run(const Array3<float>& holograms);

  [[nodiscard]] const PipelineTimings& timings() const;
  // Host time of the constructor: allocations and cuFFT plans.
  [[nodiscard]] double setup_ms() const;
  // Device memory held, including the cuFFT work area; 0 on CPU.
  [[nodiscard]] std::size_t device_bytes() const;
  [[nodiscard]] std::array<int, 3> output_shape() const;

  class Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace qoraholo
