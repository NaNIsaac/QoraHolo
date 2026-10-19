#pragma once

// Device stages shared by the per-call backend functions (host in, host out) and by the
// device-resident Pipeline. Stages own their cuFFT plans and scratch buffers; the caller owns
// inputs, outputs and the shared cuFFT work area.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "backends.hpp"
#include "cuda/util.cuh"

namespace qoraholo::cuda {

// Hologram stack -> cropped sideband spectra -> fields. The holograms are real, so their
// spectra are half spectra (R2C); bins with k_x < 0 are read through Hermitian symmetry.
class FieldStage {
 public:
  // batch: holograms per R2C call; the hologram buffer passed to spectra() must hold a whole
  // batch even when fewer images are valid. count: fields per inverse call.
  FieldStage(int n, int batch, int count, const CropPlan& plan);

  [[nodiscard]] std::span<FftPlan* const> plans() { return plan_list_; }
  void set_stream(cudaStream_t s);

  // Spectra of `batch` holograms, cropped into images first .. first + valid - 1 of fields.
  void crop(const float* holograms, int valid, float2* fields, int first);
  // In-place inverse FFT of `count` cropped spectra. The fields still carry the centre-origin
  // sign (-1)^(x+y) and a factor n^2, which cancel in a sample / background ratio.
  void inverse(float2* fields);
  // Removes that sign and factor.
  void demodulate(float2* fields, int count);

  [[nodiscard]] int out() const { return plan_.out; }

 private:
  int n_;
  int batch_;
  CropPlan plan_;
  cudaStream_t stream_ = nullptr;
  DeviceBuffer<float2> spectra_;
  FftPlan r2c_;
  FftPlan inverse_;
  FftPlan* plan_list_[2];
};

// psi = arg(sample / background), logamp = log|sample / background|. background holds one
// field per sample field.
void ratio_phase(const float2* sample, const float2* background, std::size_t total, float* psi,
                 float* logamp, cudaStream_t s);
// The same for a ratio that is already formed.
void split_ratio(const float2* ratio, std::size_t total, float* psi, float* logamp, cudaStream_t s);

// Least-squares unwrapping and Rytov assembly for a stack of count images of ny x nx.
class Unwrapper {
 public:
  Unwrapper(int count, int ny, int nx);

  [[nodiscard]] std::span<FftPlan* const> plans() { return plan_list_; }
  void set_stream(cudaStream_t s);

  void unwrap(const float* psi, float* phi);
  // out = logamp + i (phi - 2 pi m), m fixed per image by the border mean of phi.
  void assemble_rytov(const float* logamp, const float* phi, float2* out);

 private:
  void dct(float* a, int axis, bool forward);

  int count_;
  int ny_;
  int nx_;
  cudaStream_t stream_ = nullptr;
  DeviceBuffer<float2> buf_;
  DeviceBuffer<double> sums_;
  FftPlan plan_x_;
  FftPlan plan_y_;
  FftPlan* plan_list_[2];
};

// Rytov fields -> refractive index: 2D FFT, Ewald mapping, normalisation, a missing-cone
// solver (Gerchberg-Papoulis on C2C or R2C/C2R spectra, or FISTA-TV), potential -> index.
class OdtSolver {
 public:
  OdtSolver(const detail::OdtGrid& grid, int count, const Geometry& geometry,
            const OdtOptions& options);

  [[nodiscard]] std::span<FftPlan* const> plans() { return plan_list_; }
  void set_stream(cudaStream_t s);
  void set_illuminations(std::span<const Illumination> illum);

  // Marks recorded by run() after each of its stages, in this order.
  enum Mark { fft2, clear, mapping, normalize, solve, finish, marks };

  // fields: (count, n, n) Rytov fields on the device, overwritten by their spectra. clock marks
  // first + Mark when given.
  void run(float2* fields, StageClock* clock = nullptr, int first = 0);

  // Refractive index rows of n floats, `pitch` floats apart; valid after run().
  [[nodiscard]] const float* index() const { return index_; }
  [[nodiscard]] std::size_t pitch() const { return pitch_; }
  // Number of k-space voxels that received data, on the device.
  [[nodiscard]] const unsigned long long* filled() const { return filled_.get(); }
  // Device memory held by the solver's buffers, without cuFFT plans and work area.
  [[nodiscard]] std::size_t bytes() const;

 private:
  detail::OdtGrid g_;
  int count_;
  int iterations_;
  Mapping mapping_;
  bool tv_;
  bool real_;
  int tv_inner_;
  float lambda_ = 0;
  std::size_t half_;
  float nm2_;
  float inv_k02_ = 0;
  cudaStream_t stream_ = nullptr;
  DeviceBuffer<float3> illum_;
  DeviceBuffer<float2> potential_;
  DeviceBuffer<float> weight_;
  DeviceBuffer<float2> measured_;
  DeviceBuffer<std::uint8_t> hits_;
  DeviceBuffer<float> tv_buffers_;
  DeviceBuffer<unsigned long long> filled_;
  const float* index_ = nullptr;
  std::size_t pitch_ = 0;
  FftPlan plan2_;
  std::optional<FftPlan> plan3_;
  std::optional<FftPlan> r2c_;
  std::optional<FftPlan> c2r_;
  std::vector<FftPlan*> plan_list_;
};

// Copies a pitched device volume of nz * n rows of n floats to dense host memory.
void download_rows(const float* src, std::size_t pitch, int rows, int n, float* host,
                   cudaStream_t s);

}  // namespace qoraholo::cuda
