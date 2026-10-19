// Per-stage timing of the reconstruction pipeline on CPU and CUDA.
//
//   qoraholo_bench [--field N] [--angles A] [--gp G] [--repeat R]
//
// Holograms are 2N x 2N, fields N x N, the volume N^3. CUDA field/Rytov times are wall times
// including host-device copies; ODT sub-stages on CUDA are cudaEvent times without copies.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "backends.hpp"
#include "qoraholo/qoraholo.hpp"

#ifdef QORAHOLO_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace qoraholo;

namespace {

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double time_ms(const std::function<void()>& f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Row {
  std::string stage;
  double cpu = 0;
  double gpu = -1;
};

[[maybe_unused]] double contrast_error(const Array3<float>& a, const Array3<float>& b, float nm) {
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = double(a.data[i]) - b.data[i];
    num += d * d;
    den += (double(b.data[i]) - nm) * (double(b.data[i]) - nm);
  }
  return std::sqrt(num / den);
}

}  // namespace

int main(int argc, char** argv) {
  int n = 256;
  int angles = 49;
  int gp = 10;
  int repeat = 3;
  for (int i = 1; i + 1 < argc; i += 2) {
    const int v = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--field")) n = v;
    if (!std::strcmp(argv[i], "--angles")) angles = v;
    if (!std::strcmp(argv[i], "--gp")) gp = v;
    if (!std::strcmp(argv[i], "--repeat")) repeat = v;
  }
  const int holo_n = 2 * n;
  const bool gpu = cuda_available();

  // 55 nm hologram pixels, 110 nm field pixels and voxels; a cell-like phantom.
  Geometry holo_geom{0.532f, 1.337f, 1.2f, 0.055f};
  Geometry geom = holo_geom;
  geom.pixel = 2.0f * holo_geom.pixel;
  const float scale = static_cast<float>(n) / 256.0f;
  const std::vector<Sphere> cell{
      {0.0f, 0.0f, 0.0f, 6.0f * scale, 1.360f},
      {0.5f * scale, -1.0f * scale, 1.0f * scale, 3.0f * scale, 1.375f},
      {-2.0f * scale, 3.0f * scale, -2.5f * scale, 0.8f * scale, 1.390f},
      {2.0f * scale, 2.5f * scale, 3.0f * scale, 0.6f * scale, 1.390f},
  };
  const auto illum = circular_scan(angles, 0.9f, geom);
  std::printf("field %d, holograms %d x %d, angles %d, volume %d^3, GP iterations %d\n", n, holo_n,
              holo_n, angles, n, gp);
#ifdef QORAHOLO_HAS_CUDA
  if (gpu) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s\n", prop.name);
  }
#endif

  const auto ratio = forward_fields(cell, illum, holo_geom, holo_n, ScatteringModel::rytov);
  HologramOptions hopt;
  hopt.noise = 0.01f;
  const auto holo = synthesize_holograms(ratio, illum, holo_geom, hopt);
  const auto bg =
      synthesize_holograms(Array3<cfloat>(angles, holo_n, holo_n, 1.0f), illum, holo_geom, hopt);
  const float carrier_px = 5.0f * static_cast<float>(holo_n) * holo_geom.pixel;
  FieldOptions fo;
  fo.center = {carrier_px, carrier_px};
  fo.radius = std::ceil(holo_geom.na / holo_geom.wavelength * holo_n * holo_geom.pixel) + 1.0f;
  fo.output_size = n;

  std::vector<Row> rows;
  auto bench = [&](const char* name, const std::function<void(Backend)>& f) {
    Row r{name};
    std::vector<double> t;
    for (int i = 0; i < repeat; ++i) t.push_back(time_ms([&] { f(Backend::cpu); }));
    r.cpu = median(t);
    if (gpu) {
      f(Backend::cuda);  // context, plans
      t.clear();
      for (int i = 0; i < repeat; ++i) t.push_back(time_ms([&] { f(Backend::cuda); }));
      r.gpu = median(t);
    }
    rows.push_back(r);
  };

  Array3<cfloat> fields;
  bench("field extraction (sample + background)", [&](Backend b) {
    fields = field_ratio(extract_fields(holo, fo, b), extract_fields(bg, fo, b));
  });
  Array3<cfloat> ryt;
  bench("Rytov phase incl. LS unwrap", [&](Backend b) { ryt = rytov(fields, b); });

  const auto grid = make_grid(n, n, geom);
  OdtTimings cpu_t;
  OdtResult cpu_result;
  {
    std::vector<OdtTimings> runs(repeat);
    for (auto& tm : runs) cpu_result = cpu::reconstruct(ryt, illum, grid, geom, gp, &tm);
    cpu_t = runs.back();
  }
  OdtTimings gather_t;
  OdtTimings scatter_t;
  double odt_err = -1;
  double scatter_err = -1;
#ifdef QORAHOLO_HAS_CUDA
  if (gpu) {
    (void)cuda::reconstruct(ryt, illum, grid, geom, gp, Mapping::gather);
    std::vector<double> g_map;
    std::vector<double> s_map;
    OdtResult g_res;
    OdtResult s_res;
    for (int i = 0; i < repeat; ++i) {
      g_res = cuda::reconstruct(ryt, illum, grid, geom, gp, Mapping::gather, &gather_t);
      g_map.push_back(gather_t.mapping);
      s_res = cuda::reconstruct(ryt, illum, grid, geom, gp, Mapping::scatter, &scatter_t);
      s_map.push_back(scatter_t.mapping);
    }
    gather_t.mapping = median(g_map);
    scatter_t.mapping = median(s_map);
    odt_err = contrast_error(g_res.ri, cpu_result.ri, geom.n_medium);
    scatter_err = contrast_error(s_res.ri, cpu_result.ri, geom.n_medium);
  }
#endif

  auto gpu_or_dash = [&](double v) { return gpu ? v : -1.0; };
  rows.push_back({"ODT: 2D FFT of Rytov fields", cpu_t.fft2, gpu_or_dash(gather_t.fft2)});
  rows.push_back({"ODT: Ewald mapping (gather)", cpu_t.mapping, gpu_or_dash(gather_t.mapping)});
  rows.push_back({"ODT: Ewald mapping (scatter, atomics)", -1, gpu_or_dash(scatter_t.mapping)});
  rows.push_back({"ODT: normalise", cpu_t.normalize, gpu_or_dash(gather_t.normalize)});
  rows.push_back({"ODT: GP iterations + final 3D IFFT", cpu_t.gp, gpu_or_dash(gather_t.gp)});
  rows.push_back({"ODT: potential to index", cpu_t.finish, gpu_or_dash(gather_t.finish)});
  rows.push_back({"ODT: host-device copies", -1, gpu_or_dash(gather_t.upload + gather_t.download)});

  std::printf("\n| stage | CPU (ms) | CUDA (ms) |\n|---|---:|---:|\n");
  for (const auto& r : rows) {
    const auto cell_text = [](double v) {
      if (v < 0) return std::string("-");
      char buf[32];
      std::snprintf(buf, sizeof buf, v < 10 ? "%.2f" : "%.0f", v);
      return std::string(buf);
    };
    std::printf("| %s | %s | %s |\n", r.stage.c_str(), cell_text(r.cpu).c_str(),
                cell_text(r.gpu).c_str());
  }
  std::printf("\nfilled k-space fraction %.4f\n", cpu_result.filled);
  if (gpu) {
    std::printf("relative RI-contrast difference CUDA vs CPU: gather %.2e, scatter %.2e\n", odt_err,
                scatter_err);
  }
  return 0;
}
