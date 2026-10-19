// Per-stage timing of the reconstruction pipeline on CPU and CUDA.
//
//   qoraholo_bench [--field N] [--angles A] [--gp G] [--repeat R]
//
// Holograms are 2N x 2N, fields N x N, the volume N^3. In the per-call table, CUDA field/Rytov
// times are wall times including host-device copies and plan creation; ODT sub-stages on CUDA
// are cudaEvent times without copies. The end-to-end table compares the per-call chain with the
// Pipeline, whose CUDA stage times are cudaEvent intervals.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
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

void print_table(const char* title, const std::vector<Row>& rows) {
  std::printf("\n%s\n\n| stage | CPU (ms) | CUDA (ms) |\n|---|---:|---:|\n", title);
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
}

double contrast_error(const Array3<float>& a, const Array3<float>& b, float nm) {
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
  OdtOptions r2c_opt;
  r2c_opt.iterations = gp;
  OdtOptions c2c_opt = r2c_opt;
  c2c_opt.gp_transform = GpTransform::c2c;
  [[maybe_unused]] OdtOptions scatter_opt = r2c_opt;
  scatter_opt.mapping = Mapping::scatter;

  // Median of every stage over `repeat` runs.
  const auto profile = [&](auto&& run, OdtResult& result) {
    std::vector<OdtTimings> runs(repeat);
    for (auto& tm : runs) result = run(&tm);
    const auto med = [&](double OdtTimings::* f) {
      std::vector<double> v;
      for (const auto& t : runs) v.push_back(t.*f);
      return median(v);
    };
    OdtTimings out = runs.back();
    for (auto f :
         {&OdtTimings::setup, &OdtTimings::upload, &OdtTimings::fft2, &OdtTimings::mapping,
          &OdtTimings::normalize, &OdtTimings::solve, &OdtTimings::finish, &OdtTimings::download}) {
      out.*f = med(f);
    }
    return out;
  };
  OdtResult cpu_result;
  OdtResult cpu_c2c;
  const OdtTimings cpu_t =
      profile([&](OdtTimings* t) { return cpu::reconstruct(ryt, illum, grid, geom, r2c_opt, t); },
              cpu_result);
  const OdtTimings cpu_c2c_t = profile(
      [&](OdtTimings* t) { return cpu::reconstruct(ryt, illum, grid, geom, c2c_opt, t); }, cpu_c2c);
  OdtTimings gather_t;
  OdtTimings scatter_t;
  OdtTimings gpu_c2c_t;
  double odt_err = -1;
  double scatter_err = -1;
  double c2c_err = -1;
#ifdef QORAHOLO_HAS_CUDA
  if (gpu) {
    OdtResult g_res;
    OdtResult s_res;
    OdtResult c_res;
    (void)cuda::reconstruct(ryt, illum, grid, geom, r2c_opt);
    gather_t = profile(
        [&](OdtTimings* t) { return cuda::reconstruct(ryt, illum, grid, geom, r2c_opt, t); },
        g_res);
    scatter_t = profile(
        [&](OdtTimings* t) { return cuda::reconstruct(ryt, illum, grid, geom, scatter_opt, t); },
        s_res);
    gpu_c2c_t = profile(
        [&](OdtTimings* t) { return cuda::reconstruct(ryt, illum, grid, geom, c2c_opt, t); },
        c_res);
    odt_err = contrast_error(g_res.ri, cpu_result.ri, geom.n_medium);
    scatter_err = contrast_error(s_res.ri, cpu_result.ri, geom.n_medium);
    c2c_err = contrast_error(c_res.ri, cpu_c2c.ri, geom.n_medium);
  }
#endif

  auto gpu_or_dash = [&](double v) { return gpu ? v : -1.0; };
  rows.push_back({"ODT: 2D FFT of Rytov fields", cpu_t.fft2, gpu_or_dash(gather_t.fft2)});
  rows.push_back({"ODT: Ewald mapping (gather)", cpu_t.mapping, gpu_or_dash(gather_t.mapping)});
  rows.push_back({"ODT: Ewald mapping (scatter, atomics)", -1, gpu_or_dash(scatter_t.mapping)});
  rows.push_back({"ODT: normalise", cpu_t.normalize, gpu_or_dash(gather_t.normalize)});
  rows.push_back(
      {"ODT: GP iterations + final 3D IFFT, R2C/C2R", cpu_t.solve, gpu_or_dash(gather_t.solve)});
  rows.push_back(
      {"ODT: GP iterations + final 3D IFFT, C2C", cpu_c2c_t.solve, gpu_or_dash(gpu_c2c_t.solve)});
  rows.push_back({"ODT: potential to index", cpu_t.finish, gpu_or_dash(gather_t.finish)});
  rows.push_back({"ODT: host-device copies", -1, gpu_or_dash(gather_t.upload + gather_t.download)});
  rows.push_back({"ODT: allocations and cuFFT plans (host)", -1, gpu_or_dash(gather_t.setup)});
  print_table("per-call functions", rows);
  std::printf("\nfilled k-space fraction %.4f\n", cpu_result.filled);
  const auto gib = [](std::size_t b) { return double(b) / (1 << 30); };
  std::printf("reconstruct memory, R2C / C2C: CPU arrays %.2f / %.2f GiB", gib(cpu_t.bytes),
              gib(cpu_c2c_t.bytes));
  if (gpu) {
    std::printf(", CUDA device %.2f / %.2f GiB", gib(gather_t.bytes), gib(gpu_c2c_t.bytes));
  }
  std::printf("\nrelative RI-contrast difference R2C vs C2C (CPU): %.2e\n",
              contrast_error(cpu_result.ri, cpu_c2c.ri, geom.n_medium));
  const auto truth = render_phantom(cell, geom, n, n);
  // RMSE against the phantom over the volume and mean index in the inner 60% of the nucleus.
  const auto score = [&](const Array3<float>& ri) {
    double sq = 0;
    double inner = 0;
    int count = 0;
    const Sphere& nuc = cell[1];
    for (int z = 0; z < n; ++z) {
      for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
          const double e = ri(z, y, x) - truth(z, y, x);
          sq += e * e;
          const double dz = (z - n / 2) * geom.pixel - nuc.z;
          const double dy = (y - n / 2) * geom.pixel - nuc.y;
          const double dx = (x - n / 2) * geom.pixel - nuc.x;
          if (dz * dz + dy * dy + dx * dx < 0.36 * nuc.radius * nuc.radius) {
            inner += ri(z, y, x);
            ++count;
          }
        }
      }
    }
    return std::pair{std::sqrt(sq / double(ri.size())), inner / count};
  };
  {
    const auto [r2c_rmse, r2c_nuc] = score(cpu_result.ri);
    const auto [c2c_rmse, c2c_nuc] = score(cpu_c2c.ri);
    std::printf("against the phantom (nucleus %.4f): R2C RMSE %.3e, nucleus mean %.4f; ", cell[1].n,
                r2c_rmse, r2c_nuc);
    std::printf("C2C RMSE %.3e, nucleus mean %.4f\n", c2c_rmse, c2c_nuc);
  }
  if (gpu) {
    std::printf(
        "relative RI-contrast difference CUDA vs CPU: gather %.2e, scatter %.2e, C2C %.2e\n",
        odt_err, scatter_err, c2c_err);
  }

  // Missing-cone solvers on the same Rytov fields. CPU: one run; CUDA: median of `repeat`.
  {
    struct Config {
      const char* name;
      OdtOptions opt;
    };
    std::vector<Config> configs(4);
    configs[0].name = "Rytov ODT, no iterations";
    configs[1].name = "GP, 20 iterations";
    configs[1].opt.iterations = 20;
    configs[2].name = "GP, 100 iterations";
    configs[2].opt.iterations = 100;
    configs[3].name = "FISTA-TV, 20 x 10 iterations";
    configs[3].opt.iterations = 20;
    configs[3].opt.solver = Solver::tv;
    std::printf("\nmissing-cone solvers (TV weight %.0e)\n\n", double(configs[3].opt.tv_weight));
    std::printf("| solver | CPU (ms) | CUDA (ms) | RMSE | nucleus mean |\n");
    std::printf("|---|---:|---:|---:|---:|\n");
    for (auto& c : configs) {
      OdtResult r;
      const double t_cpu = time_ms([&] { r = reconstruct(ryt, illum, geom, c.opt); });
      const auto [rmse, nuc] = score(r.ri);
      double t_gpu = -1;
      if (gpu) {
        c.opt.backend = Backend::cuda;
        (void)reconstruct(ryt, illum, geom, c.opt);
        std::vector<double> t;
        for (int i = 0; i < repeat; ++i) {
          t.push_back(time_ms([&] { (void)reconstruct(ryt, illum, geom, c.opt); }));
        }
        t_gpu = median(t);
      }
      std::printf("| %s | %.0f | %.0f | %.3e | %.4f |\n", c.name, t_cpu, t_gpu, rmse, nuc);
    }
  }

  // End to end: holograms in, index out.
  PipelineOptions po;
  po.count = angles;
  po.hologram_size = holo_n;
  po.field = fo;
  po.illuminations = illum;
  po.geometry = geom;
  po.odt.iterations = gp;
  std::vector<Row> e2e;
  {
    Row r{"per-call functions, host in / host out"};
    auto chain = [&](Backend b) {
      OdtOptions o = po.odt;
      o.backend = b;
      const auto f = field_ratio(extract_fields(holo, fo, b), extract_fields(bg, fo, b));
      return reconstruct(rytov(f, b), illum, geom, o);
    };
    std::vector<double> t;
    for (int i = 0; i < repeat; ++i) t.push_back(time_ms([&] { (void)chain(Backend::cpu); }));
    r.cpu = median(t);
    if (gpu) {
      t.clear();
      (void)chain(Backend::cuda);
      for (int i = 0; i < repeat; ++i) t.push_back(time_ms([&] { (void)chain(Backend::cuda); }));
      r.gpu = median(t);
    }
    e2e.push_back(r);
  }
  std::vector<Backend> backends{Backend::cpu};
  if (gpu) backends.push_back(Backend::cuda);
  Row setup_row{"Pipeline: constructor (allocations, cuFFT plans)"};
  Row bg_row{"Pipeline: set_background"};
  Row run_row{"Pipeline: run(), copies into and out of its buffers"};
  Row process_row{"Pipeline: process(), page-locked input and output"};
  std::vector<Row> stage_rows{{"  upload of the sample holograms"},
                              {"  field extraction (from the start of the upload)"},
                              {"  ratio, LS unwrap, Rytov"},
                              {"  2D FFT"},
                              {"  Ewald mapping"},
                              {"  normalise"},
                              {"  GP iterations + final 3D IFFT"},
                              {"  potential to index"},
                              {"  download of the volume"}};
  std::size_t device_bytes = 0;
  OdtResult pipe_result;
  for (Backend b : backends) {
    po.odt.backend = b;
    std::vector<double> t_setup;
    std::unique_ptr<Pipeline> p;
    for (int i = 0; i < repeat; ++i) {
      p.reset();
      p = std::make_unique<Pipeline>(po);
      t_setup.push_back(p->setup_ms());
    }
    const double t_bg = time_ms([&] { p->set_background(bg); });
    std::vector<double> t_run;
    (void)p->run(holo);
    for (int i = 0; i < repeat; ++i) t_run.push_back(time_ms([&] { pipe_result = p->run(holo); }));
    std::copy(holo.data.begin(), holo.data.end(), p->input().begin());
    std::vector<double> t_proc;
    std::vector<PipelineTimings> stages;
    for (int i = 0; i < repeat; ++i) {
      p->process();
      t_proc.push_back(p->timings().total);
      stages.push_back(p->timings());
    }
    const auto pick = [&](double PipelineTimings::* f) {
      std::vector<double> v;
      for (const auto& s : stages) v.push_back(s.*f);
      return median(v);
    };
    const double values[] = {pick(&PipelineTimings::upload),  pick(&PipelineTimings::fields),
                             pick(&PipelineTimings::rytov),   pick(&PipelineTimings::fft2),
                             pick(&PipelineTimings::mapping), pick(&PipelineTimings::normalize),
                             pick(&PipelineTimings::solve),   pick(&PipelineTimings::finish),
                             pick(&PipelineTimings::download)};
    const bool on_gpu = b == Backend::cuda;
    (on_gpu ? setup_row.gpu : setup_row.cpu) = median(t_setup);
    (on_gpu ? bg_row.gpu : bg_row.cpu) = t_bg;
    (on_gpu ? run_row.gpu : run_row.cpu) = median(t_run);
    (on_gpu ? process_row.gpu : process_row.cpu) = median(t_proc);
    for (std::size_t k = 0; k < stage_rows.size(); ++k) {
      (on_gpu ? stage_rows[k].gpu : stage_rows[k].cpu) = values[k];
    }
    if (on_gpu) device_bytes = p->device_bytes();
  }
  // The CPU path has no transfers.
  stage_rows[0].cpu = -1;
  stage_rows[8].cpu = -1;
  e2e.push_back(setup_row);
  e2e.push_back(bg_row);
  e2e.push_back(run_row);
  e2e.push_back(process_row);
  e2e.insert(e2e.end(), stage_rows.begin(), stage_rows.end());
  print_table("end to end, sample holograms in, index out", e2e);
  if (gpu) {
    std::printf("\nPipeline device memory %.2f GiB\n", double(device_bytes) / (1 << 30));
    std::printf("relative RI-contrast difference CUDA Pipeline vs CPU: %.2e\n",
                contrast_error(pipe_result.ri, cpu_result.ri, geom.n_medium));
  }

#ifdef QORAHOLO_HAS_CUDA
  if (gpu) {
    // Copy paths on their own, and the chunked upload overlapping field extraction.
    std::printf("\n| transfer | pageable (ms) | page-locked (ms) |\n|---|---:|---:|\n");
    const auto transfer = [&](const char* name, std::size_t bytes, bool to_device) {
      std::vector<char> pageable(bytes, 1);
      char* pinned = nullptr;
      char* dev = nullptr;
      cudaMallocHost(&pinned, bytes);
      cudaMalloc(&dev, bytes);
      const auto copy = [&](char* host) {
        return time_ms([&] {
          if (to_device) {
            cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice);
          } else {
            cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost);
          }
        });
      };
      std::vector<double> a;
      std::vector<double> b;
      (void)copy(pageable.data());
      (void)copy(pinned);
      for (int i = 0; i < repeat; ++i) {
        a.push_back(copy(pageable.data()));
        b.push_back(copy(pinned));
      }
      std::printf("| %s, %.0f MiB | %.2f | %.2f |\n", name, double(bytes) / (1 << 20), median(a),
                  median(b));
      cudaFree(dev);
      cudaFreeHost(pinned);
    };
    transfer("sample holograms to device", holo.size() * sizeof(float), true);
    transfer("index volume to host", cpu_result.ri.size() * sizeof(float), false);

    std::printf("\n| upload chunks | upload + field extraction (ms) | process() (ms) |\n");
    std::printf("|---:|---:|---:|\n");
    po.odt.backend = Backend::cuda;
    for (int chunks : {1, 7, 49}) {
      if (chunks > angles) continue;
      po.upload_chunks = chunks;
      Pipeline p(po);
      p.set_background(bg);
      std::copy(holo.data.begin(), holo.data.end(), p.input().begin());
      p.process();
      std::vector<double> f;
      std::vector<double> t;
      for (int i = 0; i < repeat; ++i) {
        p.process();
        f.push_back(p.timings().fields);
        t.push_back(p.timings().total);
      }
      std::printf("| %d | %.2f | %.2f |\n", chunks, median(f), median(t));
    }
  }
#endif
  return 0;
}
