#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qoraholo/qoraholo.hpp"

namespace nb = nanobind;
using namespace qoraholo;

namespace {

template <class T>
using In3 = nb::ndarray<const T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
template <class T>
using Out = nb::ndarray<nb::numpy, T>;

template <class T>
Array3<T> to_array(const In3<T>& a) {
  Array3<T> out(static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)),
                static_cast<int>(a.shape(2)));
  std::copy(a.data(), a.data() + a.size(), out.data.begin());
  return out;
}

template <class T>
Out<T> to_numpy(Array3<T>&& a) {
  auto* owner = new std::vector<T>(std::move(a.data));
  nb::capsule free(owner, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  const std::size_t shape[3] = {std::size_t(a.nz), std::size_t(a.ny), std::size_t(a.nx)};
  return Out<T>(owner->data(), 3, shape, free);
}

std::vector<Illumination> to_illum(
    const nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>& a) {
  if (a.shape(1) != 2) throw std::invalid_argument("illuminations must have shape (count, 2)");
  std::vector<Illumination> out(a.shape(0));
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = {a(i, 0), a(i, 1)};
  return out;
}

Backend parse_backend(const std::string& s) {
  if (s == "cpu") return Backend::cpu;
  if (s == "cuda") return Backend::cuda;
  throw std::invalid_argument("backend must be 'cpu' or 'cuda'");
}

Solver parse_solver(const std::string& s) {
  if (s == "gp") return Solver::gerchberg_papoulis;
  if (s == "tv") return Solver::tv;
  throw std::invalid_argument("solver must be 'gp' or 'tv'");
}

GpTransform parse_transform(const std::string& s) {
  if (s == "r2c") return GpTransform::r2c;
  if (s == "c2c") return GpTransform::c2c;
  throw std::invalid_argument("gp_transform must be 'r2c' or 'c2c'");
}

Mapping parse_mapping(const std::string& s) {
  if (s == "gather") return Mapping::gather;
  if (s == "scatter") return Mapping::scatter;
  throw std::invalid_argument("mapping must be 'gather' or 'scatter'");
}

}  // namespace

NB_MODULE(_core, m) {
  m.def("cuda_available", &cuda_available);

  nb::class_<Geometry>(m, "Geometry")
      .def(nb::init<>())
      .def(
          "__init__",
          [](Geometry* g, float wavelength, float n_medium, float na, float pixel) {
            new (g) Geometry{wavelength, n_medium, na, pixel};
          },
          nb::arg("wavelength"), nb::arg("n_medium"), nb::arg("na"), nb::arg("pixel"))
      .def_rw("wavelength", &Geometry::wavelength)
      .def_rw("n_medium", &Geometry::n_medium)
      .def_rw("na", &Geometry::na)
      .def_rw("pixel", &Geometry::pixel)
      .def("__repr__", [](const Geometry& g) {
        return "Geometry(wavelength=" + std::to_string(g.wavelength) +
               ", n_medium=" + std::to_string(g.n_medium) + ", na=" + std::to_string(g.na) +
               ", pixel=" + std::to_string(g.pixel) + ")";
      });

  nb::class_<Sphere>(m, "Sphere")
      .def(
          "__init__",
          [](Sphere* s, float z, float y, float x, float radius, float n) {
            new (s) Sphere{z, y, x, radius, n};
          },
          nb::arg("z"), nb::arg("y"), nb::arg("x"), nb::arg("radius"), nb::arg("n"))
      .def_rw("z", &Sphere::z)
      .def_rw("y", &Sphere::y)
      .def_rw("x", &Sphere::x)
      .def_rw("radius", &Sphere::radius)
      .def_rw("n", &Sphere::n);

  nb::class_<Sideband>(m, "Sideband")
      .def_ro("iy", &Sideband::iy)
      .def_ro("ix", &Sideband::ix)
      .def_ro("y", &Sideband::y)
      .def_ro("x", &Sideband::x)
      .def_ro("magnitude", &Sideband::magnitude);

  m.def(
      "find_sideband",
      [](In3<float> holograms, float dc_exclusion) {
        auto h = to_array(holograms);
        nb::gil_scoped_release release;
        return find_sideband(h, dc_exclusion);
      },
      nb::arg("holograms"), nb::arg("dc_exclusion") = 0.0f);

  m.def(
      "extract_fields",
      [](In3<float> holograms, std::optional<std::array<float, 2>> center, float radius,
         float dc_exclusion, int output_size, const std::string& backend) {
        auto h = to_array(holograms);
        const FieldOptions opt{center, radius, dc_exclusion, output_size};
        const Backend b = parse_backend(backend);
        Array3<cfloat> out;
        {
          nb::gil_scoped_release release;
          out = extract_fields(h, opt, b);
        }
        return to_numpy(std::move(out));
      },
      nb::arg("holograms"), nb::arg("center") = nb::none(), nb::arg("radius") = 0.0f,
      nb::arg("dc_exclusion") = 0.0f, nb::arg("output_size") = 0, nb::arg("backend") = "cpu");

  m.def(
      "field_ratio",
      [](In3<cfloat> sample, In3<cfloat> background) {
        return to_numpy(field_ratio(to_array(sample), to_array(background)));
      },
      nb::arg("sample"), nb::arg("background"));

  m.def(
      "unwrap",
      [](In3<float> wrapped, const std::string& backend) {
        auto w = to_array(wrapped);
        const Backend b = parse_backend(backend);
        Array3<float> out;
        {
          nb::gil_scoped_release release;
          out = unwrap(w, b);
        }
        return to_numpy(std::move(out));
      },
      nb::arg("wrapped"), nb::arg("backend") = "cpu");

  m.def(
      "rytov",
      [](In3<cfloat> ratio, const std::string& backend) {
        auto r = to_array(ratio);
        const Backend b = parse_backend(backend);
        Array3<cfloat> out;
        {
          nb::gil_scoped_release release;
          out = rytov(r, b);
        }
        return to_numpy(std::move(out));
      },
      nb::arg("ratio"), nb::arg("backend") = "cpu");

  m.def(
      "reconstruct",
      [](In3<cfloat> fields,
         nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> illuminations,
         const Geometry& geometry, int nz, int iterations, const std::string& solver,
         const std::string& gp_transform, float tv_weight, int tv_inner, const std::string& mapping,
         const std::string& backend) {
        auto f = to_array(fields);
        const auto il = to_illum(illuminations);
        OdtOptions opt;
        opt.nz = nz;
        opt.iterations = iterations;
        opt.solver = parse_solver(solver);
        opt.gp_transform = parse_transform(gp_transform);
        opt.tv_weight = tv_weight;
        opt.tv_inner = tv_inner;
        opt.mapping = parse_mapping(mapping);
        opt.backend = parse_backend(backend);
        OdtResult r;
        {
          nb::gil_scoped_release release;
          r = reconstruct(f, il, geometry, opt);
        }
        return nb::make_tuple(to_numpy(std::move(r.ri)), r.filled);
      },
      nb::arg("fields"), nb::arg("illuminations"), nb::arg("geometry"), nb::arg("nz") = 0,
      nb::arg("iterations") = 0, nb::arg("solver") = "gp", nb::arg("gp_transform") = "r2c",
      nb::arg("tv_weight") = OdtOptions{}.tv_weight, nb::arg("tv_inner") = OdtOptions{}.tv_inner,
      nb::arg("mapping") = "gather", nb::arg("backend") = "cpu");

  nb::class_<Pipeline>(m, "Pipeline")
      .def(
          "__init__",
          [](Pipeline* p, int count, int hologram_size, std::array<float, 2> center, float radius,
             int output_size,
             nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> illuminations,
             const Geometry& geometry, int nz, int iterations, const std::string& solver,
             const std::string& gp_transform, float tv_weight, int tv_inner,
             const std::string& mapping, const std::string& backend, int upload_chunks) {
            PipelineOptions o;
            o.count = count;
            o.hologram_size = hologram_size;
            o.field.center = center;
            o.field.radius = radius;
            o.field.output_size = output_size;
            o.illuminations = to_illum(illuminations);
            o.geometry = geometry;
            o.odt.nz = nz;
            o.odt.iterations = iterations;
            o.odt.solver = parse_solver(solver);
            o.odt.gp_transform = parse_transform(gp_transform);
            o.odt.tv_weight = tv_weight;
            o.odt.tv_inner = tv_inner;
            o.odt.mapping = parse_mapping(mapping);
            o.odt.backend = parse_backend(backend);
            o.upload_chunks = upload_chunks;
            new (p) Pipeline(std::move(o));
          },
          nb::arg("count"), nb::arg("hologram_size"), nb::arg("center"), nb::arg("radius"),
          nb::arg("output_size"), nb::arg("illuminations"), nb::arg("geometry"), nb::arg("nz"),
          nb::arg("iterations"), nb::arg("solver"), nb::arg("gp_transform"), nb::arg("tv_weight"),
          nb::arg("tv_inner"), nb::arg("mapping"), nb::arg("backend"), nb::arg("upload_chunks"))
      .def(
          "set_background",
          [](Pipeline& p, In3<float> backgrounds) {
            auto b = to_array(backgrounds);
            nb::gil_scoped_release release;
            p.set_background(b);
          },
          nb::arg("backgrounds"))
      .def(
          "run",
          [](Pipeline& p, In3<float> holograms) {
            const auto in = p.input();
            if (holograms.size() != in.size()) {
              throw std::invalid_argument("Pipeline.run: shape must be (count, N, N)");
            }
            std::copy(holograms.data(), holograms.data() + holograms.size(), in.begin());
            const auto [nz, ny, nx] = p.output_shape();
            Array3<float> out(nz, ny, nx);
            {
              nb::gil_scoped_release release;
              p.process();
              const auto o = p.output();
              std::copy(o.begin(), o.end(), out.data.begin());
            }
            return to_numpy(std::move(out));
          },
          nb::arg("holograms"))
      .def_prop_ro("filled", &Pipeline::filled)
      .def_prop_ro("setup_ms", &Pipeline::setup_ms)
      .def_prop_ro("device_bytes", &Pipeline::device_bytes)
      .def_prop_ro("timings", [](const Pipeline& p) {
        const PipelineTimings& t = p.timings();
        nb::dict d;
        d["upload"] = t.upload;
        d["fields"] = t.fields;
        d["rytov"] = t.rytov;
        d["fft2"] = t.fft2;
        d["clear"] = t.clear;
        d["mapping"] = t.mapping;
        d["normalize"] = t.normalize;
        d["solve"] = t.solve;
        d["finish"] = t.finish;
        d["download"] = t.download;
        d["total"] = t.total;
        return d;
      });

  m.def(
      "circular_scan",
      [](int count, float na_illumination, const Geometry& geometry) {
        const auto il = circular_scan(count, na_illumination, geometry);
        Array3<float> out(1, count, 2);
        for (int i = 0; i < count; ++i) {
          out.data[2 * i] = il[i].fy;
          out.data[2 * i + 1] = il[i].fx;
        }
        return to_numpy(std::move(out));
      },
      nb::arg("count"), nb::arg("na_illumination"), nb::arg("geometry"));

  m.def(
      "render_phantom",
      [](const std::vector<Sphere>& spheres, const Geometry& geometry, int nz, int n) {
        return to_numpy(render_phantom(spheres, geometry, nz, n));
      },
      nb::arg("spheres"), nb::arg("geometry"), nb::arg("nz"), nb::arg("n"));

  m.def(
      "forward_fields",
      [](const std::vector<Sphere>& spheres,
         nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> illuminations,
         const Geometry& geometry, int n, const std::string& model) {
        ScatteringModel m{};
        if (model == "rytov") {
          m = ScatteringModel::rytov;
        } else if (model == "born") {
          m = ScatteringModel::born;
        } else if (model == "exact") {
          m = ScatteringModel::exact;
        } else {
          throw std::invalid_argument("model must be 'rytov', 'born' or 'exact'");
        }
        const auto il = to_illum(illuminations);
        Array3<cfloat> out;
        {
          nb::gil_scoped_release release;
          out = forward_fields(spheres, il, geometry, n, m);
        }
        return to_numpy(std::move(out));
      },
      nb::arg("spheres"), nb::arg("illuminations"), nb::arg("geometry"), nb::arg("n"),
      nb::arg("model") = "rytov");

  m.def(
      "bpm_forward",
      [](In3<float> ri,
         nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> illuminations,
         const Geometry& geometry, bool obliquity, const std::string& backend) {
        auto v = to_array(ri);
        const auto il = to_illum(illuminations);
        const BpmOptions opt{obliquity, parse_backend(backend)};
        Array3<cfloat> out;
        {
          nb::gil_scoped_release release;
          out = bpm_forward(v, il, geometry, opt);
        }
        return to_numpy(std::move(out));
      },
      nb::arg("ri"), nb::arg("illuminations"), nb::arg("geometry"), nb::arg("obliquity") = true,
      nb::arg("backend") = "cpu");

  m.def(
      "bpm_reconstruct",
      [](In3<cfloat> fields,
         nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> illuminations,
         const Geometry& geometry, In3<float> initial, int iterations, int batch, float step,
         float tv_weight, int tv_inner, bool obliquity, const std::string& backend) {
        auto f = to_array(fields);
        auto init = to_array(initial);
        const auto il = to_illum(illuminations);
        BpmSolveOptions opt;
        opt.iterations = iterations;
        opt.batch = batch;
        opt.step = step;
        opt.tv_weight = tv_weight;
        opt.tv_inner = tv_inner;
        opt.obliquity = obliquity;
        opt.backend = parse_backend(backend);
        BpmResult r;
        {
          nb::gil_scoped_release release;
          r = bpm_reconstruct(f, il, geometry, init, opt);
        }
        return nb::make_tuple(to_numpy(std::move(r.ri)), r.loss);
      },
      nb::arg("fields"), nb::arg("illuminations"), nb::arg("geometry"), nb::arg("initial"),
      nb::arg("iterations"), nb::arg("batch"), nb::arg("step"), nb::arg("tv_weight"),
      nb::arg("tv_inner"), nb::arg("obliquity"), nb::arg("backend"));

  m.def(
      "synthesize_holograms",
      [](In3<cfloat> ratio,
         nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> illuminations,
         const Geometry& geometry, std::array<float, 2> carrier, float reference_amplitude,
         float noise, std::uint64_t seed) {
        const HologramOptions opt{carrier, reference_amplitude, noise, seed};
        return to_numpy(
            synthesize_holograms(to_array(ratio), to_illum(illuminations), geometry, opt));
      },
      nb::arg("ratio"), nb::arg("illuminations"), nb::arg("geometry"),
      nb::arg("carrier") = std::array<float, 2>{5.0f, 5.0f}, nb::arg("reference_amplitude") = 1.0f,
      nb::arg("noise") = 0.0f, nb::arg("seed") = 1);
}
