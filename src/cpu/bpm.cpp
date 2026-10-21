#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "backends.hpp"
#include "cpu/fft.hpp"
#include "cpu/tv.hpp"

namespace qoraholo::cpu {

namespace {

using cvec = std::vector<cfloat>;

// exp(i phase x) for every voxel.
cvec screens(const Array3<float>& x, float phase) {
  cvec t(x.size());
  const auto total = static_cast<std::ptrdiff_t>(x.size());
#pragma omp parallel for
  for (std::ptrdiff_t i = 0; i < total; ++i) t[i] = std::polar(1.0f, phase * x.data[i]);
  return t;
}

void step(const BpmTables& t, int a, int k, const cfloat* spectrum, cfloat* s) {
  const std::size_t plane = std::size_t(t.n) * t.n;
  const cfloat* c = (k == t.nz - 1 ? t.last.data() : t.step.data()) + a * plane;
  const float* q = t.q.data() + a * plane;
  for (std::size_t i = 0; i < plane; ++i) {
    float re = s[i].real();
    float im = s[i].imag();
    detail::bpm_step(c[i].real(), c[i].imag(), q[i], spectrum[i].real(), spectrum[i].imag(), re,
                     im);
    s[i] = {re, im};
  }
}

// Envelope of illumination a at z = 0 into y (n * n). store: nz * n * n, receives the envelope
// in front of each screen when given.
void forward_angle(const BpmTables& t, const cvec& screen, int a, cfloat* store, cfloat* y) {
  const int n = t.n;
  const std::size_t plane = std::size_t(n) * n;
  const float inv = 1.0f / static_cast<float>(plane);
  cvec s(plane);
  cvec w(plane, cfloat(1.0f));
  cvec v(plane);
  s[0] = static_cast<float>(plane);  // spectrum of the unit envelope
  for (int k = 0; k < t.nz; ++k) {
    if (k > 0) {
      std::copy(s.begin(), s.end(), w.begin());
      fft2(w.data(), n, false, inv);
    }
    if (store) std::copy(w.begin(), w.end(), store + k * plane);
    const cfloat* tk = screen.data() + k * plane;
    for (std::size_t i = 0; i < plane; ++i) v[i] = tk[i] * w[i];
    fft2(v.data(), n, true);
    step(t, a, k, v.data(), s.data());
  }
  std::copy(s.begin(), s.end(), y);
  fft2(y, n, false, inv);
}

// grad += J_a^T r for illumination a, from the envelopes stored by forward_angle.
void adjoint_angle(const BpmTables& t, const cvec& screen, int a, const cfloat* store,
                   const cfloat* r, float* grad) {
  const int n = t.n;
  const std::size_t plane = std::size_t(n) * n;
  const float inv = 1.0f / static_cast<float>(plane);
  cvec g(r, r + plane);
  cvec b(plane);
  cvec back(plane);
  fft2(g.data(), n, true);
  for (int k = t.nz - 1; k >= 0; --k) {
    const cfloat* c = (k == t.nz - 1 ? t.last.data() : t.step.data()) + a * plane;
    const float* q = t.q.data() + a * plane;
    for (std::size_t i = 0; i < plane; ++i) {
      cfloat gi = g[i] + (k < t.nz - 1 ? back[i] : cfloat{});
      float gre = gi.real();
      float gim = gi.imag();
      float are = 0;
      float aim = 0;
      detail::bpm_step_adjoint(c[i].real(), c[i].imag(), q[i], gre, gim, are, aim);
      g[i] = {gre, gim};
      b[i] = {are, aim};
    }
    fft2(b.data(), n, false, inv);
    const cfloat* tk = screen.data() + k * plane;
    const cfloat* wk = store + k * plane;
    float* gk = grad + k * plane;
    for (std::size_t i = 0; i < plane; ++i) {
      const cfloat v = tk[i] * wk[i];
      gk[i] += t.phase * (v.real() * b[i].imag() - v.imag() * b[i].real());
      back[i] = std::conj(tk[i]) * b[i];
    }
    fft2(back.data(), n, true);
  }
}

// Gradient of 0.5 / (last - first) sum_{a in [first, last)} |bpm_a(x) - fields_a|^2 and that
// data term with the factor 0.5 / count instead.
double gradient(const BpmTables& t, const Array3<float>& x, const Array3<cfloat>& fields, int first,
                int last, std::vector<float>& grad) {
  const std::size_t plane = std::size_t(t.n) * t.n;
  const cvec screen = screens(x, t.phase);
  std::fill(grad.begin(), grad.end(), 0.0f);
  double loss = 0;
#pragma omp parallel
  {
    std::vector<float> local(grad.size(), 0.0f);
    cvec store(std::size_t(t.nz) * plane);
    cvec y(plane);
#pragma omp for schedule(dynamic) reduction(+ : loss)
    for (int a = first; a < last; ++a) {
      forward_angle(t, screen, a, store.data(), y.data());
      const cfloat* d = fields.slice(a);
      for (std::size_t i = 0; i < plane; ++i) {
        y[i] -= d[i];
        loss += std::norm(std::complex<double>(y[i]));
      }
      adjoint_angle(t, screen, a, store.data(), y.data(), local.data());
    }
#pragma omp critical
    for (std::size_t i = 0; i < grad.size(); ++i) grad[i] += local[i];
  }
  const float scale = 1.0f / static_cast<float>(last - first);
  for (float& g : grad) g *= scale;
  return 0.5 * loss / t.count;
}

}  // namespace

Array3<cfloat> bpm_forward(const Array3<float>& x, const BpmTables& t) {
  const cvec screen = screens(x, t.phase);
  Array3<cfloat> out(t.count, t.n, t.n);
#pragma omp parallel for schedule(dynamic)
  for (int a = 0; a < t.count; ++a) forward_angle(t, screen, a, nullptr, out.slice(a));
  return out;
}

Array3<cfloat> bpm_jvp(const Array3<float>& x, const Array3<float>& h, const BpmTables& t) {
  const int n = t.n;
  const std::size_t plane = std::size_t(n) * n;
  const float inv = 1.0f / static_cast<float>(plane);
  const cvec screen = screens(x, t.phase);
  Array3<cfloat> out(t.count, n, n);
  for (int a = 0; a < t.count; ++a) {
    cvec s(plane);
    cvec ds(plane);
    cvec w(plane, cfloat(1.0f));
    cvec dw(plane);
    cvec v(plane);
    cvec dv(plane);
    s[0] = static_cast<float>(plane);
    for (int k = 0; k < t.nz; ++k) {
      if (k > 0) {
        w = s;
        dw = ds;
        fft2(w.data(), n, false, inv);
        fft2(dw.data(), n, false, inv);
      }
      for (std::size_t i = 0; i < plane; ++i) {
        const cfloat tk = screen[k * plane + i];
        v[i] = tk * w[i];
        dv[i] = tk * (dw[i] + cfloat(0.0f, t.phase * h.data[k * plane + i]) * w[i]);
      }
      fft2(v.data(), n, true);
      fft2(dv.data(), n, true);
      step(t, a, k, v.data(), s.data());
      step(t, a, k, dv.data(), ds.data());
    }
    fft2(ds.data(), n, false, inv);
    std::copy(ds.begin(), ds.end(), out.slice(a));
  }
  return out;
}

Array3<float> bpm_vjp(const Array3<float>& x, const Array3<cfloat>& r, const BpmTables& t) {
  const std::size_t plane = std::size_t(t.n) * t.n;
  const cvec screen = screens(x, t.phase);
  Array3<float> grad(t.nz, t.n, t.n);
  cvec store(std::size_t(t.nz) * plane);
  cvec y(plane);
  for (int a = 0; a < t.count; ++a) {
    forward_angle(t, screen, a, store.data(), y.data());
    adjoint_angle(t, screen, a, store.data(), r.slice(a), grad.data.data());
  }
  return grad;
}

BpmResult bpm_reconstruct(const Array3<cfloat>& fields, Array3<float> x, const BpmTables& t,
                          const BpmSolveOptions& options) {
  const int count = t.count;
  const int batch = options.batch <= 0 || options.batch >= count ? count : options.batch;
  const float alpha = options.step / t.lipschitz;
  const bool tv = options.tv_weight > 0;
  const detail::Volume vol{t.nz, t.n, t.n, t.n};
  const std::size_t size = x.size();
  for (float& v : x.data) v = std::max(v, 0.0f);

  std::vector<float> grad(size);
  std::vector<float> z(size);
  Dual dual(tv ? size : 0);
  // z = point - alpha grad, then x_out = prox(z).
  const auto prox_step = [&](const Array3<float>& point, std::vector<float>& out) {
    for (std::size_t i = 0; i < size; ++i) z[i] = point.data[i] - alpha * grad[i];
    if (tv) {
      tv_prox(z.data(), out.data(), dual, vol, alpha * options.tv_weight, options.tv_inner);
    } else {
      for (std::size_t i = 0; i < size; ++i) out[i] = std::max(z[i], 0.0f);
    }
  };

  BpmResult result;
  if (batch == count) {
    // FISTA: gradient at the extrapolated point y.
    Array3<float> y = x;
    std::vector<float> next(size);
    float tk = 1.0f;
    for (int it = 0; it < options.iterations; ++it) {
      result.loss.push_back(gradient(t, y, fields, 0, count, grad));
      prox_step(y, next);
      const float tn = detail::fista_t(tk);
      const float m = (tk - 1.0f) / tn;
      for (std::size_t i = 0; i < size; ++i) {
        y.data[i] = next[i] + m * (next[i] - x.data[i]);
        x.data[i] = next[i];
      }
      tk = tn;
    }
  } else {
    for (int it = 0; it < options.iterations; ++it) {
      double loss = 0;
      for (int first = 0; first < count; first += batch) {
        loss += gradient(t, x, fields, first, std::min(first + batch, count), grad);
        prox_step(x, x.data);
      }
      result.loss.push_back(loss);
    }
  }
  result.ri = std::move(x);
  return result;
}

}  // namespace qoraholo::cpu
