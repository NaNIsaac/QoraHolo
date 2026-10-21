#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "backends.hpp"
#include "cuda/tv.cuh"
#include "cuda/util.cuh"

namespace qoraholo::cuda {

namespace {

// Kernels run on a chunk of nb illuminations: arrays of nb planes, one thread per element.

// v = exp(i phase x) * scale * w; the scaled envelope is also written to stored when given.
__global__ void bpm_screen(const float2* w, float scale, const float* x, float phase,
                           std::size_t plane, std::size_t total, float2* stored, float2* v) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  const float2 e = cscale(w[i], scale);
  if (stored) stored[i] = e;
  float s;
  float c;
  sincosf(phase * x[i % plane], &s, &c);
  v[i] = cmul({c, s}, e);
}

__global__ void bpm_propagate(const float2* v, const float2* coef, const float* q,
                              std::size_t total, float2* s) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  float2 r = s[i];
  detail::bpm_step(coef[i].x, coef[i].y, q[i], v[i].x, v[i].y, r.x, r.y);
  s[i] = r;
}

// Spectrum of the unit envelope (n^2 at the zero bin) and the unnormalised envelope n^2.
__global__ void bpm_start(float2* s, float2* w, std::size_t plane, std::size_t total) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  s[i] = {i % plane == 0 ? float(plane) : 0.0f, 0.0f};
  w[i] = {float(plane), 0.0f};
}

// r = scale * y - data, sum |r|^2 added to *loss.
__global__ void bpm_residual(const float2* y, float scale, const float2* data, std::size_t total,
                             float2* r, double* loss) {
  __shared__ double partial[256];
  const std::size_t i = grid_index();
  double sq = 0;
  if (i < total) {
    const float2 v = {scale * y[i].x - data[i].x, scale * y[i].y - data[i].y};
    r[i] = v;
    sq = double(v.x) * v.x + double(v.y) * v.y;
  }
  partial[threadIdx.x] = sq;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(loss, partial[0]);
}

// g holds the adjoint after the step (plus back, the screen's contribution, when given); leaves
// the part for the spectrum before the step in g and the part for the screen in b.
__global__ void bpm_adjoint_split(float2* g, const float2* back, const float2* coef, const float* q,
                                  std::size_t total, float2* b) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  float2 gi = g[i];
  if (back) {
    gi.x += back[i].x;
    gi.y += back[i].y;
  }
  float2 a;
  detail::bpm_step_adjoint(coef[i].x, coef[i].y, q[i], gi.x, gi.y, a.x, a.y);
  g[i] = gi;
  b[i] = a;
}

// One thread per pixel of slice k, summing over the nb illuminations: grad += gscale * phase *
// Im(conj(T w) b), and b <- conj(T) b, the adjoint passed on to the envelope.
__global__ void bpm_gradient(float2* b, float scale, const float2* stored, const float* x,
                             float phase, int nb, std::size_t plane, float gscale, float* grad) {
  const std::size_t i = grid_index();
  if (i >= plane) return;
  float s;
  float c;
  sincosf(phase * x[i], &s, &c);
  float acc = 0;
  for (int j = 0; j < nb; ++j) {
    const std::size_t o = j * plane + i;
    const float2 bv = cscale(b[o], scale);
    const float2 v = cmul({c, s}, stored[o]);
    acc += v.x * bv.y - v.y * bv.x;
    b[o] = cmul({c, -s}, bv);
  }
  grad[i] += gscale * phase * acc;
}

__global__ void gradient_step(const float* point, const float* grad, float alpha, std::size_t total,
                              float* z) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  z[i] = point[i] - alpha * grad[i];
}

__global__ void clamp_nonnegative(const float* z, std::size_t total, float* out) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  out[i] = fmaxf(z[i], 0.0f);
}

__global__ void extrapolate(const float* next, const float* x, float momentum, std::size_t total,
                            float* y) {
  const std::size_t i = grid_index();
  if (i >= total) return;
  y[i] = next[i] + momentum * (next[i] - x[i]);
}

// Device BPM for the illuminations of the tables, run on chunks of at most `chunk`
// illuminations at a time. With storage, forward() keeps the envelope in front of every screen
// for the following adjoint().
class Bpm {
 public:
  Bpm(const BpmTables& t, std::span<const int> chunk_sizes, bool storage)
      : t_(t),
        plane_(std::size_t(t.n) * t.n),
        chunk_(*std::max_element(chunk_sizes.begin(), chunk_sizes.end())),
        step_(t.step.size()),
        last_(t.last.size()),
        q_(t.q.size()),
        s_(chunk_ * plane_),
        w_(chunk_ * plane_),
        v_(chunk_ * plane_),
        stored_(storage ? std::size_t(t.nz) * chunk_ * plane_ : 0) {
    step_.upload(t.step.data());
    last_.upload(t.last.data());
    q_.upload(t.q.data());
    const std::array dims{t.n, t.n};
    std::vector<FftPlan*> list;
    for (int nb : std::set<int>(chunk_sizes.begin(), chunk_sizes.end())) {
      auto& p = plans_[nb];
      p = std::make_unique<FftPlan>(2, dims.data(), nb, CUFFT_C2C);
      list.push_back(p.get());
    }
    work_ = FftWorkArea(list);
  }

  // Envelopes at z = 0 of illuminations first .. first + nb - 1 at contrast x, unnormalised
  // (times n^2), left in w().
  void forward(const float* x, int first, int nb) {
    const FftPlan& plan = *plans_.at(nb);
    const std::size_t total = nb * plane_;
    const unsigned grid = blocks_for(total);
    const std::size_t off = first * plane_;
    bpm_start<<<grid, 256>>>(s_.get(), w_.get(), plane_, total);
    for (int k = 0; k < t_.nz; ++k) {
      if (k > 0) plan.c2c(s_.get(), w_.get(), CUFFT_INVERSE);
      float2* stored = stored_.size() ? stored_.get() + k * total : nullptr;
      bpm_screen<<<grid, 256>>>(w_.get(), 1.0f / float(plane_), x + k * plane_, t_.phase, plane_,
                                total, stored, v_.get());
      plan.forward(v_.get());
      const float2* coef = (k == t_.nz - 1 ? last_.get() : step_.get()) + off;
      bpm_propagate<<<grid, 256>>>(v_.get(), coef, q_.get() + off, total, s_.get());
    }
    plan.c2c(s_.get(), w_.get(), CUFFT_INVERSE);
    check_launch("bpm forward");
  }

  // grad += gscale * J^T (w / n^2 - data) for the chunk of the last forward(); sum |r|^2 is
  // added to *loss.
  void adjoint(const float* x, const float2* data, int first, int nb, float gscale, float* grad,
               double* loss) {
    const FftPlan& plan = *plans_.at(nb);
    const std::size_t total = nb * plane_;
    const unsigned grid = blocks_for(total);
    const std::size_t off = first * plane_;
    const float inv = 1.0f / float(plane_);
    float2* g = s_.get();
    float2* b = v_.get();
    bpm_residual<<<grid, 256>>>(w_.get(), inv, data + off, total, g, loss);
    plan.forward(g);
    for (int k = t_.nz - 1; k >= 0; --k) {
      const float2* coef = (k == t_.nz - 1 ? last_.get() : step_.get()) + off;
      bpm_adjoint_split<<<grid, 256>>>(g, k == t_.nz - 1 ? nullptr : b, coef, q_.get() + off, total,
                                       b);
      plan.inverse(b);
      bpm_gradient<<<blocks_for(plane_), 256>>>(b, inv, stored_.get() + k * total, x + k * plane_,
                                                t_.phase, nb, plane_, gscale, grad + k * plane_);
      plan.forward(b);
    }
    check_launch("bpm adjoint");
  }

  [[nodiscard]] const float2* w() const { return w_.get(); }

 private:
  const BpmTables& t_;
  std::size_t plane_;
  int chunk_;
  DeviceBuffer<float2> step_;
  DeviceBuffer<float2> last_;
  DeviceBuffer<float> q_;
  DeviceBuffer<float2> s_;
  DeviceBuffer<float2> w_;
  DeviceBuffer<float2> v_;
  DeviceBuffer<float2> stored_;
  std::map<int, std::unique_ptr<FftPlan>> plans_;
  FftWorkArea work_;
};

// Consecutive ranges [first, last) of at most `size` illuminations within [begin, end).
std::vector<std::pair<int, int>> ranges(int begin, int end, int size) {
  std::vector<std::pair<int, int>> out;
  for (int a = begin; a < end; a += size) out.emplace_back(a, std::min(a + size, end));
  return out;
}

// Illuminations per chunk such that the stored envelopes take at most half the free memory.
int chunk_for(const BpmTables& t, int batch) {
  std::size_t free = 0;
  std::size_t total = 0;
  check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  const std::size_t per = std::size_t(t.nz) * t.n * t.n * sizeof(float2);
  return std::clamp(static_cast<int>(free / 2 / per), 1, batch);
}

}  // namespace

Array3<cfloat> bpm_forward(const Array3<float>& x, const BpmTables& t) {
  const int sizes[] = {t.count};
  Bpm bpm(t, sizes, false);
  DeviceBuffer<float> dx(x.size());
  dx.upload(x.data.data());
  bpm.forward(dx.get(), 0, t.count);
  Array3<cfloat> out(t.count, t.n, t.n);
  check(cudaMemcpy(out.data.data(), bpm.w(), out.size() * sizeof(float2), cudaMemcpyDeviceToHost),
        "download fields");
  const float inv = 1.0f / (float(t.n) * float(t.n));
  for (auto& v : out.data) v *= inv;
  return out;
}

Array3<float> bpm_vjp(const Array3<float>& x, const Array3<cfloat>& r, const BpmTables& t) {
  // adjoint() differentiates 0.5 |w - data|^2; with data = w - r its residual is r.
  const int sizes[] = {t.count};
  Bpm bpm(t, sizes, true);
  DeviceBuffer<float> dx(x.size());
  dx.upload(x.data.data());
  bpm.forward(dx.get(), 0, t.count);
  Array3<cfloat> w(t.count, t.n, t.n);
  check(cudaMemcpy(w.data.data(), bpm.w(), w.size() * sizeof(float2), cudaMemcpyDeviceToHost),
        "download fields");
  const float inv = 1.0f / (float(t.n) * float(t.n));
  for (std::size_t i = 0; i < w.size(); ++i) w.data[i] = w.data[i] * inv - r.data[i];
  DeviceBuffer<float2> data(w.size());
  data.upload(w.data.data());
  DeviceBuffer<float> grad(x.size());
  grad.zero();
  DeviceBuffer<double> loss(1);
  loss.zero();
  bpm.adjoint(dx.get(), data.get(), 0, t.count, 1.0f, grad.get(), loss.get());
  Array3<float> out(t.nz, t.n, t.n);
  grad.download(out.data.data());
  return out;
}

BpmResult bpm_reconstruct(const Array3<cfloat>& fields, Array3<float> x0, const BpmTables& t,
                          const BpmSolveOptions& options, int chunk) {
  const int count = t.count;
  const int batch = options.batch <= 0 || options.batch >= count ? count : options.batch;
  const bool fista = batch == count;
  const float alpha = options.step / t.lipschitz;
  const bool tv = options.tv_weight > 0;
  const std::size_t size = x0.size();
  const unsigned vgrid = blocks_for(size);

  DeviceBuffer<float2> data(fields.size());
  data.upload(fields.data.data());
  // x, the extrapolated point, the prox output, the gradient and z = point - alpha grad.
  DeviceBuffer<float> volumes(5 * size);
  float* x = volumes.get();
  float* y = x + size;
  float* next = x + 2 * size;
  float* grad = x + 3 * size;
  float* z = x + 4 * size;
  DeviceBuffer<float> dual(tv ? 9 * size : 0);
  if (tv) check(cudaMemset(dual.get(), 0, 3 * size * sizeof(float)), "cudaMemset");
  DeviceBuffer<double> loss(std::max(options.iterations, 1));
  loss.zero();
  check(cudaMemcpy(z, x0.data.data(), size * sizeof(float), cudaMemcpyHostToDevice), "upload");
  clamp_nonnegative<<<vgrid, 256>>>(z, size, x);
  check(cudaMemcpy(y, x, size * sizeof(float), cudaMemcpyDeviceToDevice), "copy");

  const auto batches = ranges(0, count, batch);
  if (chunk <= 0) chunk = chunk_for(t, batch);
  chunk = std::min(chunk, batch);
  std::vector<int> sizes;
  for (const auto& [b0, b1] : batches) {
    for (const auto& [c0, c1] : ranges(b0, b1, chunk)) sizes.push_back(c1 - c0);
  }
  Bpm bpm(t, sizes, true);

  const detail::Volume vol{t.nz, t.n, t.n, t.n};
  const auto step = [&](const float* point, int b0, int b1, float* out, double* l) {
    check(cudaMemsetAsync(grad, 0, size * sizeof(float)), "cudaMemset");
    for (const auto& [c0, c1] : ranges(b0, b1, chunk)) {
      bpm.forward(point, c0, c1 - c0);
      bpm.adjoint(point, data.get(), c0, c1 - c0, 1.0f / float(b1 - b0), grad, l);
    }
    gradient_step<<<vgrid, 256>>>(point, grad, alpha, size, z);
    if (tv) {
      float* p = dual.get();
      tv_prox(z, 1.0f, out, p, p + 3 * size, p + 6 * size, size, alpha * options.tv_weight,
              options.tv_inner, vol, TvKernel::fused, nullptr);
    } else {
      clamp_nonnegative<<<vgrid, 256>>>(z, size, out);
    }
  };

  float tk = 1.0f;
  for (int it = 0; it < options.iterations; ++it) {
    if (fista) {
      step(y, 0, count, next, loss.get() + it);
      const float tn = detail::fista_t(tk);
      extrapolate<<<vgrid, 256>>>(next, x, (tk - 1.0f) / tn, size, y);
      std::swap(x, next);
      tk = tn;
    } else {
      for (const auto& [b0, b1] : batches) step(x, b0, b1, x, loss.get() + it);
    }
  }
  check_launch("bpm reconstruct");

  BpmResult result;
  result.ri = Array3<float>(t.nz, t.n, t.n);
  check(cudaMemcpy(result.ri.data.data(), x, size * sizeof(float), cudaMemcpyDeviceToHost),
        "download volume");
  result.loss.resize(options.iterations);
  if (options.iterations > 0) loss.download(result.loss.data());
  for (double& l : result.loss) l *= 0.5 / count;
  return result;
}

}  // namespace qoraholo::cuda
