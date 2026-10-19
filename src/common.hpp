#pragma once

// Index and physics helpers shared by the CPU and CUDA backends, so both evaluate the same
// expressions in the same order.

#include <cmath>

#ifdef __CUDACC__
#define QH_HD __host__ __device__ __forceinline__
#else
#define QH_HD inline
#endif

namespace qoraholo::detail {

inline constexpr float kPi = 3.14159265358979323846f;

// (-1)^i for integer i.
QH_HD float parity(int i) { return (i & 1) ? -1.0f : 1.0f; }

// Signed frequency index of DFT bin k of an n-point transform.
QH_HD int signed_bin(int k, int n) { return k < n / 2 ? k : k - n; }

// Makhoul reordering for a DCT-II of length n: x[i] goes to v[dct_position(i, n)].
QH_HD int dct_position(int i, int n) { return (i & 1) ? n - 1 - (i >> 1) : (i >> 1); }

// Eigenvalue of the 5-point Neumann Laplacian for DCT-II mode (ky, kx).
QH_HD float neumann_eigenvalue(int ky, int kx, int ny, int nx) {
  return 2.0f * cosf(kPi * static_cast<float>(ky) / static_cast<float>(ny)) +
         2.0f * cosf(kPi * static_cast<float>(kx) / static_cast<float>(nx)) - 4.0f;
}

QH_HD float wrap_phase(float p) { return p - 2.0f * kPi * floorf((p + kPi) / (2.0f * kPi)); }

// Constants of the k-space grid used by the Ewald mapping. Frequencies in cycles / um.
struct OdtGrid {
  int n;       // lateral side of fields and volume
  int nz;      // axial side of the volume
  float df;    // lateral frequency step, 1 / (n * pixel)
  float dfz;   // axial frequency step, 1 / (nz * pixel)
  float fm2;   // (n_medium / wavelength)^2
  float fna2;  // (na / wavelength)^2
  float gain;  // 4 pi / pixel; a mapped value is -i * gain * nu_z * spectrum
};

struct EwaldTarget {
  int kz;     // axial bin, wrapped to [0, nz)
  float nuz;  // axial frequency of the scattered wave
};

// Where the spectrum value at lateral bin (ky, kx) of an illumination with frequency
// (fiy, fix, fiz) lands in 3D k-space: the nearest axial bin. Linear splitting between two bins
// with per-voxel normalisation was tried and copies the value up to a full bin away, which near
// the cap apex (where the spectrum of a cell-sized object changes sign within a few bins)
// inflated those voxels by tens of percent. Returns false outside the pupil or the grid.
QH_HD bool ewald_target(const OdtGrid& g, int ky, int kx, float fiy, float fix, float fiz,
                        EwaldTarget& t) {
  const float fy = static_cast<float>(signed_bin(ky, g.n)) * g.df + fiy;
  const float fx = static_cast<float>(signed_bin(kx, g.n)) * g.df + fix;
  const float rho2 = fy * fy + fx * fx;
  if (rho2 > g.fna2 || rho2 >= g.fm2) return false;
  t.nuz = sqrtf(g.fm2 - rho2);
  const int k = static_cast<int>(floorf((t.nuz - fiz) / g.dfz + 0.5f));
  if (k < -g.nz / 2 || k >= g.nz / 2) return false;
  t.kz = k < 0 ? k + g.nz : k;
  return true;
}

// Bin of -k on an n-point DFT grid.
QH_HD int negative_bin(int k, int n) { return k == 0 ? 0 : n - k; }

// Gerchberg-Papoulis on a real potential, k-space step on the half spectrum. The C2C loop takes
// the real part after its inverse FFT, which keeps the Hermitian part (Y(k) + conj Y(-k)) / 2
// of the restored spectrum Y (data where measured, the estimate E elsewhere). E is Hermitian,
// so that part is a + b E with
//   a = (P(k) + conj P(-k)) / 2   (P: mapped data, zero where unmeasured)
//   b = 1 - (M(k) + M(-k)) / 2    (M: 1 where measured)
// and the R2C loop reproduces the C2C iterates. hits = M(k) + M(-k).
QH_HD void hermitian_data(float pre, float pim, float nre, float nim, float& are, float& aim) {
  are = 0.5f * (pre + nre);
  aim = 0.5f * (pim - nim);
}
QH_HD void hermitian_restore(float are, float aim, int hits, float& re, float& im) {
  const float b = 1.0f - 0.5f * static_cast<float>(hits);
  re = are + b * re;
  im = aim + b * im;
}

// Volume of nz * ny rows of nx values, rows `pitch` values apart (pitch > nx for the padded
// real layout of in-place R2C transforms).
struct Volume {
  int nz;
  int ny;
  int nx;
  int pitch;
  QH_HD long long at(int z, int y, int x) const {
    return (static_cast<long long>(z) * ny + y) * pitch + x;
  }
};

// Total-variation proximal step by fast gradient projection on the dual (Beck & Teboulle 2009,
// algorithm FGP): min_x 0.5 |x - b|^2 + lambda TV(x) subject to x >= 0, isotropic TV with
// forward differences. The dual variable p holds one vector per voxel; its component along an
// axis is zero on the last plane of that axis.
//
// L(p) at voxel i: sum over axes of p_a(i) - p_a(i - e_a), the negative divergence.
QH_HD float tv_adjoint(const float* pz, const float* py, const float* px, const Volume& v, int z,
                       int y, int x) {
  const long long i = v.at(z, y, x);
  float s = 0;
  if (z < v.nz - 1) s += pz[i];
  if (z > 0) s -= pz[v.at(z - 1, y, x)];
  if (y < v.ny - 1) s += py[i];
  if (y > 0) s -= py[v.at(z, y - 1, x)];
  if (x < v.nx - 1) s += px[i];
  if (x > 0) s -= px[i - 1];
  return s;
}

// Primal point for dual p: max(b - lambda L(p), 0).
QH_HD float tv_primal(float b, float lambda, float adjoint) {
  const float x = b - lambda * adjoint;
  return x > 0.0f ? x : 0.0f;
}

// One FGP dual step at a voxel. g = L^T z there (z(i) - z(i + e_a) per axis), step = 1 / (12
// lambda) since |L|^2 <= 12 in 3D, momentum = (t_k - 1) / t_{k+1}. Updates p and r in place.
QH_HD void tv_dual_step(const float g[3], float step, float momentum, float p[3], float r[3]) {
  float q[3];
  float norm2 = 0;
  for (int a = 0; a < 3; ++a) {
    q[a] = r[a] + step * g[a];
    norm2 += q[a] * q[a];
  }
  const float scale = norm2 > 1.0f ? 1.0f / sqrtf(norm2) : 1.0f;
  for (int a = 0; a < 3; ++a) {
    const float next = q[a] * scale;
    r[a] = next + momentum * (next - p[a]);
    p[a] = next;
  }
}

// L^T z at a voxel: forward differences z(i) - z(i + e_a), zero on the last plane of each axis.
QH_HD void tv_gradient(const float* zv, const Volume& v, int z, int y, int x, float g[3]) {
  const long long i = v.at(z, y, x);
  const float c = zv[i];
  g[0] = z < v.nz - 1 ? c - zv[v.at(z + 1, y, x)] : 0.0f;
  g[1] = y < v.ny - 1 ? c - zv[v.at(z, y + 1, x)] : 0.0f;
  g[2] = x < v.nx - 1 ? c - zv[i + 1] : 0.0f;
}

// t_{k+1} of FISTA and FGP.
QH_HD float fista_t(float t) { return 0.5f * (1.0f + sqrtf(1.0f + 4.0f * t * t)); }

// Real part of the principal sqrt(n_m^2 + O / k0^2): refractive index from scattering potential.
QH_HD float index_from_potential(float re, float im, float nm2, float inv_k02) {
  const float a = nm2 + re * inv_k02;
  const float b = im * inv_k02;
  return sqrtf(0.5f * (hypotf(a, b) + a));
}

}  // namespace qoraholo::detail
