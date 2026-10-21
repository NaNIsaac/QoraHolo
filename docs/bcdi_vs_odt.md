# Bragg CDI and optical diffraction tomography: what is shared and what is not

Bragg coherent diffraction imaging (BCDI) and optical diffraction tomography (ODT) both
reconstruct a 3D object from Fourier-space data on Ewald spheres, under a single-scattering
model. That makes them look like one problem at two wavelengths. They are not: they measure
different quantities, so one inverse problem is non-convex and the other is linear, and they
lack different parts of Fourier space. This note sets the two side by side as they are
implemented here. On the BCDI side that is [QoraPhase](https://github.com/NaNIsaac/QoraPhase)
(phase retrieval), [CXD](https://github.com/NaNIsaac/CXD) (detector stacks) and a synthetic
Bragg benchmark in rocking_xrd_sim (not public yet; its findings on QoraPhase are written up in
QoraPhase#21). On the ODT side it is QoraHolo, including the first-Born operator of #2 and the
reconstruction study of #4. A bare #n below is a QoraHolo pull request. For each
repository it also says what the current tests establish and what they do not.

## At a glance

| | BCDI (QoraPhase) | ODT (QoraHolo) |
|---|---|---|
| Probe | hard X-rays, 5–15 keV, λ ≈ 0.08–0.25 nm (5.55 keV in the benchmark) | visible light, λ = 0.532 µm in the examples |
| Object | complex Bragg density ρ(r) = s(r) exp(i G·u(r)) of one crystal and one reflection G | scattering potential V(r) = k0²(n(r)² − n_m²), set by the refractive index n |
| Forward model | kinematic (first Born), far field: I(q) = \|F[ρ](q − G)\|² | first Born or Rytov field on the Ewald cap of each illumination; BPM beyond first order |
| Measured | intensity only, photon counts | complex field: amplitude and phase from an off-axis hologram |
| Illumination | one direction; the crystal is rocked through the Bragg condition | many directions (31 to 49 in the examples), sample fixed |
| Fourier coverage | a small box around G, \|Δq\| ≪ k | a union of caps through K = 0, laterally out to (NA_ill + NA)/λ, with a missing cone along the optical axis |
| Inverse problem | non-convex: modulus and support projections (ER, HIO, RAAR, shrinkwrap) | linear in V: Rytov gather, Landweber, Gerchberg–Papoulis, FISTA-TV |
| Ambiguities | global phase, translation, twin, phase ramp | none in the measured region; the null space is the missing cone |
| What fills the gaps | the support constraint, with oversampling ≥ 2 per axis | priors: a real-valued V, non-negativity, total variation |
| Noise | Poisson counts | camera noise on the hologram; Gaussian in the examples |
| The phase means | G·u: the displacement along G, i.e. strain | optical path delay: δn integrated along the beam |

## Forward models

**BCDI.** A crystal of shape s(r), with its lattice displaced by u(r), scatters near a
reciprocal-lattice vector G with the amplitude

  A(q) = ∫ s(r) exp(i G·u(r)) exp(−i (q − G)·r) dr,  I(q) ∝ \|A(q)\|²

(sign conventions differ between codes; they decide which of the two twins is "the" object).
This is the first Born approximation in the far field. It needs the crystal to be small
against the extinction length (typically micrometres), the illumination to be coherent across
the crystal, and the strain to be small enough that G·u carries all of it. The particles in
the benchmark are 120–250 nm. A thick, nearly perfect crystal breaks the first condition, and
then dynamical diffraction (the Takagi–Taupin equations, as in [TraTT](https://github.com/NaNIsaac/TraTT)) replaces this model.

Each detector frame samples the Ewald sphere of radius k = 2π/λ passing through G. The
measured box is tiny against k: in the benchmark's integration case, 64 pixels span 0.019 Å⁻¹
against k = 2.8 Å⁻¹. Each frame is therefore close to a plane through G, and rocking sweeps
that plane through the peak. The resulting 3D grid is uniform in (rocking angle, pixel).
That grid is linear in q only to first order, and it is sheared: the q steps form a
non-orthogonal matrix B. The linear fit leaves a residual of 0.3–0.6 % on the benchmark grids
(the benchmark's linear fit). The real-space voxel of a detector-frame reconstruction is
then R = 2π (B⁻¹)ᵀ diag(1/N). QoraPhase has no geometry of its own: the
input contract ([QoraPhase#9](https://github.com/NaNIsaac/QoraPhase/pull/9))
leaves B to the caller.

**ODT.** A transparent sample with refractive index n(r) in a medium n_m is illuminated by
plane waves k_in. To first order, the 2D spectrum of the scattered field at the detector gives
V̂ on a cap of the Ewald sphere, K = k_s − k_in with \|k_s\| = k_m = 2π n_m/λ. This is the
Fourier diffraction theorem. In 2D, QoraHolo#2 implements it as
R(μ′) = i/(4π ν_z) · V̂(μ′, ν_z − ν_in,z). Two first-order models are used:

- **Born:** the scattered field u_s itself is the linear quantity. It is valid while the
  phase the object adds is small, \|δn\| k0 L ≪ 1 (`born_phase` in #2).
- **Rytov:** u_in · ln(u/u_in) stands in for u_s. It holds while δn is small and the
  scattered phase φ_s varies slowly over a wavelength, roughly δn ≫ (λ\|∇φ_s\|/2π)²
  (Chen and Stamnes, 1998; Devaney). The condition does not involve the object's size, so
  Rytov tolerates a large accumulated phase where Born fails. This is why `rytov` comes
  before `reconstruct` in QoraHolo's pipeline. The logarithm needs the phase unwrapped (least squares in QoraHolo).

Each illumination contributes a cap through K = 0, shifted by −k_in. Tilting the
illumination moves the cap, and the union of the tilted caps fills a 3D region, minus the
missing cone around the axis. Because each cap is a curved surface, not a plane, the
mapping onto a Cartesian K grid is done by an Ewald-sphere gather. Alternatively, as in
#2, both forward and adjoint are evaluated as exact sums at the cap frequencies. When the
first order fails (a thick cell, a large δn), the multi-slice beam propagation method (BPM)
models forward multiple scattering. #2 measures exactly that failure: the Born–BPM gap
doubles when δn doubles (2.11 % at \|δn\| k0 L = 0.047, 4.22 % at 0.094), the scaling of the
neglected second order.

What is shared is the first-order (single-scattering) structure: in both, the data are the
Fourier transform of the object on Ewald spheres. What is not shared is everything that
follows from the scale. BCDI samples a flat neighbourhood of one G. ODT samples curved caps
through the origin of K, out to a large fraction of 2k.

## What is measured

**BCDI records \|A\|² only.** The phase of the far field is lost, and with it half of the
data. Phase retrieval can recover it because the pattern is oversampled: at more than twice
the Nyquist rate of the object's extent, the number of independent intensities exceeds the
number of unknowns inside a known support. The counts are Poisson, so the natural misfit is
on √I. For counts well above one, var(√I) ≈ 1/4, which sets the noise floor of the Fourier
error (QoraPhase#12 checks the floor against √(N/4/ΣI)).

**ODT records the complex field.** An off-axis hologram interferes the object wave with a
tilted reference. The sideband of its 2D spectrum is the field, which is cropped, shifted to
DC and inverse transformed (`extract_fields`). A background hologram without the sample gives
u_in, and the ratio u/u_in removes the illumination. The phase is measured directly, up to
2π wraps. Nothing has to be retrieved; the work is in carrier estimation, unwrapping and the
validity of Born or Rytov.

This is the root of every later difference. Measuring the phase makes ODT's data linear in
V. Measuring only intensity makes BCDI's data quadratic in ρ.

## Inverse problems

**BCDI: a non-convex feasibility problem.** QoraPhase looks for an object that has the
measured Fourier magnitude and vanishes outside a support. Each constraint has a cheap
projection: replace \|F[ρ]\| by √I where measured, and zero ρ outside the support. The
magnitude set is not convex, so the iteration can stall or land in different solutions from
different starts. The algorithms are compositions of the two projections chosen to escape
stagnation: ER, HIO, RAAR, and shrinkwrap to tighten the support. The default recipe runs
RAAR, a detwin step, RAAR with shrinkwrap, then ER, from several random starts. The FFT is
unitary, so no adjoint has to be derived.

Four transformations leave \|F[ρ]\| unchanged: a global phase, a translation, the twin
conj(ρ(−r)), and a linear phase ramp, which is the q-centring of the pattern. Any score has to
take them out first (the benchmark's `align`). The ramp has a cost. A slipped crop is a q
shift, and hence exactly such a ramp, so the shape scores cannot see one. The benchmark
scorer adds `q_offset_px`, the centroid distance between the two diffraction patterns, for that
reason: 0.03–0.05 voxel on correct runs, about 0.9 per voxel of slip.

Choosing among starts is part of the problem. On the strained benchmark cases, the per-start
phase error is bimodal, near 0.08 rad and 0.18–0.26 rad, while the Fourier errors of the two
modes differ by about 1e-3. Picking the start with the lowest Fourier error is then close to a
coin toss (QoraPhase#21). Picking the start closest to the consensus of the aligned starts
removes the bad picks; that is QoraPhase#20.

**ODT: a linear inverse problem.** With Born or Rytov, the data are d = A V with A linear, so
the problem is least squares. Its non-uniqueness lies entirely in the null space of A, the
unmeasured K.

- Rytov ODT in QoraHolo is a direct gather. Each measured cap value is divided by the
  transfer factor (i/(4π ν_z) in 2D) and placed at the nearest K voxel. Values landing on the
  same voxel are averaged ("normalise" in the data flow of `ARCHITECTURE.md`). An inverse 3D
  FFT follows, then n = √(n_m² + V/k0²).
- Landweber (#4) iterates V ← V + τ Aᴴ(d − A V), restricted to a real potential. This needs
  the exact adjoint Aᴴ, because a wrong adjoint moves the fixed point or makes the iteration diverge. #2 tests ⟨Av, u⟩ =
  ⟨v, Aᴴu⟩ to 1e-12 for exactly this reason.
- Gerchberg–Papoulis alternates between the measured K values and an object-space constraint
  (non-negativity).
- FISTA-TV adds a total-variation penalty.

The iterations converge, but where to depends on the start through the null space. Landweber
from zero goes to the minimum-norm solution (#4). GP goes to a point of the intersection of
its two convex sets. TV is convex but not strictly convex, so FISTA-TV need not have a unique
minimiser either. Only a strictly convex objective, e.g. one with a Tikhonov term, makes the
minimiser unique. There is no twin and no translation ambiguity.

Gerchberg–Papoulis and BCDI's error reduction are the same alternating-projection loop: data
consistency in Fourier space, a constraint in object space. The difference is in the Fourier
constraint. In GP it fixes known complex values, an affine set; the object-space set
(V ≥ 0) is convex too, so the iteration converges to a point of their intersection. In ER it fixes only the magnitude, which is not convex. That one
difference is the gap between the two inverse problems.

## Missing information

| | BCDI | ODT |
|---|---|---|
| Never measured | the phase, everywhere | nothing on the measured caps |
| Unmeasured regions | detector gaps, hot or saturated pixels (masked); beyond the detector edge and the rocking range | the missing cone along the optical axis, outside the illumination and detection NA |
| How the code treats them | masked voxels are left free by the modulus projection, never forced to zero (QoraPhase input contract) | outside the coverage, V̂ is not constrained by the data |
| What puts information back | the support constraint and oversampling | priors only: a real-valued V (no absorption), which mirrors the measured caps to −K; non-negativity (GP); TV (FISTA-TV) |
| Visible effect | resolution set by the q extent of the crop (R above); weak high-q tails lost to noise | axial elongation: PSF 0.502 µm against 0.151 µm laterally (#4); halo and underestimated δn |

In BCDI the direct beam does not reach a detector parked at a Bragg angle. So there is no
beamstop hole at the centre, unlike forward-scattering CDI. What is missing is the phase, plus
whatever the gaps and the finite box cut out. In ODT the phase is measured, but a whole cone
of K is not, and only priors put information there. The most physical of them is a
real-valued V (no absorption), which mirrors the measured caps to −K: in #4 it raises the
rank from 715 to 1005 of 4096. #4 makes the rest quantitative. Of a 36 % underestimate of δn
for a 2 µm disc, 23.6 % is information the data do not carry (projection onto the row space,
at a cutoff of 1e-5 of the largest singular value), 12.0 % is early stopping and 0.18 % is the Born model error.

## What each repository's validation establishes

**QoraPhase**

Established:
- The default path reconstructs a known noiseless 64² object to 7.9e-7 (README).
- On a 24³ object with Poisson counts at 1e6 photons in the brightest voxel, the default
  recipe ends at the noise floor: 0.005037 against a floor of 0.005121, accepted between 0.9
  and 1.05 times the floor (QoraPhase#12).
- The input contract (axis order, centring, mask polarity, NaN handling) is checked rule by
  rule on synthetic stacks (QoraPhase#9).

Not established:
- phase accuracy on strained crystals;
- robustness of the choice among starts (see above; QoraPhase#20 addresses it);
- behaviour on measured data (partial coherence, detector response, drifts).

**rocking_xrd_sim benchmark**

Established:
- A detector-frame truth whose \|DFT\|² is the noiseless frames to 2e-4.
- On a 12-case grid, QoraPhase reaches the Fourier error of the truth cut to a compact
  support (0.05 noiseless): the remaining misfit comes from the final shrinkwrap support, not
  from stagnation. Started from the truth, the default recipe ends at the same value
  (QoraPhase#21). This support-limited floor is specific to BCDI: phase retrieval needs a
  compact support, while the band-limited truth has tails over the whole grid. ODT's linear
  inversion has no such floor; its limit is the missing cone (#4).
- The phase error grows with strain: about 0.03 rad unstrained, 0.06–0.19 rad at 1.2 rad,
  0.17–0.21 rad at 3.0 rad.
- One synthetic case runs from TIFF files through CXD and QoraPhase to a score, with the CXD
  q steps matching the xrayutilities geometry to 1e-5.
- Swapped or reversed detector axes fail the score, and a slipped crop is flagged.

Not established:
- the comparison with a MATLAB reference pipeline, which needs MATLAB and a CUDA GPU;
- the full 270-case grid;
- the lab-frame transform beyond a linear-map IoU check;
- any dynamical-diffraction regime.

**QoraHolo**

Established:
- CPU and CUDA agree to 4e-6 to 4e-5 relative.
- On a synthetic cell with 1 % Gaussian noise, the nucleus index is 1.3683 (Rytov), 1.3757
  (GP) and 1.3762 (FISTA-TV) against 1.3750 (README). The TV weight was tuned on this phantom.
- The 2D Born operator has an exact adjoint (1e-12) and matches a closed-form spectrum (1e-9).
  Its error against BPM scales as the neglected second order (#2).
- Resolution follows the coverage, and the δn deficit is split into null space, early stopping
  and model error (#4).

Not established:
- measured holograms;
- a 3D Born operator (#2 is 2D);
- the strongly multiple-scattering regime beyond BPM;
- recovering δn to 5 % without a prior. #4 keeps that as a strict xfail (#3).

## Words that mean different things

| Word | In BCDI | In ODT |
|---|---|---|
| phase | G·u, lattice displacement along G (strain) | optical path delay from δn, integrated along the beam |
| support | hard constraint that makes phase retrieval unique; shrinkwrapped | optional prior; the measured caps fix V̂ without it |
| error | Fourier error on √I with the best scale | residual of the complex field (\|\|d − A V\|\| / \|\|d\|\|) |
| resolution | q extent of the crop: R = 2π (B⁻¹)ᵀ diag(1/N) | lateral cutoff frequency (NA_ill + NA)/λ; PSF FWHM ≈ 0.6 λ/(NA_ill + NA) for a uniformly filled band (0.151 µm measured in #4); worse axially from the missing cone (0.502 µm) |
| oversampling | > 2 per axis, needed for phase retrieval | not needed; Nyquist sampling of the field suffices |
| Born | the kinematic far-field model, valid for crystals well below the extinction length | the first-order field model, valid for \|δn\| k0 L ≪ 1 |

Moving a method across without its assumption causes problems in both directions. A TV prior
helps BCDI only if the strain field is piecewise smooth, which a dislocation is not. A
support-only reconstruction cannot fill ODT's missing cone, because that region is not
constrained by oversampling.
