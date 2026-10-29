# Findings

What this repository has shown so far, newest first. Each item links its evidence.

Every result here comes from synthetic data and model comparisons; none involves a measurement.

## F3: The sphere forward models agree with independent codes; Rytov's error grows with Δn
- Question: Do QoraHolo's sphere forward fields agree with independent codes (qpsphere, vector Mie theory)?
- Result: Rytov agrees with qpsphere to 1.3e-6 at the main point and to at most 7.3e-5 over all 30 sweep points, and first Born to 1.2e-6. The scalar `exact` model differs from the co-polarised vector Mie field by 0.2 % at NA 0.2 and 1.6 % at NA 1.2.
- Figure: none
- Data and script: [checks/results/](../checks/results), [sphere_crosscheck.py](../checks/sphere_crosscheck.py)
- Status: checked
- Limits: One on-axis illumination, no objective, synthetic only; the vector reference shares QoraHolo's refocusing relation but no code with it. Checked: two independent codes; [test_crosscheck.py](../python/tests/test_crosscheck.py).
- Report: [Cross-check of the sphere forward models](crosscheck.md) (2026-10-26)

## F5: The prior n ≥ n_m recovers the core index that the data do not carry, within 5 % up to moderate noise
- Question: Can a positivity prior (n ≥ n_m) recover the core index that a 2D first-Born reconstruction without a prior leaves out?
- Result: On noiseless Born data the core δn comes out at 0.00410 against a truth of 0.004 (0.00408 on BPM data). With BPM data and noise it stays within 5 % of the truth at η ≤ 0.3 and is 0.00376 ± 0.00022 (−6 %) at η = 1.0.
- Figure: [figure.png](../examples/born2d_reconstruction/figure.png)
- Data and script: [noise.json](../examples/born2d_reconstruction/noise.json), [expected.json](../examples/born2d_reconstruction/expected.json), [run.py](../examples/born2d_reconstruction/run.py), [noise.py](../examples/born2d_reconstruction/noise.py)
- Status: checked
- Limits: A best case: the object is sparse and positive on a background of exactly n_m, so n ≥ n_m acts almost like a known support; a region below n_m is clipped to n_m. 2D, scalar, white complex Gaussian noise only. The convergence criterion set beforehand (1500 → 3000 iteration change below 1e-5) was missed (2.4e-5), and TV did not meet its noise criterion. Checked: pass/fail criteria written before the runs; [test_example_born2d_reconstruction.py](../python/tests/test_example_born2d_reconstruction.py).
- Report: [Reconstructing a known 2D object with the first Born operator](../examples/born2d_reconstruction/README.md#with-the-prior-n--n_m-qoraholo3) (2026-10-23)

## F4: Without a prior, most of the missing index is information the data do not carry
- Question: How much of a known 2D object does a first-Born reconstruction recover from 31 illuminations, and where does the rest go?
- Result: The core δn (truth 0.00400) is 0.00305 for the information limit (−23.6 %, missing cone and band limit) and 0.00258 after 300 Landweber iterations (−12.0 % more, early stopping); the PSF FWHM is 0.151 µm lateral and 0.502 µm axial.
- Figure: [figure.png](../examples/born2d_reconstruction/figure.png)
- Data and script: [expected.json](../examples/born2d_reconstruction/expected.json), [run.py](../examples/born2d_reconstruction/run.py), [figure.py](../examples/born2d_reconstruction/figure.py)
- Status: checked
- Limits: 2D only, scalar model, synthetic data; Landweber without a prior is the baseline, not a recommended method. The information limit depends on the singular-value cutoff (0.00305 at 1e-5, 0.00315 at 1e-10). Checked: [test_example_born2d_reconstruction.py](../python/tests/test_example_born2d_reconstruction.py) against the physical expectations in expected.json; BPM data measure the Born model error.
- Report: [Reconstructing a known 2D object with the first Born operator](../examples/born2d_reconstruction/README.md#recovered-index-where-the-missing-36--goes) (2026-10-22)

## F2: A BPM inversion halves the RMSE of the best first-order reconstruction on multiple-scattering phantoms
- Question: How much do multiple-scattering (BPM) reconstructions improve on first-order ones for a large bead and three stacked beads?
- Result: RMSE 2.50e-3 against 4.93e-3 for Rytov + FISTA-TV on the bead (exact data), and 2.00e-3 against 3.89e-3 on the stack (finer-grid BPM data); plain Rytov underestimates the two objects by 19 % and 31 % of the contrast.
- Figure: [bpm.png](bpm.png)
- Data and script: no saved data, the script recomputes; [bpm_figure.py](../examples/bpm_figure.py)
- Status: draft
- Limits: No noise; the missing-cone elongation along z remains; the stack data come from the same physics model (BPM on a finer grid), so it tests the solver and discretisation only. The README records that CPU and CUDA agree to the printed digits, but no independent check of the numbers.
- Report: [Multiple scattering: beam propagation](../README.md#multiple-scattering-beam-propagation) (2026-10-21)

## F1: On a synthetic cell, the missing-cone solvers bring the nucleus index from 1.3683 to 1.3757-1.3762 (true 1.3750)
- Question: How close does the hologram-to-index pipeline come to a known synthetic cell, with and without a missing-cone solver?
- Result: Mean index in the nucleus (true 1.3750) is 1.3683 from Rytov ODT, 1.3757 after 20 Gerchberg-Papoulis iterations and 1.3762 after 20 FISTA-TV iterations; RMSE over the volume 2.37e-3, 9.25e-4 and 7.69e-4.
- Figure: [reconstruction.png](reconstruction.png)
- Data and script: no saved data, the script recomputes; [readme_figure.py](../examples/readme_figure.py)
- Status: draft
- Limits: Synthetic data (49 illuminations at NA 0.9, Gaussian noise of 1 % of the mean intensity); the TV weight was picked from a sweep on this phantom, so the TV numbers are tuned to it. The README records CPU and CUDA differing by 4e-6 to 4e-5 relative on the benchmark volumes, but no independent check of these numbers.
- Report: [QoraHolo README](../README.md#qoraholo) (2026-10-20)
