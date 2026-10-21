# TV solver: the FGP step on the GPU

Each outer FISTA iteration of the TV solver (see ARCHITECTURE.md) runs `tv_inner` steps of
the fast gradient projection on the TV dual. At the default 20 x 10 iterations that is 200
FGP steps against 40 3D FFTs, so the FGP step sets the cost of the solver. This note records
how the step was made faster and what did not help. All times are cudaEvent intervals on the
RTX 5070, measured with `qoraholo_bench --field N --tv 1` while holding the shared GPU
benchmark lock, with no other compute process on the GPU (`nvidia-smi`, sampled every 5 s).
Profiler counters from Nsight Compute 2026.3.1 at 256³ are in "Profiler counters" below; the
"traffic model" columns are a byte count from the algorithm divided by the event time, kept as
a cross-check.

## What one step does

For every voxel i, with b the result of the gradient step, r the momentum copy of the dual and
p the dual (three components each):

```
z(j)  = max(b(j) - λ L(r)(j), 0)          L(r)(j) = Σ_a r_a(j) - r_a(j - e_a)
g_a   = z(i) - z(i + e_a)
q     = r(i) + g / (12 λ);  p' = q / max(1, |q|);  r(i) = p' + m (p' - p(i));  p(i) = p'
```

The update at i needs z at i and at its three upper neighbours, and each z needs r at the
voxel and its three lower neighbours.

## Variants

| variant | kernels per step | volumes moved per step (model) |
|---|---:|---:|
| separate: z written by one kernel, read by the next | 2 | 18: read b, r (3), write z; read z, p (3), r (3), write p (3), r (3) |
| fused: z recomputed at the voxel and its three upper neighbours | 1 | 13: read b, r (3), p (3), write p (3), r (3) |
| fused, 16 x 8 x 4 tiles: b and r for the tile and a one-voxel halo in shared memory, z computed once per tile voxel | 1 | 13 |
| fused, 32 x 8 x-y tiles marching along z: two planes of r and z in shared memory | 1 | 13 |

The fused variants read r from one buffer and write the update into another, because
neighbouring threads still need the old r; that costs three more volumes of memory. All
variants do the same arithmetic in the same order (`tv_adjoint`, `tv_primal`, `tv_gradient`,
`tv_dual_step` in `common.hpp`, or the same expressions on shared memory).

## Result at 256³ (cudaEvent)

20 outer x 10 dual iterations on the README phantom. Per FGP step = (solve time with 10 dual
steps - solve time with 0) / 200. Medians of 5 runs.

| FGP step | solve (ms) | per FGP step (ms) | traffic model (GB/s) | max abs diff vs separate | RMSE |
|---|---:|---:|---:|---:|---:|
| separate (2 kernels) | 493.5 | 2.131 | 567 | 0.0e+00 | 7.686e-04 |
| fused (1 kernel) | 391.5 | 1.618 | 539 | 0.0e+00 | 7.686e-04 |
| fused, 16x8x4 shared-memory tiles | 396.8 | 1.645 | 530 | 1.2e-07 | 7.686e-04 |
| fused, 32x8 tiles marching along z | 397.4 | 1.649 | 529 | 0.0e+00 | 7.686e-04 |

"max abs diff" is the largest difference of the refractive index from the two-kernel version.
The fused and marching kernels give bitwise the same volume. The cube-tiled one differs by
about one float ulp of the index, presumably from a different contraction of a multiply-add
in that kernel.

At 512³ (medians of 3 runs, same configuration):

| FGP step | solve (ms) | per FGP step (ms) | traffic model (GB/s) | max abs diff vs separate | RMSE |
|---|---:|---:|---:|---:|---:|
| separate (2 kernels) | 3932.3 | 16.816 | 575 | 0.0e+00 | 8.066e-04 |
| fused (1 kernel) | 3090.8 | 12.607 | 554 | 0.0e+00 | 8.066e-04 |
| fused, 16x8x4 shared-memory tiles | 3380.2 | 14.053 | 497 | 2.4e-07 | 8.066e-04 |
| fused, 32x8 tiles marching along z | 3322.8 | 13.777 | 507 | 0.0e+00 | 8.066e-04 |

## What made the difference

The steps in order, each measured at 256³ on the build of that moment:

| change | per FGP step (ms) |
|---|---:|
| separate kernels (the version before this work) | 2.14 |
| fused | 1.91 |
| fused, 16 x 8 x 4 shared-memory tiles | 2.64 |
| fused, 32 x 8 marching tiles, 32 planes per block | 1.88 |
| marching tiles, loads of the voxel's own p issued before the barriers | 1.73 |
| same, 8 planes per block (sweep below) | 1.66 |
| cube tiles with the same early loads | 1.66 |
| fused without shared memory, own p and r loaded first | 1.63 |

Marching tile shapes, with the early loads, ms per FGP step: 32x8x32 1.73, 32x8x16 1.70,
32x8x8 1.66, 32x8x4 1.65, 32x8x2 1.64, 32x16x16 1.69, 32x16x8 1.65, 32x4x32 1.77,
64x4x16 1.67, 64x8x8 1.66, 64x4x8 1.65. Beyond 8 planes the shape hardly matters.

Two things helped: fusing the two kernels, which removes the write and the read of z (five
volumes of the eighteen), and issuing the loads of the voxel's own p and r before anything
that waits, so their latency overlaps the neighbour reads. Shared-memory tiling did not help.
The stencil reads each value of r a few times, from neighbouring threads, and those reads
already come from L1 and L2; staging them in shared memory adds the halo loads and barriers
without removing DRAM traffic. The first cube-tiled version was slower than the plain kernels
because each thread's p load came after two barriers. With that fixed, both tiled kernels
match the plain fused kernel within 2% at 256³ and are 11% (cube) and 9% (marching) slower
at 512³, so the
plain fused kernel is the default. The tiled kernels stay in the code, selectable through
the internal `TvKernel` enum, so the comparison above can be rerun.

## Profiler counters

Nsight Compute, 256³, `qoraholo_bench --field 256 --tv 1` under the GPU lock, each kernel
profiled on 4 to 6 launches after skipping the first ones; medians. DRAM bytes are
`dram__bytes.sum` (the read / write split, `dram__bytes_read.sum` and `dram__bytes_write.sum`,
reports n/a on this GPU). Bandwidth is bytes / `gpu__time_duration.sum`. A padded real volume
at 256³ is 64.5 MiB.

| kernel | time (µs) | DRAM (MiB) | volumes | GB/s | % of DRAM peak | L1 hit | L2 hit | achieved / theoretical occupancy | registers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separate: primal | 537 | 311 | 4.8 | 606 | 91.8% | 29.0% | 34.6% | 84.5% / 100% | 26 |
| separate: dual | 1461 | 811 | 12.6 | 582 | 87.9% | 43.5% | 56.0% | 86.7% / 100% | 30 |
| fused | 1543 | 821 | 12.7 | 558 | 84.3% | 47.3% | 68.2% | 70.8% / 83.3% | 48 |
| fused, 16x8x4 tiles | 1583 | 825 | 12.8 | 547 | 82.7% | 5.1% | 64.8% | 96.5% / 100% | 40 |
| fused, 32x8 marching | 1568 | 824 | 12.8 | 551 | 83.2% | 6.8% | 51.0% | 82.4% / 83.3% | 47 |

What this shows:

- The traffic model holds. The separate step moves 4.8 + 12.6 = 17.4 volumes (model 5 + 13
  = 18), the fused step 12.7 (model 13). The small deficit is L2 keeping part of a volume
  between kernels.
- Neighbour reuse already hits the caches: the fused kernel reads r at the voxel and at three
  lower neighbours and recomputes z at four points, yet moves no more than the 13 volumes it
  must touch once. That was the hypothesis for why tiling cannot help, and the counters
  confirm it: both tiled kernels move the same bytes (12.8 volumes) as the plain fused
  kernel. Their L1 hit rate drops to 5-7% because the reuse moved into shared memory, which
  the L1 counters do not include; DRAM traffic does not change.
- All variants run at 83-92% of the DRAM peak. With no traffic left to remove, the time per
  step follows the bytes: the fused step is cheaper than the separate pair because it moves
  27% fewer bytes, not because it is faster per byte (558 against 582-606 GB/s).
- Occupancy is not the limit: the fused kernel's 48 registers cap it at 5 blocks of 256
  threads per SM (83% theoretical, 71% achieved), the cube-tiled one reaches 97%, and it is
  not faster.

The 512³ profile was not done. Nsight Compute's kernel replay saves and restores the device
buffers in host memory, and the 512³ TV run (about 7 GiB of device buffers) ran the 32 GB
host out of memory, so the run was stopped. The 512³ times above are cudaEvent intervals.

The model cross-check: by the traffic model the fused kernel moves 13 n³ floats per step; at 256³ and 512³ that is
539 and 554 GB/s, against 567 and 575 GB/s for the 18-volume separate version
and 672 GB/s in the card's specification. The profiler numbers above
confirm that the step is limited by DRAM traffic at its compulsory minimum; the remaining ways to
make it cheaper are to move fewer bytes, for example the dual in half
precision or fewer dual steps per outer iteration, both of which change the result.
