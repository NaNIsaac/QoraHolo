# TV solver: the FGP step on the GPU

Each outer FISTA iteration of the TV solver (see ARCHITECTURE.md) runs `tv_inner` steps of
the fast gradient projection on the TV dual. At the default 20 x 10 iterations that is 200
FGP steps against 40 3D FFTs, so the FGP step sets the cost of the solver. This note records
how the step was made faster and what did not help. All times are cudaEvent intervals on the
RTX 5070, measured with `qoraholo_bench --field N --tv 1` while holding the shared GPU
benchmark lock, with no other compute process on the GPU (`nvidia-smi`, sampled every 5 s).
Nsight Compute is not available on this machine (`ERR_NVGPUCTRPERM`), so there are no
profiler counters; the bandwidth column is a model.

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

## Result at 256³

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

By the traffic model the fused kernel moves 13 n³ floats per step; at 256³ and 512³ that is
539 and 554 GB/s, against 567 and 575 GB/s for the 18-volume separate version
and 672 GB/s in the card's specification. That the fused step runs at nearly the same rate
as the separate kernels suggests it is still limited by memory traffic; the remaining ways to
make it cheaper are to move fewer bytes, for example the dual in half
precision or fewer dual steps per outer iteration, both of which change the result.
