# CUDA (@cuda.jit) analysis notes for worksheet

## 1) Optimal block size and warp-size-multiple rule

In CUDA, a warp has 32 threads. Block sizes that are multiples of 32 total threads
(e.g., 128, 256, 512) typically give better occupancy and less scheduling waste.
In this assignment, we test multiple 2D block shapes (`8x8`, `16x8`, `16x16`, `32x8`, etc.)
and compare kernel execution times.

Rule of thumb:
- Prefer block sizes where `threads_per_block % 32 == 0`
- Keep `threads_per_block <= 1024`
- Then find the best one experimentally (memory behavior and divergence also matter)

## 2) Why no shared memory is needed for core Mandelbrot

Mandelbrot is embarrassingly parallel: each pixel is independent and only needs its
own `(x, y)` values and local `z` iteration state. No thread needs intermediate results
from another thread, so shared memory and `cuda.syncthreads()` are not required for the
core kernel.

Where shared memory *would* help:
- Reduction kernels (e.g., compute mean/variance of iteration counts)
- Supersampling or neighborhood post-processing where nearby pixels are reused

## 3) Transfer time vs kernel time

`cuda.to_device()` and `.copy_to_host()` add PCIe overhead. We therefore report:
- **kernel-only time** (CUDA events)
- **end-to-end time** (H2D + kernel + D2H)

This avoids unfair comparisons between pure GPU compute time and full CPU pipelines.

## 4) Asynchronous kernel launches

Kernel launches are asynchronous. Timing must use either:
- CUDA events (`start.record()`, `end.record()`, `end.synchronize()`), or
- host timer + `cuda.synchronize()` before stop.

Also, the first launch includes JIT compilation, so do a warm-up run before timing.

## 5) Warp divergence comment

Near the Mandelbrot boundary, threads in the same warp run different iteration counts:
some escape quickly, some keep iterating. This causes warp divergence and reduces SIMD
efficiency. This is expected behavior for Mandelbrot, not a bug.

## 6) Float32 vs Float64 precision tradeoff

Mandelbrot is numerically sensitive near the set boundary. Small coordinate rounding
differences can change the escape iteration count for some pixels.

Implication for this project:
- Using `float32` for `x` and `y` on GPU is usually faster and uses less memory, but can produce small differences vs a CPU baseline that uses `float64`.
- Using `float64` improves agreement with a `float64` CPU reference, but can reduce performance on many GPUs.

For correctness checks:
- If exact equality is required (`array_equal`), use matching precision on both CPU and GPU (typically `float64` on both).
- If the goal is performance-oriented `float32` GPU runs, report mismatch statistics (e.g., mismatch count/ratio) instead of only strict pass/fail.
