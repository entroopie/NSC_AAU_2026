## Submission checklist

- [x] Does the submission include docstrings for 2 or more functions, with full explanation of purpose and input/output variables?
  - Evidence: [mini_project_1/utils/functions.py](mini_project_1/utils/functions.py)
- [x] Does the code include unit testing based on Doctest, unittest, or py.test, for at least 3 test cases?
  - Evidence: [mini_project_1/test_mandelbrot_unittest.py](mini_project_1/test_mandelbrot_unittest.py) (4 tests)
- [x] Does the submission include a working CUDA/Numba implementation of the Mandelbrot algorithm?
  - Evidence: [mini_project_1/cuda_numba_benchmark.py](mini_project_1/cuda_numba_benchmark.py)
- [x] Does the CUDA implementation correctly use 2D grid/block configuration, including out-of-bounds checks, and discuss the effect of block size on performance?
  - Evidence: kernel indexing + out-of-bounds guard in [mini_project_1/cuda_numba_benchmark.py](mini_project_1/cuda_numba_benchmark.py);
    block-size sweep outputs [benchmark_outputs/cuda_blocksize_scan.csv](benchmark_outputs/cuda_blocksize_scan.csv) and [benchmark_outputs/cuda_blocksize_scan.png](benchmark_outputs/cuda_blocksize_scan.png)
- [x] Do the code and worksheet include benchmarking and scaling results, comparing CUDA against previous implementations using consistent parameters and speedups?
  - Evidence: scaling dataset [benchmark_outputs/scaling_analysis_timings.csv](benchmark_outputs/scaling_analysis_timings.csv) and scaling plots
    [benchmark_outputs/scaling_cuda_r3_fixed/scaling_times.png](benchmark_outputs/scaling_cuda_r3_fixed/scaling_times.png),
    [benchmark_outputs/scaling_cuda_large_r1/scaling_times.png](benchmark_outputs/scaling_cuda_large_r1/scaling_times.png)
- [x] Does the worksheet discuss important CUDA-specific performance aspects, such as CPU-GPU data transfers, correct timing/synchronization, memory-type choices, and/or warp divergence?
  - Evidence: [mini_project_1/cuda_notes.md](mini_project_1/cuda_notes.md)
- [x] Does the submission include any extra features worth bonus points, such as shared-memory reduction, profiling, improved visualization, or comparison across devices? (If yes, please briefly describe them in comments)
  - Yes (improved visualization): interactive Mandelbrot zoom explorer that auto-seeks a high-variance boundary sub-region when zooming.
    Implemented in [mini_project_1/utils/interactive_plot.py](mini_project_1/utils/interactive_plot.py) and used from [mini_project_1/mandelbrot_test.py](mini_project_1/mandelbrot_test.py).

# Mandelbrot Set – CPU vs CUDA Performance & Scaling Report

## 1. The Problem

The Mandelbrot set is a classic fractal defined in the complex plane. For each point $c = x + iy$, we apply the iteration

$$z_{n+1} = z_n^2 + c, \quad z_0 = 0$$

and say that a point belongs to the set if $|z|$ never exceeds 2. In practice we cap the iteration count with `max_iter` and record the number of iterations before escape (or `max_iter` if it never escaped).

Mandelbrot is a good HPC benchmark because the work is embarrassingly parallel at the pixel level (each pixel is independent), while still having branch divergence (different pixels escape after different iteration counts).

**Common configuration used in the experiments:**

| Parameter | Value |
|---|---|
| Domain | x ∈ [−2.0, 1.0], y ∈ [−1.5, 1.5] |
| Max iterations | 256 |
| Primary scaling sizes | 128 – 4096 (repeat=3, median) |
| Large scaling sizes | 6144, 8192, 12288 (repeat=1) |
| CUDA thread block (2D) | 16x8 |
| CUDA coordinate dtype | float32 (performance) |


## 2. Implementations

### 2.1 Naive CPU (pure Python)

Two nested Python loops over pixels + per-pixel `while` loop. This is the most readable baseline but extremely slow due to interpreter overhead.

### 2.2 NumPy vectorized CPU

Builds a complex grid and iterates over the whole grid using NumPy array operations plus a boolean mask. This reduces Python overhead but uses large intermediate arrays (memory pressure) and still runs a Python loop over iterations.

Note: this implementation records the escape iteration in a slightly different convention (0-based `n` for escaped points), while the naive/numba implementations record 1-based `n`. Comparisons in the scaling analysis are adjusted to account for this.

### 2.3 Numba CPU (`@njit`)

Compiles a scalar nested-loop implementation to native code. This keeps the per-pixel `while` loop structure (early exit when escaped) but removes Python overhead. A small warm-up call is used so compilation time is excluded from timings.

### 2.4 Multiprocessing CPU

Splits the image into row-blocks (chunks) and computes each chunk with a vectorized block function in multiple processes. This can speed up large grids but has overhead (process startup/IPC and assembling blocks).

### 2.5 Dask local CPU

Uses `dask.array.map_blocks` with a block function that computes a chunk of rows. This provides a high-level parallel programming model but introduces scheduling overhead, especially for smaller problem sizes.

### 2.6 CUDA GPU (Numba `@cuda.jit`)

Implements a 2D CUDA kernel where each thread computes one pixel.

Important CUDA details (also discussed in the notes):

- Explicit grid and block configuration:
  `blocks_per_grid = ((H + tx - 1)//tx, (W + ty - 1)//ty)`
- Out-of-bounds guard in the kernel.
- No shared memory is needed for the core Mandelbrot kernel because pixels are independent.
- Kernel timing uses CUDA events (kernel-only time) and a separate end-to-end measurement that includes H2D/D2H transfers.


## 3. Benchmarking Methodology

### 3.1 Timing

- CPU methods are timed with `time.perf_counter()` and reported as the **median** over `repeat` runs.
- CUDA is timed in two ways:
  - **Kernel-only:** CUDA events (`start.record()`, `end.record()`, `end.synchronize()`)
  - **End-to-end:** host timer around H2D + kernel + D2H and `cuda.synchronize()`
- Warm-up runs are used to exclude JIT compilation time (Numba CPU and Numba CUDA).

### 3.2 Scaling experiment design

To demonstrate a clear GPU advantage and reach problem sizes where CPU becomes slow:

- **Small → mid sweep:** N = 128, 256, 512, 1024, 2048, 4096 with `repeat=3`.
  - Naive is included only up to N=512.
  - NumPy vectorized is included up to N=4096.
- **Large sweep:** N = 6144, 8192, 12288 with `repeat=1`.
  - Naive and full-grid NumPy vectorized are skipped.

### 3.3 Experimental data (deliverable)

All timing measurements used in the scaling analysis are stored in:

- [benchmark_outputs/scaling_analysis_timings.csv](benchmark_outputs/scaling_analysis_timings.csv)

This file contains all (method, size) timings across both the small/mid sweep and the large sweep.


## 4. Experimental Results

### 4.1 Scaling plots

- Small → mid scaling plot: [benchmark_outputs/scaling_cuda_r3_fixed/scaling_times.png](benchmark_outputs/scaling_cuda_r3_fixed/scaling_times.png)
- Large scaling plot: [benchmark_outputs/scaling_cuda_large_r1/scaling_times.png](benchmark_outputs/scaling_cuda_large_r1/scaling_times.png)

### 4.2 Representative timings and speedups

All numbers below come from the scaling CSVs (medians where applicable).

| N (NxN) | cpu_numba (s) | cpu_numpy_vectorized (s) | cpu_multiprocessing_p8 (s) | cpu_dask_processes_w4 (s) | cuda_kernel_float32_16x8 (s) | cuda_e2e_float32_16x8 (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.139315 | 0.684437 | 0.614072 | 0.955336 | 0.000926 | 0.001612 |
| 4096 | 2.275016 | 17.586730 | 1.796750 | 3.880975 | 0.010801 | 0.016401 |
| 12288 | 20.354264 | – | 14.138216 | 26.164792 | 0.094814 | 0.153619 |

Speedups vs `cpu_numba` at N=12288:

- CUDA kernel-only: 20.354264 / 0.094814 ≈ 214.68×
- CUDA end-to-end: 20.354264 / 0.153619 ≈ 132.50×
- cpu_multiprocessing_p8: 20.354264 / 14.138216 ≈ 1.44×
- cpu_dask_processes_w4: 20.354264 / 26.164792 ≈ 0.78×


## 5. Reasoning and Interpretation

### 5.1 Why CUDA is so fast here

Mandelbrot is embarrassingly parallel at the pixel level: each thread performs scalar arithmetic and writes one output value. The GPU can schedule many thousands of threads, hiding latency and delivering very high throughput.

### 5.2 Kernel-only vs end-to-end

For CUDA, the difference between kernel-only and end-to-end time is the host-device transfer overhead (PCIe + driver overhead). Reporting both avoids an unfair comparison between “GPU compute only” and “CPU full pipeline”.

### 5.3 Why multiprocessing/Dask help only at large N

For small N, multiprocessing and Dask are dominated by overhead (process management and scheduling). As N grows, compute starts to dominate and multiprocessing can overtake single-process CPU Numba (e.g., at N=4096 and above in the measurements).

### 5.4 Divergence

Near the Mandelbrot boundary, different pixels escape at different iteration counts, so warps contain threads that execute different loop trip counts (warp divergence). This reduces GPU efficiency relative to an ideal uniform workload, but is inherent to the algorithm.

### 5.5 Correctness and floating point precision

- CPU implementations largely agree. Tiny mismatch ratios can occur at large sizes (on the order of $10^{-6}$ vs the CPU Numba reference), which mainly affects boundary pixels.
- CUDA uses float32 coordinates for performance. In the scaling runs, CUDA float32 shows a small but consistent mismatch ratio vs the float64 CPU Numba reference of about 0.12–0.13% (e.g., 0.00124 at N=1024 and 0.00128 at N=4096/12288). Switching CUDA coordinate arrays to float64 eliminates mismatches but may reduce GPU performance.


## 6. Conclusion

- CPU Numba is a strong baseline and scales predictably with $N^2$.
- Multiprocessing can help at larger N but has overhead that makes it worse than single-process Numba at smaller N.
- Dask (local) adds additional scheduling overhead and only becomes competitive for sufficiently large workloads and careful chunking.
- CUDA provides the clearest advantage: at N=12288, CUDA is >100× faster end-to-end than CPU Numba in the measured setup.
