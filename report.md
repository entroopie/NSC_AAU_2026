# Mandelbrot Set – Performance Scaling Report

---

## 1. The Problem

The Mandelbrot set is a classic fractal defined in the complex plane. For each point
`c = x + iy`, we repeatedly apply the iteration:

```
z_(n+1) = z_n² + c,   z_0 = 0
```

A point belongs to the Mandelbrot set if `|z|` never exceeds 2, no matter how many
iterations we run. In practice we set a maximum iteration count (`max_iter`). If the
point has not escaped after `max_iter` steps, we treat it as part of the set.

Computing the Mandelbrot set for a full image grid is computationally expensive:
every pixel is an independent, iterative calculation. This makes it a good benchmark
for comparing different Python computing strategies.

**Configuration used in all experiments:**

| Parameter | Value |
|-----------|-------|
| Domain | x ∈ [−2.0, 1.0], y ∈ [−1.5, 1.5] |
| Max iterations | 256 |
| Grid sizes tested | 64 – 4096 (powers of two) |

---

## 2. Implementations

Three methods were implemented, each representing a common pattern in numerical Python.

### 2.1 Naïve (Pure Python loops)

```python
for i in range(height):
    for j in range(width):
        c = complex(x[j], y[i])
        z = 0
        n = 0
        while n < max_iter and abs(z) <= 2:
            z = z ** 2 + c
            n += 1
        mandelbrot_set[i, j] = n
```

Two nested Python `for` loops iterate over every pixel. The inner iteration runs
as pure Python bytecode. This is the simplest and most readable version, but the
slowest: every arithmetic operation goes through Python's interpreter overhead.

The escape condition uses `abs(z) <= 2`, which computes a square root. This is
acceptable here because Python's built-in `abs()` on a complex scalar calls a
single C function and is actually faster than spelling out
`z.real**2 + z.imag**2 <= 4` in pure Python bytecode.

### 2.2 Vectorized (NumPy)

```python
C = X + Y * 1j                          # complex grid, all at once
Z = np.zeros(C.shape, dtype=complex)
mask = np.ones(C.shape, dtype=bool)

for n in range(max_iter):
    Z[mask] = Z[mask] ** 2 + C[mask]
    escape = mask & (np.abs(Z) > 2)
    mandelbrot_set[escape] = n
    mask[escape] = False
```

Instead of looping pixel by pixel, the entire grid is stored as a NumPy array and
updated in one operation per iteration step. A boolean `mask` tracks which pixels
have not yet escaped, so escaped pixels are dropped from further computation.

The escape check uses `np.abs(Z) > 2`.
Over a large array `Z.real**2 + Z.imag**2 > 4` may give computing a square root for every element
a genuine speedup at the NumPy level.

The outer `for n in range(max_iter)` loop is still Python, but each pass processes
all active pixels at once using NumPy's C-compiled array operations.

### 2.3 Numba (JIT-compiled)

```python
@njit
def numba(xmin, xmax, ymin, ymax, height, width, max_iter):
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            z = complex(0, 0)
            n = 0
            while n < max_iter and (z.real**2 + z.imag**2) <= 4:
                z = z * z + c
                n += 1
            mandelbrot_set[i, j] = n
```

The `@njit` decorator from Numba compiles the function to native machine code the
first time it is called (just-in-time compilation). The code structure is identical
to the naïve version - nested loops, scalar arithmetic - but it runs at C-like speed.

Here the `z.real**2 + z.imag**2 <= 4` check *does* pay off: in compiled code,
avoiding a square root is a concrete gain. Similarly, `z * z` is used instead of
`z ** 2` to avoid the overhead of a general power function.

Because compilation happens on the first call, a **warm-up run** is performed before
any timing begins.

---

## 3. Benchmarking Methodology

- **Grid sizes:** 64, 128, 256, 512, 1024, 2048, 4096 (each an N×N grid).
  The naïve method is included at all sizes but becomes very slow at the largest.
- **Timing:** `time.perf_counter()` - wall-clock time with the highest resolution
  available on the system.
- **Single run per size:** one timed execution per (method, size) combination.
  The naïve method is deterministic and its runtimes are very stable, so a single
  measurement is representative.
- **Numba warm-up:** the JIT compiler is triggered on a tiny 32×32 grid before any
  measurements, so compilation time is excluded from the results.
- **Results saved:** all timings are written to `benchmark_results.csv` alongside
  derived throughput (million pixels per second).

---

## 4. Experimental Results

### 4.1 Raw timings (seconds)

| Grid size | Pixels | Naïve (s) | Vectorized (s) | Numba (s) |
|-----------|--------|-----------|----------------|-----------|
| 64×64 | 4 096 | 0.0327 | 0.0029 | 0.0004 |
| 128×128 | 16 384 | 0.1282 | 0.0082 | 0.0015 |
| 256×256 | 65 536 | 0.5175 | 0.0346 | 0.0062 |
| 512×512 | 262 144 | 2.0653 | 0.1262 | 0.0269 |
| 1024×1024 | 1 048 576 | 5.6283 | 0.7288 | 0.1001 |
| 2048×2048 | 4 194 304 | 25.8827 | 3.7981 | 0.3874 |
| 4096×4096 | 16 777 216 | 103.1887 | 16.7817 | 1.5300 |

### 4.2 Throughput (million pixels per second)

| Method | Typical throughput |
|--------|--------------------|
| Naïve | ~0.13 Mpx/s |
| Vectorized | ~1.0–2.1 Mpx/s |
| Numba | ~10–11 Mpx/s |

Numba achieves roughly **constant throughput** across all grid sizes (~10 Mpx/s),
which is a hallmark of a method that is compute-bound with no overhead growing
with size. Vectorized throughput drops slightly at larger sizes, likely due to
increased memory pressure as the full complex array no longer fits in cache.

### 4.3 Speedup over naïve

| Grid size | Vectorized speedup | Numba speedup |
|-----------|--------------------|---------------|
| 64×64 | 11× | 83× |
| 512×512 | 16× | 77× |
| 4096×4096 | 6× | 67× |

---

## 5. Interpretation

### Why is Naïve so slow?

Python executes each operation one at a time through its interpreter. Every
arithmetic step - `z ** 2`, `z + c`, `abs(z)` - involves Python object
allocation, type checking, and function dispatch. For a 4096×4096 grid with
up to 256 iterations per pixel, this adds up to billions of such overhead events.

### Why does Vectorized help?

NumPy moves the inner arithmetic into compiled C/Fortran code. A single call like
`Z[mask] ** 2 + C[mask]` processes millions of values in a tight C loop with no
Python overhead per element. The outer `for n in range(max_iter)` loop is still
Python, but it runs at most 256 times regardless of grid size - so its cost is
negligible.

The vectorized approach is still not ideal because it cannot exit early for a
pixel once it escapes. It always runs `max_iter` outer iterations (operating on a
shrinking set of active pixels). Numba's per-pixel `while` loop can stop as soon
as a point escapes.

### Why is Numba the fastest?

Numba compiles the exact same loop structure as the naïve version to native machine
code. The CPU can execute it with full SIMD and branch-prediction optimisation,
just like a hand-written C program. Additionally, Numba's per-pixel `while` loop
exits early when a point escapes, skipping all remaining iterations - something
the vectorized version cannot do per-pixel. The near-constant throughput across
all grid sizes shows that Numba is not bottlenecked by memory or Python overhead;
it is purely compute-limited.

### Scaling behaviour

All three methods scale approximately as **O(N²)** - doubling the grid size
quadruples the number of pixels and therefore roughly quadruples the runtime.
This is visible in the log-log plot as straight, parallel lines. The difference
between methods is a constant factor (the slope is the same; the intercept differs),
not a difference in algorithmic complexity.

---

## 6. Conclusion

| Method | Best for | Drawback |
|--------|----------|----------|
| Naïve | Clarity, teaching | Very slow |
| Vectorized | No extra dependencies | Memory pressure at large sizes |
| Numba | Production speed | Requires compilation warm-up |

For interactive or production use, Numba is the clear winner - roughly **70× faster**
than naïve and **10× faster** than vectorized at typical grid sizes, with no change
to the algorithm's logic. Vectorized NumPy is a good middle ground when Numba is not
available or the grid is small.
