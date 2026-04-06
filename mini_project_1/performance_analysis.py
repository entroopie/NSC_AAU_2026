import csv
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from parallel_dask_benchmark import mandelbrot_multiprocessing, mandelbrot_dask
from utils.functions import naive, vectorized, numba


METHODS = {
    "naive": "#e05252",
    "vectorized": "#5b8dd9",
    "numba": "#57b26a",
    "multiprocessing": "#f0a202",
    "dask_local": "#8d6fd1",
}


def warmup(config):
    print("Warming up Numba JIT...")
    numba(config.xmin, config.xmax, config.ymin, config.ymax, 32, 32, config.max_iter)
    print("Done.\n")


def _timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run_benchmark(config):
    bounds = (config.xmin, config.xmax, config.ymin, config.ymax)
    results = {
        "naive": {"sizes": [], "times": []},
        "vectorized": {"sizes": [], "times": []},
        "numba": {"sizes": [], "times": []},
        "multiprocessing": {"sizes": [], "times": []},
        "dask_local": {"sizes": [], "times": []},
    }

    # Naive (limited sizes)
    print("[naive]")
    for size in config.SIZES_NAIVE:
        elapsed = _timeit(lambda: naive(config.xmin, config.xmax, config.ymin, config.ymax, size, size, config.max_iter))
        results["naive"]["sizes"].append(size)
        results["naive"]["times"].append(elapsed)
        print(f"  {size} x {size} ---> {elapsed:.4f} s")
    print()

    # Vectorized / Numba / MP / Dask local on fast sizes
    for size in config.SIZES_FAST:
        tv = _timeit(lambda size=size: vectorized(config.xmin, config.xmax, config.ymin, config.ymax, size, size, config.max_iter))
        tn = _timeit(lambda size=size: numba(config.xmin, config.xmax, config.ymin, config.ymax, size, size, config.max_iter))
        tm = _timeit(
            lambda size=size: mandelbrot_multiprocessing(
                size,
                config.max_iter,
                bounds,
                processes=config.MP_PROCESSES,
                chunk_rows=config.MP_CHUNK_ROWS,
            )
        )
        td = _timeit(
            lambda size=size: mandelbrot_dask(
                size,
                config.max_iter,
                bounds,
                chunk_rows=config.DASK_CHUNK_ROWS,
                scheduler="processes",
                num_workers=config.DASK_LOCAL_WORKERS,
            )
        )

        results["vectorized"]["sizes"].append(size)
        results["vectorized"]["times"].append(tv)
        results["numba"]["sizes"].append(size)
        results["numba"]["times"].append(tn)
        results["multiprocessing"]["sizes"].append(size)
        results["multiprocessing"]["times"].append(tm)
        results["dask_local"]["sizes"].append(size)
        results["dask_local"]["times"].append(td)

        print(f"[{size} x {size}] vec={tv:.4f}s numba={tn:.4f}s mp={tm:.4f}s dask={td:.4f}s")
    print()

    return results


def save_csv(results, config, path="benchmark_results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Mandelbrot scaling benchmark"])
        writer.writerow([f"# date: {datetime.date.today()}"])
        writer.writerow([f"# domain: xmin={config.xmin} xmax={config.xmax} ymin={config.ymin} ymax={config.ymax}"])
        writer.writerow([f"# max_iter: {config.max_iter}"])
        writer.writerow([f"# multiprocessing: P={config.MP_PROCESSES}, chunk_rows={config.MP_CHUNK_ROWS}"])
        writer.writerow([f"# dask_local: workers={config.DASK_LOCAL_WORKERS}, chunk_rows={config.DASK_CHUNK_ROWS}"])
        writer.writerow(["method", "grid_size_N", "total_pixels", "time_s", "mpx_per_s"])
        for name, data in results.items():
            for sz, t in zip(data["sizes"], data["times"]):
                mpps = round(sz * sz / t / 1e6, 4)
                writer.writerow([name, sz, sz * sz, round(t, 6), mpps])
    print(f"Benchmark data saved --> {path}")


def plot_scaling(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Mandelbrot – Performance Scaling (All Methods)", fontsize=13, fontweight="bold")

    # left: wall-clock time vs grid size (log-log)
    ax = axes[0]
    for name, color in METHODS.items():
        sizes = results[name]["sizes"]
        times = results[name]["times"]
        if not sizes:
            continue
        ax.loglog(sizes, times, marker="o", color=color, linewidth=2, markersize=5, label=name)
    ax.set_xlabel("Grid size N x N")
    ax.set_ylabel("Time (s)")
    ax.set_title("Time vs. Grid size")
    ax.legend()
    all_sizes = sorted(set(sz for data in results.values() for sz in data["sizes"]))
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([str(s) for s in all_sizes], rotation=30, ha="right")
    ax.grid(True, which="both", alpha=0.3)

    # right: speedup over naive (where naive exists)
    ax = axes[1]
    naive_lookup = dict(zip(results["naive"]["sizes"], results["naive"]["times"]))
    naive_sizes = set(results["naive"]["sizes"])

    for name, color in METHODS.items():
        if name == "naive":
            continue
        common = [(sz, t) for sz, t in zip(results[name]["sizes"], results[name]["times"]) if sz in naive_sizes]
        if not common:
            continue
        szs, tms = zip(*common)
        speedup = [naive_lookup[sz] / t for sz, t in zip(szs, tms)]
        ax.plot(szs, speedup, marker="o", color=color, linewidth=2, markersize=5, label=name)
        ax.annotate(f" x{speedup[-1]:.1f}", xy=(szs[-1], speedup[-1]), fontsize=8, color=color, va="center")

    ax.set_xlabel("Grid size N x N")
    ax.set_ylabel("Speedup over naive")
    ax.set_title("Speedup relative to naive")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    naive_sizes_sorted = sorted(naive_sizes)
    ax.set_xticks(naive_sizes_sorted)
    ax.set_xticklabels([str(s) for s in naive_sizes_sorted], rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig("performance_scaling.png", dpi=150, bbox_inches="tight")
    print("Plot saved --> performance_scaling.png")
    plt.show()


if __name__ == "__main__":
    config = Config()
    warmup(config)
    results = run_benchmark(config)
    save_csv(results, config)
    plot_scaling(results)
