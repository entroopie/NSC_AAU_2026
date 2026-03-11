import csv
import datetime
import time

import numpy as np
import matplotlib.pyplot as plt

from config import Config
from utils.functions import naive, vectorized, numba


METHODS = {
    "naive":      (naive,      "#e05252"),
    "vectorized": (vectorized, "#5b8dd9"),
    "numba":      (numba,      "#57b26a"),
}


def warmup(config):
    print("Warming up Numba JIT...")
    numba(config.xmin, config.xmax, config.ymin, config.ymax, 32, 32, config.max_iter)
    print("Done.\n")


def run_benchmark(config):
    results = {}
    sizes = {"naive": config.SIZES_NAIVE, "vectorized": config.SIZES_FAST, "numba": config.SIZES_FAST}

    for name, (func, _) in METHODS.items():
        print(f"[{name}]")
        times = []
        for size in sizes[name]:
            t0 = time.perf_counter()
            func(config.xmin, config.xmax, config.ymin, config.ymax, size, size, config.max_iter)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  {size} x {size} ---> {elapsed:.4f} s")
        results[name] = {"sizes": sizes[name], "times": times}
        print()

    return results


def save_csv(results, config, path="benchmark_results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Mandelbrot scaling benchmark"])
        writer.writerow([f"# date: {datetime.date.today()}"])
        writer.writerow([f"# domain: xmin={config.xmin} xmax={config.xmax} ymin={config.ymin} ymax={config.ymax}"])
        writer.writerow([f"# max_iter: {config.max_iter}"])
        writer.writerow(["method", "grid_size_N", "total_pixels", "time_s", "mpx_per_s"])
        for name, data in results.items():
            for sz, t in zip(data["sizes"], data["times"]):
                mpps = round(sz * sz / t / 1e6, 4)
                writer.writerow([name, sz, sz * sz, round(t, 6), mpps])
    print(f"Benchmark data saved --> {path}")


def plot_scaling(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Mandelbrot – Performance Scaling", fontsize=13, fontweight="bold")

    # left: wall-clock time vs grid size (log-log)
    ax = axes[0]
    for name, (_, color) in METHODS.items():
        sizes = results[name]["sizes"]
        times = results[name]["times"]
        ax.loglog(sizes, times, marker="o", color=color, linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Grid size N x N")
    ax.set_ylabel("Time (s)")
    ax.set_title("Time vs. Grid size")
    ax.legend()
    all_sizes = sorted(set(sz for data in results.values() for sz in data["sizes"]))
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([str(s) for s in all_sizes], rotation=30, ha="right")
    ax.grid(True, which="both", alpha=0.3)

    # right: speedup over naive
    ax = axes[1]
    naive_lookup = dict(zip(results["naive"]["sizes"], results["naive"]["times"]))
    naive_sizes  = set(results["naive"]["sizes"])
    for name, (_, color) in METHODS.items():
        if name == "naive":
            continue
        common = [(sz, t) for sz, t in zip(results[name]["sizes"], results[name]["times"])
                  if sz in naive_sizes]
        szs, tms = zip(*common)
        speedup = [naive_lookup[sz] / t for sz, t in zip(szs, tms)]
        ax.plot(szs, speedup, marker="o", color=color, linewidth=2, markersize=6, label=name)
        ax.annotate(f" ×{speedup[-1]:.0f}", xy=(szs[-1], speedup[-1]),
                    fontsize=9, color=color, va="center")
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
