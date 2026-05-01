import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

from config import Config
from utils.functions import naive, vectorized, numba
from parallel_dask_benchmark import mandelbrot_multiprocessing, mandelbrot_dask
from cuda_numba_benchmark import mandelbrot_cuda_kernel


def vec_like_mismatch_ratio_vs_numba(vec_like: np.ndarray, numba_ref: np.ndarray, max_iter: int) -> float:
    """Mismatch ratio vs numba/naive reference for a vectorized-style output.

    `vectorized`/`vectorized_block` store escaped points as `n` (0-based), while
    `naive`/`numba` store escaped points as `n+1` (1-based). Non-escaped points are
    `max_iter` in both.
    """

    if vec_like.shape != numba_ref.shape:
        return 1.0

    escaped = vec_like < max_iter
    mism = 0
    if np.any(escaped):
        mism += int(np.count_nonzero(numba_ref[escaped] != (vec_like[escaped] + 1)))
    if np.any(~escaped):
        mism += int(np.count_nonzero(numba_ref[~escaped] != vec_like[~escaped]))
    return float(mism / vec_like.size)


def direct_mismatch_ratio(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 1.0
    return float(np.count_nonzero(a != b) / a.size)


def parse_sizes(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_block(s: str):
    s = s.strip().lower()
    tx, ty = s.split("x")
    return int(tx), int(ty)


def time_median(fn, repeat: int):
    vals = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals)), out


def cuda_time(bounds, size, max_iter, block, repeat, coord_dtype):
    xmin, xmax, ymin, ymax = bounds

    x = np.linspace(xmin, xmax, size, dtype=coord_dtype)
    y = np.linspace(ymin, ymax, size, dtype=coord_dtype)

    threads = block
    blocks = (
        (size + threads[0] - 1) // threads[0],
        (size + threads[1] - 1) // threads[1],
    )

    # warmup / compile
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_out = cuda.device_array((size, size), dtype=np.int32)
    mandelbrot_cuda_kernel[blocks, threads](d_x, d_y, max_iter, d_out)
    cuda.synchronize()

    # kernel-only timings
    kernel_s = []
    for _ in range(repeat):
        start = cuda.event(timing=True)
        end = cuda.event(timing=True)
        start.record()
        mandelbrot_cuda_kernel[blocks, threads](d_x, d_y, max_iter, d_out)
        end.record()
        end.synchronize()
        kernel_s.append(cuda.event_elapsed_time(start, end) / 1000.0)

    # end-to-end timings
    e2e_s = []
    out_host = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        d_x_rt = cuda.to_device(x)
        d_y_rt = cuda.to_device(y)
        d_out_rt = cuda.device_array((size, size), dtype=np.int32)
        mandelbrot_cuda_kernel[blocks, threads](d_x_rt, d_y_rt, max_iter, d_out_rt)
        cuda.synchronize()
        out_host = d_out_rt.copy_to_host()
        e2e_s.append(time.perf_counter() - t0)

    return float(np.median(kernel_s)), float(np.median(e2e_s)), out_host


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_scaling(csv_path: Path, out_png: Path):
    # read rows
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    # group by method
    methods = {}
    for r in rows:
        m = r["method"]
        methods.setdefault(m, {"size": [], "time_s": []})
        if r["time_s"] == "":
            continue
        methods[m]["size"].append(int(r["size"]))
        methods[m]["time_s"].append(float(r["time_s"]))

    plt.figure(figsize=(9.5, 6.0))
    for m, d in methods.items():
        if not d["size"]:
            continue
        xs = np.array(d["size"], dtype=np.int32)
        ys = np.array(d["time_s"], dtype=np.float64)
        order = np.argsort(xs)
        plt.loglog(xs[order], ys[order], marker="o", linewidth=2, label=m)

    plt.xlabel("Grid size N (NxN pixels)")
    plt.ylabel("Median time (s)")
    plt.title("Mandelbrot scaling: CPU methods vs CUDA")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)


def main():
    p = argparse.ArgumentParser(description="Scaling analysis: CPU implementations vs CUDA")
    p.add_argument("--sizes", type=str, default="128,256,512,1024,2048,4096")
    p.add_argument("--max-iter", type=int, default=256)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="../benchmark_outputs/scaling_cuda")

    p.add_argument("--mp-processes", type=int, default=8)
    p.add_argument("--mp-chunk-rows", type=int, default=32)

    p.add_argument("--dask-workers", type=int, default=4)
    p.add_argument("--dask-chunk-rows", type=int, default=32)
    p.add_argument("--dask-scheduler", type=str, default="processes", choices=["threads", "processes"])

    p.add_argument("--cuda-block", type=str, default="16x8")
    p.add_argument("--cuda-coord-dtype", type=str, default="float32", choices=["float32", "float64"])

    p.add_argument("--naive-max-size", type=int, default=512)
    p.add_argument("--skip-vectorized", action="store_true")

    args = p.parse_args()

    if not cuda.is_available():
        raise RuntimeError("CUDA not available for Numba in this environment")

    cfg = Config()
    bounds = (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax)
    xmin, xmax, ymin, ymax = bounds

    sizes = parse_sizes(args.sizes)
    block = parse_block(args.cuda_block)
    coord_dtype = np.float32 if args.cuda_coord_dtype == "float32" else np.float64

    # warmup numba cpu
    numba(xmin, xmax, ymin, ymax, 32, 32, args.max_iter)

    out_dir = Path(args.out_dir)
    out_csv = out_dir / "scaling_times.csv"
    out_png = out_dir / "scaling_times.png"

    rows = []

    for n in sizes:
        # CPU baselines
        cpu_numba_t, cpu_numba_out = time_median(
            lambda n=n: numba(xmin, xmax, ymin, ymax, n, n, args.max_iter),
            repeat=args.repeat,
        )
        rows.append(
            {
                "method": "cpu_numba",
                "size": n,
                "max_iter": args.max_iter,
                "repeat": args.repeat,
                "time_s": round(cpu_numba_t, 6),
                "correct_vs_cpu_numba": True,
                "mismatch_ratio_vs_cpu_numba": 0.0,
                "note": "",
            }
        )

        if n <= args.naive_max_size:
            naive_t, naive_out = time_median(
                lambda n=n: naive(xmin, xmax, ymin, ymax, n, n, args.max_iter),
                repeat=max(1, min(args.repeat, 2)),
            )
            rows.append(
                {
                    "method": "cpu_naive",
                    "size": n,
                    "max_iter": args.max_iter,
                    "repeat": max(1, min(args.repeat, 2)),
                    "time_s": round(naive_t, 6),
                    "correct_vs_cpu_numba": bool(np.array_equal(naive_out, cpu_numba_out)),
                    "mismatch_ratio_vs_cpu_numba": round(direct_mismatch_ratio(naive_out, cpu_numba_out), 8),
                    "note": "",
                }
            )

        if not args.skip_vectorized:
            try:
                vec_t, vec_out = time_median(
                    lambda n=n: vectorized(xmin, xmax, ymin, ymax, n, n, args.max_iter),
                    repeat=args.repeat,
                )
                vec_mr = vec_like_mismatch_ratio_vs_numba(vec_out, cpu_numba_out, args.max_iter)
                rows.append(
                    {
                        "method": "cpu_numpy_vectorized",
                        "size": n,
                        "max_iter": args.max_iter,
                        "repeat": args.repeat,
                        "time_s": round(vec_t, 6),
                        "correct_vs_cpu_numba": vec_mr == 0.0,
                        "mismatch_ratio_vs_cpu_numba": round(vec_mr, 8),
                        "note": "escape-index convention adjusted",
                    }
                )
            except MemoryError:
                rows.append(
                    {
                        "method": "cpu_numpy_vectorized",
                        "size": n,
                        "max_iter": args.max_iter,
                        "repeat": args.repeat,
                        "time_s": "",
                        "correct_vs_cpu_numba": "",
                        "mismatch_ratio_vs_cpu_numba": "",
                        "note": "MemoryError",
                    }
                )

        # Multiprocessing
        mp_t, mp_out = time_median(
            lambda n=n: mandelbrot_multiprocessing(
                n,
                args.max_iter,
                bounds,
                processes=args.mp_processes,
                chunk_rows=args.mp_chunk_rows,
            ),
            repeat=args.repeat,
        )
        mp_mr = vec_like_mismatch_ratio_vs_numba(mp_out, cpu_numba_out, args.max_iter)
        rows.append(
            {
                "method": f"cpu_multiprocessing_p{args.mp_processes}",
                "size": n,
                "max_iter": args.max_iter,
                "repeat": args.repeat,
                "time_s": round(mp_t, 6),
                "correct_vs_cpu_numba": mp_mr == 0.0,
                "mismatch_ratio_vs_cpu_numba": round(mp_mr, 8),
                "note": f"chunk_rows={args.mp_chunk_rows} (escape-index convention adjusted)",
            }
        )

        # Dask local
        dask_t, dask_out = time_median(
            lambda n=n: mandelbrot_dask(
                n,
                args.max_iter,
                bounds,
                chunk_rows=args.dask_chunk_rows,
                scheduler=args.dask_scheduler,
                num_workers=args.dask_workers,
            ),
            repeat=args.repeat,
        )
        dask_mr = vec_like_mismatch_ratio_vs_numba(dask_out, cpu_numba_out, args.max_iter)
        rows.append(
            {
                "method": f"cpu_dask_{args.dask_scheduler}_w{args.dask_workers}",
                "size": n,
                "max_iter": args.max_iter,
                "repeat": args.repeat,
                "time_s": round(dask_t, 6),
                "correct_vs_cpu_numba": dask_mr == 0.0,
                "mismatch_ratio_vs_cpu_numba": round(dask_mr, 8),
                "note": f"chunk_rows={args.dask_chunk_rows} (escape-index convention adjusted)",
            }
        )

        # CUDA
        cuda_kernel_t, cuda_e2e_t, cuda_out = cuda_time(
            bounds,
            n,
            args.max_iter,
            block,
            repeat=args.repeat,
            coord_dtype=coord_dtype,
        )
        cuda_mr = direct_mismatch_ratio(cuda_out, cpu_numba_out)
        rows.append(
            {
                "method": f"cuda_kernel_{args.cuda_coord_dtype}_{block[0]}x{block[1]}",
                "size": n,
                "max_iter": args.max_iter,
                "repeat": args.repeat,
                "time_s": round(cuda_kernel_t, 6),
                "correct_vs_cpu_numba": bool(np.array_equal(cuda_out, cpu_numba_out)),
                "mismatch_ratio_vs_cpu_numba": round(cuda_mr, 8),
                "note": "kernel_only",
            }
        )
        rows.append(
            {
                "method": f"cuda_e2e_{args.cuda_coord_dtype}_{block[0]}x{block[1]}",
                "size": n,
                "max_iter": args.max_iter,
                "repeat": args.repeat,
                "time_s": round(cuda_e2e_t, 6),
                "correct_vs_cpu_numba": bool(np.array_equal(cuda_out, cpu_numba_out)),
                "mismatch_ratio_vs_cpu_numba": round(cuda_mr, 8),
                "note": "includes H2D+D2H",
            }
        )

        print(
            f"N={n}: cpu_numba={cpu_numba_t:.4f}s mp={mp_t:.4f}s dask={dask_t:.4f}s "
            f"cuda_e2e={cuda_e2e_t:.6f}s cuda_kernel={cuda_kernel_t:.6f}s"
        )

    write_csv(out_csv, rows)
    plot_scaling(out_csv, out_png)

    print(f"CSV saved: {out_csv.resolve()}")
    print(f"Plot saved: {out_png.resolve()}")


if __name__ == "__main__":
    main()
