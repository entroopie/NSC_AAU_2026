import argparse
import csv
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from multiprocessing import get_context

from utils.functions import vectorized, vectorized_block


# MANDELBROT WRAPPERS
def mandelbrot_numpy(size: int, max_iter: int, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    return vectorized(xmin, xmax, ymin, ymax, size, size, max_iter)


# MULTIPROCESSING
_G_X = None
_G_MAX_ITER = None


def _mp_init(x: np.ndarray, max_iter: int):
    global _G_X, _G_MAX_ITER
    _G_X = x
    _G_MAX_ITER = max_iter


def _mp_worker(y_block: np.ndarray) -> np.ndarray:
    return vectorized_block(y_block, _G_X, _G_MAX_ITER)


def _chunk_rows(y: np.ndarray, chunk_rows: int) -> List[np.ndarray]:
    return [y[i : i + chunk_rows] for i in range(0, len(y), chunk_rows)]


def mandelbrot_multiprocessing(
    size: int,
    max_iter: int,
    bounds: Tuple[float, float, float, float],
    processes: int,
    chunk_rows: int,
) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, size, dtype=np.float64)
    y = np.linspace(ymin, ymax, size, dtype=np.float64)
    chunks = _chunk_rows(y, chunk_rows)

    ctx = get_context("spawn")
    with ctx.Pool(processes=processes, initializer=_mp_init, initargs=(x, max_iter)) as pool:
        blocks = pool.map(_mp_worker, chunks)

    return np.vstack(blocks)


# DASK MAP BLOCKS
def _ensure_dask_imports():
    import dask.array as da
    from dask import config as dask_config
    from dask.distributed import Client

    return da, dask_config, Client


def mandelbrot_dask(
    size: int,
    max_iter: int,
    bounds: Tuple[float, float, float, float],
    chunk_rows: int,
    scheduler: str = "threads",
    num_workers: int = 4,
    scheduler_address: str = "",
) -> np.ndarray:
    da, dask_config, Client = _ensure_dask_imports()

    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, size, dtype=np.float64)

    y = da.linspace(ymin, ymax, size, chunks=chunk_rows, dtype=np.float64)

    chunks_2d = (y.chunks[0], (size,))

    def _block_func(y_block, x_arr=None, max_iter_local=None):
        # Keep this function self-contained so scheduler/workers do not need local project imports.
        C = x_arr[np.newaxis, :] + 1j * y_block[:, np.newaxis]
        Z = np.zeros(C.shape, dtype=np.complex128)
        out = np.zeros(C.shape, dtype=np.int32)
        mask = np.ones(C.shape, dtype=bool)
        for n in range(max_iter_local):
            Z[mask] = Z[mask] * Z[mask] + C[mask]
            escape = mask & (np.abs(Z) > 2)
            out[escape] = n
            mask[escape] = False
            if not mask.any():
                break
        out[mask] = max_iter_local
        return out

    arr = y.map_blocks(
        _block_func,
        x_arr=x,
        max_iter_local=max_iter,
        dtype=np.int32,
        new_axis=1,
        chunks=chunks_2d,
    )

    if scheduler == "distributed":
        if not scheduler_address:
            raise ValueError("scheduler='distributed' requires --scheduler-address")
        client = Client(scheduler_address)
        try:
            return arr.compute()
        finally:
            client.close()

    with dask_config.set(num_workers=num_workers):
        return arr.compute(scheduler=scheduler)

def time_call(fn, repeat: int = 1):
    vals = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals)), out


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    p = argparse.ArgumentParser(description="Mandelbrot multiprocessing + Dask benchmark")
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--max-iter", type=int, default=256)
    p.add_argument("--xmin", type=float, default=-2.0)
    p.add_argument("--xmax", type=float, default=1.0)
    p.add_argument("--ymin", type=float, default=-1.5)
    p.add_argument("--ymax", type=float, default=1.5)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--processes", type=str, default="1,2,4,8")
    p.add_argument("--chunks", type=str, default="16,32,64,128,256,512")
    p.add_argument("--dask-local-scheduler", type=str, default="processes", choices=["threads", "processes"])
    p.add_argument("--dask-local-workers", type=int, default=3)
    p.add_argument("--scheduler-address", type=str, default="")
    p.add_argument("--out-dir", type=str, default="benchmark_outputs")
    p.add_argument("--skip-dask", action="store_true")
    p.add_argument("--skip-cluster", action="store_true")
    args = p.parse_args()

    bounds = (args.xmin, args.xmax, args.ymin, args.ymax)
    processes_list = [int(x) for x in args.processes.split(",") if x.strip()]
    chunk_list = [int(x) for x in args.chunks.split(",") if x.strip()]

    out_dir = Path(args.out_dir)

    print(f"Running baseline NumPy vectorized: size={args.size}, max_iter={args.max_iter}")
    t_numpy, img_numpy = time_call(
        lambda: mandelbrot_numpy(args.size, args.max_iter, bounds),
        repeat=args.repeat,
    )
    print(f"NumPy time: {t_numpy:.4f}s")

    # --- Multiprocessing chunk sweep ---
    mp_rows = []
    best_per_p = {}

    print("\n[Multiprocessing chunk-size sweep]")
    for proc in processes_list:
        best_t = None
        best_chunk = None
        for chunk in chunk_list:
            t_mp, img_mp = time_call(
                lambda proc=proc, chunk=chunk: mandelbrot_multiprocessing(
                    args.size, args.max_iter, bounds, proc, chunk
                ),
                repeat=args.repeat,
            )
            ok = bool(np.array_equal(img_mp, img_numpy))
            speedup = t_numpy / t_mp
            mp_rows.append(
                {
                    "size": args.size,
                    "max_iter": args.max_iter,
                    "processes": proc,
                    "chunk_rows": chunk,
                    "time_s": round(t_mp, 6),
                    "speedup_vs_numpy": round(speedup, 4),
                    "correct_vs_numpy": ok,
                }
            )
            print(
                f"P={proc:>2}, chunk={chunk:>4} -> {t_mp:>8.4f}s, "
                f"speedup={speedup:>6.2f}x, correct={ok}"
            )
            if best_t is None or t_mp < best_t:
                best_t = t_mp
                best_chunk = chunk

        best_per_p[proc] = (best_chunk, best_t)

    write_csv(
        out_dir / "multiprocessing_chunk_scan.csv",
        mp_rows,
        ["size", "max_iter", "processes", "chunk_rows", "time_s", "speedup_vs_numpy", "correct_vs_numpy"],
    )

    mp_best_rows = []
    print("\n[Multiprocessing best chunk per process count]")
    for proc in processes_list:
        best_chunk, best_t = best_per_p[proc]
        speedup = t_numpy / best_t
        mp_best_rows.append(
            {
                "size": args.size,
                "max_iter": args.max_iter,
                "processes": proc,
                "best_chunk_rows": best_chunk,
                "best_time_s": round(best_t, 6),
                "speedup_vs_numpy": round(speedup, 4),
            }
        )
        print(f"P={proc:>2}: best_chunk={best_chunk:>4}, best_time={best_t:.4f}s, speedup={speedup:.2f}x")

    write_csv(
        out_dir / "multiprocessing_best_by_p.csv",
        mp_best_rows,
        ["size", "max_iter", "processes", "best_chunk_rows", "best_time_s", "speedup_vs_numpy"],
    )

    # --- Dask local ---
    if not args.skip_dask:
        dask_local_rows = []
        print("\n[Dask local chunk sweep via map_blocks]")
        for chunk in chunk_list:
            t_dask, img_dask = time_call(
                lambda chunk=chunk: mandelbrot_dask(
                    args.size,
                    args.max_iter,
                    bounds,
                    chunk_rows=chunk,
                    scheduler=args.dask_local_scheduler,
                    num_workers=args.dask_local_workers,
                ),
                repeat=args.repeat,
            )
            ok = bool(np.array_equal(img_dask, img_numpy))
            speedup = t_numpy / t_dask
            dask_local_rows.append(
                {
                    "size": args.size,
                    "max_iter": args.max_iter,
                    "scheduler": args.dask_local_scheduler,
                    "num_workers": args.dask_local_workers,
                    "chunk_rows": chunk,
                    "time_s": round(t_dask, 6),
                    "speedup_vs_numpy": round(speedup, 4),
                    "correct_vs_numpy": ok,
                }
            )
            print(
                f"chunk={chunk:>4} -> {t_dask:>8.4f}s, speedup={speedup:>6.2f}x, correct={ok}"
            )

        write_csv(
            out_dir / "dask_local_chunk_scan.csv",
            dask_local_rows,
            [
                "size",
                "max_iter",
                "scheduler",
                "num_workers",
                "chunk_rows",
                "time_s",
                "speedup_vs_numpy",
                "correct_vs_numpy",
            ],
        )

    # --- Dask cluster ---
    if (not args.skip_dask) and (not args.skip_cluster) and args.scheduler_address:
        dask_cluster_rows = []
        print("\n[Dask cluster chunk sweep]")
        for chunk in chunk_list:
            t_dask_c, img_dask_c = time_call(
                lambda chunk=chunk: mandelbrot_dask(
                    args.size,
                    args.max_iter,
                    bounds,
                    chunk_rows=chunk,
                    scheduler="distributed",
                    scheduler_address=args.scheduler_address,
                ),
                repeat=args.repeat,
            )
            ok = bool(np.array_equal(img_dask_c, img_numpy))
            speedup = t_numpy / t_dask_c
            dask_cluster_rows.append(
                {
                    "size": args.size,
                    "max_iter": args.max_iter,
                    "scheduler_address": args.scheduler_address,
                    "chunk_rows": chunk,
                    "time_s": round(t_dask_c, 6),
                    "speedup_vs_numpy": round(speedup, 4),
                    "correct_vs_numpy": ok,
                }
            )
            print(
                f"cluster chunk={chunk:>4} -> {t_dask_c:>8.4f}s, speedup={speedup:>6.2f}x, correct={ok}"
            )

        write_csv(
            out_dir / "dask_cluster_chunk_scan.csv",
            dask_cluster_rows,
            ["size", "max_iter", "scheduler_address", "chunk_rows", "time_s", "speedup_vs_numpy", "correct_vs_numpy"],
        )

    print(f"\nDone. CSV outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
