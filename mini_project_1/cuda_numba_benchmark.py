import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

from config import Config
from utils.functions import numba


@cuda.jit
def mandelbrot_cuda_kernel(x, y, max_iter, out):
    """CUDA kernel: compute Mandelbrot iteration counts for each pixel."""
    i, j = cuda.grid(2)
    h = out.shape[0]
    w = out.shape[1]

    # Out-of-bounds guard is mandatory when grid does not divide image dimensions exactly
    if i >= h or j >= w:
        return

    cr = x[j]
    ci = y[i]
    zr = 0.0
    zi = 0.0
    n = 0

    # Same iteration convention as CPU numba() implementation (count starts at 1 for escaped points)
    while n < max_iter and (zr * zr + zi * zi) <= 4.0:
        zr_new = zr * zr - zi * zi + cr
        zi = 2.0 * zr * zi + ci
        zr = zr_new
        n += 1

    out[i, j] = n


def parse_block_list(blocks_str):
    """Parse block list string like '8x8,16x8,16x16,32x8'."""
    blocks = []
    for item in blocks_str.split(','):
        item = item.strip().lower()
        if not item:
            continue
        tx, ty = item.split('x')
        blocks.append((int(tx), int(ty)))
    return blocks


def benchmark_cpu_numba(bounds, size, max_iter):
    xmin, xmax, ymin, ymax = bounds

    # Warm-up JIT
    numba(xmin, xmax, ymin, ymax, 32, 32, max_iter)

    t0 = time.perf_counter()
    out = numba(xmin, xmax, ymin, ymax, size, size, max_iter)
    elapsed = time.perf_counter() - t0
    return elapsed, out


def benchmark_cuda(bounds, size, max_iter, block, repeat):
    xmin, xmax, ymin, ymax = bounds
    h = w = size

    x = np.linspace(xmin, xmax, w, dtype=np.float32)
    y = np.linspace(ymin, ymax, h, dtype=np.float32)

    # --- End-to-end timing (includes PCIe transfers) ---
    e2e_times = []

    # --- Kernel-only timing (GPU compute only) ---
    kernel_times_ms = []

    threads_per_block = block
    blocks_per_grid = (
        (h + threads_per_block[0] - 1) // threads_per_block[0],
        (w + threads_per_block[1] - 1) // threads_per_block[1],
    )

    # Pre-allocate for kernel-only timing
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_out = cuda.device_array((h, w), dtype=np.int32)

    # Warm-up launch (includes first-time JIT compilation)
    mandelbrot_cuda_kernel[blocks_per_grid, threads_per_block](d_x, d_y, max_iter, d_out)
    cuda.synchronize()

    # Kernel-only timing with CUDA events
    for _ in range(repeat):
        start = cuda.event(timing=True)
        end = cuda.event(timing=True)

        start.record()
        mandelbrot_cuda_kernel[blocks_per_grid, threads_per_block](d_x, d_y, max_iter, d_out)
        end.record()
        end.synchronize()

        kernel_times_ms.append(cuda.event_elapsed_time(start, end))

    # End-to-end timing with host clock + synchronize
    out_host = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        d_x_rt = cuda.to_device(x)
        d_y_rt = cuda.to_device(y)
        d_out_rt = cuda.device_array((h, w), dtype=np.int32)

        mandelbrot_cuda_kernel[blocks_per_grid, threads_per_block](d_x_rt, d_y_rt, max_iter, d_out_rt)
        cuda.synchronize()

        out_host = d_out_rt.copy_to_host()
        e2e_times.append(time.perf_counter() - t0)

    return {
        'kernel_ms': float(np.median(kernel_times_ms)),
        'end_to_end_s': float(np.median(e2e_times)),
        'output': out_host,
        'blocks_per_grid': blocks_per_grid,
        'threads_per_block': threads_per_block,
    }


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                'size', 'max_iter',
                'block_x', 'block_y', 'threads_per_block_total', 'warp_multiple',
                'grid_x', 'grid_y',
                'kernel_ms', 'end_to_end_s',
                'cpu_numba_s', 'speedup_vs_cpu_kernel_only', 'speedup_vs_cpu_end_to_end',
                'correct_vs_cpu_numba',
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_results(rows, out_png):
    labels = [f"{r['block_x']}x{r['block_y']}" for r in rows]
    kernel_ms = [float(r['kernel_ms']) for r in rows]
    e2e_s = [float(r['end_to_end_s']) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(labels, kernel_ms, color='#5b8dd9')
    axes[0].set_title('CUDA kernel time (lower is better)')
    axes[0].set_ylabel('ms')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(labels, e2e_s, color='#57b26a')
    axes[1].set_title('CUDA end-to-end time (transfer included)')
    axes[1].set_ylabel('s')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle('Mandelbrot CUDA block-size sweep', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='CUDA Mandelbrot block-size sweep with Numba @cuda.jit')
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--max-iter', type=int, default=256)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--blocks', type=str, default='8x8,10x10,16x8,16x16,32x8,32x16')
    parser.add_argument('--out-dir', type=str, default='benchmark_outputs')
    args = parser.parse_args()

    if not cuda.is_available():
        raise RuntimeError(
            'CUDA is not available. This script requires an NVIDIA GPU + CUDA driver/runtime.'
        )

    cfg = Config()
    bounds = (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax)

    print(f'Running CPU Numba baseline: size={args.size}, max_iter={args.max_iter}')
    cpu_s, cpu_out = benchmark_cpu_numba(bounds, args.size, args.max_iter)
    print(f'CPU numba time: {cpu_s:.4f}s')

    blocks = parse_block_list(args.blocks)
    rows = []

    for block in blocks:
        tx, ty = block
        tpb_total = tx * ty

        # Skip invalid block sizes
        if tpb_total > 1024:
            print(f'Skipping {tx}x{ty} (>{1024} threads per block)')
            continue

        result = benchmark_cuda(bounds, args.size, args.max_iter, block, repeat=args.repeat)
        correct = bool(np.array_equal(result['output'], cpu_out))

        row = {
            'size': args.size,
            'max_iter': args.max_iter,
            'block_x': tx,
            'block_y': ty,
            'threads_per_block_total': tpb_total,
            'warp_multiple': (tpb_total % 32 == 0),
            'grid_x': result['blocks_per_grid'][0],
            'grid_y': result['blocks_per_grid'][1],
            'kernel_ms': round(result['kernel_ms'], 4),
            'end_to_end_s': round(result['end_to_end_s'], 6),
            'cpu_numba_s': round(cpu_s, 6),
            'speedup_vs_cpu_kernel_only': round(cpu_s / (result['kernel_ms'] / 1000.0), 4),
            'speedup_vs_cpu_end_to_end': round(cpu_s / result['end_to_end_s'], 4),
            'correct_vs_cpu_numba': correct,
        }
        rows.append(row)

        print(
            f"block={tx}x{ty} (threads={tpb_total}, warp-multiple={row['warp_multiple']}) "
            f"kernel={row['kernel_ms']} ms e2e={row['end_to_end_s']} s "
            f"speedup(kernel)={row['speedup_vs_cpu_kernel_only']}x "
            f"speedup(e2e)={row['speedup_vs_cpu_end_to_end']}x correct={correct}"
        )

    out_dir = Path(args.out_dir)
    csv_path = out_dir / 'cuda_blocksize_scan.csv'
    png_path = out_dir / 'cuda_blocksize_scan.png'

    write_csv(rows, csv_path)
    plot_results(rows, png_path)

    best_kernel = min(rows, key=lambda r: float(r['kernel_ms']))
    print('\nBest block size by kernel time:')
    print(
        f"{best_kernel['block_x']}x{best_kernel['block_y']} "
        f"({best_kernel['threads_per_block_total']} threads, "
        f"warp-multiple={best_kernel['warp_multiple']}) -> {best_kernel['kernel_ms']} ms"
    )
    print(f'CSV saved: {csv_path}')
    print(f'Plot saved: {png_path}')


if __name__ == '__main__':
    main()
