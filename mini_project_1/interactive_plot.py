import numpy as np
import time
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt

def find_boundary(compute_func, xmin, xmax, ymin, ymax, max_iter):
    """Quick low-res sample to find the most detailed sub-region (highest variance)."""
    sample_size = 64
    sample = compute_func(xmin, xmax, ymin, ymax, sample_size, sample_size, max_iter)

    x_coords = np.linspace(xmin, xmax, sample_size)
    y_coords = np.linspace(ymin, ymax, sample_size)

    block = 16
    step = 4
    best_var = -1
    best_cx = (xmin + xmax) / 2
    best_cy = (ymin + ymax) / 2

    for i in range(0, sample_size - block, step):
        for j in range(0, sample_size - block, step):
            region = sample[i:i + block, j:j + block]
            v = np.var(region)
            if v > best_var:
                best_var = v
                best_cx = x_coords[j + block // 2]
                best_cy = y_coords[i + block // 2]

    return best_cx, best_cy


def interactive_plot(xmin, xmax, ymin, ymax, height, width, max_iter, compute_func):
    fig, ax = plt.subplots()

    original_x_span = xmax - xmin

    def draw(xmin, xmax, ymin, ymax):
        zoom_level = original_x_span / (xmax - xmin)
        current_max_iter = int(max_iter + 150 * np.log2(max(zoom_level, 1)))
        mandelbrot_set = compute_func(xmin, xmax, ymin, ymax, height, width, current_max_iter)
        ax.clear()
        ax.imshow(mandelbrot_set, cmap='magma', extent=[xmin, xmax, ymin, ymax])
        ax.set_title(f"Zoom: {zoom_level:.1f}x | max_iter: {current_max_iter}")
        fig.canvas.draw()

    # Store current bounds in a mutable container so the callback can update them
    bounds = [xmin, xmax, ymin, ymax]

    def on_click(event):
        if event.inaxes != ax:
            return

        cx, cy = event.xdata, event.ydata  # clicked coordinates in data space
        zoom_factor = 0.5  # each click halves the visible range

        # Current span
        x_span = (bounds[1] - bounds[0]) * zoom_factor
        y_span = (bounds[3] - bounds[2]) * zoom_factor

        # Tentative new bounds centered on click
        new_xmin = cx - x_span / 2
        new_xmax = cx + x_span / 2
        new_ymin = cy - y_span / 2
        new_ymax = cy + y_span / 2

        # Auto-seek the boundary within the zoomed region
        zoom_level = original_x_span / x_span
        search_iter = int(max_iter + 150 * np.log2(max(zoom_level, 1)))
        bcx, bcy = find_boundary(compute_func, new_xmin, new_xmax, new_ymin, new_ymax, search_iter)

        # Re-center on the most interesting point
        bounds[0] = bcx - x_span / 2
        bounds[1] = bcx + x_span / 2
        bounds[2] = bcy - y_span / 2
        bounds[3] = bcy + y_span / 2

        draw(*bounds)

    fig.canvas.mpl_connect('button_press_event', on_click)
    draw(*bounds)
    plt.show()