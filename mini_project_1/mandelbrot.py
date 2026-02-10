import numpy as np
import time
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt

def mandelbrot_naive(xmin, xmax, ymin, ymax, height, width, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    mandelbrot_set = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            z = 0
            n = 0
            while n < max_iter and abs(z) <= 2:
                z = z ** 2 + c
                n += 1
            mandelbrot_set[i, j] = n

    return mandelbrot_set

def mandelbrot_vectorized(xmin, xmax, ymin, ymax, height, width, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    X, Y = np.meshgrid(x, y)
    C = X + Y*1j #fancy complex matrix (in python 1j represents imaginary unit)

    Z = np.zeros(C.shape, dtype=complex)
    
    mandelbrot_set = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)

    for n in range(max_iter):
        Z[mask] = Z[mask]**2 + C[mask]
        escape = mask & (np.abs(Z) > 2)
        mandelbrot_set[escape] = n
        mask[escape] = False

    mandelbrot_set[mask] = max_iter #set to max_iter to make not escaped points bright
    return mandelbrot_set
    

def stopwatch(func, *args):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    print(f"{func.__name__}: {elapsed:.4f} seconds")
    return result


if __name__ == '__main__':
    max_iter = 100
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    height, width = 1024, 1024  

    params = (xmin, xmax, ymin, ymax, height, width, max_iter)

    naive_set = stopwatch(mandelbrot_naive, *params)
    vectorized_set = stopwatch(mandelbrot_vectorized, *params)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(naive_set, cmap='magma', extent=[xmin, xmax, ymin, ymax])
    axes[0].set_title("Naive")

    axes[1].imshow(vectorized_set, cmap='magma', extent=[xmin, xmax, ymin, ymax])
    axes[1].set_title("Vectorized")

    plt.tight_layout()
    plt.show()