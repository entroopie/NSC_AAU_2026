import numpy as np
import time
from numba import jit, njit

def stopwatch(func, *args):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    print(f"{func.__name__}: {elapsed:.6f} seconds")
    return result


def naive(xmin, xmax, ymin, ymax, height, width, max_iter):
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

def vectorized(xmin, xmax, ymin, ymax, height, width, max_iter):
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

@njit
def numba(xmin, xmax, ymin, ymax, height, width, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    mandelbrot_set = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            z = complex(0, 0)
            n = 0
            while n < max_iter and (z.real**2 + z.imag**2) <= 4:
                z = z * z + c
                n += 1
            mandelbrot_set[i, j] = n

    return mandelbrot_set