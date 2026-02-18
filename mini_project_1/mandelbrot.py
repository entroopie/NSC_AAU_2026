import numpy as np
import time
import matplotlib

from interactive_plot import interactive_plot
import matplotlib.pyplot as plt


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
    

def stopwatch(func, *args):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    print(f"{func.__name__}: {elapsed:.6f} seconds")
    return result

def classic_plot(set, func_name):
    plt.imshow(set, cmap='magma', extent=[xmin, xmax, ymin, ymax])
    plt.colorbar()
    plt.title(f"Mandelbrot Set + {func_name}")
    plt.show()

if __name__ == '__main__':
    max_iter = 100
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    height, width = 512, 512  

    params = (xmin, xmax, ymin, ymax, height, width, max_iter)

    naive_set = stopwatch(naive, *params)
    vectorized_set = stopwatch(vectorized, *params)
    # numby_set = stopwatch(None)


    # classic_plot(naive_set, "naive")
    # classic_plot(vectorized_set, "vectorized")

    # interactive_plot(*params, vectorized) # just for fun