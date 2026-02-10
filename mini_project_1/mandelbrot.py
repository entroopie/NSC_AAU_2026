import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt

def mandelbrot(xmin, xmax, ymin, ymax, height, width, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    mandelbrot_set = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            z = 0
            n = 0
            while n < max_iter and abs(z) <= 2:
                z = z*z + c
                n += 1
            mandelbrot_set[i, j] = n

    image = mandelbrot_set
    plt.imshow(image, cmap='magma', extent=[xmin, xmax, ymin, ymax])
    plt.colorbar()
    plt.title("Mandelbrot Set")
    plt.show()


if __name__ == '__main__':
    max_iter = 100
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    height, width = 1024, 1024  

    mandelbrot(xmin, xmax, ymin, ymax, height, width, max_iter)