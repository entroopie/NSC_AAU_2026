from matplotlib import pyplot as plt

def classic_plot(set, func_name, xmin, xmax, ymin, ymax):
    plt.imshow(set, cmap='magma', extent=[xmin, xmax, ymin, ymax])
    plt.colorbar()
    plt.title(f"Mandelbrot Set ({func_name})")
    plt.show()