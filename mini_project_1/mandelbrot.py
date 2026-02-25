from config import Config
from utils.functions import stopwatch ,naive, vectorized, numba
from utils.classic_plot import classic_plot
# from utils.interactive_plot import interactive_plot

import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load configuration
    config = Config()
    params = (config.xmin, config.xmax, config.ymin, config.ymax, config.height, config.width, config.max_iter)

    # From utils.functions
    functions = [naive, vectorized, numba]

    # warm up numba
    numba(*params)

    for func in functions:
        stopwatch(func, *params)





    # Plotting
    # for func in functions:
    #     classic_plot(func(*params), func.__name__, *params[:4])

    # interactive_plot(*params, vectorized) # just for fun