import random as r
import time
import numpy as np

def ex_5(n, x):
    x = np.array(x)
    start_time = time.time()
    y = np.diff(x)
    end_time = time.time()
    print(f"5 (numpy.diff): {end_time - start_time} seconds")

def ex_4(n, x):
    x = np.array(x)
    y = np.zeros(n-1)
    start_time = time.time()

    x_now = x[0]
    for i in range(n-1):
        x_next = x[i+1]
        y[i] = x_next - x_now
        x_now = x_next

    end_time = time.time()
    print(f"4: {end_time - start_time} seconds")

def ex_3(n, x):
    x = np.array(x)
    y = np.zeros(n-1)

    start_time = time.time()
    for i in range(n-1):
        y[i] = x[i+1] - x[i]

    end_time = time.time()
    print(f"3: {end_time - start_time} seconds")

def ex_2(n, x):
    y = []
    start_time = time.time()

    x_now = x[0]
    for i in range(n-1):
        x_next = x[i+1]
        y.append(x_next - x_now)
        x_now = x_next

    end_time = time.time()
    print(f"2: {end_time - start_time} seconds")

    return None


def ex_1(n, x):
    y = []
    start_time = time.time()

    for i in range(n-1):
        y.append(x[i+1] - x[i])

    end_time = time.time()
    print(f"1: {end_time - start_time} seconds")

    return None

if __name__ == "__main__":
    x = []
    n = int(1e8)
    for i in range(n):   
        x.append(r.randint(0, n))

    ex_1(n, x)
    ex_2(n, x)
    ex_3(n, x)
    ex_4(n, x)
    ex_5(n, x)