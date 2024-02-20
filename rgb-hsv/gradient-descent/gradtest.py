# https://realpython.com/gradient-descent-algorithm-python/
import numpy as np
import pandas as pd

# valor25,valor80,valor154
dataframe = pd.read_csv("/home/ivan/Descargas/csv_test.csv")
dataframe.drop(axis='index',index=1)
array = np.array(dataframe)
print(array)

def gradient_descent(gradient, start, learn_rate, n_iter):
    vector = start.astype(float)  # float64
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        vector += diff
    return vector

def gradient(y):
    return np.array([y])

# # Funció gradient
# def gradient(y):
#     # grad_y1 = 0.1302857995 * y[0] - 0.6172515
#     # grad_y2 = 0.1302857995 * y[1] - 0.45261446
#     grad_y1 = 0.1302857995 
#     grad_y2 = 0.1302857995
#     return np.array([grad_y1, grad_y2])

# # Funció gradient
# def gradient(y):
#     m = len(y)
#     grad_y1 = (1/m) * np.sum(0.1302857995 * y[0] - 0.6172515)
#     grad_y2 = (1/m) * np.sum(0.1302857995 * y[1] - 0.45261446)
#     return np.array([grad_y1, grad_y2])

# Funció gradient
# def gradient(y):
#     m = len(y)
#     grad_y1 = (1/m) * (0.1302857995 * y[0] - 0.6172515)
#     grad_y2 = (1/m) * (0.1302857995 * y[1] - 0.45261446)
#     return np.array([grad_y1, grad_y2])



# Call gradient_descent amb la funció gradient, punt d'inici, lr i nº iteracions
# start = np.array([0, 0])
start = array
learn_rate = 1
n_iter = 250
result = gradient_descent(gradient, start, learn_rate, n_iter)
print(f"Result: {result}")